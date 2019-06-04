import utility.visualize as visualize
from robot.vrep_robot import VrepRobot
from vision.tracker import Tracker
from settings import Settings
from utility.util_functions import vrep_ports, timeit, save_fitness_moea, euclidean_distance
from utility.visualize import plot_single_run
from evolution.eval_genomes import \
    eval_genomes_simulation, \
    eval_genomes_hardware, \
    eval_genome, \
    eval_genome_hardware
from subprocess import Popen
from functools import partial
from neat import ParallelEvaluator
from keras.models import Sequential
from keras.layers import Dense
from keras.initializers import normal
from keras.utils import plot_model
from deap import base, creator, tools, algorithms
import numpy as np
import vrep.vrep as vrep
import neat
import pickle
import warnings
import os
import time
import sys
import random
try:
    from robot.evolved_robot import EvolvedRobot
except ImportError as error:
    print(error.__class__.__name__ + ": " + 'DBus works only on linux!')
try:
    # pylint: disable=import-error
    import Queue as queue
except ImportError:
    # pylint: disable=import-error
    import queue

try:
    import threading
except ImportError:  # pragma: no cover
    import dummy_threading as threading
    HAVE_THREADS = False
else:
    HAVE_THREADS = True


class ThreadedEvolution(object):
    """
    A threaded genome evaluator.
    Useful on python implementations without GIL (Global Interpreter Lock).
    """

    def __init__(self, clients, settings, num_workers, eval_function):
        """
        eval_function should take two arguments (a genome object and the
        configuration) and return a single float (the genome's fitness).
        """
        self.num_workers = num_workers
        self.eval_function = eval_function
        self.workers = []
        self.working = False
        self.clients = clients
        self.settings = settings
        self.inqueue = queue.Queue()
        self.outqueue = queue.Queue()

        if not HAVE_THREADS:  # pragma: no cover
            warnings.warn(
                "No threads available; use ParallelEvaluator, not ThreadedEvaluator")

    def __del__(self):
        """
        Called on deletion of the object. We stop our workers here.
        WARNING: __del__ may not always work!
        Please stop the threads explicitly by calling self.stop()!
        TODO: ensure that there are no reference-cycles.
        """
        if self.working:
            self.stop()

    @timeit
    def start(self):
        """Starts the worker threads each connected to specific vrep server"""
        if self.working:
            return
        self.working = True
        for i in range(self.num_workers):
            w = threading.Thread(
                name="Worker Thread #{i}".format(i=i),
                target=self._worker,
                args=(self.clients[i], self.settings,),
            )
            w.daemon = True
            w.start()
            print("{0} client_id = {1}".format(
                w.getName(), self.clients[i]))
            self.workers.append(w)

    def stop(self):
        """Stops the worker threads and waits for them to finish"""
        self.working = False
        for w in self.workers:
            w.join()
        self.workers = []

    def _worker(self, client_id, settings):
        """The worker function"""
        while self.working:
            try:
                genome_id, genome, config = self.inqueue.get(
                    block=True,
                    timeout=0.2,
                )
            except queue.Empty:
                continue
            f = self.eval_function(client_id, settings,
                                   genome_id, genome, config)
            self.outqueue.put((genome_id, genome, f))
            self.inqueue.task_done()

    @timeit
    def evaluate(self, genomes, config):
        """Evaluate the genomes"""
        if not self.working:
            self.start()
        p = 0
        for genome_id, genome in genomes:
            p += 1
            self.inqueue.put((genome_id, genome, config))

        self.inqueue.join()
        # assign the fitness back to each genome
        while p > 0:
            p -= 1
            genome_id, genome, fitness = self.outqueue.get()
            assert genome_id == genome.key
            genome.fitness = fitness


class Simulation(object):

    def __init__(self,
                 settings,
                 config_file,
                 eval_function,
                 simulation_type,
                 threaded=False,
                 checkpoint=None,
                 genome_path=None,
                 headless=False,
                 multiobjective=False,
                 n_layers=1,
                 input_dim=7,
                 neurons=5
                 ):

        self.config_file = config_file
        self.threaded = threaded
        self.settings = settings
        self.simulation_type = simulation_type
        self.eval_function = eval_function
        self.parallel_eval = None
        self.threaded_eval = None
        self.checkpoint = checkpoint
        self.genome_path = genome_path
        self.headless = headless
        self.vrep_scene = None
        self.multiobjective = multiobjective
        self.n_layers = n_layers
        self.input_dim = input_dim
        self.neurons = neurons
        self.disparity = Disparity()
        self._init_vrep()
        self._init_network()
        self._init_agent()
        self._init_genome()
        # self._init_vision()

    def _init_vrep(self):
        """initialize vrep simulator"""
        vrep.simxFinish(-1)
        self.fnull = open(os.devnull, 'w')

        if self.threaded:
            self.ports = vrep_ports()
        elif self.simulation_type == 'transferability':
            self.ports = vrep_ports()[:2]
        else:
            self.ports = [vrep_ports()[0]]

        if self.headless:
            h = '-h'
        else:
            h = ''

        if self.simulation_type == 'thymio':
            self.vrep_scene = os.getcwd() + '/scenes/thymio_hw.ttt'

        elif self.simulation_type == 'transferability':
            self.vrep_bot_scene = os.getcwd() + '/scenes/thymio_v_infrared.ttt'
            self.thymio_bot_scene = os.getcwd() + '/scenes/thymio_hw.ttt'
            self.scenes = [self.vrep_bot_scene, self.thymio_bot_scene]
        else:
            self.vrep_scene = os.getcwd() + '/scenes/thymio_v_infrared.ttt'

        # if not self.genome_path and self.simulation_type != 'transferability':
        #     self.vrep_servers = [Popen(
        #         ['{0} {1} -gREMOTEAPISERVERSERVICE_{2}_TRUE_TRUE {3}'
        #             .format(self.settings.vrep_abspath, h, port, self.vrep_scene)],
        #         shell=True, stdout=self.fnull) for port in self.ports]
        #     time.sleep(5)

        if self.simulation_type == 'transferability':
            self.vrep_servers = [Popen(
                ['{0} {1} -gREMOTEAPISERVERSERVICE_{2}_TRUE_TRUE {3}'
                 .format(self.settings.vrep_abspath, h, port, self.scenes[scene])],
                shell=True, stdout=self.fnull) for scene, port in enumerate(self.ports)]
            time.sleep(5)

        self.clients = [vrep.simxStart(
            '127.0.0.1',
            port,
            True,
            True,
            5000,
            5) for port in self.ports]

        if not all(c >= 0 for c in self.clients):
            sys.exit('Some clients were not correctly initialized!')

        if len(self.clients) == 1:
            self.settings.client_id = self.clients[0]

        self.client_initialized = True

    def _init_network(self):
        """initialize NEAT"""
        # load the confifuration file
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  self.config_file)
        if self.settings.save_data and not self.multiobjective:
            self.config.save(self.settings.path + 'config.ini')

        if self.checkpoint:
            # restore population from a checkpoint
            self.restored_population = neat.Checkpointer.restore_checkpoint(
                self.checkpoint)
            self.population = self.restored_population
        else:
            # initialize the network and population
            self.population = neat.Population(self.config)
        # Add a stdout reporter to show progress in the terminal.
        self.stats = neat.StatisticsReporter()
        self.population.add_reporter(neat.StdOutReporter(True))
        self.population.add_reporter(self.stats)
        self.population.add_reporter(neat.Checkpointer(1))
        self.network_initialized = True

    def _init_agent(self):
        if self.simulation_type == 'vrep':
            self.individual = VrepRobot(
                client_id=self.settings.client_id,
                id=None,
                op_mode=self.settings.op_mode,
                robot_type=self.settings.robot_type
            )
        elif self.simulation_type == 'thymio':
            self.individual = EvolvedRobot(
                'thymio-II',
                client_id=self.settings.client_id,
                id=None,
                op_mode=self.settings.op_mode,
                chromosome=None,
                robot_type=self.settings.robot_type
            )
        elif self.simulation_type == 'transferability':
            self.vrep_bot = VrepRobot(
                client_id=self.clients[0],
                id=None,
                op_mode=self.settings.op_mode,
                robot_type=self.settings.robot_type
            )
            self.thymio_bot = EvolvedRobot(
                'thymio-II',
                client_id=self.clients[1],
                id=None,
                op_mode=self.settings.op_mode,
                chromosome=None,
                robot_type=self.settings.robot_type
            )
        else:
            self.individual = None
            return
        self.agent_initialized = True

    def _init_genome(self):
        if self.genome_path:
            with open(self.genome_path, 'rb') as f:
                genome = pickle.load(f)
            self.winner = genome
            print(genome)
        return

    def _init_vision(self):
        if self.simulation_type == 'thymio' or self.simulation_type == 'transferability':
            self.vision_thread = Tracker(mid=5,
                                         transform=None,
                                         mid_aux=0,
                                         video_source=-1,
                                         capture=False,
                                         show=True,
                                         debug=False,
                                         )

            self.vision_thread.start()

            while self.vision_thread.cornersDetected is not True:
                time.sleep(2)

            self.vision_initialized = True
        return

    def _stop_vrep(self):
        # stop vrep simulations
        _ = [vrep.simxFinish(client) for client in self.clients]

    def _kill_vrep(self):
        # kill vrep server instances
        _ = [server.kill() for server in self.vrep_servers]

    def build_model(self, n_layers, input_dim, neurons, activation='sigmoid', initializer=None):
        if isinstance(neurons, list):
            assert len(neurons) == n_layers
        else:
            neurons = [neurons] * n_layers

        if initializer is None:
            # Uses normal initializer
            initializer = normal(mean=0, stddev=0.1, seed=13)

        model = Sequential()

        # Adds first hidden layer with input_dim parameter
        model.add(Dense(units=neurons[0],
                        input_shape=(input_dim,),
                        activation=activation,
                        kernel_initializer=initializer,
                        name='hidden_layer'))

        # Adds output layer
        model.add(Dense(units=2, activation=activation,
                        kernel_initializer=initializer, name='net_output'))

        # Compiles the model
        model.compile(loss='mean_squared_error', optimizer='Adam',
                      metrics=['mean_squared_error'])

        return model

    @timeit
    def start(self, simulation):
        default = None
        return getattr(self, simulation, lambda: default)()

    def stop(self):
        if self.threaded_eval:
            self.threaded_eval.stop()
        if self.parallel_eval:
            self.parallel_eval.__del__()
        self._stop_vrep()
        time.sleep(5)
        self._kill_vrep()

    @timeit
    def simulation(self):
        # run simulation in vrep
        try:
            self.winner = self.population.run(partial(self.eval_function,
                                                      self.individual,
                                                      self.settings),
                                              self.settings.n_gen)
        except neat.CompleteExtinctionException() as ex:
            print("Extinction: {0}".format(ex))

        return self.config, self.stats, self.winner

    @timeit
    def simulation_transferability(self):
        # run simulation in vrep
        try:
            self.winner = self.population.run(partial(self.eval_function,
                                                      self.vrep_bot,
                                                      self.thymio_bot,
                                                      self.settings),
                                              self.settings.n_gen)
        except neat.CompleteExtinctionException() as ex:
            print("Extinction: {0}".format(ex))

        return self.config, self.stats, self.winner

    @timeit
    def simulation_threaded(self):
        """run simulation using threads in vrep"""
        # Run for up to N generations.
        t = threading.currentThread()
        print('Main Thread: {}'.format(str(t.getName())))

        self.threaded_eval = ThreadedEvolution(self.clients, self.settings, len(
            self.clients), self.eval_function)
        try:
            self.winner = self.population.run(
                self.threaded_eval.evaluate, self.settings.n_gen)
        except neat.CompleteExtinctionException() as ex:
            print("Extinction: {0}".format(ex))

        self.stop()
        return self.config, self.stats, self.winner

    def simulation_multiobjective(self):
        """Multiobjective optmization 2. Genome is represented as
        an (concatenated) array of connections weights for each layer in the NN.
        Weights are concatenated into a single list of floating points.
        The weights than are reshaped and adjusted in the eval_function in
        order to update the model weights correctly.
        """
        model = self.build_model(self.n_layers, self.input_dim, self.neurons)

        def init_individual_ndim(cls, model):
            """
            Genome is represented as
            ndarrays of connections weights for each layer in the NN.
            [
                2d_array(layer_1),
                1d_array(bias_hidden_layer),
                2d_array(layer2),
                1d_array(bias_output_layer)
            ]
            """
            ind = cls([np.random.permutation(w.flat).reshape(
                w.shape) for w in model.get_weights()])

            ind.weights_shape = [tuple(weights.shape)
                                 for weights in model.get_weights()]
            ind.features = None
            ind.weights = None
            ind.key = None

            return ind

        def init_individual(cls, model):
            """Concatenated weights of the Keras model
            Weights are concatenated into a single list of floating points.
            The weights than are reshaped and adjusted in the eval_function in
            order to update the model weights correctly.
            """
            ind = cls(np.concatenate(tuple(weight.flatten()
                                           for weight in model.get_weights())).tolist())
            ind.weights_shape = [tuple(weights.shape)
                                 for weights in model.get_weights()]
            ind.features = None
            ind.weights = None
            ind.str_disparity = None
            ind.diversity = None
            ind.task_fitness = None
            ind.evaluation = None
            ind.position = None
            ind.gen = None
            ind.key = None

            return ind

        def mutate_individual(individual, indpb):
            """Mutation for the init_individual_ndim individual"""
            for i, weight in enumerate(individual):
                w, = tools.mutFlipBit(weight.flatten().tolist(), indpb=indpb)
                individual[i] = np.array(w).reshape(weight.shape)
            return individual

        def mate_individuals(ind1, ind2):
            """Crossover for the init_individual_ndim individual"""
            for i, (w1, w2) in enumerate(zip(ind1, ind2)):
                cx1, cx2 = tools.cxTwoPoint(
                    w1.flatten().tolist(), w2.flatten().tolist())
                ind1[i], ind2[i] = np.array(cx1).reshape(
                    w1.shape), np.array(cx2).reshape(w2.shape)
            return ind1, ind2

        def eq(ind1, ind2):
            """Required for the HallOfFame and ParetoFront. Comparison of the 2
            layers. Bias layer is ommited. For the init_individual_ndim individual."""
            return np.array_equal(ind1[0], ind2[0]) and np.array_equal(ind1[2], ind2[2])

        # Creating the appropriate type of the problem
        creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0, 1.0))
        creator.create("Individual", list,
                       fitness=creator.FitnessMax, model=None)

        toolbox = base.Toolbox()
        history = tools.History()

        toolbox.register("individual", init_individual,
                         creator.Individual, model=model)
        # register the crossover operator
        toolbox.register('mate', tools.cxTwoPoint)
        # register the mutation operator
        toolbox.register('mutate', tools.mutFlipBit, indpb=0.5)
        # register the evaluation function
        if self.simulation_type == 'transferability':
            toolbox.register('evaluate', partial(
                self.eval_function, self.vrep_bot, self.settings, model))
        else:
            toolbox.register('evaluate', partial(
                self.eval_function, self.individual, self.settings, model))
        # register NSGA-II multiobjective optimization algorithm
        toolbox.register("select", tools.selNSGA2)
        # instantiate the population
        toolbox.register('population', tools.initRepeat,
                         list, toolbox.individual)
        # maintain stats of the evolution
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register('avg', np.mean)
        stats.register('std', np.std)
        stats.register('min', np.min)
        stats.register('max', np.max)

        logbook = tools.Logbook()
        logbook.header = "gen", "evals", "std", "min", "avg", "max"

        best_inds, best_inds_fitness = np.array([]), np.array([])

        # Decorate the variation operators
        # toolbox.decorate("mate", history.decorator)
        # toolbox.decorate("mutate", history.decorator)

        # create an initial population of N individuals
        pop = toolbox.population(n=self.settings.pop)
        history.update(pop)

        # object that contain the best individuals
        # hof = tools.ParetoFront(eq)
        hof = tools.ParetoFront()

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in pop if not ind.fitness.valid]

        # Assigng ids to individuals
        UNIQ_ID = 1
        for ind in invalid_ind:
            ind.key = UNIQ_ID
            ind.gen = 0
            UNIQ_ID += 1

        if self.simulation_type == 'transferability':
            # take the first individual c0 and evaluate in simulation
            c0 = toolbox.clone(invalid_ind[0])
            _ = toolbox.evaluate(c0)

            # clone the evaluated indvidual c0
            controller_0 = toolbox.clone(c0)

            del (
                controller_0.features,
                controller_0.task_fitness,
                controller_0.evaluation,
                controller_0.position
            )

            # transfer controller c0 to thymio
            _ = eval_genome_hardware(
                self.thymio_bot, self.settings, controller_0, model)
            # Add the controller c0 to the transfered controllers set
            self.disparity.add(controller_0, c0)
            # compute the surrogate model
            self.disparity.comptute(c0)

        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            if self.simulation_type == 'transferability':
                diversity = self.disparity.diversity(ind)
                ind.fitness.values = (
                    fit, self.disparity.disparity_value, diversity)
                print('fitness: {0} disparity: {1} diversity: {2}'.format(
                    fit, self.disparity.disparity_value, diversity))
            else:
                ind.fitness.values = (fit, 0.0, 1.0)

            # Save Deap Individual
            with open(self.settings.path + 'deap_inds/' + str(ind.key) + "_genome_.pkl", "wb") as ind_file:
                pickle.dump(ind, ind_file)

        # save the fitness of the initial population
        save_fitness_moea(invalid_ind, 0, self.settings.path)

        # This is just to assign the crowding distance to the individuals
        # no actual selection is done
        pop = toolbox.select(pop, len(pop))
        record = stats.compile(pop)
        logbook.record(gen=0, evals=len(invalid_ind), **record)
        print(logbook.stream)
        hof.update(pop)

        # Begin the generational process
        for gen in range(1, self.settings.n_gen+1):
            # Vary the population & crete the offspring
            offspring = tools.selTournamentDCD(pop, len(pop))
            offspring = [toolbox.clone(ind) for ind in offspring]

            for ind1, ind2 in zip(offspring[::2], offspring[1::2]):
                if random.random() <= self.settings.CXPB:
                    toolbox.mate(ind1, ind2)

                toolbox.mutate(ind1)
                toolbox.mutate(ind2)

                del (
                    ind1.fitness.values,
                    ind1.features, ind1.key,
                    ind1.weights, ind1.task_fitness,
                    ind1.evaluation, ind1.diversity
                )

                del (
                    ind2.fitness.values,
                    ind2.features, ind2.key,
                    ind2.weights, ind2.task_fitness,
                    ind2.evaluation, ind2.diversity
                )

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

            # Assigng ids to individuals
            for ind in invalid_ind:
                ind.key = UNIQ_ID
                ind.gen = gen
                UNIQ_ID += 1

            # Calculate the fitness & assigned it
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                if self.simulation_type == 'transferability':
                    diversity = self.disparity.diversity(ind)
                    ind.fitness.values = (
                        fit, self.disparity.disparity_value, diversity)
                    print('fitness: {0} disparity: {1} diversity: {2}'.format(
                        fit, self.disparity.disparity_value, diversity))
                else:
                    ind.fitness.values = (fit, 0.0, 1.0)

                # Save Deap Individual
                with open(self.settings.path + 'deap_inds/' + str(ind.key) + "_genome_.pkl", "wb") as ind_file:
                    pickle.dump(ind, ind_file)

            # transfer controllers to reality based on diversity threshold
            if self.simulation_type == 'transferability':
                # filter controllers that we transfer to thymio
                transfer_simulation = list(
                    filter(lambda x: x.diversity > self.settings.STR, invalid_ind))
                # clone simulation controllers
                transfered_controllers = list(
                    map(lambda x: toolbox.clone(x), transfer_simulation))

                for sim, trans in zip(transfer_simulation, transfered_controllers):
                    del trans.features, trans.task_fitness, trans.diversity, trans.evaluation
                    # evaluation on the thymio
                    _ = eval_genome_hardware(
                        self.thymio_bot, self.settings, trans, model)
                    # add the controller to the transfered_set and compute D*(controller)
                    self.disparity.add(trans, sim)
                    # update surrogate Model
                    self.disparity.comptute(sim)

            # save the fitness of the population
            save_fitness_moea(invalid_ind, gen, self.settings.path)

            # add the best individual for each generation
            best_ind = tools.selBest(pop, 1)[0]
            best_inds = np.append(best_inds, best_ind)
            best_inds_fitness = np.append(
                best_inds_fitness, best_ind.fitness.values)

            # Select the next generation population
            pop = toolbox.select(pop + offspring, len(offspring))
            record = stats.compile(pop)
            logbook.record(gen=gen, evals=len(invalid_ind), **record)
            print(logbook.stream)
            hof.update(pop)

        # dump transfered genemos
        self.disparity.dump(self.settings.path)

        # log Statistics
        with open(self.settings.path + 'ea_fitness.dat', 'w') as s:
            s.write(logbook.__str__())

        # best individuals each generation
        with open(self.settings.path + 'best_genomes.pkl', 'wb') as fp:
            pickle.dump(best_inds, fp)

        # save the best individual
        with open(self.settings.path + 'winner_{0}.pkl'.format(hof[0].key), 'wb') as winner:
            pickle.dump(hof[0], winner)

        # Evolution records as a chronological list of dictionaries
        gen = logbook.select('gen')
        fit_mins = logbook.select('min')
        fit_avgs = logbook.select('avg')
        fit_maxs = logbook.select('max')

        plot_single_run(
            gen,
            fit_mins,
            fit_avgs,
            fit_maxs,
            ratio=0.35,
            save=self.settings.path + 'evolved-obstacle.pdf')

        return pop, hof, logbook, best_inds, best_inds_fitness

    @timeit
    def restore_genome(self, N=2):
        """restore genome and re-run simulation"""

        if N == 1:

            if self.simulation_type == 'thymio':
                _ = eval_genome_hardware(
                    self.individual, self.settings, self.winner, model=None, config=self.config, generation=-1)
            else:
                self.winner = self.eval_function(
                    self.individual, self.settings, [(self.winner.key, self.winner)], self.config)
        else:
            if not os.path.exists('./data/neat/restored_genomes/'):
                os.makedirs('./data/neat/restored_genomes/')
            
            self.settings.path = './data/neat/restored_genomes/'
            toolbox = base.Toolbox()
            # genomes = [toolbox.clone(self.winner) for _ in range(0, N)]
            for i in range(0, N):
                genome = toolbox.clone(self.winner)

                del (
                    genome.features,
                    genome.fitness,
                    # genome.position
                )
                _ = eval_genome_hardware(
                    self.individual, self.settings, genome, model=None, config=self.config, generation=-1)

                result = np.concatenate(([genome.key], [genome.fitness], genome.features))

                with open(self.settings.path + 'restored_genome_{}_fitness.txt'.format(genome.key), 'a') as w:
                    np.savetxt(w, (result,), delimiter=',', fmt='%s')

                with open(self.settings.path + '{}_{}_restored_genome_.pkl'.format(genome.key, i), 'wb') as ind_file:
                    pickle.dump(genome, ind_file)
        return

    def post_eval(self):
        """post evalution of genome"""
        self.individual = self.eval_function(
            self.individual, self.settings, self.winner, self.config)
        return self.individual

    def log_statistics(self):
        """log results and save best/winner genomes"""
        # Write run statistics to file.
        self.stats.save_genome_fitness(
            filename=self.settings.path+'fitnesss_history.csv')
        self.stats.save_species_count(
            filename=self.settings.path+'speciation.csv')
        self.stats.save_species_fitness(
            filename=self.settings.path+'species_fitness.csv')
        # log the winner network
        with open(self.settings.path + 'winner_network.txt', 'w') as s:
            s.write('\nBest genome:\n{!s}'.format(self.winner))
            s.write('\nBest genomes:\n{!s}'.format(
                print(self.stats.best_genomes(5))))
        # Save the winner.
        with open(self.settings.path + 'winner_genome', 'wb') as f:
            pickle.dump(self.winner, f)
        # save the 10 best genomes
        for i, best in enumerate(self.stats.best_genomes(10)):
            with open(self.settings.path + 'best_genome_{0}'.format(i), 'wb') as g:
                pickle.dump(best, g)

    def visualize_results(self):
        """Visualize network topology, species, results"""
        node_names = {-1: 'A', -2: 'B', -3: 'C', -4: 'D', -5: 'E',
                      -6: 'F', -7: 'G', 0: 'LEFT', 1: 'RIGHT', }

        visualize.draw_net(self.config, self.winner, True, node_names=node_names,
                           filename=self.settings.path+'network')

        visualize.plot_stats(self.stats, ylog=False, view=False,
                             filename=self.settings.path+'feedforward-fitness.svg')

        visualize.plot_species(
            self.stats, view=False, filename=self.settings.path+'feedforward-speciation.svg')

        visualize.draw_net(self.config, self.winner, view=False, node_names=node_names,
                           filename=self.settings.path+'winner-feedforward.gv')

        visualize.draw_net(self.config, self.winner, view=False, node_names=node_names,
                           filename=self.settings.path+'winner-feedforward-enabled.gv', show_disabled=False)

        visualize.draw_net(self.config, self.winner, view=False, node_names=node_names,
                           filename=self.settings.path+'winner-feedforward-enabled-pruned.gv', show_disabled=False, prune_unused=False)


class Disparity(object):

    def __init__(self):
        self.transfered_set = []
        self.disparity_value = 0.0

    def add(self, transfer_controller, controller_sim):
        transfer_controller.str_disparity = euclidean_distance(
            controller_sim.position, transfer_controller.position)
        self.transfered_set.append(transfer_controller)

    def comptute(self, controller):
        numerator = np.sum([(ind.str_disparity) * np.power(euclidean_distance(
            ind.features, controller.features), -2) for ind in self.transfered_set])
        denominator = np.sum([np.power(euclidean_distance(
            ind.features, controller.features), -2) for ind in self.transfered_set])

        self.disparity_value = numerator / denominator

    def diversity(self, controller):
        diversity = np.amin([euclidean_distance(
            t.features, controller.features) for t in self.transfered_set])
        controller.diversity = diversity
        return diversity

    def dump(self, path):
        """dump transfered controllers"""
        for ind in self.transfered_set:
            with open(path + 'deap_inds/' + str(ind.key) + "_transformed_genome_.pkl", "wb") as ind_file:
                pickle.dump(ind, ind_file)
