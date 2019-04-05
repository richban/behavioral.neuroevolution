import utility.visualize as visualize
from robot.vrep_robot import VrepRobot
from vision.tracker import Tracker
from settings import Settings
from utility.util_functions import vrep_ports, timeit
from evolution.eval_genomes import \
    eval_genomes_simulation, \
    eval_genomes_hardware, \
    eval_genome
from subprocess import Popen
from functools import partial
import vrep.vrep as vrep
import neat
import pickle
import warnings
import os
import time
import sys
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


class ParrallelEvolution(object):
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
            self.inqueue.task_done()
            self.outqueue.put((genome_id, genome, f))

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
            _, genome, fitness = self.outqueue.get()
            genome.fitness = fitness


class Simulation(object):

    def __init__(self, settings, config_file, eval_function,
                 simulation_type, threaded=False, checkpoint=None, genome_path=None, headless=False):
        self.config_file = config_file
        self.threaded = threaded
        self.settings = settings
        self.simulation_type = simulation_type
        self.eval_function = eval_function
        self.parallel_eval = None
        self.checkpoint = checkpoint
        self.genome_path = genome_path
        self.headless = headless
        self.vrep_scene = None
        self._init_vrep()
        self._init_network()
        self._init_agent()
        self._init_genome()
        self._init_vision()

    def _init_vrep(self):
        """initialize vrep simulator"""
        vrep.simxFinish(-1)
        self.fnull = open(os.devnull, 'w')

        if self.threaded:
            self.ports = vrep_ports()
        else:
            self.ports = [vrep_ports()[0]]

        if self.headless:
            h = '-h'
        else:
            h = ''

        if self.simulation_type == 'thymio':
            self.vrep_scene = os.getcwd() + '/scenes/thymio_hw.ttt'
        else:
            self.vrep_scene = os.getcwd() + '/scenes/thymio_v.ttt'

        self.vrep_servers = [Popen(
            ['{0} {1} -gREMOTEAPISERVERSERVICE_{2}_TRUE_TRUE {3}'
                .format(self.settings.vrep_abspath, h, port, self.vrep_scene)],
            shell=True, stdout=self.fnull) for port in self.ports]

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
        """initialize the neural network"""
        # load the confifuration file
        self.config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                  neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                  self.config_file)
        if self.settings.save_data:
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
        if self.simulation_type == 'thymio':
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

    @timeit
    def start(self, simulation):
        default = None
        return getattr(self, simulation, lambda: default)()

    def stop(self):
        if self.parallel_eval:
            self.parallel_eval.stop()
        self._stop_vrep()
        time.sleep(3)
        self._kill_vrep()

    @timeit
    def simulation(self):
        # run simulation in vrep
        self.winner = self.population.run(partial(self.eval_function,
                                                  self.individual,
                                                  self.settings),
                                          self.settings.n_gen)
        return self.config, self.stats, self.winner

    @timeit
    def simulation_parallel(self):
        """run simulation using threads in vrep"""
        # Run for up to N generations.
        self.parallel_eval = ParrallelEvolution(self.clients, self.settings, len(
            self.clients), self.eval_function)
        self.winner = self.population.run(
            self.parallel_eval.evaluate, self.settings.n_gen)
        self.stop()
        return self.config, self.stats, self.winner

    @timeit
    def simulation_genome(self):
        """restore genome and re-run simulation"""
        self.winner = self.eval_function(
            self.individual, self.settings, [(self.winner.key, self.winner)], self.config)
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
