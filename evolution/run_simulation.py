from subprocess import Popen
from functools import partial
import vrep.vrep as vrep
import neat
import pickle
import yaml
import warnings
import os
import time
from robot.vrep_robot import VrepRobot
from evolution.eval_genomes import eval_genomes_simulation, eval_genomes_hardware, eval_genome
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
            print("thread_id = {0} client_id = {1}".format(
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
            f = self.eval_function(client_id, settings, genome, config)
            self.outqueue.put((genome_id, genome, f))

    def evaluate(self, genomes, config):
        """Evaluate the genomes"""
        if not self.working:
            self.start()
        p = 0
        for genome_id, genome in genomes:
            p += 1
            self.inqueue.put((genome_id, genome, config))

        # assign the fitness back to each genome
        while p > 0:
            p -= 1
            ignored_genome_id, genome, fitness = self.outqueue.get()
            genome.fitness = fitness


def vrep_ports():
    """Load the vrep ports"""
    with open("ports.yml", 'r') as f:
        portConfig = yaml.load(f)
    return portConfig['ports']


def run_vrep_simluation(settings, config_file):
    print('Neuroevolutionary program started!')
    vrep.simxFinish(-1)
    settings.client_id = vrep.simxStart(
        '127.0.0.1',
        settings.port_num,
        True,
        True,
        5000,
        5)

    if settings.client_id == -1:
        return

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    config.save(settings.path + 'config.ini')

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    # Add a stdout reporter to show progress in the terminal.
    stats = neat.StatisticsReporter()
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    individual = VrepRobot(
        client_id=settings.client_id,
        id=None,
        op_mode=settings.op_mode,
        robot_type=settings.robot_type
    )

    # Run for up to N_GENERATIONS generations.
    winner = p.run(partial(eval_genomes_simulation,
                           individual, settings), settings.n_gen)

    return config, stats, winner


def run_vrep_parallel(settings, config_file):
    print('Neuroevolutionary program started in parallel!')
    vrep.simxFinish(-1)

    ports = vrep_ports()
    FNULL = open(os.devnull, 'w')

    spawns multiple vrep instances
    vrep_servers = [Popen(
        ['{0} -gREMOTEAPISERVERSERVICE_{1}_TRUE_TRUE {2}'
            .format(settings.vrep_abspath, port, settings.vrep_scene)],
        shell=True, stdout=FNULL) for port in ports]

    time.sleep(5)

    clients = [vrep.simxStart(
        '127.0.0.1',
        port,
        True,
        True,
        5000,
        5) for port in ports]

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    config.save(settings.path + 'config.ini')

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)
    # Add a stdout reporter to show progress in the terminal.
    stats = neat.StatisticsReporter()
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))

    # Run for up to N generations.
    pe = ParrallelEvolution(clients, settings, len(clients), eval_genome)
    winner = p.run(pe.evaluate, settings.n_gen)

    # stop the workers
    pe.stop()

    # stop vrep simulation
    _ = [vrep.simxFinish(client) for client in clients]
    # kill vrep instances
    # _ = [server.kill() for server in vrep_servers]

    return config, stats, winner


def run_hardware_simulation(settings, config_file):
    print('Neuroevolutionary program started!')
    vrep.simxFinish(-1)
    settings.client_id = vrep.simxStart(
        settings.address,
        settings.port_num,
        True,
        True,
        5000,
        5)  # Connect to V-REP

    if settings.client_id == -1:
        print('Failed connecting to remote API server')
        print('Program ended')
        return

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    stats = neat.StatisticsReporter()
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(stats)

    individual = EvolvedRobot(
        'thymio-II',
        client_id=settings.client_id,
        id=None,
        op_mode=settings.op_mode,
        chromosome=None,
        robot_type=settings.robot_type
    )

    # Run for up to N_GENERATIONS generations.
    winner = p.run(partial(eval_genomes_hardware,
                           individual, settings), settings.n_gen)

    return config, stats, winner


def restore_vrep_simulation(settings, config_file, checkpoint=None, path=None):
    print('Restore neuroevolutionary program started!')
    vrep.simxFinish(-1)
    settings.client_id = vrep.simxStart(
        '127.0.0.1',
        settings.port_num,
        True,
        True,
        5000,
        5)

    if settings.client_id == -1:
        return

    individual = VrepRobot(
        client_id=settings.client_id,
        id=None,
        op_mode=settings.op_mode,
        robot_type=settings.robot_type
    )

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    stats = neat.StatisticsReporter()

    if checkpoint:
        population = neat.Checkpointer.restore_checkpoint(checkpoint)
        p = neat.population.Population(
            config, (population.population, population.species, settings.n_gen))
        # Add a stdout reporter to show progress in the terminal.
        p.add_reporter(neat.StdOutReporter(True))
        p.add_reporter(stats)
        # Run for up to N_GENERATIONS generations.
        winner = p.run(partial(eval_genomes_simulation,
                               individual, settings), settings.n_gen)
        return config, stats, winner
    elif path:
        with open(path + '/winner_genome', 'rb') as f:
            c = pickle.load(f)
        print('Loaded genome:')
        print(c)
        eval_genomes_simulation(individual, settings, [(c.key, c)], config)
        return

    return
