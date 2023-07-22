import numpy as np
from env.envs import LazyAgentsCentralized  # environment
from utils.metaheuristics import GetLazinessBySLPSO  # optimizer
import copy  # for preserving the environment object in the results for later use
import ray  # for parallelization
import pickle  # for saving results
import os  # for creating directories
from datetime import datetime  # for getting current date and time


if __name__ == '__main__':
    # Params
    num_agents = 20  # Fixed number of agents
    num_experiments = 5  # Number of experiments
    num_cpus = 14  # Number of CPUs to use for parallelization by Ray

    # Get start time
    start_time = datetime.now()

    # Get a config dict for the test environment
    config = {
        "num_agents_max": num_agents,  # Maximum number of agents
        "num_agents_min": num_agents,  # Minimum number of agents

        # Optional parameters
        "speed": 15,  # Speed in m/s. Default is 15
        "predefined_distance": 60,  # Predefined distance in meters. Default is 60
        "communication_decay_rate": 1 / 3,  # Communication decay rate. Default is 1/3
        "cost_weight": 1,  # Cost weight. Default is 1
        "inter_agent_strength": 5,  # Inter agent strength. Default is 5
        "bonding_strength": 1,  # Bonding strength. Default is 1
        "k1": 1,  # K1 coefficient. Default is 1
        "k2": 3,  # K2 coefficient. Default is 3
        "max_turn_rate": 8 / 15,  # Maximum turn rate in rad/s. Default is 8/15
        "initial_position_bound": 250,  # Initial position bound in meters. Default is 250
        "dt": 0.1,  # Delta time in seconds. Default is 0.1
        "network_topology": "fully_connected",  # Network topology. Default is "fully_connected"

        # Tune the following parameters for your environment
        "std_pos_converged": 45,  # Standard position when converged. Default is 0.7*R
        "std_vel_converged": 0.1,  # Standard velocity when converged. Default is 0.1
        "std_pos_rate_converged": 0.1,  # Standard position rate when converged. Default is 0.1
        "std_vel_rate_converged": 0.2,  # Standard velocity rate when converged. Default is 0.2
        "max_time_step": 2000,  # Maximum time steps. Default is 1000,
        "incomplete_episode_penalty": -700,  # Penalty for incomplete episode. Default is -600

        # Step mode
        "auto_step": False,  # If True, the env will step automatically (i.e. episode length==1). Default: False

        # Ray config
        "use_custom_ray": True,  # If True, immutability of the env will be ensured. Default: False
    }

    # Initialize ray
    #   : pso.compute_single_particle_cost used as a remote function in pso._cost_func()
    ray.init(num_cpus=num_cpus)

    # For storing results
    results = []

    # Run experiments
    fully_active_action = np.ones(num_agents, dtype=np.float32)
    for exp in range(num_experiments):
        # Create environment
        env = LazyAgentsCentralized(config)
        env.reset()

        pso = GetLazinessBySLPSO()
        pso.set_env(copy.deepcopy(env))
        laziness, cost, _ = pso.run(see_updates=True, see_time=True)

        env_fully_active = copy.deepcopy(env)
        _, fully_active_episode_reward, _, _ = env_fully_active.auto_step(fully_active_action)

        print(f"\nExperiment {exp + 1}/{num_experiments}")
        print("Laziness: ", laziness)
        print("Cost_lazy: ", cost)
        print("Cost_fully_active: ", -fully_active_episode_reward)

        # Append environment and results to results list
        results.append({"env": env,
                        "laziness": laziness,
                        "cost": cost,
                        "cost_fully_active": -fully_active_episode_reward,})

    # Get current date and time
    current_directory = os.path.dirname(os.path.abspath(__file__))  # current directory where the script is located
    base_directory = os.path.dirname(current_directory)  # get the parent directory
    data_directory = os.path.join(base_directory, "data")  # data directory is in the base directory
    now = datetime.now()
    date_time = now.strftime("%y_%m_%d_%H%M")
    path_to_create = os.path.join(data_directory, date_time)

    # Create directory with date and time as name
    os.makedirs(path_to_create, exist_ok=True)

    # Save results to a pickle file
    file_path = os.path.join(path_to_create, "results.pkl")
    with open(file_path, "wb") as f:  # wb stands for write binary
        pickle.dump({"results": results, "config": config}, f)  # dump the results and config to the pickle file

    # Print the path to the pickle file
    print(f"\nData collection completed at", now.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Results are saved to {file_path}")

    # Tell the duration of the experiment
    end_time = now
    duration = end_time - start_time
    seconds = int(duration.total_seconds())
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)

    is_plural = lambda x: '' if x == 1 else 's'  # for printing plural forms

    print(f"\n\n Duration of the {num_experiments} experiments:")
    print("    It took {} day{}, {} hour{}, {} minute{}, and {} second{}".format(days, is_plural(days), hours,
                                                                                 is_plural(hours), minutes,
                                                                                 is_plural(minutes), seconds,
                                                                                 is_plural(seconds)))

    ray.shutdown()
    print("\n\nRay shutdown!")
    print("\nDone")

""" How to use the results l8r
import pickle
import os

# Get the path to the pickle file; please update the path accordingly
data_path = "./data"  # Get your data path
folder_name = "23_01_01_0000"
file_name = "results.pkl"
path_to_read = os.path.join(data_path, folder_name, file_name)

# Load the pickle file
with open(path_to_read, "rb") as f:
    data = pickle.load(f)

# Get the results and config
results = data["results"]
config = data["config"]

# Now you can access the results
# For example, results[i]["laziness"] gives the laziness of the i-th experiment (i starts from 0)
for result in results:
    print("Laziness: ", result["laziness"])
    print("Cost: ", result["cost"])
    print("Environment: ", result["env"])

# And you can access the config
print("Config: ", config)
"""
