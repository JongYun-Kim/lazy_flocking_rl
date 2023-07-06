import numpy as np
import matplotlib.pyplot as plt
# import time
# from gym.spaces import Box, Dict
from env.envs import LazyAgentsCentralized  # Import your custom environment


def plot_trajectories(agent_positions_history, current_states, fig, ax):
    ax.clear()
    ax.set_title('Agent Trajectories')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')

    # extract current positions and velocities
    positions = current_states[:, :2]
    velocities = current_states[:, 2:4]

    # plot each agent's trajectory
    for i in range(len(current_states)):
        agent_trajectory = agent_positions_history[i]  # (T, 2)
        ax.plot(agent_trajectory[:, 0], agent_trajectory[:, 1], '-')
        # plot start positions as green dot
        ax.plot(*agent_trajectory[0], 'go')
        # plot current positions with velocity arrow
        ax.quiver(*positions[i], *velocities[i], angles='xy', scale_units='xy', scale=1, color='r')

    ax.grid(True)

    plt.draw()
    plt.pause(0.001)


def plot_agents(agents_state, fig, ax):
    ax.clear()
    ax.set_title('Agent Positions and Velocities')
    ax.set_xlabel('X position')
    ax.set_ylabel('Y position')

    # extract position and velocity
    positions = agents_state[:, :2]
    velocities = agents_state[:, 2:4]

    # # Compute the min and max coordinates considering both position and velocity vectors
    # min_coords = np.min(positions + velocities, axis=0)
    # max_coords = np.max(positions + velocities, axis=0)
    # # Leave a little extra space on the edges
    # ax.set_xlim(min_coords[0] - 20, max_coords[0] + 20)
    # ax.set_ylim(min_coords[1] - 20, max_coords[1] + 20)

    # Compute the min and max coordinates considering both position and velocity vectors
    min_coords = np.min(positions + velocities, axis=0) - 1
    max_coords = np.max(positions + velocities, axis=0) + 1

    # Create invisible boundary points
    ax.plot([min_coords[0], max_coords[0]], [min_coords[1], max_coords[1]], alpha=0.0)

    # plot start positions as green dots
    ax.scatter(positions[:, 0], positions[:, 1], color='g')

    # plot velocity vectors as arrows
    ax.quiver(positions[:, 0], positions[:, 1], velocities[:, 0], velocities[:, 1], angles='xy', scale_units='xy',
              scale=1)

    ax.grid(True)

    plt.draw()
    plt.pause(0.001)


def plot_std_history(std_vel_history, std_pos_history, fig, ax):
    ax.clear()
    ax.set_title('Standard Deviation History')
    ax.set_xlabel('Time step')
    ax.set_ylabel('Std. Dev.')
    ax.plot(std_vel_history, label='Velocity Std. Dev.')
    ax.plot(std_pos_history, label='Position Std. Dev.')
    ax.legend()

    ax.grid(True)

    plt.draw()
    plt.pause(0.001)


if __name__ == '__main__':
    print("Running env_test.py")

    num_experiment = 30

    for exp in range(num_experiment):

        num_agent = 20
        action = np.ones(num_agent, dtype=np.float32)
        config = {"num_agent_max": num_agent,
                  "num_agent_min": num_agent,
                  #
                  "std_pos_converged": 42,  # Standard position when converged. Default is R/2
                  "std_vel_converged": 0.1,  # Standard velocity when converged. Default is 0.1
                  "std_pos_rate_converged": 0.1,  # Standard position rate when converged. Default is 0.1
                  "std_vel_rate_converged": 0.2,  # Standard velocity rate when converged. Default is 0.01
                  "max_time_step": 1000  # Maximum time steps. Default is 1000,
                  }

        env = LazyAgentsCentralized(config)
        obs = env.reset()
        agent_states = obs['agent_embeddings']

        # std_vel_history = []
        # std_pos_history = []
        agent_positions_history = np.zeros((num_agent, env.max_time_step, 2), dtype=np.float32)

        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        fig3, ax3 = plt.subplots()

        done = False
        while not done:
            obs, reward, done, info = env.step(action)

            # extract agent_states from the observations
            agent_states = obs['agent_embeddings']

            # update agent_positions_history with current positions
            agent_positions_history[:, env.time_step-1] = agent_states[:, :2]

            # # plot trajectories and std dev history
            # std_pos_history = env.std_pos_hist[:env.time_step]
            # std_vel_history = env.std_vel_hist[:env.time_step]
            # plot_agents(agent_states, fig1, ax1)
            # if env.time_step % 50 == 0:
            #     plot_std_history(std_vel_history, std_pos_history, fig2, ax2)
            #     plot_trajectories(agent_positions_history[:, :env.time_step], agent_states, fig3, ax3)

            # if env.time_step % 100 == 0:
            #     print(f"Time step: {env.time_step}")
            #     time.sleep(0.01)

        # plot trajectories and std dev history
        std_pos_history = env.std_pos_hist[:env.time_step]
        std_vel_history = env.std_vel_hist[:env.time_step]
        plot_agents(agent_states, fig1, ax1)
        plot_std_history(std_vel_history, std_pos_history, fig2, ax2)
        plot_trajectories(agent_positions_history[:, :env.time_step], agent_states, fig3, ax3)

        print("Done!")

        if env.time_step == env.max_time_step:
            print("Max time step reached!")

        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)

