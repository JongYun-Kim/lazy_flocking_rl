import gym
from gym.spaces import Box, Dict
import numpy as np
# from utils.metaheuristics import SLPSO as PSO


class LazyAgentsCentralized(gym.Env):
    """
    Lazy Agents environment
    - Note (1): with the default settings, the average reward per step was -0.387909 in a 10000-episode test
    - Note (2): with the default settings, the maximum number of time steps was 1814 in the test
    - Note (3): with the default settings, the average number of time steps was 572.4 in the test
    """
    # TODO (0): [x] Should we use RELATIVE position and angle as the embeddings (observation)?
    # TODO (1): [o] Check agent update method (t vs t+1)
    # TODO (2): [△] Check if the reference variables in the state changes after assignment (e.g. self.is_padded)
    #   TODO (2-1): [o] Fully connected network cases -> working; ok!
    #   TODO (2-2): [o] No changes in the network topology -> working; ok!
    #   TODO (2-3): [x] Network changes -> not working; yet! (basics are implemented tho...)
    # TODO (3): [△] Determine the terminal condition
    #   TODO (3-1): [o] Check the paper's method
    #   TODO (3-2): [o] Check the code's method:
    #                   (1) position error, (2) changes in std of pos and (3) vel in the pas 50 iters
    # TODO (4): [x] Normalize the state, particularly with respect to the agents' initial positions
    # TODO (5): [x] Normalize the reward, particularly with respect to the number of agents
    # TODO (6): [ing] Add comments of every array's shape
    # TODO (7): [x] Take control of the data types! np.int32 or np.float32!
    # TODO (8): [Pending] Include checking if there's a network separation
    # TODO (9): [o] Implement angle WRAPPING <- not necessary as we use the relative angle and sin/cos(θ)
    #               just added the function to the env class (wrapped_ang = self.wrap_to_pi(ang))
    # TODO (10): [o] Configuration template
    # TODO (11): [o] Allow auto_step as well as the norminal step, single_step

    def __init__(self, config):
        """
        :param config: dict
        - config template:
            config = {
                "num_agents_max": 20,  # Maximum number of agents
                "num_agents_min": 2,  # Minimum number of agents

                # Optional parameters
                "speed": 15,  # Speed in m/s. Default is 15
                "predefined_distance": 60,  # Predefined distance in meters. Default is 60
                "communication_decay_rate": 1/3,  # Communication decay rate. Default is 1/3
                "cost_weight": 1,  # Cost weight. Default is 1
                "inter_agent_strength": 5,  # Inter agent strength. Default is 5
                "bonding_strength": 1,  # Bonding strength. Default is 1
                "k1": 1,  # K1 coefficient. Default is 1
                "k2": 3,  # K2 coefficient. Default is 3
                "max_turn_rate": 8/15,  # Maximum turn rate in rad/s. Default is 8/15
                "initial_position_bound": 250,  # Initial position bound in meters. Default is 250
                "dt": 0.1,  # Delta time in seconds. Default is 0.1
                "network_topology": "fully_connected",  # Network topology. Default is "fully_connected"

                # Tune the following parameters for your environment
                "std_pos_converged": 45,  # Standard position when converged. Default is 0.7*R
                "std_vel_converged": 0.1,  # Standard velocity when converged. Default is 0.1
                "std_pos_rate_converged": 0.1,  # Standard position rate when converged. Default is 0.1
                "std_vel_rate_converged": 0.2,  # Standard velocity rate when converged. Default is 0.2
                "max_time_step": 2000,  # Maximum time steps. Default is 2000,
                #                         Note: With the default settings, albeit 1000 was insufficient sometimes,
                #                               2000 would be sufficient for convergence of fully active agents.
                #                               In fact, 1814 was the maximum time step out of the 10000 episodes.
                "incomplete_episode_penalty": -600,  # Penalty for incomplete episode. Default is -600

                # Step mode
                "auto_step": False,  # If True, the env will step automatically (i.e. episode length==1). Default: False

                # Ray config
                "use_custom_ray": False,  # If True, immutability of the env will be ensured. Default: False

                # For RLlib models
                "use_preprocessed_obs": True,  # If True, the env will return preprocessed obs. Default: True
            }
        """

        super().__init__()

        # Configurations
        self.config = config
        # Mandatory parameters
        if "num_agents_max" in self.config:
            self.num_agents_max = self.config["num_agents_max"]
        else:
            raise ValueError("num_agents_max not found in env_config")
        self.num_agents_min = self.config["num_agents_min"] if "num_agents_min" in self.config else 2
        assert self.num_agents_min <= self.num_agents_max, "num_agents_min should be less than or eq. to num_agents_max"
        assert self.num_agents_min > 1, "num_agents_min must be greater than 1"

        # Optional parameters
        self.v = self.config["speed"] if "speed" in self.config else 15  # m/s
        self.R = self.config["predefined_distance"] if "predefined_distance" in self.config else 60  # m
        self.beta = self.config["communication_decay_rate"] if "communication_decay_rate" in self.config else 1/3
        self.rho = self.config["cost_weight"] if "cost_weight" in self.config else 1
        self.lambda_ = self.config["inter_agent_strength"] if "inter_agent_strength" in self.config else 5
        self.sigma = self.config["bonding_strength"] if "bonding_strength" in self.config else 1
        self.k1 = self.config["k1"] if "k1" in self.config else 1
        self.k2 = self.config["k2"] if "k2" in self.config else 3
        self.u_max = self.config["max_turn_rate"] if "max_turn_rate" in self.config else 8/15  # rad/s
        self.l_bound = self.config["initial_position_bound"] if "initial_position_bound" in self.config else 250  # m
        self.dt = self.config["dt"] if "dt" in self.config else 0.1  # s
        self.net_type = self.config["network_topology"] if "network_topology" in self.config else "fully_connected"

        # Tunable parameters (tune them for convergence conditions)
        self.std_pos_converged = self.config["std_pos_converged"] \
            if "std_pos_converged" in self.config else 0.7*self.R  # m
        # Note: if we use std(x)+std(y), it goes to 1.0*R; but we use sqrt(V(x)+V(y)). So, it goes to 0.7*R
        self.std_vel_converged = self.config["std_vel_converged"] \
            if "std_vel_converged" in self.config else 0.1  # m/s
        self.std_pos_rate_converged = self.config["std_pos_rate_converged"] \
            if "std_pos_rate_converged" in self.config else 0.1  # m
        self.std_vel_rate_converged = self.config["std_vel_rate_converged"] \
            if "std_vel_rate_converged" in self.config else 0.2  # m/s
        self.max_time_step = self.config["max_time_step"] if "max_time_step" in self.config else 2000
        self.incomplete_episode_penalty = self.config["incomplete_episode_penalty"] \
            if "incomplete_episode_penalty" in self.config else -600
        assert self.incomplete_episode_penalty <= 0, "incomplete_episode_penalty must be less than 0 (or 0)"

        # Step mode
        self.do_auto_step = self.config["auto_step"] if "auto_step" in self.config else False

        # Ray config (SL-PSO checks the immutability of the env; so, make it true if you want to use ray in SL-PSO)
        # If with RLlib, you probably need to set it to False.
        self.use_custom_ray = self.config["use_custom_ray"] if "use_custom_ray" in self.config else False

        # For RLlib models
        self.use_preprocessed_obs = self.config["use_preprocessed_obs"] \
            if "use_preprocessed_obs" in self.config else True

        # Define action space
        # Laziness vector; padding included
        # TODO: continuous actions? VS discrete actions?
        #       If continuous, what's the action distribution?
        self.action_space = Box(low=0, high=1, shape=(self.num_agents_max,), dtype=np.float32)

        # Define observation space
        self.d_v = 6  # data dim; [x, y, vx, vy, theta, omega]
        # low_bound_single_agent = np.array([-np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -self.u_max])
        # high_bound_single_agent = np.array([np.inf, np.inf, np.inf, np.inf, np.inf, self.u_max])
        # # TODO: Check the bounds; tighter bounds better? [inf, inf, v, v, pi, u_max]?
        # low_bound_all_agents = np.tile(low_bound_single_agent, (self.num_agents_max, 1))
        # high_bound_all_agents = np.tile(high_bound_single_agent, (self.num_agents_max, 1))
        low_bound_all_agents = -np.inf
        high_bound_all_agents = np.inf

        self.observation_space = Dict({
            "agent_embeddings": Box(low=low_bound_all_agents, high=high_bound_all_agents,
                                    shape=(self.num_agents_max, self.d_v), dtype=np.float32),
            "net_topology": Box(low=0, high=1, shape=(self.num_agents_max, self.num_agents_max), dtype=np.int32),
            "pad_tokens": Box(low=0, high=1, shape=(self.num_agents_max,), dtype=np.int32),
        })
        self.is_padded = None  # 1 if the task is padded, 0 otherwise; shape (num_agents_max,)

        # Define variables
        self.num_agents = None  # number of agents; scalar
        self.agent_pos = None  # shape (num_agents_max, 2)
        self.agent_vel = None  # shape (num_agents_max, 2)
        self.agent_ang = None  # shape (num_agents_max, 1)
        self.agent_omg = None  # shape (num_agents_max, 1)
        self.agent_states = None  # shape (num_agents_max, 6)
        self.adjacency_matrix = None   # w/o padding;  shape (num_agents, num_agents)
        self.net_topology_full = None  # with padding; shape (num_agents_max, num_agents_max)
        self.center_in_star_net = None  # center agent index of star network; shape (1,)? or ()
        self.u = None  # control input; shape (num_agents_max,)
        self.time_step = None
        self.clock_time = None

        # Check convergence
        self.std_pos_hist = None  # shape (self.max_time_step,)
        self.std_vel_hist = None  # shape (self.max_time_step,)

        # Backward compatibility
        self.num_agent = None
        self.num_agent_min = None
        self.num_agent_max = None

    def reset(self):
        # Reset agent number
        self.num_agents = np.random.randint(self.num_agents_min, self.num_agents_max + 1)

        # Initialize padding
        self.is_padded = np.zeros(shape=(self.num_agents_max,), dtype=np.int32)
        self.is_padded[self.num_agents:] = 1

        # Initialize network topology
        self.net_topology_full = np.zeros((self.num_agents_max, self.num_agents_max), dtype=np.int32)
        self.update_topology_from_padding(init=True)

        # Initialize agent states
        # (1) position: shape (num_agents_max, 2)
        self.agent_pos = np.random.uniform(-self.l_bound / 2, self.l_bound / 2, (self.num_agents_max, 2))
        # (2) angle: shape (num_agents_max, 1); it's a 2-D array !!!!
        self.agent_ang = np.random.uniform(-np.pi, np.pi, (self.num_agents_max, 1))
        # (3) velocity v = [vx, vy] = [v*cos(theta), v*sin(theta)]; shape (num_agents_max, 2)
        self.agent_vel = self.v * np.concatenate((np.cos(self.agent_ang), np.sin(self.agent_ang)), axis=1)
        # (4) angular velocity; shape (num_agents_max, 1); it's a 2-D array !!!!
        self.agent_omg = np.zeros((self.num_agents_max, 1))

        # Pad agent states: it concatenates agent states with padding
        self.agent_states = self.pad_states()

        # Initialize control input
        # c_init = np.ones(self.num_agents_max, dtype=np.float32)
        # self.u = self.update_u(c=c_init)  # TODO: think about the timing of this: zero vs get it from current state
        # There's nothing to initialize for control input because it's a function of the state in the previous time step
        # Hence, it doesn't exist at the initial time step (t=0); self.u == None
        # Also, w_0 == zeros; No control at all, you dumb ass :-(

        # Get observation in a dict
        observation = self.get_observation(get_preprocessed_obs=self.use_preprocessed_obs)

        # Initialize convergence check variables
        self.std_pos_hist = np.zeros(self.max_time_step)
        self.std_vel_hist = np.zeros(self.max_time_step)

        # Update time step
        self.time_step = 0
        self.clock_time = 0

        return observation

    def get_observation(self, get_preprocessed_obs, mask=None):
        # Get agent embeddings
        if get_preprocessed_obs:
            obs = self.preprocess_obs(mask=mask)
        else:
            agent_embeddings = self.agent_states[:, :self.d_v]

            # Get network topology
            net_topology = self.net_topology_full

            # Get padding tokens
            pad_tokens = self.is_padded

            # Get observation in a dict
            obs = {
                "agent_embeddings": agent_embeddings,
                "net_topology": net_topology,
                "pad_tokens": pad_tokens,
            }

        return obs

    def preprocess_obs(self, mask=None):
        # Processes the observation
        # Here, particularly, agent_embeddings; net_topology and pad_tokens are intact, right now.
        # agent_embeddings = [x, y, vx, vy, heading, heading rate]
        # Move agent_embeddings in the original coordinate to another coordinate
        # The new coordinate is translated by the center of the agents (CoM of the agents' positions)
        # and rotated by the average heading of the agents.

        # Get mask; (shows non-padding agents)
        mask = self.is_padded == 0 if mask is None else mask  # shape (num_agents_max,)

        # Get agent states
        agent_positions = self.agent_pos[mask, :]  # shape (num_agents, 2); masked
        agent_angles = self.agent_ang[mask, :]  # shape (num_agents, 1); masked
        agent_angles = self.wrap_to_pi(agent_angles)  # shape (num_agents, 1)
        agent_angular_velocities = self.agent_omg[mask, :]  # shape (num_agents, 1); masked

        # Get the center of the agents (CoM of the agents' positions)
        center = np.mean(agent_positions, axis=0)  # shape (2,)
        # Get the average heading of the agents
        avg_heading = np.mean(agent_angles)  # shape (); scalar

        # Get the rotation matrix
        rot_mat = np.array([[np.cos(avg_heading), -np.sin(avg_heading)],
                            [np.sin(avg_heading), np.cos(avg_heading)]])  # shape (2, 2)

        # Translate the center to the origin
        agent_positions_translated = agent_positions - center  # shape (num_agents, 2)
        # Rotate the translated positions
        agent_positions_transformed = np.dot(agent_positions_translated, rot_mat)  # shape (num_agents, 2)
        # Translate the agent angles
        agent_angles_transformed = agent_angles - avg_heading  # shape (num_agents, 1)
        # Wrap the agent angles
        agent_angles_transformed = self.wrap_to_pi(agent_angles_transformed)  # shape (num_agents, 1)

        # Calculate the relative velocity = [v*cos(ang_new), v*sin(ang_new)]  # shape (num_agents, 2)
        # This is omitted for faster computation (concatenating them in the next step)
        # agent_velocities_transformed = self.v * np.concatenate((np.cos(agent_angles_transformed),
        #                                                         np.sin(agent_angles_transformed)), axis=1)

        # Concatenate the transformed states as embeddings
        agent_embeddings_transformed_masked = np.concatenate(
            (
                agent_positions_transformed,
                self.v * np.cos(agent_angles_transformed),
                self.v * np.sin(agent_angles_transformed),
                agent_angles_transformed,  # wrapped angles
                agent_angular_velocities
            ),
            axis=1
        )  # shape (num_agents, 6)

        # Pad the transformed states
        # agent_embeddings_transformed_unmasked: shape (num_agents_max, 6)
        agent_embeddings_transformed_unmasked = np.zeros((self.num_agents_max, 6), dtype=np.float32)
        agent_embeddings_transformed_unmasked[mask, :] = agent_embeddings_transformed_masked

        # Get the preprocessed observation in a dict
        obs_preprocessed = {
            "agent_embeddings": agent_embeddings_transformed_unmasked,
            "net_topology": self.net_topology_full,
            "pad_tokens": self.is_padded,
        }

        return obs_preprocessed

    def pad_states(self, mask=None):
        # Hard-pad agent states
        pad_mask = self.is_padded == 1 if mask is None else mask  # Only shows padding (virtual) agents
        self.agent_pos[pad_mask, :] = 0
        self.agent_vel[pad_mask, :] = 0
        self.agent_ang[pad_mask, :] = 0
        self.agent_omg[pad_mask, :] = 0

        # Pad network topology
        # self.net_topology_full = np.pad(self.adjacency_matrix,
        #                                 ((0, self.num_agents_max - self.num_agents),
        #                                  (0, self.num_agents_max - self.num_agents)),
        #                                 mode='constant')

        # Pad agent states
        out = self.concat_states(self.agent_pos, self.agent_vel, self.agent_ang, self.agent_omg)

        return out

    def concat_states(self, agent_pos, agent_vel, agent_ang, agent_omg):
        # Check dimensions
        assert agent_pos.shape == (self.num_agents_max, 2)
        assert agent_vel.shape == (self.num_agents_max, 2)
        assert agent_ang.shape == (self.num_agents_max, 1)
        assert agent_omg.shape == (self.num_agents_max, 1)

        # Concatenate agent states
        out = np.concatenate((agent_pos, agent_vel, agent_ang, agent_omg), axis=1)

        return out

    def update_topology(self,
                        changes_in_agents=None  # shape: (num_agents_max,); 1: gain; -1: loss; 0: no change
                        # TODO: Check if this is the correct shape
                        #       and if you really need None
                        ):
        if changes_in_agents is None:
            return None

        # You must update padding before updating the network topology because the padding is used in the update
        # Dim check!
        assert changes_in_agents.shape == (self.num_agents_max,)
        # Check if there is any change
        # if np.sum(change_in_agents**2) == 0:
        #     return None

        # Update padding from the changes_in_agents
        # shapes: (num_agents_max,) in both cases
        gain_mask, loss_mask = self.update_padding_from_agent_changes(changes_in_agents)

        # Update network topology
        if np.sum(gain_mask) + np.sum(loss_mask) > 0:  # if there is any change
            self.update_topology_from_padding()

    def update_padding_from_agent_changes(self, changes_in_agents):
        # This function updates the padding from the changes in agents
        # Dim check!
        assert changes_in_agents.shape == (self.num_agents_max,)
        assert changes_in_agents.dtype == np.int32

        # gain_idx = np.where(change_in_agents == 1)
        gain_mask = changes_in_agents == 1
        # loss_idx = np.where(change_in_agents == -1)
        loss_mask = changes_in_agents == -1

        # Check if there's an invalid change (e.g. gain of existing agent, loss of non-existing agent)
        # (1) Loss of agents in padding
        assert np.sum(loss_mask[self.is_padded == 1]) == 0, "Cannot lose agents in padding."
        # (2) Gain of agents in non-padding (i.e. existing agents)
        assert np.sum(gain_mask[self.is_padded == 0]) == 0, "Cannot gain agents in non-padding."

        # Update padding
        self.is_padded[gain_mask] = 0
        self.is_padded[loss_mask] = 1

        # Update the number of agents
        self.num_agents += np.sum(gain_mask) - np.sum(loss_mask)
        assert self.num_agents <= self.num_agents_max, "Number of agents exceeds maximum number of agents."
        assert self.num_agents > 0, "Number of agents must be positive."
        # Check if padding equals to the number of agents
        assert np.sum(self.is_padded) == self.num_agents_max - self.num_agents, \
            "Padding does not equal to the number of agents."

        return gain_mask, loss_mask

    def update_topology_from_padding(self, init=False):
        # Get the adjacency matrix
        # adj_mat = self.view_adjacency_matrix()
        adj_mat = np.zeros((self.num_agents, self.num_agents), dtype=np.int32)

        # Get the network topology
        if self.net_type == "fully_connected":
            # A fully connected graph has an edge between every pair of vertices.
            # Also add self-loop connections.
            adj_mat = np.ones((self.num_agents, self.num_agents), dtype=np.int32)
        elif self.net_type == "star":
            # A star graph has one center node connected to all other nodes.
            # The center node: the first non-padded node in this case.
            self.center_in_star_net = np.where(self.is_padded == 0)[0][0]
            adj_mat[self.center_in_star_net, :] = 1
            adj_mat[:, self.center_in_star_net] = 1
            # Adding self-loop connections.
            np.fill_diagonal(adj_mat, 1)
        # elif self.net_type == "ring":
        #     # In a ring network, each node is connected to its two neighbors and itself.
        #     # So we create 1s on the diagonal (self-loops), sub-diagonal and super-diagonal.
        #     # We also need to connect the first and last nodes to close the ring.
        #     np.fill_diagonal(adj_mat, 1)
        #     adj_mat = adj_mat \
        #               + np.diag(np.ones(self.num_agents - 1), k=-1) \
        #               + np.diag(np.ones(self.num_agents - 1), k=1)
        #     adj_mat[0, -1] = adj_mat[-1, 0] = 1
        # elif self.net_type == "random":
        #     # Start with a spanning tree (a ring network is a simple example of a spanning tree)
        #     np.fill_diagonal(adj_mat, 1)
        #     adj_mat += np.diag(np.ones(self.num_agents - 1), k=-1)
        #     adj_mat += np.diag(np.ones(self.num_agents - 1), k=1)
        #     adj_mat[0, -1] = adj_mat[-1, 0] = 1
        #
        #     # Now add extra edges randomly until we reach a specified number of total edges.
        #     # This will be more than n_nodes because we started with a spanning tree.
        #     n_edges = self.num_agents \
        #               + np.random.randint(self.num_agents, self.num_agents * (self.num_agents - 1) // 2)
        #     while np.sum(adj_mat) // 2 < n_edges:
        #         i, j = np.random.choice(self.num_agents_max, 2)
        #         adj_mat[i, j] = adj_mat[j, i] = 1
        else:
            raise ValueError(f"Invalid network type: {self.net_type}; The network type is not supported.")

        if init:  # just for efficiency; you can also use init=False in the initialization
            self.adjacency_matrix = adj_mat
            self.net_topology_full[:self.num_agents, :self.num_agents] = adj_mat
        else:
            # Assign the network topology
            self.adjacency_matrix = adj_mat
            self.assign_adjacency_to_full_net(adj_mat)
            # Pad the network topology
            self.pad_topology()

    def assign_adjacency_to_full_net(self, adj_mat, mask=None):
        # Assign the adjacency matrix
        assert adj_mat.shape == (self.num_agents, self.num_agents)
        mask = self.is_padded == 0 if mask is None else mask
        self.net_topology_full[mask, :][:, mask] = adj_mat

    def pad_topology(self, mask=None):  # Be careful: the mask is the opposite of the is_padded mask!
        assert self.net_topology_full is not None
        mask = self.is_padded == 1 if mask is None else mask
        self.net_topology_full[mask, :] = 0
        self.net_topology_full[:, mask] = 0

    def view_adjacency_matrix(self, mask=None):
        # Note: You must have the latest padding and the full network topology before viewing the adjacency matrix

        # Get the adjacency matrix
        assert self.is_padded is not None
        assert self.net_topology_full is not None  # TODO: Check if it's duplicated
        mask = self.is_padded == 0 if mask is None else mask
        adj_mat = self.net_topology_full[mask, :][:, mask]  # shape: (num_agents, num_agents); (without padding)
        # TODO: check if you actually get the shape in the run-time

        return adj_mat  # a copy of the adjacency matrix; not linked to the original object (i.e. self.adjacency_matrix)

    def step(self,
             action: np.ndarray,
             ):
        obs, reward, done, info = self.auto_step(action) if self.do_auto_step else self.single_step(action)
        # if self.do_auto_step:
        #     obs, reward, done, info = self.auto_step(action)
        # else:
        #     obs, reward, done, info = self.single_step(action)

        return obs, reward, done, info

    def auto_step(self,
                  action: np.ndarray,
                  ):
        # The given action is continuously used in all single_step calls until the episode is done.

        obs = self.get_observation(get_preprocessed_obs=self.use_preprocessed_obs)
        episode_reward = 0
        done = False
        info = {}
        constant_action = action
        while not done:
            obs, reward, done, info = self.single_step(constant_action)
            episode_reward += reward

        return obs, episode_reward, done, info

    def single_step(self,
                    action: np.ndarray,
                    ):
        # Note: action is not used as control input;
        #       it is used to weight the control input (i.e. laziness)

        # Get the mask
        # TODO: Check if the mask is silently updated in the run-time; if yes, then u d better use a copy of it
        mask = self.is_padded == 0

        # Get the laziness
        c_lazy = action  # TODO: think about the dimensionality of the action

        # Update the control input:
        #   >>  u_{t+1} = f(x_t, y_t, vx_t, vy_t, θ_t, w_t);  != f(s_{t+1})
        #   Be careful! the control input uses the previous state (i.e. x_t, y_t, vx_t, vy_t, θ_t)
        #     This is because the current state may not be observable
        #     (e.g. in a decentralized network; forward compatibility)
        self.u = self.update_u(c_lazy, mask=mask)  # state transition 1/2

        # Update the state (==agent_embeddings) based on the control input
        self.update_state(u=self.u, mask=mask)  # state transition 2/2

        # Get observation
        obs = self.get_observation(get_preprocessed_obs=self.use_preprocessed_obs, mask=mask)

        # Compute reward
        # TODO: When should we compute reward?
        #  1. After updating the network topology (i.e. after changing the agent loss and gain)
        #  2. Before updating the network topology (i.e. before changing the agent loss and gain) <- current option
        reward = self.compute_reward()
        reward = self.penalize_incomplete_task(reward)

        # Check if done,
        # done = True if self.is_done(mask=mask) else False
        done = True if self.is_done_in_paper(mask=mask) else False

        # Update the clock time and the time step count
        self.clock_time += self.dt
        self.time_step += 1
        # Here, the time step virtually already done, so we increment the time step count by 1
        # as we may not notice the changes in agents at the very this time step (especially in a decentralized network)

        # Get agent changes
        #  We lose or gain agents after updating the state
        #  This is because the agents may notice the changes in the next time step
        changes_in_agents = self.get_agent_changes()  # TODO: Not supported yet

        # Update the network topology
        #  We may lose or gain edges from the changes of the swarm.
        #  So we need to update the network topology, accordingly.
        self.update_topology(changes_in_agents)  # This changes padding; be careful with silent updates (e.g. mask)

        # Get info
        info = {}

        return obs, reward, done, info

    def update_u(self, c, mask=None):
        # c: laziness; shape: (num_agents_max,)

        # Get mask
        mask = self.is_padded == 0 if mask is None else mask
        assert c.shape == mask.shape
        assert c.shape == (self.num_agents_max,)
        # assert c.dtype == np.float32  # TODO: c comes from outer scope; so it may not be float32; may need astype???
        assert mask.dtype == np.bool  # np.bool_ depending on your ray (or numpy) version
        assert np.sum(mask) == self.num_agents  # TODO: remove the asserts once the code is stable

        # Get variables (local infos)
        # rel_pos ((x_j-x_i), (y_j-y_i)): relative position; shape: (num_agents, num_agents, 2)
        # rel_dist (r_ij): relative distance; all positive (0 inc); shape: (num_agents, num_agents)
        rel_pos, rel_dist = self.get_relative_info(self.agent_pos, get_dist=True, mask=mask, get_local=True)
        # rel_ang (θ_j - θ_i): relative angle; shape: (num_agents, num_agents): 2D (i.e. dist info)
        _, rel_ang = self.get_relative_info(self.agent_ang, get_dist=True, mask=mask, get_local=True)
        # rel_vel ((vx_j-vx_i), (vy_j-vy_i)): relative velocity; shape: (num_agents, num_agents, 2)
        rel_vel, _ = self.get_relative_info(self.agent_vel, get_dist=False, mask=mask, get_local=True)

        # 1. Compute alignment control input
        # u_cs = (lambda/n(N_i)) * sum_{j in N_i}[ psi(r_ij)sin(θ_j - θ_i) ],
        # where N_i is the set of neighbors of agent i,
        # psi(r_ij) = 1/(1+r_ij^2)^(beta),
        # r_ij = ||X_j - X_i||, X_i = (x_i, y_i),
        psi = (1 + rel_dist**2) ** (-self.beta)  # shape: (num_agents, num_agents)
        alignment_error = np.sin(rel_ang)  # shape: (num_agents, num_agents)
        Neighbors = self.view_adjacency_matrix(mask=mask)  # shape: (num_agents, num_agents); from full network topology
        # u_cs: shape: (num_agents,)
        u_cs = (self.lambda_ / Neighbors.sum(axis=1)) * (Neighbors * psi * alignment_error).sum(axis=1)

        # 2. Compute cohesion control input
        # u_coh[i] = (sigma/N*V)
        #            * sum_(j in N_i)
        #               [
        #                   {
        #                       (K1/(2*r_ij^2))*<-rel_vel, -rel_pos> + (K2/(2*r_ij^2))*(r_ij-R)
        #                   }
        #                   * <[-sin(θ_i), cos(θ_i)]^T, rel_pos>
        #               ]
        # where N_i is the set of neighbors of agent i,
        # r_ij = ||X_j - X_i||, X_i = (x_i, y_i),
        # rel_vel = (vx_j - vx_i, vy_j - vy_i),
        # rel_pos = (x_j - x_i, y_j - y_i),
        sig_NV = self.sigma / (self.v * Neighbors.sum(axis=1))  # shape: (num_agents,)
        r_ij = rel_dist + (np.eye(self.num_agents) * np.finfo(float).eps)  # shape: (num_agents, num_agents)
        k1_2rij2 = self.k1 / (2 * r_ij**2)  # shape: (num_agents, num_agents)
        k2_2rij = self.k2 / (2 * r_ij)  # shape: (num_agents, num_agents)
        v_dot_p = np.einsum('ijk,ijk->ij', rel_vel, rel_pos)  # shape: (num_agents, num_agents)
        rij_R = rel_dist - self.R  # shape: (num_agents, num_agents)
        ang_vec = np.concatenate([-np.sin(self.agent_ang[mask]), np.cos(self.agent_ang[mask])], axis=1)  # (num_a, 2)
        ang_vec = np.tile(ang_vec[:, np.newaxis, :], (1, self.num_agents, 1))  # (num_agents, num_agents, 2)
        ang_dot_p = np.einsum('ijk,ijk->ij', ang_vec, rel_pos)  # shape: (num_agents, num_agents)
        need_sum = (k1_2rij2 * v_dot_p + k2_2rij * rij_R) * ang_dot_p  # shape: (num_agents, num_agents)
        u_coh = sig_NV * (Neighbors * need_sum).sum(axis=1)  # shape: (num_agents,)

        # 3. Compute separation control input
        # TODO: implement separation control input when it is needed; for now, no separation control input
        # u_sep =

        # 4. Compute communication control input
        # Not implemented yet as we use fixed communication topology (regardless of actual distances)
        # u_comm =

        # 5. Compute control input
        u_local = u_cs + u_coh  # + u_sep + u_comm  # shape: (num_agents,)
        # Saturation
        u_local = np.clip(u_local, -self.u_max, self.u_max)  # shape: (num_agents,)
        # Consider laziness
        u = np.zeros(self.num_agents_max, dtype=np.float32)  # shape: (num_agents_max,)
        # Starting from zeros is important to avoid non-padded agents to have non-zero control inputs.
        # This is because the model's output isn't perfectly masked as the std of the output cannot be an absolute zero.
        # If you are dealing with a Gaussian action distribution in your RL, the output is zero with a zero probability,
        # which can still happen though. (An event with zero probability can still happen, mathematically speaking.)
        u[mask] = c[mask] * u_local

        # print(rel_dist[0, :].reshape(4, 5))
        # print(rel_ang[0, :].reshape(4, 5))

        return u  # shape: (num_agents_max,)

    def get_relative_info(self, data, get_dist=False, mask=None, get_local=False):
        # Returns the relative information of the agents (e.g. relative position, relative angle, etc.)
        # data: shape (num_agents_max, data_dim)
        # mask: shape (num_agents_max, num_agents_max)

        # Define mask; enables to take into account non-padded data only
        mask = self.is_padded == 0 if mask is None else mask
        assert mask.dtype == np.bool_  # np.bool probably deprecated (check your numpy/ray versions; not sure...)

        # Get dimension of the data
        assert data.ndim == 2  # we use a 2D array for the data
        assert data[mask].shape[0] == self.num_agents  # TODO: do we have to check both dims?
        assert data.shape[0] == self.num_agents_max
        data_dim = data.shape[1]

        # Compute relative data
        # rel_data: shape (num_agents_max, num_agents_max, data_dim); rel_data[i, j] = data[j] - data[i]
        # rel_data_local: shape (num_agents, num_agents, data_dim)
        # rel_data_local = data[mask] - data[mask, np.newaxis, :]
        rel_data_local = data[np.newaxis, mask, :] - data[mask, np.newaxis, :]
        if get_local:
            rel_data = rel_data_local
        else:
            rel_data = np.zeros((self.num_agents_max, self.num_agents_max, data_dim), dtype=np.float32)
            rel_data[np.ix_(mask, mask, np.arange(data_dim))] = rel_data_local
            # rel_data[mask, :, :][:, mask, :] = rel_data_local  # not sure; maybe 2-D array (not 3-D) if num_true = 1

        # Compute relative distances
        # rel_dist: shape (num_agents_max, num_agents_max)
        # Note: data are all non-negative!!
        if get_dist:
            rel_dist = np.linalg.norm(rel_data, axis=2) if data_dim > 1 else rel_data.squeeze()
        else:
            rel_dist = None

        # get_local==False: (num_agents_max, num_agents_max, data_dim), (num_agents_max, num_agents_max)
        # get_local==True: (num_agents, num_agents, data_dim), (num_agents, num_agents)
        # get_dist==False: (n, n, d), None
        return rel_data, rel_dist

    def update_state(self, u, mask):
        # Update the state based on the control input
        if self.use_custom_ray:  # memory inefficient! but ray requires immutable objects..
            self.agent_omg = np.copy(self.agent_omg)  # TODO: testing.. ray
            self.agent_pos = np.copy(self.agent_pos)  # TODO: testing.. ray
            self.agent_ang = np.copy(self.agent_ang)  # TODO: testing.. ray
            self.agent_vel = np.copy(self.agent_vel)  # TODO: testing.. ray

        # (1) Update angular velocity:
        # >>  w_{t+1} = u_{t+1}
        # Note: u_{t+1} = f(p_t, ang_t, v_t)
        self.agent_omg[mask] = u[mask, np.newaxis]

        # (2) Update position:
        # >>  p_{t+1} = p_t + v_t * dt;  Be careful: v_t is the velocity at time t, not t+1
        self.agent_pos[mask] = self.agent_pos[mask] + self.agent_vel[mask] * self.dt

        # (3) Update angle:
        # >>  ang_{t+1} = ang_t + w_t * dt;  Be careful: w_t is the angular velocity at time t, not t+1
        self.agent_ang[mask] = self.agent_ang[mask] + u[mask, np.newaxis] * self.dt
        # Wrap the angle to [-pi, pi]  <- not necessary, right now
        # self.agent_ang[mask] = self.wrap_to_pi(self.agent_ang[mask])

        # (4) Update velocity:
        # >>  v_{t+1} = V * [cos(ang_{t+1}), sin(ang_{t+1})]; Be careful: ang_{t+1} is the angle at time t+1
        # TODO: think about the masking because cos(0) is not 0; may cause issues when the net's not fully-connected
        self.agent_vel[mask] = self.v * np.concatenate(
            [np.cos(self.agent_ang[mask]), np.sin(self.agent_ang[mask])], axis=1)

        # Update the state
        self.agent_states = self.concat_states(self.agent_pos, self.agent_vel, self.agent_ang, self.agent_omg)
        # self.agent_states = self.pad_states()  # TODO: Do we (effectively) pad the states twice?

    def get_agent_changes(self):
        # loss or gain agents after updating the state
        # changes_in_agents[i, 0] ==  0: no change in agent_i's existence
        #                         ==  1: agent_i is gained as a new agent
        #                         == -1: agent_i is lost

        # TODO: Implement your swarm group dynamics here
        # TODO: For now, no changes in the swarm
        # changes_in_agents = np.zeros((self.num_agents_max,), dtype=np.int32)
        changes_in_agents = None  # effectively same as the above: the zeros array

        return changes_in_agents  # shape: (num_agents_max,) dtype: int32

    def compute_reward(self):
        # J_{energy}   = [V/N * sum_{i=1}^{N} sum_{k=1}^{t_c} ||u_i(k)||^2] + t_c : total cost throughout the time
        # J_{energy}_t = [V/N * sum_{i=1}^{N} ||u_i(t)||^2] + dt : cost at time t

        # Get control input data: shape (num_agents_max,)
        u_t = self.u

        # Check the control input of the padding agents
        assert np.all(u_t[self.is_padded == 1] == 0)  # TODO: delete this line after debugging or stabilizing the code

        # Compute reward of all agents into a scalar: centralized
        # control_cost = (self.v / self.num_agents) * np.sum(u_t ** 2)
        control_cost = self.dt * (self.v / self.num_agents) * np.sum(np.abs(u_t))  # from the paper
        fuel_cost = self.dt
        # Shape of reward: (1,)  TODO: check the data types!
        total_cost = control_cost + self.rho * fuel_cost
        reward = - total_cost  # maximize the reward == minimize the cost
        # i.e. reward is negative in most cases in this environment

        # Check the data type of the reward
        assert isinstance(reward, float) or isinstance(reward, np.float32)  # TODO: delete this line as well, l8r

        return reward

    def penalize_incomplete_task(self, reward):
        # Penalize the RL-agent if the task is not completed
        if self.time_step >= self.max_time_step-1:
            reward = self.incomplete_episode_penalty
            # print("Task was not completed within the maximum time step!")

        return reward

    def is_done(self, mask=None):  # almost deprecated
        # Check if the swarm converged
        # Mine (but paper actually said these):
        #   (1) velocity deviation is smaller than a threshold v_th
        #   (2) position deviation is smaller than a threshold p_th
        #   (3) (independently) maximum number of time steps is reached

        # Get mask
        mask = self.is_padded == 0 if mask is None else mask

        # 1. Check velocity standard deviation
        vel_distribution = self.agent_vel[mask]  # shape: (num_agents, 2)
        vel_std = np.std(vel_distribution, axis=0)  # shape: (2,)
        vel_std = np.mean(vel_std)  # shape: (1,) or (,)
        # vel_converged = True if np.all(vel_std < self.vel_err_allowed) else False
        vel_converged = True if vel_std < self.std_vel_converged else False

        # 2. Check position deviation
        pos_distribution = self.agent_pos[mask]  # shape: (num_agents, 2)
        pos_std = np.std(pos_distribution, axis=0)  # shape: (2,)
        pos_std = np.mean(pos_std)  # shape: (1,) or (,)
        # pos_converged = True if np.all(pos_std < self.pos_err_allowed) else False
        pos_converged = True if pos_std < self.std_pos_converged else False

        done = True if vel_converged and pos_converged else False

        # 3. Check if the maximum number of time steps is reached
        if self.time_step >= self.max_time_step:
            done = True

        print(f"time_step: {self.time_step}, max_time_step: {self.max_time_step}")
        print(f"  vel_deviation: {vel_std},\n  pos_deviation: {pos_std},\n  done: {done}")

        return done

    def is_done_in_paper(self, mask=None):
        # Check if the swarm converged or the episode is done
        # The code of the paper implemented:
        #   (1) position std: sqrt(V(x)+V(y)) < std_p_converged
        #   (2) change in position std: max of (sqrt(V(x)+V(y))) - min of (sqrt(V(x)+V(y))) < std_p_rate_converged
        #       in the last 50 iterations
        #   (3) change in velocity std: max of (sqrt(V(vx)+V(vy))) - min of (sqrt(V(vx)+V(vy))) < std_v_rate_converged
        #       in the last 50 iterations
        #   (4) (independently) maximum number of time steps is reached
        # Note: if you move this method in step(), you can use self.time_step instead of self.time_step-1
        #       Check your time_step in step() in such case

        # Get mask
        mask = self.is_padded == 0 if mask is None else mask
        done = False

        # Get standard deviations of position and velocity
        pos_distribution = self.agent_pos[mask]  # shape: (num_agents, 2)
        pos_std = np.sqrt(np.sum(np.var(pos_distribution, axis=0)))  # shape: (1,) or (,)
        vel_distribution = self.agent_vel[mask]  # shape: (num_agents, 2)
        vel_std = np.sqrt(np.sum(np.var(vel_distribution, axis=0)))  # shape: (1,) or (,)
        # Store the standard deviations
        if self.use_custom_ray:  # memory inefficient... but ray requires immutable objects
            self.std_pos_hist = np.copy(self.std_pos_hist)  # TODO: ray...
            self.std_vel_hist = np.copy(self.std_vel_hist)  # TODO: ray...
        self.std_pos_hist[self.time_step] = pos_std
        self.std_vel_hist[self.time_step] = vel_std

        # 1. Check position standard deviation
        pos_converged = True if pos_std < self.std_pos_converged else False

        # Check 2 and 3 only if the position standard deviation is smaller than the threshold
        if pos_converged:
            # Only proceed if there have been at least 50 time steps
            if self.time_step >= 49:
                # 2. Check change in position standard deviation
                # Get the last 50 iterations
                last_n_pos_std = self.std_pos_hist[self.time_step - 49: self.time_step + 1]
                # Get the maximum and minimum of the last 50 iterations
                max_last_n_pos_std = np.max(last_n_pos_std)
                min_last_n_pos_std = np.min(last_n_pos_std)
                # Check if the change in position standard deviation is smaller than the threshold
                pos_rate_converged = \
                    True if (max_last_n_pos_std-min_last_n_pos_std) < self.std_pos_rate_converged else False

                # Check 3 only if the change in position standard deviation is smaller than the threshold
                if pos_rate_converged:
                    # 3. Check change in velocity standard deviation
                    # Get the last 50 iterations
                    last_n_vel_std = self.std_vel_hist[self.time_step - 49: self.time_step + 1]
                    # Get the maximum and minimum of the last 50 iterations
                    max_last_n_vel_std = np.max(last_n_vel_std)
                    min_last_n_vel_std = np.min(last_n_vel_std)
                    # Check if the change in velocity standard deviation is smaller than the threshold
                    vel_rate_converged = \
                        True if (max_last_n_vel_std-min_last_n_vel_std) < self.std_vel_rate_converged else False
                    # Check if the swarm converged in the sense of std_pos, std_pos_rate, and std_vel_rate
                    done = vel_rate_converged

        # 4. Check if the maximum number of time steps is reached
        if self.time_step >= self.max_time_step-1:
            done = True

        # # Print
        # print(f"time_step: {self.time_step}, max_time_step: {self.max_time_step}")
        # print(f"      pos_std: {pos_std},\n"
        #       f"      vel_std: {vel_std},\n"
        #       # f"      pos_rate_deviation: {max_last_n_pos_std-min_last_n_pos_std},\n"
        #       # f"      vel_rate_deviation: {max_last_n_vel_std-min_last_n_vel_std},\n"
        #       f"      done: {done}")

        return done

    @staticmethod
    def wrap_to_pi(angles):
        # Wrap angles to [-pi, pi]
        return (angles + np.pi) % (2 * np.pi) - np.pi

    def render(self, mode="human"):
        # Render the environment
        pass

    def update_env_object(self, do_auto_step=False, use_custom_ray=False):  # for backward compatibility
        # Enables the old environment instance is compatible with the new version
        # when you use serialized environments of the old version in the new version.
        if hasattr(self, "num_agents"):
            if hasattr(self, "num_agent"):
                print("The environment is already the newer version.")
            else:
                self.num_agent = None
                self.num_agent_max = None
                self.num_agent_min = None
                assert hasattr(self, "auto_step") and hasattr(self, "use_custom_ray")
                print("The environment is a bit old version. Just num_agent(_x) weren't defined.")
        else:
            if hasattr(self, "num_agent"):
                self.num_agents = self.num_agent
                self.num_agents_max = self.num_agent_max
                self.num_agents_min = self.num_agent_min
                self.do_auto_step = do_auto_step
                self.use_custom_ray = use_custom_ray
            else:
                print("No num_agent, no num_agents? wtf!")


