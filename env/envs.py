import gym
from gym.spaces import Box, Dict
from gym.utils import seeding
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

try:
    matplotlib.use('TkAgg')  # To avoid the MacOS backend; but just follow your needs
except (ImportError, ModuleNotFoundError):
    matplotlib.use('Agg')

# Compatibility layer for np.bool_ and np.bool
if not hasattr(np, 'bool_'):
    np.bool_ = np.bool

class LazyAgentsCentralized(gym.Env):

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
                "normalize_obs": False,  # If True, the env will normalize the obs. Default: False\
                "use_fixed_horizon": False,  # If True, the env will use fixed horizon. Default: False
                "use_L2_norm": False,  # If True, the env will use L2 norm. Default: False

                # Heuristic policy
                "use_heuristics": False,  # If True, the env will use heuristic policy. Default: False
                "_use_fixed_lazy_idx": True,  # If True, the env will use fixed lazy idx. Default: True

                # Step mode
                "auto_step": False,  # If True, the env will step automatically (i.e. episode length==1). Default: False

                # Ray config
                "use_custom_ray": False,  # If True, immutability of the env will be ensured. Default: False

                # For RLlib models
                "use_preprocessed_obs": True,  # If True, the env will return preprocessed obs. Default: True
                "use_mlp_settings": False,  # If True, flatten obs used without topology and padding. Default: False
                #                             Note: No padding applied to the MLP settings for now

                # Plot config
                "get_state_hist": False,  # If True, state_hist stored. Use this for plotting. Default: False
                # try to leave it empty in your config unless you explicitly want to plot
                # as it's gonna be False by default, use more memory, and slow down the training/evaluation
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
            if "incomplete_episode_penalty" in self.config else 0
        assert self.incomplete_episode_penalty <= 0, "incomplete_episode_penalty must be less than 0 (or 0)"
        self.normalize_obs = self.config["normalize_obs"] if "normalize_obs" in self.config else False
        self.use_fixed_horizon = self.config["use_fixed_horizon"] if "use_fixed_horizon" in self.config else False
        self.use_L2_norm = self.config["use_L2_norm"] if "use_L2_norm" in self.config else False

        # Heuristic policy
        self.use_heuristics = self.config["use_heuristics"] if "use_heuristics" in self.config else False
        self._use_fixed_lazy_idx = self.config["_use_fixed_lazy_idx"] if "_use_fixed_lazy_idx" in self.config else True

        # Step mode
        self.do_auto_step = self.config["auto_step"] if "auto_step" in self.config else False

        # Ray config (SL-PSO checks the immutability of the env; so, make it true if you want to use ray in SL-PSO)
        # If with RLlib, you probably need to set it to False.
        self.use_custom_ray = self.config["use_custom_ray"] if "use_custom_ray" in self.config else False

        # For RLlib models
        self.use_preprocessed_obs = self.config["use_preprocessed_obs"] \
            if "use_preprocessed_obs" in self.config else True
        self.use_mlp_settings = self.config["use_mlp_settings"] if "use_mlp_settings" in self.config else False

        # Plot config
        self.get_state_hist = self.config["get_state_hist"] if "get_state_hist" in self.config else False

        # Define action space
        # Laziness vector; padding included
        self.action_space = Box(low=0, high=1, shape=(self.num_agents_max,), dtype=np.float32)

        # Define observation space
        self.d_v = 6  # data dim; [x, y, vx, vy, theta, omega]
        low_bound_all_agents = -np.inf
        high_bound_all_agents = np.inf

        if self.use_mlp_settings:
            self.observation_space = Box(low=low_bound_all_agents, high=high_bound_all_agents,
                                         shape=(self.num_agents_max * (self.d_v-1),), dtype=np.float32)
        else:
            self.observation_space = Dict({
                "agent_embeddings": Box(low=low_bound_all_agents, high=high_bound_all_agents,
                                        shape=(self.num_agents_max, self.d_v), dtype=np.float32),
                "pad_tokens": Box(low=0, high=1, shape=(self.num_agents_max,), dtype=np.int32),
            })
        self.is_padded = None  # 1 if the task is padded, 0 otherwise; shape (num_agents_max,)

        # Define variables
        self.num_agents = None  # number of agents; scalar
        self.agent_pos = None  # shape (num_agents_max, 2)
        self.agent_vel = None  # shape (num_agents_max, 2)
        self.agent_ang = None  # shape (num_agents_max, 1)
        self.agent_omg = None  # shape (num_agents_max, 1)
        self.agent_states = None  # shape (num_agents_max, 6==d_v)
        self.adjacency_matrix = None   # w/o padding;  shape (num_agents, num_agents)
        self.net_topology_full = None  # with padding; shape (num_agents_max, num_agents_max)
        self.center_in_star_net = None  # center agent index of star network; shape (1,)? or ()
        self.u_lazy = None  # control input; shape (num_agents_max,)  == self.u before
        self.u_fully_active = None  # control input; shape (num_agents_max,) in fully active control
        # u_decomposed: (u_cs, u_coh)
        self.u_fully_active_decomposed = None  # control input; shape (num_agents_max, 2) in fully active control
        self.time_step = None
        self.clock_time = None

        # For heuristic policy
        self.r_max = None
        self.alpha = None
        self.gamma = None
        self.heuristic_fitness = None
        self.initial_lazy_indices = None

        # Check convergence
        self.std_pos_hist = None  # shape (self.max_time_step,)
        self.std_vel_hist = None  # shape (self.max_time_step,)

        # Skip observation computation (for PSO where obs is discarded)
        self._skip_obs = False

        # Cached topology arrays (set in _cache_topology, called from reset)
        self._cache_neighbors = None
        self._cache_neighbor_counts = None
        self._cache_eye_eps = None
        self._cache_lambda_over_nc = None
        self._cache_sig_nv = None

        # For plotting
        self.state_hist = None

        # Seed
        self.seed_value = None

    def seed(self, seed=None):
        self.seed_value = seed
        np.random.seed(seed)
        super().seed(seed)

    def get_seed(self):
        return self.seed_value

    def reset(self):
        # Reset agent number
        self.num_agents = np.random.randint(self.num_agents_min, self.num_agents_max + 1)

        # Initialize padding
        self.is_padded = np.zeros(shape=(self.num_agents_max,), dtype=np.int32)
        self.is_padded[self.num_agents:] = 1

        # Initialize network topology
        self.net_topology_full = np.zeros((self.num_agents_max, self.num_agents_max), dtype=np.int32)
        self.update_topology_from_padding(init=True)
        self._cache_topology()

        # Initialize agent states
        # (1) position: shape (num_agents_max, 2)
        self.agent_pos = np.random.uniform(-self.l_bound / 2, self.l_bound / 2, (self.num_agents_max, 2))
        # (2) angle: shape (num_agents_max, 1); it's a 2-D array !!!!
        self.agent_ang = np.random.uniform(-np.pi, np.pi, (self.num_agents_max, 1))
        # (3) velocity v = [vx, vy] = [v*cos(theta), v*sin(theta)]; shape (num_agents_max, 2)
        self.agent_vel = self.v * np.concatenate((np.cos(self.agent_ang), np.sin(self.agent_ang)), axis=1)
        # (4) angular velocity; shape (num_agents_max, 1); it's a 2-D array !!!!
        self.agent_omg = np.zeros((self.num_agents_max, 1))

        self.u_fully_active, self.u_fully_active_decomposed = self.get_u(get_decomposed_u=True)

        # Pad agent states: it concatenates agent states with padding
        self.agent_states = self.pad_states()

        # No control input at t=0: w_0 = 0 because u is a function of the previous state

        # Get observation in a dict
        observation = self.get_observation(get_preprocessed_obs=self.use_preprocessed_obs)

        # Initialize convergence check variables
        self.std_pos_hist = np.zeros(self.max_time_step)
        self.std_vel_hist = np.zeros(self.max_time_step)

        # Update time step
        self.time_step = 0
        self.clock_time = 0

        # Get initial state if needed
        if self.get_state_hist:
            self.state_hist = np.zeros((self.max_time_step, self.num_agents_max, self.d_v), dtype=np.float32)
            self.state_hist[self.time_step, :, :] = self.agent_states

        # Update r_max
        if self.use_heuristics:
            mask = self.is_padded == 0
            self.r_max = np.zeros((self.num_agents_max,), dtype=np.float32)
            self.alpha, self.gamma = self.update_r_max_and_get_alpha_n_gamma(mask=mask, init_update=True)
            self.initial_lazy_indices = np.argsort(-self.alpha * self.gamma)  # shape: (num_agents_max)
        if not self._use_fixed_lazy_idx:
            for _ in range(4):
                print("    The env uses variable lazy index !!!  Not recommended but for testing purposes !!!")

        return observation

    def get_observation(self, get_preprocessed_obs, mask=None):
        # Get agent embeddings
        if get_preprocessed_obs:
            obs = self.preprocess_obs(mask=mask)
        else:
            agent_embeddings = self.agent_states[:, :self.d_v]

            # Get padding tokens
            pad_tokens = self.is_padded

            # Get observation in a dict
            obs = {
                "agent_embeddings": agent_embeddings,
                "pad_tokens": pad_tokens,
            }

        # Replace heading dim with u_fully_active (ACS control input as feature)
        obs["agent_embeddings"][:, 4] = self.u_fully_active

        if self.normalize_obs:
            # Normalize the positions by the half of the length of boundary
            obs["agent_embeddings"][:, :2] = obs["agent_embeddings"][:, :2] / (self.l_bound / 2.0)
            # Normalize the velocities by the maximum velocity
            obs["agent_embeddings"][:, 2:4] = obs["agent_embeddings"][:, 2:4] / self.v
            # Normalize u_fully_active by the maximum turn rate
            obs["agent_embeddings"][:, 4] = obs["agent_embeddings"][:, 4] / self.u_max
            # Normalize the angular velocity by the maximum angular velocity
            obs["agent_embeddings"][:, 5] = obs["agent_embeddings"][:, 5] / self.u_max

        if self.use_mlp_settings:
            # Use only agent_embeddings
            obs = obs["agent_embeddings"][:, :5]  # shape (num_agents_max, d_v-1)
            # Flatten the observation
            obs = obs.flatten()  # shape (num_agents_max * d_v-1,)

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
                        ):
        if changes_in_agents is None:
            return None

        # You must update padding before updating the network topology because the padding is used in the update
        # Dim check!
        assert changes_in_agents.shape == (self.num_agents_max,)
        # Update padding from the changes_in_agents
        # shapes: (num_agents_max,) in both cases
        gain_mask, loss_mask = self.update_padding_from_agent_changes(changes_in_agents)

        # Update network topology
        if np.sum(gain_mask) + np.sum(loss_mask) > 0:  # if there is any change
            self.update_topology_from_padding()
            self._cache_topology()

    def update_padding_from_agent_changes(self, changes_in_agents):
        # This function updates the padding from the changes in agents
        # Dim check!
        assert changes_in_agents.shape == (self.num_agents_max,)
        assert changes_in_agents.dtype == np.int32

        gain_mask = changes_in_agents == 1
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
        assert self.net_topology_full is not None
        mask = self.is_padded == 0 if mask is None else mask
        adj_mat = self.net_topology_full[mask, :][:, mask]  # shape: (num_agents, num_agents); (without padding)

        return adj_mat  # a copy of the adjacency matrix; not linked to the original object (i.e. self.adjacency_matrix)

    def _cache_topology(self, mask=None):
        mask = self.is_padded == 0 if mask is None else mask
        neighbors = self.net_topology_full[mask, :][:, mask]
        self._cache_neighbors = neighbors.astype(np.float64)                     # (num_agents, num_agents)
        self._cache_neighbor_counts = neighbors.sum(axis=1).astype(np.float64)   # (num_agents,)
        self._cache_eye_eps = np.eye(self.num_agents) * np.finfo(float).eps      # (num_agents, num_agents)
        self._cache_lambda_over_nc = self.lambda_ / self._cache_neighbor_counts  # (num_agents,)
        self._cache_sig_nv = self.sigma / (self.v * self._cache_neighbor_counts) # (num_agents,)

    def step(self,
             action: np.ndarray,
             ):
        obs, reward, done, info = self.auto_step(action) if self.do_auto_step else self.single_step(action)
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

    def auto_step_fully_active(self):
        # auto_step with fully active actions (array with ones for alive agents, zeros for dead agents)

        # Define fully active action
        fully_active_action = self.get_fully_active_action()

        # Call auto_step
        obs, episode_reward, done, info = self.auto_step(fully_active_action)

        return obs, episode_reward, done, info

    def get_fully_active_action(self):
        # Get fully active action (array with ones for alive agents, zeros for dead agents)
        # Please override this method if you want to use a different fully active action    `

        # Define fully active action
        fully_active_action = np.ones(self.num_agents_max)
        fully_active_action[self.is_padded == 1] = 0

        return fully_active_action

    def single_step(self,
                    action: np.ndarray,
                    ):
        # Note: action is not used as control input;
        #       it is used to weight the control input (i.e. laziness)

        # Get the mask
        mask = self.is_padded == 0

        # We view the action as the laziness of the agents
        c_lazy = self.interpret_action(model_output=action, mask=mask)

        # Update the control input:
        #   >>  u_{t+1} = f(x_t, y_t, vx_t, vy_t, θ_t, w_t);  != f(s_{t+1})
        #   Be careful! the control input uses the previous state (i.e. x_t, y_t, vx_t, vy_t, θ_t)
        #     This is because the current state may not be observable
        #     (e.g. in a decentralized network; forward compatibility)
        u_lazy_wo_disturbance = self.apply_laziness(
            c_lazy=c_lazy,
            u_fully_active=self.u_fully_active,
            u_fully_active_decomposed=self.u_fully_active_decomposed,
            mask=mask
        )
        # Get disturbance
        u_noise = self.get_disturbance(u_lazy=u_lazy_wo_disturbance, mask=mask)

        # Apply disturbance
        self.u_lazy = u_lazy_wo_disturbance if u_noise is NotImplemented else u_lazy_wo_disturbance + u_noise

        # Update the state (==agent_embeddings) based on the control input
        self.update_state(u=self.u_lazy, mask=mask)  # state transition 2/2

        # Get observation
        self.u_fully_active, self.u_fully_active_decomposed = self.get_u(mask=mask, get_decomposed_u=True)
        obs = None if self._skip_obs else \
            self.get_observation(get_preprocessed_obs=self.use_preprocessed_obs, mask=mask)  # next_observation

        # Get state history: get it before updating the time step (overriding the initial state from the reset method)
        if self.get_state_hist:
            if self.time_step < self.max_time_step-1:
                self.state_hist[self.time_step+1, :, :] = self.agent_states

        # Check if done
        done = True if self.is_done_in_paper(mask=mask) else False

        # Compute reward
        rewards = self.compute_reward()
        rewards = self.penalize_incomplete_task(rewards)
        target_reward = rewards[1] if self.use_L2_norm else rewards[0]
        auxiliary_reward = self.compute_auxiliary_reward(rewards=rewards, target_reward=target_reward,)

        std_pos = self.std_pos_hist[self.time_step]
        std_vel = self.std_vel_hist[self.time_step]

        # Update r_max, alpha, and gamma for the heuristic policy, if necessary
        if self.use_heuristics:
            self.alpha, self.gamma = self.update_r_max_and_get_alpha_n_gamma(mask=mask)

        # Update the clock time and the time step count
        self.clock_time += self.dt
        self.time_step += 1

        # Get info
        info = {
            "std_pos": std_pos,
            "std_vel": std_vel,
            "original_rewards": rewards,
        }
        # Switch the reward to the auxiliary reward if it is computed
        reward = auxiliary_reward if auxiliary_reward is not NotImplemented else target_reward

        return obs, reward, done, info

    def interpret_action(self, model_output, mask):
        # This method is used to interpret the action
        # Please override this method in your task

        # Interpretation: model_output -> interpreted_action
        interpreted_action = model_output
        # Mask out the padding agents (only when it needs to be masked out for efficiency)
        if self.num_agents < self.num_agents_max:
            interpreted_action = model_output.copy()  # Ray requires a copy
            interpreted_action[~mask] = 0
        # Validation: interpreted_action -> c_lazy
        interpreted_and_validated_action = self.validate_action(interpreted_action, mask)

        return interpreted_and_validated_action

    def validate_action(self, interpreted_action, mask):
        # This method is used to validate the action
        # Please override this method in your task and include it in the interpret_action method

        # Clip the interpreted action into the laziness range
        laziness_lower_bound = 0
        laziness_upper_bound = 1
        interpreted_and_validated_action = np.clip(interpreted_action, laziness_lower_bound, laziness_upper_bound)

        return interpreted_and_validated_action

    def apply_laziness(self, c_lazy, u_fully_active, u_fully_active_decomposed, mask):
        # This defines how to generate u_lazy from the given c_lazy from the model and the u_fa_s from the control law
        # Please extend this method for your needs

        # Here, we get u_lazy = c_lazy * u_fa
        u_lazy = u_fully_active * c_lazy

        return u_lazy

    def get_disturbance(self, u_lazy, mask):
        # Extend your disturbance mechanism (here, it's an placeholder; identity process)
        return NotImplemented

    def compute_auxiliary_reward(
            self,
            *,
            rewards,        # shape: (2,); 0: L1 norm; 1: L2 norm
            target_reward,  # scalar; either L1 or L2 norm depending on self.use_L2_norm
    ):
        # Override in subclass to define a shaped reward. The original reward is logged
        # via info["original_rewards"] — use RLlib callbacks to track it on tensorboard.
        return NotImplemented

    def get_u(self, mask=None, get_decomposed_u=False):
        # mask: mask for the non-padding agents; shape: (num_agents_max,)
        # get_decomposed_u: if True, returns u_decomposed; otherwise, returns u

        # Get mask
        mask = self.is_padded == 0 if mask is None else mask
        assert mask.dtype == np.bool  # np.bool_ depending on your ray (or numpy) version
        assert np.sum(mask) == self.num_agents

        # Relative quantities via single broadcast
        pos = self.agent_pos[mask]   # (num_agents, 2)
        ang = self.agent_ang[mask]   # (num_agents, 1)
        vel = self.agent_vel[mask]   # (num_agents, 2)
        rel_pos = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]    # (num_agents, num_agents, 2)
        rel_ang = ang.ravel()[np.newaxis, :] - ang.ravel()[:, np.newaxis]  # (num_agents, num_agents)
        rel_vel = vel[np.newaxis, :, :] - vel[:, np.newaxis, :]    # (num_agents, num_agents, 2)
        rel_dist = np.sqrt(rel_pos[:, :, 0]**2 + rel_pos[:, :, 1]**2)     # (num_agents, num_agents)

        # Use cached topology arrays (set in reset / update_topology_from_padding)
        Neighbors = self._cache_neighbors
        lambda_over_nc = self._cache_lambda_over_nc
        sig_NV = self._cache_sig_nv
        eye_eps = self._cache_eye_eps

        # 1. Alignment: u_cs = (λ/|N_i|) * Σ_j∈N_i [ψ(r_ij) sin(θ_j - θ_i)]
        psi = (1 + rel_dist ** 2) ** (-self.beta)  # (n, n)
        u_cs = lambda_over_nc * (Neighbors * psi * np.sin(rel_ang)).sum(axis=1)  # (n,)

        # 2. Cohesion: u_coh
        r_ij = rel_dist + eye_eps  # (n, n)
        k1_2rij2 = self.k1 / (2 * r_ij ** 2)  # (n, n)
        k2_2rij = self.k2 / (2 * r_ij)  # (n, n)
        v_dot_p = np.einsum('ijk,ijk->ij', rel_vel, rel_pos)  # (n, n)
        rij_R = rel_dist - self.R  # (n, n)
        ang_flat = ang.ravel()
        ang_vec = np.empty((self.num_agents, 2))
        ang_vec[:, 0] = -np.sin(ang_flat)
        ang_vec[:, 1] = np.cos(ang_flat)
        ang_dot_p = np.einsum('ik,ijk->ij', ang_vec, rel_pos)  # (num_agents, num_agents) — no tile
        need_sum = (k1_2rij2 * v_dot_p + k2_2rij * rij_R) * ang_dot_p  # (num_agents, num_agents)
        u_coh = sig_NV * (Neighbors * need_sum).sum(axis=1)  # (num_agents,)

        # Compute control input
        if not get_decomposed_u:
            u_local = u_cs + u_coh  # + u_sep + u_comm  # shape: (num_agents,)
            # Saturation
            u_local = np.clip(u_local, -self.u_max, self.u_max)  # shape: (num_agents,)
            # Get u
            u = np.zeros(self.num_agents_max, dtype=np.float32)  # shape: (num_agents_max,)
            u[mask] = u_local

            return u  # shape: (num_agents_max,)
        else:
            # Define local control inputs (decomposed and summed)
            u_local_decomposed = np.stack((u_cs, u_coh), axis=1)  # shape: (num_agents, 2)
            u_local_sum = np.sum(u_local_decomposed, axis=1)  # shape: (num_agents,)

            # Find the agents where the sum exceeds the maximum allowable control input
            mask_exceed = np.abs(u_local_sum) > self.u_max  # shape: (num_agents,)
            # Note: u_local_sum[mask_exceed] never includes zero elements once self.u_max is set to a non-zero value
            # So, it's safe to divide something by u_local_sum[mask_exceed]

            # Scale down the u_local components for those agents
            scaling_factor = self.u_max / np.abs(u_local_sum[mask_exceed])  # shape: (num_agents,) > 0
            u_local_decomposed[mask_exceed] *= scaling_factor[:, np.newaxis]  # shape: (num_agents, 2)
            # Update u_local_sum (==saturation)
            u_local_sum = np.sum(u_local_decomposed, axis=1)  # shape: (num_agents,)

            # Get u
            u = np.zeros(self.num_agents_max, dtype=np.float32)  # shape: (num_agents_max,)
            u_decomposed = np.zeros((self.num_agents_max, 2), dtype=np.float32)  # shape: (num_agents_max, 2)
            u[mask] = u_local_sum
            u_decomposed[mask] = u_local_decomposed

            return u, u_decomposed  # shape: (num_agents_max,)

    def update_state(self, u, mask):
        # Update the state based on the control input
        if self.use_custom_ray:  # memory inefficient! but ray requires immutable objects..
            self.agent_omg = np.copy(self.agent_omg)
            self.agent_pos = np.copy(self.agent_pos)
            self.agent_ang = np.copy(self.agent_ang)
            self.agent_vel = np.copy(self.agent_vel)

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

        # (4) Update velocity:
        # >>  v_{t+1} = V * [cos(ang_{t+1}), sin(ang_{t+1})]; Be careful: ang_{t+1} is the angle at time t+1
        self.agent_vel[mask] = self.v * np.concatenate(
            [np.cos(self.agent_ang[mask]), np.sin(self.agent_ang[mask])], axis=1)

        # Update the state
        self.agent_states = self.concat_states(self.agent_pos, self.agent_vel, self.agent_ang, self.agent_omg)

    def compute_reward(self):
        # J_{energy}   = [V/N * sum_{i=1}^{N} sum_{k=1}^{t_c} ||u_i(k)||^2] + t_c : total cost throughout the time
        # J_{energy}_t = [V/N * sum_{i=1}^{N} ||u_i(t)||^2] + dt : cost at time t

        # Get control input data: shape (num_agents_max,)
        u_t = self.u_lazy

        # Check the control input of the padding agents
        assert np.all(u_t[self.is_padded == 1] == 0)

        # Compute reward of all agents into a scalar: centralized
        control_cost_L1 = self.dt * (self.v / self.num_agents) * np.linalg.norm(u_t, 1)  # from the paper
        control_cost_L2 = self.dt * (self.v / self.num_agents) * np.sum(u_t ** 2)  # sum of squares; matches the paper's J control term
        fuel_cost = self.dt
        total_cost_L1 = control_cost_L1 + self.rho * fuel_cost
        total_cost_L2 = control_cost_L2 + self.rho * fuel_cost
        reward_L1 = - total_cost_L1  # maximize the reward == minimize the cost
        reward_L2 = - total_cost_L2  # maximize the reward == minimize the cost
        # i.e. reward is negative in most cases in this environment

        # Check the data type of the reward
        assert isinstance(reward_L1, float) or isinstance(reward_L1, np.float32)

        rewards = np.array([reward_L1, reward_L2], dtype=np.float32)

        return rewards  # shape: (2,)

    def penalize_incomplete_task(self, rewards):
        # Penalize the RL-agent if the task is not completed
        # time: before update time step in the step func
        t = self.time_step
        if t >= self.max_time_step-1:
            if self.incomplete_episode_penalty == -666666:
                pos_penalty = -1.0 * np.maximum(self.std_pos_hist[t]-40, 0)
                vel_penalty = -10.0 * np.maximum(self.std_vel_hist[t]-0.1, 0)
                expectation_penalty = 2
                rewards = rewards + expectation_penalty*(pos_penalty + vel_penalty)
            else:
                rewards = rewards + self.incomplete_episode_penalty

        return rewards

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
            self.std_pos_hist = np.copy(self.std_pos_hist)
            self.std_vel_hist = np.copy(self.std_vel_hist)
        self.std_pos_hist[self.time_step] = pos_std
        self.std_vel_hist[self.time_step] = vel_std

        # 1. Check position standard deviation
        pos_converged = True if pos_std < self.std_pos_converged else False
        vel_converged = True if vel_std < self.std_vel_converged else False
        std_val_converged = True if pos_converged and vel_converged else False

        # Check 2 and 3 only if both std values are below threshold
        if std_val_converged and not self.use_fixed_horizon:
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

        return done

    @staticmethod
    def wrap_to_pi(angles):
        # Wrap angles to [-pi, pi]
        return (angles + np.pi) % (2 * np.pi) - np.pi

    def render(self, mode="human"):
        # Render the environment
        pass

    def compute_heuristic_action(self, mask=None, use_fixed_lazy=None, num_lazy_agents=1):
        # This method computes the heuristic action based on the current state.

        # Get mask
        mask = self.is_padded == 0 if mask is None else mask  # shape: (num_agents_max,)

        # Get fitness vector
        self.heuristic_fitness = np.zeros((self.num_agents_max,), dtype=np.float32)  # shape: (num_agents_max,)
        self.heuristic_fitness[mask] = self.alpha[mask] * self.gamma[mask]  # shape: (num_agents_max,)
        # Assert the fitness is in the range [0, 1] unless self.time_step==0
        if self.time_step > 0:
            assert np.all(0 <= self.heuristic_fitness) and np.all(self.heuristic_fitness <= 1)

        # Get who's lazy: index of the lazy agent (highest fitness)
        assert (num_lazy_agents == 1) or (isinstance(num_lazy_agents, int) and num_lazy_agents > 1),\
            "Invalid value for num_lazy_agents"
        assert num_lazy_agents <= self.num_agents  # do NOT request non-existing lazy agents
        use_fixed_lazy = self._use_fixed_lazy_idx if use_fixed_lazy is None else use_fixed_lazy

        if use_fixed_lazy:
            top_lazy_agent_indices = self.initial_lazy_indices[:num_lazy_agents]  # shape: (num_lazy_agents,)
        else:  # `-`: for descending order; shape: (num_lazy_agents,)
            top_lazy_agent_indices = np.argsort(-self.heuristic_fitness)[:num_lazy_agents]
        # Assert the lazy agent is alive
        assert np.all(mask[top_lazy_agent_indices])

        # Get the laziness value of the lazy agent (sole-lazy heuristic in this case)
        G = 0.8  # gain from the paper; prevents C_lazy(t) from decreasing near zero
        # Assert the laziness is in the range [0, 1-G]
        eps = 1e-7  # to prevent numerical errors
        if self.time_step > 0:
            laziness_values = 1 - G * self.heuristic_fitness[top_lazy_agent_indices]  # shape: (num_lazy_agents,)
            assert np.all(1 - G - eps <= laziness_values) and np.all(laziness_values <= 1 + eps)
        elif self.time_step == 0:  # shape: (num_lazy_agents,)
            laziness_values = np.full(num_lazy_agents, 1 - G)  # starts from 1-G (minimum laziness for all)
        else:
            raise ValueError(f"self.time_step must be non-negative! But self.time_step=={self.time_step}")

        # Get lazy action
        lazy_action = np.zeros((self.num_agents_max,), dtype=np.float32)  # shape: (num_agents_max,)
        lazy_action[mask] = 1.0  # locally fully active
        lazy_action[top_lazy_agent_indices] = laziness_values  # assigns the laziness to the lazy agents

        return lazy_action


    def update_r_max_and_get_alpha_n_gamma(self, mask=None, init_update=False):
        # Get mask
        # Make sure to use the mask for the live agents only; arr[mask] shows the living agents only
        mask = self.is_padded == 0 if mask is None else mask

        # Get center coordinates
        center = np.mean(self.agent_pos[mask, :], axis=0)  # shape: (2,)

        # Get current r
        r_vec = np.zeros((self.num_agents_max, 2), dtype=np.float32)  # shape: (num_agents_max, 2)
        r_vec[mask, :] = center - self.agent_pos[mask, :]
        r = np.zeros((self.num_agents_max,), dtype=np.float32)  # shape: (num_agents_max,)
        r[mask] = np.linalg.norm(r_vec[mask, :], axis=1)

        # Update max r
        self.r_max = np.maximum(self.r_max, r)  # shape: (num_agents_max,)

        # Get alpha
        alpha = np.zeros((self.num_agents_max,), dtype=np.float32)  # shape: (num_agents_max,)
        if init_update:
            alpha[mask] = r[mask]
        else:
            alpha[mask] = r[mask] / self.r_max[mask]

        # Get current v
        v_vec = np.zeros((self.num_agents_max, 2), dtype=np.float32)  # shape: (num_agents_max, 2)
        v_vec[mask, :] = self.agent_vel[mask, :]
        v = self.v  # must be a scalar

        # Get gamma from gamma = 0.5*(1-cos(phi)), where cos(phi) = |v_vec * r_vec| / (|v_vec|*|r_vec|)
        cos_phi = np.zeros((self.num_agents_max,), dtype=np.float32)  # shape: (num_agents_max,)
        cos_phi[mask] = np.sum(v_vec[mask] * r_vec[mask], axis=1) / (v * r[mask])
        gamma = np.zeros((self.num_agents_max,), dtype=np.float32)  # shape: (num_agents_max,)
        gamma[mask] = 0.5 * (1 - cos_phi[mask])
        # Clip to [0, 1] to handle floating point error
        gamma[mask] = np.clip(gamma[mask], 0, 1)

        return alpha, gamma


class LazyAgentsCentralizedPendReward(LazyAgentsCentralized):

    def __init__(self, config):
        super().__init__(config)

        # Get pendulum reward weight from config
        # Note: the default values are not optimal; they were determined for backward compatibility
        self.w_control = self.config["w_control"] if "w_control" in self.config else 0.02  # 0.001
        self.w_pos = self.config["w_pos"] if "w_pos" in self.config else 1.0  # 1.0
        self.w_vel = self.config["w_vel"] if "w_vel" in self.config else 0.2  # 0.2

    def compute_auxiliary_reward(  # should not be static when overriding
            self,
            *,  # enforce keyword arguments to avoid confusion
            rewards,  # shape: (2,); 0: L1 norm; 1: L2 norm
            target_reward,  # scalar; either L1 or L2 norm depending on the self.use_L2_norm
    ):
        # Get control reward
        control_reward = target_reward + self.rho * self.dt  # remove fuel cost

        # Get position error reward
        std_pos = self.std_pos_hist[self.time_step]
        std_pos_target = self.std_pos_converged - 2.5
        std_pos_error = (std_pos - std_pos_target)**2  # (100-40)**2 = 3600
        pos_error_reward = - (1/3600) * np.maximum(std_pos_error, 0.0)
        # Get velocity error reward
        std_vel = self.std_vel_hist[self.time_step]
        std_vel_target = self.std_vel_converged - 0.05
        std_vel_error = (std_vel - std_vel_target)**2  # (15-0.05)**2 = 223.5052
        vel_error_reward = - (1/220) * np.maximum(std_vel_error, 0.0)
        # Get auxiliary reward
        w_con = 1.0 * self.w_control  # 0.02
        w_pos = 1.0 * self.w_pos  # 1.0
        w_vel = 1.0 * self.w_vel  # 0.2
        auxiliary_reward = (
                w_con * control_reward +
                w_pos * pos_error_reward +
                w_vel * vel_error_reward
        )

        return auxiliary_reward



