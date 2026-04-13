"""
Networked variant of the LazyAgents environments.

Adds support for arbitrary (non-fully-connected) communication topologies and an
optional ego-observation mode for decentralized / parameter-sharing policies.

Design notes
------------
The base ``LazyAgentsCentralized`` from ``env/envs.py`` is left untouched. This
module subclasses it (and the pendulum-reward training variant) and overrides
the topology-related methods plus, optionally, the observation builder.

The topology only affects:

  1. The ACS control law in ``get_u()`` (via ``view_adjacency_matrix``).
  2. The neighbor set used by the ego observation (when ``obs_mode == "ego"``).

When ``obs_mode == "central"`` the observation shape and content are *exactly*
the same as the original env (modulo ``u_fully_active`` which is the topology-
dependent local control), so a model trained on the original centralized env
can still consume it.

Topologies
~~~~~~~~~~

* ``fully_connected``  -- identical to the base env, included for symmetry.
* ``ring``             -- 1-D circular ring of width ``k`` (each agent connects
  to its ``k`` left and ``k`` right neighbors in index order). Static.
* ``k_nearest``        -- each agent connects to its ``k`` nearest other agents
  (by Euclidean distance). Symmetrized with logical OR. Dynamic.
* ``disk``             -- each agent connects to all other agents within range
  ``comm_range``. Dynamic.
* ``er_random``        -- Erdos-Renyi random graph with edge probability ``p``,
  resampled at every reset (and optionally per step). Static by default.

Self-loops are always added so that ``Neighbors.sum(axis=1) >= 1`` (avoids the
``get_u`` divide-by-zero path).

Observation modes
~~~~~~~~~~~~~~~~~

* ``central`` -- same as the base env: a single ``(N, d_v)`` tensor in the
  global CoM/avg-heading frame. ``u_fully_active`` is recomputed under the
  current topology.
* ``ego``     -- per-agent local-frame view: ``(N, N, d_v)`` tensor where row
  ``i`` is agent ``i``'s view of the swarm in agent ``i``'s own neighbor frame
  (CoM and average heading taken over agent ``i``'s neighbors only, including
  self). Slots that are not neighbors of ``i`` are zero-padded. The companion
  mask ``ego_neighbor_mask`` (shape ``(N, N)``, 1 = valid neighbor, 0 = pad)
  carries the per-agent pad info.

The action space is unchanged from the parent: a per-agent laziness vector of
shape ``(num_agents_max,)``.

Reward stays centralized (CTDE).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from gym.spaces import Box, Dict

from env.envs import LazyAgentsCentralized, LazyAgentsCentralizedPendReward


# Topology types this module knows how to build.
SUPPORTED_TOPOLOGIES = (
    "fully_connected",
    "ring",
    "k_nearest",
    "disk",
    "er_random",
    "star",         # delegated to parent
    "line",         # path graph (ring without wraparound)
    "wheel",        # ring_k1 + center node
    "binary_tree",  # heap-style binary tree
    "mst",          # minimum spanning tree of pairwise distances (dynamic)
    "delaunay",     # Delaunay triangulation of positions (dynamic)
)

# Topologies whose adjacency depends on agent positions and must therefore be
# rebuilt every simulation step.
DYNAMIC_TOPOLOGIES = ("k_nearest", "disk", "mst", "delaunay")


def _normalize_topology_config(cfg) -> dict:
    """Accepts either a string or a dict and returns a canonical dict.

    Examples
    --------
    >>> _normalize_topology_config("ring")
    {'type': 'ring', 'params': {}}
    >>> _normalize_topology_config({"type": "k_nearest", "k": 4})
    {'type': 'k_nearest', 'params': {'k': 4}}
    """

    if cfg is None:
        return {"type": "fully_connected", "params": {}}
    if isinstance(cfg, str):
        return {"type": cfg, "params": {}}
    if not isinstance(cfg, dict):
        raise TypeError(
            f"network_topology must be a string or dict, got {type(cfg).__name__}"
        )
    if "type" not in cfg:
        raise KeyError("network_topology dict requires a 'type' field")
    out_type = cfg["type"]
    out_params = {k: v for k, v in cfg.items() if k != "type"}
    return {"type": out_type, "params": out_params}


class NetworkedLazyAgents(LazyAgentsCentralized):
    """Topology-aware extension of ``LazyAgentsCentralized``.

    Extra config keys
    -----------------
    network_topology : str or dict
        See module docstring. A string is treated as ``{"type": str, "params":{}}``.
        Topology-specific parameters live under ``params`` (or top-level keys when
        a dict is given). Examples::

            "fully_connected"
            "ring"
            {"type": "ring", "k": 2}
            {"type": "k_nearest", "k": 3}
            {"type": "disk", "comm_range": 90.0}
            {"type": "er_random", "p": 0.3, "resample_each_step": False}

    obs_mode : {"central", "ego"}
        ``central`` (default) keeps the original observation. ``ego`` returns
        per-agent local-frame views as described above.
    ego_use_local_frame : bool, default True
        In ego mode, whether to translate/rotate each agent's neighbor view into
        the agent's local frame (True) or just slice the global frame (False).
    """

    def __init__(self, config: dict):
        # Capture our extra fields BEFORE the parent constructor runs because the
        # parent constructor reads ``config["network_topology"]`` to set
        # ``self.net_type``. We translate the dict form into the parent's string
        # form (using "fully_connected" as a benign default) and then override
        # the per-call topology builder.
        topology_cfg = _normalize_topology_config(config.get("network_topology"))
        topology_type = topology_cfg["type"]
        topology_params = topology_cfg["params"]

        if topology_type not in SUPPORTED_TOPOLOGIES:
            raise ValueError(
                f"Unsupported network_topology type '{topology_type}'. "
                f"Supported: {SUPPORTED_TOPOLOGIES}"
            )

        # Hand the parent a topology name it understands so its __init__ doesn't
        # crash. We override the actual builder anyway.
        config = dict(config)  # don't mutate caller
        if topology_type in ("fully_connected", "star"):
            config["network_topology"] = topology_type
        else:
            config["network_topology"] = "fully_connected"

        super().__init__(config)

        # Restore the *real* topology type after parent init.
        self._topology_type = topology_type
        self._topology_params = dict(topology_params)
        self._is_dynamic_topology = topology_type in DYNAMIC_TOPOLOGIES
        # ER random gets a per-episode seedable RNG so trials are repeatable.
        self._topology_rng = np.random.RandomState(self._topology_params.get("seed", None))

        # Observation mode.
        self.obs_mode = config.get("obs_mode", "central")
        if self.obs_mode not in ("central", "ego"):
            raise ValueError(f"obs_mode must be 'central' or 'ego', got {self.obs_mode}")
        self.ego_use_local_frame = bool(config.get("ego_use_local_frame", True))

        if self.obs_mode == "ego":
            # Per-agent local view of all agents in the swarm.
            self.observation_space = Dict(
                {
                    "agent_embeddings": Box(
                        low=-np.inf,
                        high=np.inf,
                        shape=(self.num_agents_max, self.num_agents_max, self.d_v),
                        dtype=np.float32,
                    ),
                    "ego_neighbor_mask": Box(
                        low=0,
                        high=1,
                        shape=(self.num_agents_max, self.num_agents_max),
                        dtype=np.int32,
                    ),
                    "pad_tokens": Box(
                        low=0, high=1, shape=(self.num_agents_max,), dtype=np.int32
                    ),
                }
            )

    # ------------------------------------------------------------------ #
    # Topology construction
    # ------------------------------------------------------------------ #

    def update_topology_from_padding(self, init: bool = False):
        """Builds the adjacency matrix according to ``self._topology_type``.

        For dynamic topologies (``k_nearest``, ``disk``, ``mst``, ``delaunay``)
        we set a fully-self-looped adjacency here as a placeholder; the *real*
        topology is rebuilt from positions in :meth:`update_dynamic_topology`
        after positions are initialized in ``reset`` and after each state
        update.
        """

        # Dynamic topologies will be filled in once positions exist.
        if self._is_dynamic_topology:
            adj_mat = np.eye(self.num_agents, dtype=np.int32)
            self._assign(adj_mat, init=init)
            return

        if self._topology_type == "fully_connected":
            adj_mat = np.ones((self.num_agents, self.num_agents), dtype=np.int32)
        elif self._topology_type == "star":
            # Use the parent's logic to populate `center_in_star_net` etc.
            return super().update_topology_from_padding(init=init)
        elif self._topology_type == "ring":
            adj_mat = self._build_ring(self.num_agents, k=int(self._topology_params.get("k", 1)))
        elif self._topology_type == "line":
            adj_mat = self._build_line(self.num_agents)
        elif self._topology_type == "wheel":
            adj_mat = self._build_wheel(self.num_agents)
        elif self._topology_type == "binary_tree":
            adj_mat = self._build_binary_tree(self.num_agents)
        elif self._topology_type == "er_random":
            adj_mat = self._build_er(
                self.num_agents,
                p=float(self._topology_params.get("p", 0.3)),
            )
        else:
            raise ValueError(f"Unhandled static topology: {self._topology_type}")

        self._assign(adj_mat, init=init)

    def update_dynamic_topology(self):
        """Rebuilds adjacency for position-dependent topologies."""

        if not self._is_dynamic_topology:
            return

        mask = self.is_padded == 0
        positions = self.agent_pos[mask]  # (n, 2)
        n = positions.shape[0]
        if n == 0:
            return

        if self._topology_type == "k_nearest":
            k = int(self._topology_params.get("k", 3))
            adj_mat = self._build_knn(positions, k=k)
        elif self._topology_type == "disk":
            comm_range = float(self._topology_params.get("comm_range", self.R * 1.5))
            adj_mat = self._build_disk(positions, comm_range=comm_range)
        elif self._topology_type == "mst":
            adj_mat = self._build_mst(positions)
        elif self._topology_type == "delaunay":
            adj_mat = self._build_delaunay(positions)
        else:
            return

        self._assign(adj_mat, init=False)

    def _assign(self, adj_mat: np.ndarray, init: bool):
        """Assigns ``adj_mat`` to the env's adjacency state, with self-loops."""

        np.fill_diagonal(adj_mat, 1)  # always self-loops
        self.adjacency_matrix = adj_mat
        if init:
            self.net_topology_full[: self.num_agents, : self.num_agents] = adj_mat
        else:
            # Clear and rewrite the live block.
            self.net_topology_full[:] = 0
            self.net_topology_full[: self.num_agents, : self.num_agents] = adj_mat

    # ---- topology builders -------------------------------------------- #

    @staticmethod
    def _build_ring(n: int, k: int = 1) -> np.ndarray:
        """Symmetric ring of width ``k`` (each node connects to k nearest indices)."""
        adj = np.zeros((n, n), dtype=np.int32)
        if n <= 1:
            return adj
        for offset in range(1, k + 1):
            idx = np.arange(n)
            adj[idx, (idx + offset) % n] = 1
            adj[idx, (idx - offset) % n] = 1
        return adj

    def _build_er(self, n: int, p: float) -> np.ndarray:
        """Symmetric Erdos-Renyi graph (no isolated agents are forced; self-loops added later)."""
        if n <= 1:
            return np.zeros((n, n), dtype=np.int32)
        upper = (self._topology_rng.random_sample((n, n)) < p).astype(np.int32)
        upper = np.triu(upper, k=1)
        adj = upper + upper.T
        return adj

    @staticmethod
    def _build_knn(positions: np.ndarray, k: int) -> np.ndarray:
        """k-nearest neighbors (excluding self), symmetrized with logical OR."""
        n = positions.shape[0]
        if n <= 1:
            return np.zeros((n, n), dtype=np.int32)
        diffs = positions[:, None, :] - positions[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        np.fill_diagonal(dists, np.inf)  # exclude self in neighbor selection
        kk = min(k, n - 1)
        # argpartition picks the kk smallest distances per row
        nn_idx = np.argpartition(dists, kth=kk - 1, axis=1)[:, :kk]
        adj = np.zeros((n, n), dtype=np.int32)
        rows = np.repeat(np.arange(n), kk)
        cols = nn_idx.reshape(-1)
        adj[rows, cols] = 1
        # Symmetrize: i -- j if i in NN(j) OR j in NN(i)
        adj = np.maximum(adj, adj.T)
        return adj

    @staticmethod
    def _build_disk(positions: np.ndarray, comm_range: float) -> np.ndarray:
        """Disk graph: connect all agent pairs within ``comm_range``."""
        n = positions.shape[0]
        if n <= 1:
            return np.zeros((n, n), dtype=np.int32)
        diffs = positions[:, None, :] - positions[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        adj = (dists <= comm_range).astype(np.int32)
        np.fill_diagonal(adj, 0)  # the _assign step will re-add self-loops
        return adj

    @staticmethod
    def _build_line(n: int) -> np.ndarray:
        """Path graph: ring_k1 minus the wraparound. Always 1 component (n>=1)."""
        adj = np.zeros((n, n), dtype=np.int32)
        if n <= 1:
            return adj
        idx = np.arange(n - 1)
        adj[idx, idx + 1] = 1
        adj[idx + 1, idx] = 1
        return adj

    @staticmethod
    def _build_wheel(n: int) -> np.ndarray:
        """Wheel: ring_k1 over the n-1 outer nodes, plus a center node 0
        connected to every outer. Always 1 component for n>=1.

        Index 0 is the hub; indices 1..n-1 are the outer ring.
        """
        adj = np.zeros((n, n), dtype=np.int32)
        if n <= 1:
            return adj
        # Hub-to-outer
        outer = np.arange(1, n)
        adj[0, outer] = 1
        adj[outer, 0] = 1
        # Outer ring (only if there are at least 2 outer nodes)
        if n >= 3:
            for i in range(1, n):
                j = ((i - 1) + 1) % (n - 1) + 1  # next outer index
                adj[i, j] = 1
                adj[j, i] = 1
        return adj

    @staticmethod
    def _build_binary_tree(n: int) -> np.ndarray:
        """Heap-style binary tree: parent of i (i>=1) is (i-1)//2."""
        adj = np.zeros((n, n), dtype=np.int32)
        for i in range(1, n):
            parent = (i - 1) // 2
            adj[i, parent] = 1
            adj[parent, i] = 1
        return adj

    @staticmethod
    def _build_mst(positions: np.ndarray) -> np.ndarray:
        """Minimum spanning tree of the pairwise-distance graph."""
        n = positions.shape[0]
        if n <= 1:
            return np.zeros((n, n), dtype=np.int32)
        from scipy.sparse.csgraph import minimum_spanning_tree
        diffs = positions[:, None, :] - positions[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        # csgraph treats 0 as "no edge"; replace the diagonal so the MST
        # routine doesn't think pairs of identical points are absent edges.
        # In practice the diagonal is already 0 and that's fine; the routine
        # also handles non-zero pairs as edges.
        mst = minimum_spanning_tree(dists).toarray()
        adj = ((mst + mst.T) > 0).astype(np.int32)
        np.fill_diagonal(adj, 0)
        return adj

    @staticmethod
    def _build_delaunay(positions: np.ndarray) -> np.ndarray:
        """Delaunay triangulation of the live agents (in 2D).

        Falls back to a path graph for degenerate cases (n<3 or collinear).
        Always yields a connected graph for n>=2.
        """
        n = positions.shape[0]
        if n <= 1:
            return np.zeros((n, n), dtype=np.int32)
        if n == 2:
            adj = np.zeros((n, n), dtype=np.int32)
            adj[0, 1] = adj[1, 0] = 1
            return adj
        from scipy.spatial import Delaunay
        try:
            from scipy.spatial import QhullError  # newer scipy
        except ImportError:
            from scipy.spatial.qhull import QhullError  # older scipy
        adj = np.zeros((n, n), dtype=np.int32)
        try:
            tri = Delaunay(positions)
            for simplex in tri.simplices:
                # Each simplex is a triangle (3 indices); add its 3 edges.
                for u, v in ((simplex[0], simplex[1]),
                             (simplex[1], simplex[2]),
                             (simplex[2], simplex[0])):
                    adj[u, v] = 1
                    adj[v, u] = 1
        except (QhullError, ValueError):
            # Degenerate (e.g. collinear) -- fall back to a sorted-x path so
            # the swarm stays connected.
            order = np.argsort(positions[:, 0])
            for a, b in zip(order[:-1], order[1:]):
                adj[a, b] = 1
                adj[b, a] = 1
        return adj

    # ------------------------------------------------------------------ #
    # Reset / step plumbing
    # ------------------------------------------------------------------ #

    def reset(self):
        # Reseed the topology RNG so ER graphs are repeatable per episode.
        if self.seed_value is not None:
            self._topology_rng = np.random.RandomState(self.seed_value)
        else:
            self._topology_rng = np.random.RandomState(self._topology_params.get("seed", None))

        obs = super().reset()

        # The parent runs `update_topology_from_padding(init=True)` BEFORE
        # positions are sampled, so dynamic topologies need a fresh build now.
        if self._is_dynamic_topology:
            self.update_dynamic_topology()
            # u_fully_active was computed under the placeholder topology; rerun.
            self.u_fully_active, self.u_fully_active_decomposed = self.get_u(get_decomposed_u=True)
            self.agent_states = self.pad_states()
            obs = self.get_observation(get_preprocessed_obs=self.use_preprocessed_obs)

        # Ego mode rebuilds the observation from scratch.
        if self.obs_mode == "ego":
            obs = self._build_ego_observation()
        return obs

    def update_state(self, u, mask):
        """Update state then refresh dynamic topologies before the next step."""
        super().update_state(u, mask)
        if self._is_dynamic_topology:
            self.update_dynamic_topology()

    # ------------------------------------------------------------------ #
    # Observation
    # ------------------------------------------------------------------ #

    def get_observation(self, get_preprocessed_obs, mask=None):
        if self.obs_mode == "central":
            return super().get_observation(get_preprocessed_obs, mask=mask)
        return self._build_ego_observation(mask=mask)

    def _build_ego_observation(self, mask: Optional[np.ndarray] = None) -> dict:
        """Per-agent local-frame view of the swarm.

        Returns
        -------
        dict with keys
            agent_embeddings : (N, N, d_v) float32
            ego_neighbor_mask : (N, N) int32 -- 1 = valid neighbor of agent i
            pad_tokens : (N,) int32           -- 1 = i is padded (unused)
        """

        N = self.num_agents_max
        D = self.d_v
        live_mask = self.is_padded == 0 if mask is None else mask

        ego_embeddings = np.zeros((N, N, D), dtype=np.float32)
        ego_neighbor_mask = np.zeros((N, N), dtype=np.int32)
        pad_tokens = self.is_padded.copy()

        if not live_mask.any():
            return {
                "agent_embeddings": ego_embeddings,
                "ego_neighbor_mask": ego_neighbor_mask,
                "pad_tokens": pad_tokens,
            }

        # Pull global state for live agents.
        positions = self.agent_pos  # (N, 2)
        velocities = self.agent_vel  # (N, 2)
        angles = self.wrap_to_pi(self.agent_ang)  # (N, 1)
        omegas = self.agent_omg  # (N, 1)
        u_fa = self.u_fully_active  # (N,)

        adj_full = self.net_topology_full  # (N, N) padded; ints

        live_indices = np.where(live_mask)[0]
        for i in live_indices:
            # Neighbor set of i (always includes self because of self-loops).
            neighbor_row = adj_full[i].astype(bool)
            neighbor_row &= live_mask  # exclude padded slots
            n_idx = np.where(neighbor_row)[0]
            if n_idx.size == 0:
                continue

            n_pos = positions[n_idx]  # (k, 2)
            n_ang = angles[n_idx]  # (k, 1)
            n_omg = omegas[n_idx]  # (k, 1)
            n_uf = u_fa[n_idx]  # (k,)

            if self.ego_use_local_frame:
                center = np.mean(n_pos, axis=0)  # local CoM (over neighbors)
                avg_heading = np.mean(n_ang)
                rot = np.array(
                    [
                        [np.cos(avg_heading), -np.sin(avg_heading)],
                        [np.sin(avg_heading), np.cos(avg_heading)],
                    ]
                )
                n_pos_t = (n_pos - center) @ rot
                n_ang_t = self.wrap_to_pi(n_ang - avg_heading)
            else:
                n_pos_t = n_pos
                n_ang_t = n_ang

            n_vx = self.v * np.cos(n_ang_t)
            n_vy = self.v * np.sin(n_ang_t)

            # Build the d_v=6 embedding for this neighborhood.
            row_embed = np.concatenate(
                [n_pos_t, n_vx, n_vy, n_ang_t, n_omg], axis=1
            )  # (k, 6)
            # Replace the angle slot (col index 4) with the local fully-active control,
            # matching the parent env's contract.
            row_embed[:, 4] = n_uf

            ego_embeddings[i, n_idx, :] = row_embed
            ego_neighbor_mask[i, n_idx] = 1

        if self.normalize_obs:
            ego_embeddings[..., :2] = ego_embeddings[..., :2] / (self.l_bound / 2.0)
            ego_embeddings[..., 2:4] = ego_embeddings[..., 2:4] / self.v
            ego_embeddings[..., 4] = ego_embeddings[..., 4] / self.u_max
            ego_embeddings[..., 5] = ego_embeddings[..., 5] / self.u_max

        return {
            "agent_embeddings": ego_embeddings,
            "ego_neighbor_mask": ego_neighbor_mask,
            "pad_tokens": pad_tokens,
        }

    # ------------------------------------------------------------------ #
    # Per-agent observation (for decentralized policy evaluation)
    # ------------------------------------------------------------------ #

    def build_per_agent_central_obs(self) -> dict:
        """Per-agent observations in the centralized observation format.

        Each live agent *i* gets an ``(N_max, d_v)`` observation identical in
        structure to ``preprocess_obs`` output, but with:

        * **Local frame**: CoM and average heading computed from *i*'s
          neighbors only (as defined by the current topology adjacency).
        * **Neighbor masking**: slots for non-neighbors are zeroed and their
          ``pad_tokens`` entry set to 1, so the transformer's attention mask
          naturally ignores them.

        This allows the existing centralized checkpoint to be evaluated in a
        truly decentralized setting: each agent's forward pass sees only local
        information, and the per-agent outputs are assembled into one joint
        action vector.

        Returns
        -------
        dict with keys
            agent_embeddings : ndarray, shape ``(N_live, N_max, d_v)``
            pad_tokens       : ndarray, shape ``(N_live, N_max)``, int32
            live_indices     : ndarray, shape ``(N_live,)``, int64
        """

        N_max = self.num_agents_max
        D = self.d_v
        live_mask = self.is_padded == 0
        live_indices = np.where(live_mask)[0]
        N_live = len(live_indices)

        all_embeddings = np.zeros((N_live, N_max, D), dtype=np.float32)
        all_pad_tokens = np.ones((N_live, N_max), dtype=np.int32)

        if N_live == 0:
            return {
                "agent_embeddings": all_embeddings,
                "pad_tokens": all_pad_tokens,
                "live_indices": live_indices,
            }

        positions = self.agent_pos       # (N_max, 2)
        angles = self.wrap_to_pi(self.agent_ang)  # (N_max, 1)
        omegas = self.agent_omg          # (N_max, 1)
        u_fa = self.u_fully_active       # (N_max,)
        adj_full = self.net_topology_full  # (N_max, N_max)

        for batch_idx, i in enumerate(live_indices):
            neighbor_row = adj_full[i].astype(bool) & live_mask
            n_idx = np.where(neighbor_row)[0]
            if len(n_idx) == 0:
                continue

            n_pos = positions[n_idx]     # (k, 2)
            n_ang = angles[n_idx]        # (k, 1)
            n_omg = omegas[n_idx]        # (k, 1)
            n_uf = u_fa[n_idx]           # (k,)

            center = np.mean(n_pos, axis=0)
            avg_heading = float(np.mean(n_ang))
            cos_h, sin_h = np.cos(avg_heading), np.sin(avg_heading)
            rot = np.array([[cos_h, -sin_h], [sin_h, cos_h]])

            n_pos_t = (n_pos - center) @ rot
            n_ang_t = self.wrap_to_pi(n_ang - avg_heading)
            n_vx = self.v * np.cos(n_ang_t)
            n_vy = self.v * np.sin(n_ang_t)

            embed = np.concatenate(
                [n_pos_t, n_vx, n_vy, n_ang_t, n_omg], axis=1
            )  # (k, 6)
            embed[:, 4] = n_uf

            all_embeddings[batch_idx, n_idx, :] = embed
            all_pad_tokens[batch_idx, n_idx] = 0

        if self.normalize_obs:
            all_embeddings[..., :2] /= (self.l_bound / 2.0)
            all_embeddings[..., 2:4] /= self.v
            all_embeddings[..., 4] /= self.u_max
            all_embeddings[..., 5] /= self.u_max

        return {
            "agent_embeddings": all_embeddings,
            "pad_tokens": all_pad_tokens,
            "live_indices": live_indices,
        }

    # ------------------------------------------------------------------ #
    # Diagnostics
    # ------------------------------------------------------------------ #
    #
    # Two notions of "flock" live side by side here:
    #
    #   * Network-connected (PRIMARY). A *single flock* means every live agent
    #     is in the same connected component of the communication topology
    #     graph -- i.e. each pair of agents is reachable from the other in
    #     some number of hops. Heading is irrelevant. Static topologies such
    #     as ring, star, fully_connected are *trivially* one flock by this
    #     definition. Dynamic topologies (k_nearest, disk) and probabilistic
    #     ones (ER random) may or may not satisfy it on a given episode.
    #     This is the metric that classifies which topologies the downstream
    #     generalization eval cares about.
    #
    #   * Spatial cluster (AUXILIARY). Two agents are joined when their
    #     Euclidean distance is below ``distance_threshold`` and clusters are
    #     the connected components of that distance-graph. This ignores the
    #     communication topology and headings, and is reported as a secondary
    #     diagnostic.
    #
    # The polar order parameter (``alignment_order_parameter``) and the
    # alignment-augmented predicate (``is_aligned_single_component``) are
    # *not* part of the single-flock definition; they are reported separately
    # as a measure of how well the swarm settled dynamically.

    # ---- network-connected (primary) ----------------------------------- #

    def network_components(self) -> list:
        """Connected components of the live-agent subgraph of the topology.

        Returns
        -------
        list of np.ndarray
            One array per component, each holding the live-agent indices (in
            the compacted live-agent index space, not the padded one) of that
            component, sorted ascending.
        """

        mask = self.is_padded == 0
        n = int(mask.sum())
        if n == 0:
            return []
        adj = self.view_adjacency_matrix(mask=mask)  # (n, n)
        # Treat the graph as undirected. The builders here always produce
        # symmetric matrices but the parent ``view_adjacency_matrix`` returns a
        # copy, so we OR with the transpose just to be safe.
        adj = (adj | adj.T).astype(bool)
        np.fill_diagonal(adj, True)

        visited = np.zeros(n, dtype=bool)
        components = []
        for start in range(n):
            if visited[start]:
                continue
            stack = [start]
            visited[start] = True
            comp = []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in np.where(adj[u])[0]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            components.append(np.array(sorted(comp), dtype=np.int64))
        return components

    def alignment_order_parameter(self, indices: Optional[np.ndarray] = None) -> float:
        """Polar order parameter ``|mean(v_i)| / speed`` for the given live agents.

        Returns 1.0 when all sampled agents are heading exactly the same way,
        0.0 when their headings cancel out perfectly. ``indices`` are
        live-agent indices (i.e. into the compacted ``agent_vel[is_padded==0]``
        array). When ``indices`` is ``None`` the calculation runs over all
        live agents.
        """

        mask = self.is_padded == 0
        live_vel = self.agent_vel[mask]  # (n_live, 2)
        if live_vel.shape[0] == 0:
            return 0.0
        if indices is None:
            sub_vel = live_vel
        else:
            sub_vel = live_vel[indices]
        if sub_vel.shape[0] == 0:
            return 0.0
        mean_vel = sub_vel.mean(axis=0)
        return float(np.linalg.norm(mean_vel) / self.v)

    def num_network_components(self) -> int:
        """Number of connected components in the topology among live agents.

        This counts components regardless of heading alignment.
        """

        return len(self.network_components())

    def is_network_connected(self) -> bool:
        """True iff every live agent is reachable from every other (any hops).

        This is the primary "single flock" predicate: a topology produces a
        single flock at this instant iff this returns ``True``. Static
        topologies (ring, star, fully_connected) return ``True`` by
        construction; dynamic and probabilistic topologies depend on the
        current adjacency matrix.
        """

        return len(self.network_components()) == 1

    # ---- alignment-augmented (auxiliary) ------------------------------- #

    def aligned_components(self, alignment_threshold: float = 0.95) -> list:
        """Connected components whose internal heading is aligned.

        Auxiliary statistic: how many sub-flocks are *actually moving as a
        flock* (heading-aligned) at the current moment.
        """

        out = []
        for comp in self.network_components():
            if self.alignment_order_parameter(comp) >= alignment_threshold:
                out.append(comp)
        return out

    def num_aligned_components(self, alignment_threshold: float = 0.95) -> int:
        """Number of (connected AND heading-aligned) sub-flocks."""

        return len(self.aligned_components(alignment_threshold=alignment_threshold))

    def is_aligned_single_component(self, alignment_threshold: float = 0.95) -> bool:
        """One connected component AND that component is heading-aligned.

        Strictly stronger than ``is_network_connected``. Reported as a
        secondary metric so we can see e.g. ring topologies which are one
        component but whose swarm is spinning rather than flocking.
        """

        comps = self.network_components()
        if len(comps) != 1:
            return False
        return self.alignment_order_parameter(comps[0]) >= alignment_threshold

    # ---- spatial (auxiliary) ------------------------------------------ #

    def is_single_flock(self, distance_threshold: Optional[float] = None) -> bool:
        """Position-based connectivity test.

        Builds a graph in which agents are connected if their Euclidean distance
        is below ``distance_threshold`` and reports whether that graph is a
        single connected component. ``distance_threshold`` defaults to
        ``2.0 * R`` which is generous enough to call near-by agents one flock
        even if some local detail differs.
        """

        mask = self.is_padded == 0
        positions = self.agent_pos[mask]
        n = positions.shape[0]
        if n <= 1:
            return True
        thr = float(distance_threshold) if distance_threshold is not None else 2.0 * self.R
        diffs = positions[:, None, :] - positions[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        adj = (dists <= thr)
        np.fill_diagonal(adj, True)

        # BFS from node 0
        visited = np.zeros(n, dtype=bool)
        queue = [0]
        visited[0] = True
        while queue:
            u = queue.pop()
            for v in np.where(adj[u])[0]:
                if not visited[v]:
                    visited[v] = True
                    queue.append(v)
        return bool(visited.all())

    def num_position_clusters(self, distance_threshold: Optional[float] = None) -> int:
        """Returns the number of position-based clusters under the same threshold."""

        mask = self.is_padded == 0
        positions = self.agent_pos[mask]
        n = positions.shape[0]
        if n <= 1:
            return 1
        thr = float(distance_threshold) if distance_threshold is not None else 2.0 * self.R
        diffs = positions[:, None, :] - positions[None, :, :]
        dists = np.linalg.norm(diffs, axis=-1)
        adj = (dists <= thr)
        np.fill_diagonal(adj, True)

        visited = np.zeros(n, dtype=bool)
        components = 0
        for start in range(n):
            if visited[start]:
                continue
            components += 1
            stack = [start]
            visited[start] = True
            while stack:
                u = stack.pop()
                for v in np.where(adj[u])[0]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
        return int(components)


class NetworkedLazyAgentsPendReward(NetworkedLazyAgents, LazyAgentsCentralizedPendReward):
    """Pendulum-reward training variant of :class:`NetworkedLazyAgents`.

    The MRO is ``NetworkedLazyAgentsPendReward -> NetworkedLazyAgents ->
    LazyAgentsCentralizedPendReward -> LazyAgentsCentralized``, so the
    auxiliary reward computation comes from the pendulum reward parent while
    topology and observation overrides come from this module.
    """

    pass
