import numpy as np
import time
from env.envs import LazyAgentsCentralized
import copy
import gym
import ray


class SLPSO:
    def __init__(self):
        self.require_reset = True
        self.cost_func = None
        self.d = None
        self.M = None
        self.lu = None
        self.stagnation_counter = None
        self.max_stagnation = None
        self._do_custom__particle_initializer = False

    def reset(self,
              cost_func_: callable,
              d: int,
              high,
              low=None,
              M=None,
              max_stagnation=50):
        """
        :param cost_func_: (callable) cost function; Or None if you define your custom cost function as a method
                input: particles (np.array); shape: (m, d)
                output: fitness (np.array); shape: (m, )
        :param d: (int) dimension of the problem
        :param low: (np.array) lower bound of the problem; shape: (d, )
        :param high: (np.array) upper bound of the problem; shape: (d, )
        :param M: (int) base population size
        :param max_stagnation: (int) maximum stagnation counter before early termination
        """
        if self.require_reset:
            self.cost_func = cost_func_
            low = -high if low is None else low
            self._check_inputs(d, low, high, M, max_stagnation)
            self.lu = np.array([low, high], dtype=np.float32)
            self.d = d
            self.M = M if M is not None else 100
            self.stagnation_counter = 0
            self.max_stagnation = max_stagnation

            self.require_reset = False
        else:
            raise Exception("Your problem is not ready for. Probably it's in the middle of the optimization process")

    def _check_inputs(self, d, low, high, M, max_stagnation):
        if self.cost_func is not None:
            assert callable(self.cost_func), "The cost function should be callable"

        assert isinstance(d, int) and d > 0, "The dimension of the problem should be a positive integer"

        assert isinstance(low, np.ndarray) and isinstance(high, np.ndarray), \
            "The lower and upper bounds should be np.array"
        assert low.shape == high.shape == (d, ), \
            "The shape of the lower and upper bounds should be (d, )" \
            "where d is the dimension of the problem"
        assert low.dtype == high.dtype == np.float32, \
            "The data type of the lower and upper bounds should be np.float32"
        assert np.all(low <= high), \
            "The lower bound should be less than or equal to the upper bound"

        assert M > 0, "The base population size should be greater than 0"

        assert max_stagnation > 0, "The maximum stagnation should be greater than 0"

    def run(self, see_time=False, see_updates=False, maxfe=None, seed=None):
        """
        :param seed: if not None, the PSO RNG is deterministically seeded with this
                     value (via np.random.default_rng) for the entire run. If None
                     (default), the legacy behavior is preserved: the global numpy
                     RNG is reseeded with the current wall-clock time before each
                     random draw, which is non-deterministic across runs.
        """
        if self.require_reset:
            raise Exception("You must reset the problem before running the optimization process")

        self.require_reset = True

        maxfe = self.d*5000 if maxfe is None else maxfe
        if maxfe < self.d*100:
            print("Warning: maxfe is too small, consider increasing it to at least d*100")
            print("maxfe is NOW set to d*5000")
            maxfe = self.d*5000

        rng = np.random.default_rng(seed) if seed is not None else None

        def _reseed():
            if rng is None:
                np.random.seed(int(time.time()*1000) % (2**32))

        def _rand(*shape):
            if rng is not None:
                return rng.random(shape)
            return np.random.rand(*shape)

        m = self.M + self.d // 10
        c3 = self.d / self.M * 0.01
        PL = np.zeros((m, 1))

        for i in range(m):
            PL[i] = (1 - i/m)**np.log(np.sqrt(np.ceil(self.d / self.M)))

        XRRmin = np.tile(self.lu[0, :], (m, 1))
        XRRmax = np.tile(self.lu[1, :], (m, 1))
        _reseed()
        p = XRRmin + (XRRmax - XRRmin) * _rand(m, self.d)  # dtype: np.float64
        p = self._custom_particle_initializer(p) if self._do_custom__particle_initializer else p
        fitness = self._cost_func(p)  # fitness evaluated m times
        v = np.zeros((m, self.d))
        best_cost_ever = 1e200
        best_p_ever = np.zeros(self.d)

        num_fit_eval = m
        gen = 0

        start_time = time.time()

        while num_fit_eval < maxfe:
            # Sort population
            rank = np.argsort(-fitness)  # descending order
            fitness = fitness[rank]
            p = p[rank, :]
            v = v[rank, :]

            best_y = fitness[m-1]
            best_p = p[m-1, :]
            if best_y < best_cost_ever:
                best_cost_ever = best_y
                best_p_ever = best_p
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
                if self.stagnation_counter >= self.max_stagnation:
                    break  # early termination

            # Center position
            center = np.ones((m, 1)) * np.mean(p, axis=0)

            # Random matrices
            _reseed()
            randco1 = _rand(m, self.d)
            _reseed()
            randco2 = _rand(m, self.d)
            _reseed()
            randco3 = _rand(m, self.d)
            winidxmask = np.tile(np.arange(m).reshape(-1,1), (1, self.d))
            winidx = winidxmask + np.ceil(_rand(m, self.d) * ((m-1) - winidxmask))
            pwin = p[winidx.astype(int), np.arange(self.d)]

            # Social learning
            lpmask = np.tile(_rand(m, 1) < PL, (1, self.d))
            lpmask[m-1, :] = 0
            v1 = 1 * (randco1 * v + randco2 * (pwin - p) + c3 * randco3 * (center - p))
            p1 = p + v1

            # Velocity and position update
            v = lpmask * v1 + (~lpmask.astype(bool)) * v
            p = lpmask * p1 + (~lpmask.astype(bool)) * p

            # Boundary handling
            p[:m - 1, :] = np.maximum(p[:m - 1, :], self.lu[0, :])
            p[:m - 1, :] = np.minimum(p[:m - 1, :], self.lu[1, :])

            # Evaluate fitness (cost)
            fitness[:m - 1] = self._cost_func(p[:m - 1, :])
            num_fit_eval = num_fit_eval + m - 1  # best particle not evaluated
            gen += 1

            if see_updates:
                print(f"\nGeneration: {gen}")
                print(f"    Best cost: {best_cost_ever}")
                print(f"    Best position: {best_p_ever}")
            if see_time:
                print(f"Time elapsed: {time.time() - start_time:.1f} s")
                print(f"    Progress: {num_fit_eval / maxfe * 100:.2f}%")

        self.require_reset = True

        self.stagnation_counter = 0

        return best_p_ever, best_cost_ever

    def _cost_func(self, x):
        if self.cost_func is not None:
            return self.cost_func(x)
        else:
            raise NotImplementedError("The cost function is not implemented. "
                                      "Otherwise, you should pass the cost function to the constructor.")

    def _custom_particle_initializer(self, p):
        """Override to customize initial particle matrix p; shape: (m, d)."""
        raise NotImplementedError("The custom particle initializer is not implemented. "
                                  "Otherwise, you should pass the custom particle initializer to the constructor.")


class GetLazinessBySLPSO(SLPSO):
    """
    This class is used to get the (sub-)optimal laziness vector by using SLPSO
    """
    def __init__(self):
        super().__init__()

        self._do_custom__particle_initializer = True
        self.env_original = None

    def set_env(self, env_initialized):
        """
        :param env_initialized: WARNING: it is assumed to maintain the number of agents in the environment in an episode
        """
        if not isinstance(env_initialized, gym.Env):
            raise TypeError("env_initialized should be a gym environment")
        if not isinstance(env_initialized, LazyAgentsCentralized):
            raise TypeError("env_initialized should be LazyAgentsCentralized class")
        assert env_initialized.num_agents is not None, "num_agents should be initialized"

        self.env_original = env_initialized
        d = env_initialized.num_agents
        low = np.zeros(d, dtype=np.float32)
        high = np.ones(d, dtype=np.float32)
        M = 100
        max_stagnation = 20

        super().reset(cost_func_=None, d=d, low=low, high=high,  M=M, max_stagnation=max_stagnation)

    def run(self, see_updates=False, see_time=False, maxfe=None, seed=None):
        """Returns (best_laziness_full, best_cost, best_laziness_small).

        :param seed: if not None, PSO is deterministically seeded. Default None
                     preserves the legacy non-deterministic behavior.
        """
        best_laziness_small, best_cost = super().run(
            see_updates=see_updates, see_time=see_time, maxfe=maxfe, seed=seed,
        )

        mask = self.env_original.is_padded == 0
        best_laziness_full = np.zeros(self.env_original.num_agents_max, dtype=np.float32)
        best_laziness_full[mask] = best_laziness_small

        return best_laziness_full, best_cost, best_laziness_small

    def _cost_func(self, p):
        """Evaluate cost via gym episodes: cost = -reward_sum for each particle. p: (m, d) → (m,)."""
        m = p.shape[0]
        cost_futures = []

        for i in range(m):
            env = copy.deepcopy(self.env_original)
            p_in = p[i, :]
            cost_future = self.compute_single_particle_cost.remote(env=env, p=p_in)
            cost_futures.append(cost_future)

        costs_got = ray.get(cost_futures)

        costs = np.array(costs_got, dtype=np.float32)  # shape: (m, )

        assert costs.shape == (m, ), "costs.shape should be (m, )"
        assert isinstance(costs, np.ndarray), "costs should be a numpy array"

        return costs

    @staticmethod
    @ray.remote
    def compute_single_particle_cost(env, p):
        assert env.use_custom_ray is True, "env.use_custom_ray should be True for parallel computing (immutable env)"

        reward_sum = 0
        done = False
        constant_action = p  # laziness vector
        while not done:
            _, reward, done, _ = env.step(constant_action)
            reward_sum += reward  # a scalar
        cost = - reward_sum  # cost of i-th particle is the negative reward_sum of the episode

        return cost

    def _custom_particle_initializer(self, p):
        """Injects a fully-active laziness vector (all ones) as the first particle."""
        d = p.shape[1]
        data_type_pop = p.dtype

        laziness_full_active = np.ones(d, dtype=data_type_pop)

        p[0, :] = laziness_full_active

        return p


class PSOActionOptimizer:
    """Finds the optimal constant action for a LazyAgentsCentralized env using PSO + Ray."""

    def __init__(self, num_cpus=None):
        self.num_cpus = num_cpus
        self._ray_initialized_here = False

    def optimize(self, env, maxfe=None, see_updates=False, see_time=False, seed=None):
        """
        :param env: LazyAgentsCentralized instance, already reset()
        :param maxfe: max function evaluations (default: d*5000 inside SLPSO)
        :param seed: if not None, PSO's internal RNG is deterministically seeded.
                     Default None preserves the legacy (non-deterministic) behavior
                     for backward compatibility with existing results.
        :return: (optimal_action, cost, elapsed_time)
                 optimal_action shape: (num_agents_max,), padded agents are 0
                 cost: negative episode reward (lower is better)
                 elapsed_time: wall-clock seconds for the PSO optimization
        """
        if not isinstance(env, LazyAgentsCentralized):
            raise TypeError("env must be a LazyAgentsCentralized instance")
        if env.num_agents is None:
            raise ValueError("env must be reset before optimization")

        self._ensure_ray()

        env_for_pso = copy.deepcopy(env)
        env_for_pso.use_custom_ray = True
        env_for_pso._skip_obs = True

        pso = GetLazinessBySLPSO()
        pso.set_env(env_for_pso)

        t_start = time.time()
        optimal_action, cost, _ = pso.run(
            see_updates=see_updates, see_time=see_time, maxfe=maxfe, seed=seed,
        )
        elapsed_time = time.time() - t_start

        return optimal_action, cost, elapsed_time

    def _ensure_ray(self):
        if not ray.is_initialized():
            ray.init(num_cpus=self.num_cpus)
            self._ray_initialized_here = True

    def shutdown(self):
        if self._ray_initialized_here and ray.is_initialized():
            ray.shutdown()
            self._ray_initialized_here = False

    def __enter__(self):
        self._ensure_ray()
        return self

    def __exit__(self, *args):
        self.shutdown()


def cost_func(p):
    """Demo cost function (sphere). p: (m, d) → (m,)."""
    return np.sum(p**2, axis=1)


if __name__ == "__main__":
    # Smoke test: minimize a 30-d sphere function
    d = 30
    lu = np.array([-100 * np.ones(d), 100 * np.ones(d)], dtype=np.float32)
    M = 100

    pso = SLPSO()
    pso.reset(cost_func_=cost_func, d=d, low=lu[0, :], high=lu[1, :], M=M)
    best_x, best_cost = pso.run(see_time=True, see_updates=True)
    print(f"Best x: {best_x}")
    print(f"Best cost: {best_cost}")
