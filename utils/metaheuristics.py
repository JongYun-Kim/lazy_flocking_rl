import numpy as np
import time
from env.envs import LazyAgentsCentralized  # Import your custom environment
import copy
import gym
import ray


class SLPSO:

    # TODO (1): [Canceled] Could do better with the initialization
    #   TODO (1-1) [Canceled] unify problem instances with the cost function e.g. cost_func returns configs as well
    # TODO (2): [x] Think about validity of the seed generation method (current resoultion is 1ms)
    # TODO (3): [o] Use both lower and upper bounds (update
    # TODO (4): [Canceled] Set default bounds to infinities
    # TODO (5): [x] Consider early termination
    # TODO (6): [Canceled] Consider parallelization with multiple cost functions <- parallelize it in the outer scope
    # TODO (7): [o] Create _cost_func as a method of the class (internal use of the cost function)
    # TODO (8): [o] Validate the inputs
    def __init__(self):
        self.require_reset = True
        self.cost_func = None
        self.d = None
        self.M = None
        self.lu = None
        self.stagnation_counter = None
        self.max_stagnation = None

    def reset(self, cost_func, d, low, high=None,  M=None, max_stagnation=50):
        """
        :param cost_func: (callable) cost function; Or None if you define your custom cost function as a method
                input: particles (np.array); shape: (m, d)
                output: fitness (np.array); shape: (m, )
        :param d: (int) dimension of the problem
        :param low: (np.array) lower bound of the problem; shape: (d, )
        :param high: (np.array) upper bound of the problem; shape: (d, )
        :param M: (int) base population size
        :param max_stagnation: (int) maximum stagnation counter (generations) before early termination (default: 100)
        """
        if self.require_reset:
            # Check inputs
            self.cost_func = cost_func
            self._check_inputs(d, low, high, M, max_stagnation)
            # Set bounds
            self.lu = np.array([low, high], dtype=np.float32)
            # Set dimension
            self.d = d
            # Set base population size
            self.M = M if M is not None else 100
            # Set stagnation counter
            self.stagnation_counter = 0
            # Set maximum stagnation
            self.max_stagnation = max_stagnation

            self.require_reset = False
        else:
            raise Exception("Your problem is not ready for. Probably it's in the middle of the optimization process")

    def _check_inputs(self, d, low, high, M, max_stagnation):
        # Check if cost_func is callable
        if self.cost_func is not None:
            assert callable(self.cost_func), "The cost function should be callable"

        # Check if d is a positive integer
        assert isinstance(d, int) and d > 0, "The dimension of the problem should be a positive integer"

        # Check if low and high are np.array
        assert isinstance(low, np.ndarray) and isinstance(high, np.ndarray), \
            "The lower and upper bounds should be np.array"
        # Check if the bounds are valid
        assert low.shape == high.shape == (d, ), \
            "The shape of the lower and upper bounds should be (d, )" \
            "where d is the dimension of the problem"
        assert low.dtype == high.dtype == np.float32, \
            "The data type of the lower and upper bounds should be np.float32"
        assert np.all(low <= high), \
            "The lower bound should be less than or equal to the upper bound"

        # Check if M is a positive integer
        assert M > 0, "The base population size should be greater than 0"

        # Check if max_stagnation is a positive integer
        assert max_stagnation > 0, "The maximum stagnation should be greater than 0"

    def run(self, see_time=False, see_updates=False, maxfe=None):
        if self.require_reset:
            raise Exception("You must reset the problem before running the optimization process")

        self.require_reset = True

        maxfe = self.d*5000 if maxfe is None else maxfe
        if maxfe < self.d*100:
            print("Warning: maxfe is too small, consider increasing it to at least d*100")
            print("maxfe is NOW set to d*5000")
            maxfe = self.d*5000

        m = self.M + self.d // 10
        c3 = self.d / self.M * 0.01
        PL = np.zeros((m, 1))

        for i in range(m):
            PL[i] = (1 - i/m)**np.log(np.sqrt(np.ceil(self.d / self.M)))

        XRRmin = np.tile(self.lu[0, :], (m, 1))
        XRRmax = np.tile(self.lu[1, :], (m, 1))
        np.random.seed(int(time.time()*1000) % (2**32))
        p = XRRmin + (XRRmax - XRRmin) * np.random.rand(m, self.d)  # dtype: np.float64
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

            # Update best cost and position
            old_best_cost = best_cost_ever
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
            np.random.seed(int(time.time()*1000) % (2**32))
            randco1 = np.random.rand(m, self.d)
            np.random.seed(int(time.time()*1000) % (2**32))
            randco2 = np.random.rand(m, self.d)
            np.random.seed(int(time.time()*1000) % (2**32))
            randco3 = np.random.rand(m, self.d)
            winidxmask = np.tile(np.arange(m).reshape(-1,1), (1, self.d))
            winidx = winidxmask + np.ceil(np.random.rand(m, self.d) * ((m-1) - winidxmask))
            pwin = p[winidx.astype(int), np.arange(self.d)]

            # Social learning
            lpmask = np.tile(np.random.rand(m, 1) < PL, (1, self.d))
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

            # Print the results
            if see_updates:
                print(f"\nGeneration: {gen}")
                print(f"    Best cost: {best_cost_ever}")
                print(f"    Best position: {best_p_ever}")
            if see_time:
                print(f"Time elapsed: {time.time() - start_time:.1f} s")
                print(f"    Progress: {num_fit_eval / maxfe * 100:.2f}%")

        self.require_reset = True

        # Reset the stagnation counter
        self.stagnation_counter = 0

        return best_p_ever, best_cost_ever

    def _cost_func(self, x):
        # Check if the cost function is implemented
        if self.cost_func is not None:
            return self.cost_func(x)
        else:  # not implemented error
            raise NotImplementedError("The cost function is not implemented. "
                                      "Otherwise, you should pass the cost function to the constructor.")


class GetLazinessBySLPSO(SLPSO):
    """
    This class is used to get the (sub-)optimal laziness vector by using SLPSO
    """
    def __init__(self):
        super().__init__()

        self.env_original = None

        # # Initialize Ray
        # ray.init(num_cpus=14, ignore_reinit_error=True)

    def set_env(self, env_initialized):
        """
        :param env_initialized: WARNING: it is assumed to maintain the number of agents in the environment in an episode
        """
        # Check if env_initialized is a gym environment
        if not isinstance(env_initialized, gym.Env):
            raise TypeError("env_initialized should be a gym environment")
        # Check if env_initialized is LazyAgentsCentralized class
        if not isinstance(env_initialized, LazyAgentsCentralized):
            raise TypeError("env_initialized should be LazyAgentsCentralized class")
        assert env_initialized.num_agent is not None, "num_agent should be initialized"

        # Store the initialized environment and use it in the cost function by using deepcopy
        self.env_original = env_initialized
        d = env_initialized.num_agent  # TODO: if you use num_agent, you should pad the laziness vector with 0s
        # d = env_initialized.num_agent_max
        low = np.zeros(d, dtype=np.float32)
        high = np.ones(d, dtype=np.float32)
        M = 100
        max_stagnation = 20  # 50

        # Reset the class with super's reset() method
        super().reset(cost_func=None, d=d, low=low, high=high,  M=M, max_stagnation=max_stagnation)

    def run(self, see_updates=False, see_time=False, maxfe=None):
        """
        :param see_updates: if True, print the best cost and position in each generation
        :param see_time: if True, print the time elapsed
        :param maxfe: maximum number of fitness evaluations
        :return: best laziness vector and best cost
        """
        best_laziness_small, best_cost = super().run(see_updates=see_updates, see_time=see_time, maxfe=maxfe)

        # Extend the laziness vector to the maximum number of agents
        mask = self.env_original.is_padded == 0
        # Fill the solution with 0s
        best_laziness_full = np.zeros(self.env_original.num_agent_max, dtype=np.float32)
        # best_laziness_full: shape: (D, ); D: maximum number of agents == self.env_original.num_agent_max
        best_laziness_full[mask] = best_laziness_small

        return best_laziness_full, best_cost, best_laziness_small

    def _cost_func(self, p):
        """
        This function is used to evaluate the cost function from the gym environment
        A cost of a particle is the sum of rewards of the gym environment episode with the particle as the constant action
        :param p: m possible solutions (m laziness vectors ; shape: (m, d))
        :return: m rewards; (shape: (m, ))
        """
        m = p.shape[0]
        # d = p.shape[1]
        cost_futures = []

        # For each particle, evaluate the reward_sum in parallel using Ray
        for i in range(m):
            env = copy.deepcopy(self.env_original)
            p_in = p[i, :]
            cost_future = self.compute_single_particle_cost.remote(env=env, p=p_in)
            cost_futures.append(cost_future)

        # Get the costs from the futures
        costs_got = ray.get(cost_futures)

        costs = np.array(costs_got, dtype=np.float32)  # shape: (m, )

        assert costs.shape == (m, ), "costs.shape should be (m, )"
        assert isinstance(costs, np.ndarray), "costs should be a numpy array"

        return costs

    @staticmethod
    @ray.remote
    def compute_single_particle_cost(env, p):
        # TODO: You can use env.auto_step() to get the reward_sum without using the while loop
        reward_sum = 0
        done = False
        constant_action = p  # laziness vector
        while not done:
            _, reward, done, _ = env.step(constant_action)
            reward_sum += reward  # a scalar
        cost = - reward_sum  # cost of i-th particle is the negative reward_sum of the episode

        return cost


def cost_func(p):
    # params: p: population; shape: (m, d)
    # Sphere function
    # You should use your implementation of the cost function and pass it to the SLPSO class
    # Make sure if the return size is (m); must be a vector
    return np.sum(p**2, axis=1)  # shape: (m, )


if __name__ == "__main__":
    # Problem configurations
    d = 30
    lu = np.array([-100 * np.ones(d), 100 * np.ones(d)])
    M = 100

    pso = SLPSO()
    pso.reset(cost_func=cost_func, d=d, low=lu[0, :], high=lu[1, :], M=M)
    best_x, best_cost = pso.run(see_time=True, see_updates=True)
    print(f"Best x: {best_x}")
    print(f"Best cost: {best_cost}")
