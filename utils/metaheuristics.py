import numpy as np
import time


class SLPSO:

    # TODO (1): [x] Could do better with the initialization
    #   TODO (1-1) [x] unify problem instances with the cost function e.g. cost_func returns configs as well
    # TODO (2): [x] Think about validity of the seed generation method
    # TODO (3): [x] Use both lower and upper bounds (update
    # TODO (4): [x] Set default bounds to infinities
    # TODO (5): [x] Consider early termination
    # TODO (6): [x] Consider parallelization with multiple cost functions
    def __init__(self, cost_func, d, lu, M):
        self.cost_func = cost_func
        self.d = d
        self.lu = lu
        self.M = M

    def run(self, see_time=False, see_updates=False):
        maxfe = self.d*5000

        m = self.M + self.d // 10
        c3 = self.d / self.M * 0.01
        PL = np.zeros((m, 1))

        for i in range(m):
            PL[i] = (1 - i/m)**np.log(np.sqrt(np.ceil(self.d / self.M)))

        XRRmin = np.tile(self.lu[0, :], (m, 1))
        XRRmax = np.tile(self.lu[1, :], (m, 1))
        np.random.seed(int(time.time()*1000) % (2**32))
        p = XRRmin + (XRRmax - XRRmin) * np.random.rand(m, self.d)
        fitness = self.cost_func(p)  # fitness evaluated m times
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
            best_y = fitness[m-1]
            best_p = p[m-1, :]
            best_cost_ever = min(best_y, best_cost_ever)
            best_p_ever = best_p if best_y < best_cost_ever else best_p_ever

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
            fitness[:m - 1] = self.cost_func(p[:m - 1, :])
            num_fit_eval = num_fit_eval + m - 1  # best particle not evaluated
            gen += 1

            if see_time:
                print(f"Time elapsed: {time.time() - start_time}")
            if see_updates:
                print('Best fitness: %e' % best_cost_ever)

        return best_p_ever, best_cost_ever


def cost_func(p):
    # params: p: population; shape: (m, d)
    # Sphere function
    # You should use your implementation of the cost function and pass it to the SLPSO class
    # Make sure the return size is (m) must be a vector
    return np.sum(p**2, axis=1)


if __name__ == "__main__":
    # Problem configurations
    d = 30
    lu = np.array([-100 * np.ones(d), 100 * np.ones(d)])
    M = 100

    pso = SLPSO(cost_func, d, lu, M)
    best_fitness = pso.run()
    print(f"Best Fitness: {best_fitness}")

