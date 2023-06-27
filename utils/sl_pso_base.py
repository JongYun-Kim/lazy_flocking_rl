import numpy as np
import time


def cost_func(p):
    return np.sum(p**2, axis=1)

# d: dimensionality
d = 30
# maxfe: maximal number of fitness evaluations
maxfe = d*5000
results = np.zeros(1)

# lu: define the upper and lower bounds of the variables
lu = np.array([-100 * np.ones(d), 100 * np.ones(d)])

# parameter initialization
M = 100
m = M + d // 10
c3 = d / M * 0.01
PL = np.zeros((m, 1))

for i in range(m):
    PL[i] = (1 - (i) / m)**np.log(np.sqrt(np.ceil(d / M)))

# initialization
XRRmin = np.tile(lu[0, :], (m, 1))
XRRmax = np.tile(lu[1, :], (m, 1))
np.random.seed(int(time.time()*1000) % (2**32))  # Seed based on current time
p = XRRmin + (XRRmax - XRRmin) * np.random.rand(m, d)
fitness = cost_func(p)
v = np.zeros((m, d))
bestever = 1e200

FES = m
gen = 0

start_time = time.time()
# main loop
while FES < maxfe:
    # population sorting
    # fitness, rank = np.sort(fitness), np.argsort(fitness)
    rank = np.argsort(-fitness)
    fitness = fitness[rank]
    p = p[rank, :]
    v = v[rank, :]
    besty = fitness[m-1]
    bestp = p[m-1, :]
    bestever = min(besty, bestever)

    # center position
    center = np.ones((m, 1)) * np.mean(p, axis=0)

    # random matrix
    np.random.seed(int(time.time()*1000) % (2**32))
    randco1 = np.random.rand(m, d)
    np.random.seed(int(time.time()*1000) % (2**32))
    randco2 = np.random.rand(m, d)
    np.random.seed(int(time.time()*1000) % (2**32))
    randco3 = np.random.rand(m, d)
    winidxmask = np.tile(np.arange(m).reshape(-1,1), (1, d))
    # winidxmask = np.repeat(np.arange(1, m + 1)[:, None], d, axis=1)
    winidx = winidxmask + np.ceil(np.random.rand(m, d) * ((m-1) - winidxmask))
    # pwin = p.copy()
    # for j in range(d):
    #     pwin[:, j] = p[winidx[:, j].astype(int), j]
    pwin = p[winidx.astype(int), np.arange(d)]

    # social learning
    lpmask = np.tile(np.random.rand(m, 1) < PL, (1, d))
    lpmask[m-1, :] = 0
    v1 = 1 * (randco1 * v + randco2 * (pwin - p) + c3 * randco3 * (center - p))
    p1 = p + v1

    v = lpmask * v1 + (~lpmask.astype(bool)) * v
    p = lpmask * p1 + (~lpmask.astype(bool)) * p

    # boundary control
    # for i in range(m - 1):
    #     p[i, :] = np.maximum(p[i, :], lu[0, :])
    #     p[i, :] = np.minimum(p[i, :], lu[1, :])
    p[:m - 1, :] = np.maximum(p[:m - 1, :], lu[0, :])
    p[:m - 1, :] = np.minimum(p[:m - 1, :], lu[1, :])

    # fitness evaluation
    fitness[:m - 1] = cost_func(p[:m - 1, :])
    print('Best fitness: %e' % bestever)
    FES = FES + m - 1
    gen += 1

print('Done!\n')
print('CPU time: ', time.time() - start_time)
results[0] = bestever
print('Best fitness: %e' % bestever)
