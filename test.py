from GA import GA
from DE import DE
from JADE import JADE
from SHADE import SHADE
import numpy as np

def spf(x):
    return sum(x**2)

def rosenbrock(x):
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

def test():
    n_trials = 1
    results = []
    func = rosenbrock
    solvers = {'GA':GA, 'DE':DE, 'JADE':JADE, 'SHADE':SHADE}
    for name, SOLVER in solvers.items():
        print(name)
        solver = SOLVER(func, ndim=10, n_eval=100000)
        solver.solve()

if __name__ == '__main__':
    test()