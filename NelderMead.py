import numpy as np
import random

class NelderMead:
    '''
    func          : Objective Function
    x0            : initial solution candidate
    max_iter      : maximum number of iteration
    '''

    def __init__(self, func, x0, max_iter=10000):
        self.func = func
        self.ndim = len(x0)
        self.max_iter = max_iter
        self.x = np.repeat(x0[None, :], self.ndim+1, axis=0)
        self.fit = np.empty(self.ndim+1, dtype=float)
        self.alpha = 1.0
        self.gamma = 2.0
        self.rho = 0.5
        self.sigma = 0.5
    def solve(self):
        # 解候補の初期化と評価
        for i in range(self.ndim+1):
            if i != 0:
                self.x[i, i-1] += 0.05
            self.fit[i] = self.func(self.x[i])
    
        # 評価回数に達するまで繰り返し
        for t in range(1, self.max_iter+1):
            if t % (self.max_iter // 10) == 0:
                f_best = self.fit.min()
                print('iteration: %d, score: %lf' % (t, f_best))
            
            # sort by fitness
            idx = np.argsort(self.fit)
            self.x = self.x[idx]
            self.fit = self.fit[idx]
            
            # calculate centroid
            centroid = np.mean(self.x[:-1], axis=0)

            # Reflection
            xr = centroid + self.alpha * (centroid - self.x[-1])
            fr = self.func(xr)
            if self.fit[0] <= fr < self.fit[-2]:
                self.x[-1] = xr
                self.fit[-1] = fr
                continue


            # Expansion
            if fr < self.fit[0]:
                xe = centroid + self.gamma * (xr - centroid)
                fe = self.func(xe)
                if fe < fr:
                    self.x[-1] = xe
                    self.fit[-1] = fe
                else:
                    self.x[-1] = xr
                    self.fit[-1] = fr
                continue

            # Contraction
            if fr >= self.fit[-2]:
                xc = centroid + self.rho * (self.x[-1] - centroid)
                fc = self.func(xc)
                if fc < self.fit[-1]:
                    self.x[-1] = xc
                    self.fit[-1] = fc
                    continue
            
            # Shrink
            for i in range(1, self.ndim+1):
                self.x[i] = self.x[0] + self.sigma * (self.x[i] - self.x[0])
                self.fit[i] = self.func(self.x[i])

    def get_best(self):
        idx = np.argmin(self.fit)
        x_best = self.x[idx]
        f_best = self.fit[idx]
        return x_best, f_best

if __name__ == '__main__':
    def rosenbrock(x):
        return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

    ndim = 10
    x0 = np.random.rand(ndim) * 200 - 100
    nm = NelderMead(rosenbrock, x0=x0, max_iter=100000)
    nm.solve()
    x, f = nm.get_best()
    print(f)
    print(x)