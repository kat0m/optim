import numpy as np
import random

class DE:
    '''
    func          : Objective Function
    ndim          : number of dimensions of search space
    upper         : 探索空間の最大値
    lower         : 探索空間の最小値
    n_pop         : 個体集団サイズ
    F             : スケーリング係数
    CR            : 交叉率
    n_eval        : 最大評価回数
    '''

    def __init__(self, func, ndim, upper=100., lower=-100., n_pop=100, F=0.7, CR=0.9, n_eval=100000):
        self.func = func
        self.ndim = ndim
        self.upper = upper
        self.lower = lower
        self.n_pop = n_pop
        self.F = F
        self.CR = CR
        self.n_eval = n_eval

    def solve(self):
        # 解候補の初期化と評価
        self.x = np.random.rand(self.n_pop, self.ndim) * (self.upper - self.lower) + self.lower
        self.fit = np.empty(self.n_pop)
        for i in range(self.n_pop):
            self.fit[i] = self.func(self.x[i])
            if i == 0:
                self.best_fit = self.fit[i]
                self.best_x = self.x[i]
            elif self.best_fit > self.fit[i]:
                self.best_fit = self.fit[i]
                self.best_x = self.x[i]
        self.e_cnt = self.n_pop
    
        # 評価回数に達するまで繰り返し
        while self.e_cnt < self.n_eval:
            u = np.empty((self.n_pop, self.ndim))
            for i in range(self.n_pop):
                # 交叉
                pidx = random.sample([k for k in range(self.n_pop) if k != i], 3)
                # 差異ベクトル計算
                v = self.x[pidx[0]] + self.F * (self.x[pidx[1]] - self.x[pidx[2]])
                # 新しい解生成
                jrand = random.randint(0, self.ndim-1)
                for j in range(self.ndim):
                    if np.random.rand() <= self.CR or j == jrand:
                        u[i,j] = v[j]
                    else:
                        u[i,j] = self.x[i,j]
            for i in range(self.n_pop):
                # 新しい解の評価と入れ替え
                fi = self.func(u[i])
                if fi < self.fit[i]:
                    self.x[i] = u[i]
                    self.fit[i] = fi
                    if fi < self.best_fit:
                        self.best_fit = fi
                        self.best_x = u[i]
            self.e_cnt += self.n_pop

            if self.e_cnt % (self.n_eval // 10) == 0:
                self.disp()
    
    def disp(self, disp_x=False):
        print('< 評価回数 > %d < 最良適合度 > %16.10f' % (self.e_cnt, self.best_fit)) 
        if disp_x:
            print(self.best_x)