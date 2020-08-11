import numpy as np
import random
from scipy.stats import cauchy

class JADE:
    '''
    func          : 最適化対象の関数
    ndim          : 問題の次元
    upper         : 探索空間の最大値
    lower         : 探索空間の最小値
    n_pop         : 個体集団サイズ
    F             : スケーリング係数
    CR            : 交叉率
    n_eval        : 最大評価回数
    '''

    def __init__(self, func, ndim, upper=100., lower=-100., n_pop=100, n_eval=100000):
        self.func = func
        self.ndim = ndim
        self.upper = upper
        self.lower = lower
        self.n_pop = n_pop
        self.MF = 0.5
        self.MCR = 0.5
        self.F = np.zeros(n_pop)
        self.CR = np.zeros(n_pop)
        self.n_eval = n_eval
        self.archive_size = 2 * self.n_pop
        self.archive = np.empty((self.archive_size, self.ndim))
        self.archive_count = 0

    def solve(self, disp=False):
        for i in range(self.n_pop):
            self.F[i] = self.generateF(self.MF)
        self.CR = np.random.normal(loc=self.MCR, scale=0.1, size=(self.n_pop))
        self.CR = np.clip(self.CR, a_min=0., a_max=1.)

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
            pbests = np.argsort(self.fit)[:self.n_pop//20]
            for i in range(self.n_pop):
                # 差異ベクトル計算
                pidx = random.choice([k for k in range(self.n_pop) if k != i])
                pbest = np.random.choice(pbests)
                paidx = random.choice([k for k in range(self.n_pop + self.archive_count) if k != i and k != pidx])
                if paidx < self.n_pop: # selected from Population
                    v = self.x[i] + self.F[i] * (self.x[pbest] - self.x[i]) + self.F[i] * (self.x[pidx] - self.x[paidx])
                else: # selected from Archive
                    v = self.x[i] + self.F[i] * (self.x[pbest] - self.x[i]) + self.F[i] * (self.x[pidx] - self.archive[paidx-self.n_pop])
                # 新しい解生成
                jrand = np.random.randint(0, self.ndim)
                for j in range(self.ndim):
                    if np.random.rand() <= self.CR[i] or j == jrand:
                        u[i,j] = v[j]
                    else:
                        u[i,j] = self.x[i,j]
            SN = 0
            SF, SF2, SCR = 0., 0., 0.
            replacement = []
            for i in range(self.n_pop):
                # 新しい解の評価と入れ替え
                fi = self.func(u[i])
                if fi < self.fit[i]:
                    if self.archive_count < self.archive_size:
                        self.archive[self.archive_count] = self.x[i]
                        self.archive_count += 1
                    else:
                        idx = np.random.randint(0, self.archive_size)
                        self.archive[idx] = self.x[i]
                    self.x[i] = u[i]
                    self.fit[i] = fi
                    SN += 1
                    SF += self.F[i]
                    SF2 += self.F[i] ** 2
                    SCR += self.CR[i]
                    replacement.append(i)
                    if fi < self.best_fit:
                        self.best_fit = fi
                        self.best_x = u[i]
            self.e_cnt += self.n_pop
            if SF != 0:
                self.MF = (1 - 0.1) * self.MF + 0.1 * SF2 / SF
            if SN != 0:
                self.MCR = (1 - 0.1) * self.MCR + 0.1 * SCR / SN
            for i in replacement:
                self.F[i] = self.generateF(self.MF)
                self.CR[i] = self.generateCR(self.MCR)
            if self.e_cnt % (self.n_eval // 10) == 0:
                self.disp()
        
    def generateF(self, loc):
        r = cauchy.rvs(loc=loc, scale=0.1)
        while r < 0:
            r = cauchy.rvs(loc=loc, scale=0.1)
        if r > 1:
            r = 1.0
        return r

    def generateCR(self, loc):
        CRi = np.random.normal(loc=loc, scale=0.1)
        if CRi < 0:
            CRi = 0.0
        elif CRi > 1:
            CRi = 1.0
        return CRi

    def disp(self, disp_x=False):
        print('< 評価回数 > %d < 最良適合度 > %16.10f' % (self.e_cnt, self.best_fit)) 
        if disp_x:
            print(self.best_x)