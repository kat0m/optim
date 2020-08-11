import numpy as np
import random
from scipy.stats import cauchy

class SHADE:
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

    def __init__(self, func, ndim, upper=100., lower=-100., n_pop=100, H=100, n_eval=100000):
        self.func = func
        self.ndim = ndim
        self.upper = upper
        self.lower = lower
        self.n_pop = n_pop
        self.pmin = 2 / n_pop
        self.H = H
        self.Hidx = 0
        self.MF = np.full(shape=H, fill_value=0.5)
        self.MCR = np.full(shape=H, fill_value=0.5)
        self.F = np.zeros(n_pop)
        self.CR = np.zeros(n_pop)
        self.n_eval = n_eval
        self.archive_size = 1 * self.n_pop
        self.archive = np.empty((self.archive_size, self.ndim))
        self.archive_count = 0

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
                # パラメータF_i, CR_i の生成
                ri = np.random.randint(0, self.H)
                self.F[i] = self.generateF(self.MF[ri])
                self.CR[i] = self.generateCR(self.MCR[ri])
                # 差異ベクトル計算
                pidx = random.choice([k for k in range(self.n_pop) if k != i])
                p_i = np.random.rand() * (abs(self.pmin - 0.2)) + min(self.pmin, 0.2)
                pbests = np.argsort(self.fit)[:int(self.n_pop * p_i)]
                pbest = np.random.choice(pbests)
                paidx = random.choice([k for k in range(self.n_pop + self.archive_count) if k != i and k != pidx])
                if paidx < self.n_pop: # selected from Population
                    v = self.x[i] + self.F[i] * (self.x[pbest] - self.x[i]) + self.F[i] * (self.x[pidx] - self.x[paidx])
                else: # selected from Archive
                    v = self.x[i] + self.F[i] * (self.x[pbest] - self.x[i]) + self.F[i] * (self.x[pidx] - self.archive[paidx-self.n_pop])
                for j in range(self.ndim):
                    if v[j] < self.lower:
                        v[j] = (self.lower + self.x[i,j]) / 2
                    elif v[j] > self.upper:
                        v[j] = (self.upper + self.x[i,j]) / 2
                # 新しい解生成
                jrand = np.random.randint(0, self.ndim)
                for j in range(self.ndim):
                    if np.random.rand() <= self.CR[i] or j == jrand:
                        u[i,j] = v[j]
                    else:
                        u[i,j] = self.x[i,j]
            SF, SCR = [], []
            delta = []
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
                    delta.append(self.fit[i] - fi)
                    self.x[i] = u[i]
                    self.fit[i] = fi
                    SF.append(self.F[i])
                    SCR.append(self.CR[i])
                    if fi < self.best_fit:
                        self.best_fit = fi
                        self.best_x = u[i]
            self.e_cnt += self.n_pop
            if SF != [] and SCR != []:
                delta = np.array(delta)
                w = delta / np.sum(delta)
                meanSF = np.sum(np.array(SF) * w)
                meanSCR = np.sum(np.array(SCR) ** 2 * w) / np.sum(np.array(SCR) * w)
                self.MF[self.Hidx] = meanSF
                self.MCR[self.Hidx] = meanSCR
                self.Hidx = (self.Hidx + 1) % self.H

            if self.e_cnt % (self.n_eval // 10) == 0:
                self.disp()
        
    def generateF(self, loc):
        Fi = cauchy.rvs(loc=loc, scale=0.1)
        while Fi < 0:
            Fi = cauchy.rvs(loc=loc, scale=0.1)
        if Fi > 1:
            Fi = 1.0
        return Fi
    
    def generateCR(self, loc):
        CRi = np.random.normal(loc=loc, scale=0.1)
        if CRi < 0:
            CRi = 0.0
        elif CRi > 1:
            CRi = 1.0
        return CRi


    def disp(self, disp_x=False):
        print('< 評価回数 > %d < 最良適合度 > %16.15f' % (self.e_cnt, self.best_fit)) 
        if disp_x:
            print(self.best_x)