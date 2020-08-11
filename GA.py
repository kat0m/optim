import numpy as np
import random

class GA:
    '''
    func          : 最適化対象の関数
    ndim          : 問題の次元
    upper         : 探索空間の最大値
    lower         : 探索空間の最小値
    n_pop         : 個体集団サイズ
    n_eval        : 最大評価回数
    crossover     : 交叉の手法('blx-alpha' or 'simplex')
    mutation_rate : 突然変異確率
    '''

    def __init__(self, func, ndim, upper=100., lower=-100., n_pop=100, n_eval=100000, crossover='blx-alpha', mutation_rate=0.01):
        self.func = func
        self.ndim = ndim
        self.upper = upper
        self.lower = lower
        self.n_pop = n_pop
        self.n_eval = n_eval
        self.crossover = crossover
        self.mutation_rate = mutation_rate

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
            # 交叉
            if self.crossover == 'blx-alpha':
                parent_idx = random.sample([i for i in range(self.n_pop)], 2)
                parent = self.x[parent_idx]
                child = self.blx_alpha(parent, 2)
            elif self.crossover == 'simplex':
                parent_idx = random.sample([i for i in range(self.n_pop)], self.ndim+1)
                parent = self.x[parent_idx]
                child = self.simplex(parent, 2)
                parent_idx = random.sample(parent_idx, 2)
                parent = self.x[parent_idx]
            # 突然変異
            child = self.mutation(child)
            # 子個体の評価
            child_fit = np.empty(2)
            child_fit[0] = self.func(child[0])
            child_fit[1] = self.func(child[1])
            self.e_cnt += 2
            if child_fit[0] < self.best_fit:
                self.best_fit = child_fit[0]
                self.best_x = child[0]
            if child_fit[1] < self.best_fit:
                self.best_fit = child_fit[1]
                self.best_x = child[1]
            # 生存選択
            family = np.concatenate([parent, child])
            family_fit = np.concatenate([self.fit[parent_idx], child_fit])
            idx = [0, 1, 2, 3]
            idx.sort(key=lambda i:family_fit[i])
            selected_idx = [idx[0]]
            # ルーレット法のためのスケーリング
            v = family_fit[idx[-1]] - family_fit[idx[0]]
            if v == 0:
                v = 1
            family_fit_ = v / (family_fit - family_fit[idx[0]] + 0.5 * v)
            # 選択確率の計算
            f_sum = np.sum(family_fit_)
            select_prb = family_fit_ / f_sum
            # ルーレット選択
            roulette_idx = np.random.choice(4, p=select_prb)
            selected_idx.append(roulette_idx)
            # 個体の入れ替え
            for i in range(2):
                self.x[parent_idx[i]] = family[selected_idx[i]]
                self.fit[parent_idx[i]] = family_fit[selected_idx[i]]

            if self.e_cnt % (self.n_eval // 10) == 0:
                self.disp()
    
    def blx_alpha(self, parent, n_child):
        alpha = 0.36
        child = np.empty((n_child, self.ndim))
        for i in range(self.ndim):
            max_p = max(parent[0,i], parent[1,i])
            min_p = min(parent[0,i], parent[1,i])
            mn = min_p - alpha * (max_p - min_p)
            mx = max_p + alpha * (max_p - min_p)
            for c in range(n_child):
                r = (mx - mn) * np.random.rand() + mn
                r = min(r, self.upper)
                r = max(r, self.lower)
                child[c][i] = r
        return child

    def simplex(self, parent, n_child):
        # 拡張率の推奨値
        eps = np.sqrt(self.ndim + 2)
        # 重心Gを求める
        G = np.mean(parent, axis=0)
        # x_k = G + eps * (P_k - G) (k = 0, ..., n)
        x = G + eps * (parent - G)
        child = np.empty((n_child, self.ndim))
        for idx in range(n_child):
            C = np.zeros((self.ndim + 1, self.ndim))
            for i in range(1, self.ndim + 1):
                r = np.random.rand() ** (1/i)
                C[i] = r * (x[i-1] - x[i] + C[i-1])
            child[idx] = x[-1] + C[-1]
        child = np.clip(child, a_min=self.lower, a_max=self.upper)
        return child
        

    def mutation(self, child):
        # 一様乱数ベース
        for i in range(child.shape[0]):
            for j in range(self.ndim):
                rand_num = np.random.rand()
                if rand_num <= self.mutation_rate:
                    child[i,j] = np.random.rand() * (self.upper - self.lower) + self.lower
        return child
    
    def disp(self, disp_x=False):
        print('< 評価回数 > %d < 最良適合度 > %16.10f' % (self.e_cnt, self.best_fit)) 
        if disp_x:
            print(self.best_x)
    
    def get_result(self):
        return self.best_fit