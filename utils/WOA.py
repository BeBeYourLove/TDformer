import numpy as np
import torch
import torch.nn as nn
import random
import math
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vmdpy import VMD
from scipy.fftpack import hilbert,fft,ifft
from math import log
import pandas as pd

class woa(nn.Module):

    def __init__(self, pop=50, MaxIter=10, dim=2, lb=[3, 100], ub=[10, 3000], tau=0, DC=0, init=1, tol=1e-7):
        super(woa, self).__init__()
        self.pop = pop
        self.MaxIter = MaxIter
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.tau =tau
        self.DC = DC
        self.init = init
        self.tol = tol

    def initial(self, pop, dim, ub, lb):
        X = np.zeros((pop, dim))
        bound = [lb, ub]
        # print(bound)
        for i in range(pop):
            for j in range(dim):
                X[i][j] = np.random.uniform(bound[0][j], bound[1][j])
            X[i] = (int(X[i, 0]), int(X[i, 1]))
        # print(X)
        return X, lb, ub

    def fun(self, K, alpha, res):
        if K < 3:
            K = 3
        if K > 10:
            K = 10

        if alpha < 100:
            alpha = 100
        if alpha > 3000:
            alpha = 3000

        # K = int(position[0])
        # alpha = position[1]
        u, u_hat, omega = VMD(res, alpha, self.tau, K, self.DC, self.init, self.tol)
        #
        EP = []
        for i in range(K):
            H = np.abs(hilbert(u[i, :]))
            e1 = []
            for j in range(len(H)):
                p = H[j] / np.sum(H)
                e = -p * log(p, 2)
                e1.append(e)
            E = np.sum(e1)
            EP.append(E)
        s = np.sum(EP) / K
        return s

    def BorderCheck(self, X, ub, lb, pop, dim):
        for i in range(pop):
            for j in range(dim):
                if X[i, j] > ub[j]:
                    X[i, j] = ub[j]
                elif X[i, j] < lb[j]:
                    X[i, j] = lb[j]
        return X

    def CaculateFitness(self, X, fun, res):
        pop = X.shape[0]
        fitness = np.zeros((pop, 1))
        for i in range(pop):
            fitness[i] = fun(int(X[i, 0]), int(X[i, 1]), res)
        return fitness

    def SortFitness(self, Fit):
        fitness = np.sort(Fit, axis=0)
        index = np.argsort(Fit, axis=0)
        return fitness, index

    def SortPosition(self, X, index):
        Xnew = np.zeros(X.shape)
        for i in range(X.shape[0]):
            Xnew[i, :] = X[index[i], :]
        return Xnew

    def WOA(self, pop, dim, lb, ub, MaxIter, fun, res):
        X, lb, ub = self.initial(pop, dim, ub, lb)  # 初始化种群
        fitness = self.CaculateFitness(X, fun, res)  # 计算适应度值
        fitness, sortIndex = self.SortFitness(fitness)  # 对适应度值排序
        X = self.SortPosition(X, sortIndex)  # 种群排序
        GbestScore = fitness[0]
        GbestPositon = np.zeros((1, dim))
        GbestPositon[0, :] = X[0, :]
        Curve = np.zeros([MaxIter, 1])
        for t in range(MaxIter):

            Leader = X[0, :]  # 领头鲸鱼
            a = 2 - t * (2 / MaxIter)  # 线性下降权重2 - 0
            a2 = -1 + t * (-1 / MaxIter)  # 线性下降权重-1 - -2
            for i in range(pop):
                r1 = random.random()
                r2 = random.random()

                A = 2 * a * r1 - a
                C = 2 * r2
                b = 1
                l = (a2 - 1) * random.random() + 1

                for j in range(dim):

                    p = random.random()
                    if p < 0.5:
                        if np.abs(A) >= 1:
                            rand_leader_index = min(int(np.floor(pop * random.random() + 1)), pop - 1)
                            X_rand = X[rand_leader_index, :]
                            D_X_rand = np.abs(C * X_rand[j] - X[i, j])
                            X[i, j] = X_rand[j] - A * D_X_rand
                        elif np.abs(A) < 1:
                            D_Leader = np.abs(C * Leader[j] - X[i, j])
                            X[i, j] = Leader[j] - A * D_Leader
                    elif p >= 0.5:
                        distance2Leader = np.abs(Leader[j] - X[i, j])
                        X[i, j] = distance2Leader * np.exp(b * l) * np.cos(l * 2 * math.pi) + Leader[j]

            X = self.BorderCheck(X, ub, lb, pop, dim)  # 边界检测
            fitness = self.CaculateFitness(X, fun, res)  # 计算适应度值
            fitness, sortIndex = self.SortFitness(fitness)  # 对适应度值排序
            X = self.SortPosition(X, sortIndex)  # 种群排序
            if fitness[0] <= GbestScore:  # 更新全局最优
                GbestScore = fitness[0]
                GbestPositon[0, :] = (int(X[0, 0]), int(X[0, 1]))
            Curve[t] = GbestScore
            print(['迭代次数为' + str(t) + ' 的迭代结果' + str(GbestScore)])
            print(['迭代次数为' + str(t) + ' 的最优参数' + str(GbestPositon)])

        return GbestScore, GbestPositon, Curve

    def forward(self, res):
        GbestScore, GbestPositon, Curve = self.WOA(self.pop, self.dim, self.lb, self.ub, self.MaxIter, self.fun, res)
        best_K = GbestPositon[0, 0]
        best_alpha = GbestPositon[0, 1]
        return best_K, best_alpha





