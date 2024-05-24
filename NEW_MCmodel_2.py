##引用库
import random
# 忽略警告
import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
from scipy.spatial.distance import cdist


# 实例化一个类来代表某类人群在平面上的运动
class Population:
    def __init__(self, N, mobility, threshold, view_distance, infect_distance, alpha, mu_=0.1, scope=10, model='SIS'):
        # 移动能力
        self.mobility = mobility
        # 人群总人数
        self.N = N
        # 人群坐标数组
        self.statement = np.zeros((5, N))
        # 单次接触后的感染概率
        self.lambda_ = threshold * mu_
        # 发病后恢复时间
        self.mu_ = mu_
        # 边界值
        self.scope = scope
        # 聚集系数
        self.alpha = alpha
        # 视野距离
        self.view_distance = view_distance
        # 感染距离
        self.infect_distance = infect_distance
        self.model = model
        self.dist_matrix = None

        # 创建一个数组作为该人群在平面上的坐标，另外包括感染状态，0为未感染，1为感染，2为已康复

    # A人群为5-19岁高危人群,默认为0
    def generate_point(self):
        # if seed:
        #     np.random.seed(seed)
        # 0. 人群编号
        # 1. 横坐标、
        # 2. 纵坐标、
        # 3. 感染状态、
        # 4. 粒子移动方向（角度制）
        self.statement[0] = np.arange(self.N)
        self.statement[1] = np.random.uniform(-self.scope, self.scope, self.N)
        self.statement[2] = np.random.uniform(-self.scope, self.scope, self.N)
        self.statement[0].astype(np.int32)
        self.statement[3].astype(np.int32)
        # self.statement[4] = np.zeros(self.N)
        # point = np.concatenate(y_point,x_point,infect_statement)
        # return y_point,x_point,infect_statement

    # 传入自定义数组
    def Special_point(self, array):
        self.statement = array

    # 计算该系统的距离矩阵
    def cal_distance(self):
        # 先筛选方格数据，再依次计算欧氏距离以筛选圈数据
        coords = self.statement[1:3, :].T
        distance_matrix = cdist(coords, coords, 'euclidean')
        return distance_matrix

    # 调用转向函数（turn_direction）并获得转向角度以进行移动
    def move(self):
        self.dist_matrix = self.cal_distance()
        for move_index in range(self.statement.shape[1]):
            theta = self.turn_direction(move_index)
            speed = np.random.normal(self.mobility, 0.1 * self.mobility)

            self.statement[1, move_index] += speed * np.cos(theta / 180 * np.pi)
            self.statement[2, move_index] += speed * np.sin(theta / 180 * np.pi)

        self.statement[1, (self.statement[1] > self.scope)] -= self.scope  # x坐标
        self.statement[1, (self.statement[1] < -self.scope)] += self.scope
        self.statement[2, (self.statement[2] > self.scope)] -= self.scope  # y坐标
        self.statement[2, (self.statement[2] < -self.scope)] += self.scope

    # 待移动agent的坐标，计算得到该点的随机移动方向与速度
    def turn_direction(self, move_index):
        # 计算每个点到i点的距离
        move_index_x = self.statement[1, int(move_index)]
        move_index_y = self.statement[2, int(move_index)]
        # 计算反三角函数得到每个点对应的夹角
        index = np.where((self.dist_matrix[move_index] < self.view_distance) & (self.dist_matrix[move_index] > 0))
        dis = np.vstack((self.dist_matrix[move_index, index], index))

        index = dis[1].astype(np.int32).tolist()
        angle_list = np.arcsin(((self.statement[2, index] - self.statement[2, int(move_index)]) / dis[0])) / np.pi * 180
        points = np.vstack((
            self.statement[:3, index], angle_list, dis[0]
        ))
        points[3, (points[1] - move_index_x < 0) & (points[2] - move_index_y < 0)] -= 90
        points[3, (points[1] - move_index_x < 0) & (points[2] - move_index_y > 0)] += 90
        # 计数每个扇形分区
        count_number = [points[:, points[3] >= 120].shape[1] + 1,
                        points[:, (points[3] < 120) & (points[3] >= 60)].shape[1] + 1,
                        points[:, (points[3] < 60) & (points[3] >= 0)].shape[1] + 1,
                        points[:, (points[3] < 0) & (points[3] >= -60)].shape[1] + 1,
                        points[:, (points[3] < -60) & (points[3] >= -120)].shape[1] + 1,
                        points[:, (points[3] < -120)].shape[1] + 1]
        # 构建区间上下限、区间概率列表
        distribution = []
        for i in range(6):
            distribution.append([180 - 60 * i, 120 - 60 * i,
                                 (count_number[i] ** self.alpha) / sum(j ** self.alpha for j in count_number)])
        # 随机生成目标区间索引
        index = np.random.choice(a=[i for i in range(6)], p=[i[2] for i in distribution])
        # if count_number[index] > 1:
        #     speed = points[5, (points[4] < distribution[index][0]) & (points[4] >= distribution[index][1])].min()
        # else: speed = self.mobility
        theta = random.uniform(distribution[index][0], distribution[index][1])
        self.statement[4, move_index] = theta
        return theta

    def infect(self, ):
        # today_infect = np.array()
        self.dist_matrix = self.cal_distance()
        infect_indexes = np.where(self.statement[3] == 1)
        for infect_index in infect_indexes[0]:
            i_distance = np.where(
                (self.dist_matrix[infect_index] < self.infect_distance) & (self.dist_matrix[infect_index] > 0))
            # 得到处于感染距离内的个体的索引
            # contact_point_index = np.where((i_distance[0] < infect_distance) & (self.statement[3] == 0))
            # 遍历这些个体，并判定其是否感染
            for j in i_distance[0]:
                # 感染过程仅对易感者生效
                if self.statement[3, j] == 0:
                    if random.uniform(0, 1) <= self.lambda_:
                        # today_infect = np.vstack(today_infect, self.statement3, j] + 1, axis=0)
                        # 修改感染状态为1，添加剩余恢复时间
                        self.statement[3, j] = 1
                        # self.statement[4, int(j)] = (np.random.normal(self.retime, self.retimesd))

    # 采用概率方法而非计算剩余感染时间来判断恢复过程
    def recover(self, status):
        # today_recover = np.array()
        recover_index = np.where(self.statement[3] == 1)
        for i in recover_index[0]:
            if random.uniform(0, 1) <= self.mu_:
                self.statement[3, i] = status

    # 依照模型所设定的状态执行感染恢复方法
    def contagion(self):
        if self.model == 'SIS':
            self.infect()
            self.recover(status=0)
        elif self.model == 'SIR':
            self.infect()
            self.recover(status=2)

    # 返回参数以供监视模型内部状态
    def return_parameters(self):
        parameters = [self.N, self.mobility, self.lambda_, self.alpha]
        return parameters

    # 绘制模型各感染状态粒子位置状态
    def draw(self, time, path, title ):
        plt.figure(figsize=(8, 6), dpi=100)
        plt.scatter(self.statement[1, (self.statement[3] == 0)], self.statement[2, (self.statement[3] == 0)],
                    label=f'S = {self.statement[:, self.statement[3,] == 0].shape[1]}', color='#9ac9db')
        plt.scatter(self.statement[1, (self.statement[3] == 1)], self.statement[2, (self.statement[3] == 1)],
                    label=f'I = {self.statement[:, self.statement[3,] == 1].shape[1]}', color='#c82423')
        if self.model == 'SIR':
            plt.scatter(self.statement[1, (self.statement[3] == 2)], self.statement[2, (self.statement[3] == 2)],
                        label=f'R = {self.statement[:, self.statement[3,] == 2].shape[1]}')
        # plt.title(title,loc = 'left',y = 0.93)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(loc=1)
        plt.savefig(f'{path}/{title}-Time = {time}.png')
        # plt.savefig(f'{path}/{title}-Time = {time}.eps', format='eps')

    # 计算模型中相变监视参数
    def polarization(self):
        theta_array = self.statement[4]
        s = np.zeros((2, self.N))
        s[0], s[1] = np.sin(theta_array), np.cos(theta_array)
        collectivity = np.linalg.norm(s.sum(axis=1)) / self.N
        return collectivity

    def network_contact(self):
        # 创建个体网络，感染者图，易感者图，总人口图各一个
        G, G_S, G_I = nx.Graph(), nx.Graph(), nx.Graph()
        # 选取人群编号作为网络节点的编号
        G.add_nodes_from(self.statement[0])
        G_S.add_nodes_from(self.statement[0, (self.statement[3] == 0)])
        G_I.add_nodes_from(self.statement[0, (self.statement[3] == 1)])
        # 遍历所有节点，并获取与之相连的节点的编号，判断两节点属性，编入相应的的网络中
        for i in range(self.N):
            index = np.where((self.dist_matrix[i] < self.view_distance) & (self.dist_matrix[i] > 0))[0]
            G.add_edges_from([[i, x] for x in index])
            if self.statement[3, i] == 0:
                G_S.add_edges_from([[i, x] for x in index if self.statement[3, int(x)] == 0])
            elif self.statement[3, i] == 1:
                G_I.add_edges_from([[i, x] for x in index if self.statement[3, int(x)] == 1])

        k_G, k_S, k_I = [g.number_of_edges() / g.number_of_nodes() if g.number_of_nodes() > 0 else 0 for g in
                         [G, G_S, G_I]]
        return k_G, k_S, k_I

    # 计算最大组件节点数占所有节点的比例
    def networkx(self):
        # 创建个体网络，感染者图，易感者图，总人口图各一个
        G, G_S, G_I = nx.Graph(), nx.Graph(), nx.Graph()
        # 选取人群编号作为网络节点的编号
        G.add_nodes_from(self.statement[0])
        G_S.add_nodes_from(self.statement[0, (self.statement[3] == 0)])
        G_I.add_nodes_from(self.statement[0, (self.statement[3] == 1)])
        # 遍历所有节点，并获取与之相连的节点的编号，判断两节点属性，编入相应的的网络中
        for i in range(self.N):
            index = np.where((self.dist_matrix[i] < self.view_distance) & (self.dist_matrix[i] > 0))[0]
            G.add_edges_from([[i, x] for x in index])
            if self.statement[3, i] == 0:
                G_S.add_edges_from([[i, x] for x in index if self.statement[3, int(x)] == 0])
            elif self.statement[3, i] == 1:
                G_I.add_edges_from([[i, x] for x in index if self.statement[3, int(x)] == 1])
        largest_cc = max(nx.connected_components(G), key=len)
        largest_cc_S = max(nx.connected_components(G_S), key=len)
        if G_I:
            largest_cc_I = max(nx.connected_components(G_I), key=len)
        else:
            largest_cc_I = []
        # fig,axes = plt.subplots(1,3,figsize = (16,5),dpi = 100)
        # nx.draw(G,ax = axes[0], pos=nx.spring_layout(G))
        # nx.draw(G_S,ax = axes[1], pos=nx.spring_layout(G))
        # nx.draw(G_I,ax = axes[2], pos=nx.spring_layout(G))
        # fig.savefig('Test/network.png')
        # 计算各网络中各个组件的平均最短路径
        G_avrg_ptlt, GI_avrg_ptlt, GS_avrg_ptlt = 0, 0, 0
        for C in (G.subgraph(c).copy() for c in nx.connected_components(G)):
            temp = nx.average_shortest_path_length(C)
            if temp > G_avrg_ptlt:
                G_avrg_ptlt = temp

        for C in (G_S.subgraph(c).copy() for c in nx.connected_components(G_S)):
            temp = nx.average_shortest_path_length(C)
            if temp > GS_avrg_ptlt:
                GS_avrg_ptlt = temp

        for C in (G_I.subgraph(c).copy() for c in nx.connected_components(G_I)):
            temp = nx.average_shortest_path_length(C)
            if temp > GI_avrg_ptlt:
                GI_avrg_ptlt = temp

        # 计算最长路径，直径
        lenth_G, lenth_S, lenth_I = dict(nx.all_pairs_shortest_path_length(G)), dict(
            nx.all_pairs_shortest_path_length(G_S)), dict(nx.all_pairs_shortest_path_length(G_I))
        Dimt_G, Dimt_S, Dimt_I = pd.DataFrame(lenth_G).max().max(), pd.DataFrame(lenth_S).max().max(), pd.DataFrame(
            lenth_I).max().max()
        k_G, k_S, k_I = [g.number_of_edges() / g.number_of_nodes() if g.number_of_nodes() > 0 else 0 for g in
                         [G, G_S, G_I]]
        Gc_G, Gc_S, Gc_I = [nx.average_clustering(g) if g.number_of_nodes() > 0 else 0 for g in [G, G_S, G_I]]
        return len(largest_cc), len(largest_cc_S), len(largest_cc_I), round(G_avrg_ptlt, 10), round(GI_avrg_ptlt, 10), \
            round(GS_avrg_ptlt, 10), Dimt_G, Dimt_S, Dimt_I, k_G, k_S, k_I, Gc_G, Gc_S, Gc_I
