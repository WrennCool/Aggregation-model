# 本脚本为模拟单一人群接触模式下，人群内部传染病传播状态
# 本文档作为批量模拟文档下的一个中转脚本，用以调用原始模型类的各个方法，并传入各种参数，传出模拟结果或监视参数

import matplotlib.pyplot as plt
from NEW_MCmodel_2 import Population
import numpy as np
import time
from multiprocessing import Pool, cpu_count
import threading
import os
import csv


def simulation(model='SIS',
               infect_distance=1,
               view_distance=1,
               n=1000,
               threshold_=0.5,
               alpha=2,
               mobility=0.3,
               scope=15,
               seed=0.99,
               id_=None
               ):
    # 初始化一些参数
    np.random.seed(1)
    PPP = Population(N=n, mobility=mobility, view_distance=view_distance,
                     infect_distance=infect_distance, threshold=threshold_, alpha=alpha, scope=scope,
                     model=model)
    PPP.generate_point()
    PPP.statement[3, :int(PPP.N * seed)] = 1
    day_count = 0
    path = 'Test'
    max_timescale = 400
    sir = []
    while True:
        PPP.move()
        PPP.contagion()
        if day_count%20==0:
            PPP.draw(time = day_count,path = path,title='')
        susceptible = PPP.statement[:, PPP.statement[3,] == 0].shape[1]
        infected = PPP.statement[:, PPP.statement[3,] == 1].shape[1]
        sir.append([day_count, susceptible, infected])
        day_count += 1
        if day_count > max_timescale:
            break
    plague_record = np.array(sir)
    infected = PPP.statement[:, PPP.statement[3,] == 1].shape[1]
    # SIR
    plt.figure(figsize=(8, 6), dpi=100)
    plt.plot(plague_record[:, 0], plague_record[:, 1] / 1000, label='S', color='#9ac9db')
    plt.plot(plague_record[:, 0], plague_record[:, 2] / 1000, label='I', color='#c82423')
    if PPP.model == 'SIR':
        plt.plot(plague_record[:, 0], plague_record[:, 3], label='R')
    # plt.title(title)
    # 绘制图片编号
    # plt.title(id, loc='left', y=0.93)
    plt.legend(loc=1)
    plt.xlabel('Timestep')
    plt.ylabel('Fraction')
    plt.savefig(f'{path}/{id_}-plague.png')
    # plt.savefig(f'{path}/{id}-plague.eps', format='eps')
    prevalence = infected / PPP.N
    # collectivity = PPP.polarization()
    nx = PPP.networkx()
    # print( prevalence,PPP.N,[i for i in nx])
    return prevalence, PPP.N, [i for i in nx]



if __name__ == '__main__':
    # print("CPU内核数:{}".format(cpu_count()))
    # print('当前母进程: {}'.format(os.getpid()))
    # start = time.time()
    #
    # #多进程方法
    # p = Pool()
    # p.map_async(simulation(),[])
    # print('等待所有子进程完成。')
    # p.close()
    # p.join()
    #
    # end = time.time()
    # print("总共用时{}秒".format(round((end - start),2)))
    result = simulation()
    print(list(result[:-1]) + result[-1])
