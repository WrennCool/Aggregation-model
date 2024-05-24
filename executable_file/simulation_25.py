# 本脚本用于模拟New_MCmodel_2脚本代码，该版本代码新增网络平均度、全局聚类系数等参数

from multiprocessing import Pool, cpu_count
import numpy as np
import os
import time
import sys
sys.path.append("../")
from Simulation_control_2 import simulation
import csv


# seed = sys.argv[1]
# save_path = sys.argv[2]
para = [[2, 0.3, 1], [2, 1.5, 2], [-2, 0.3, 3], [-2, 1.5, 4], [0, 0.3, 5]]

save_path = 'result25.csv'
seed = 0.99
# para = [[2, 0.3, 1]]

iter_count = 51

def gen_parameter():
    def gen_arr(a, v, sign):
        arr = np.zeros((6, iter_count))
        # a,t,v
        arr[0] = a
        arr[1] = np.linspace(0, 1.5, iter_count).tolist()
        # arr[1] = np.linspace(0, 1.5, 51).tolist()
        arr[2] = v
        # 不同场景编号
        arr[3] = sign
        return arr

    array = np.zeros((6, 0))
    for a, v, sign in para:
        array = np.hstack((array, gen_arr(a, v, sign)))

    array[4] = seed
    array[5] = np.arange(array.shape[1])
    return array


def onsimulation(a, t, v, sign, seed, id):
    n, scope = 1000, 15
    result = simulation(alpha=a, mobility=v, threshold_=t, n=n, scope=scope, seed=seed)
    with open(f'../simulation_result/{save_path}', 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(list(result[:-1]) + result[-1] + [a, t, v, sign, seed, id, scope])


if __name__ == '__main__':
    start = time.time()
    p = Pool()
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f'Starting computing, start_time = {start_time}')
    array = gen_parameter()
    zip_args = list(zip(array[0], array[1], array[2], array[3], array[4], array[5]))
    p.starmap_async(onsimulation, zip_args)
    print('等待所有子进程完成。')
    p.close()
    p.join()
    end = time.time()
    print("总共用时{}小时".format(round((end - start) / 3600, 3)))
