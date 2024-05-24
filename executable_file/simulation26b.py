# 本脚本为MCmodel第一次模拟，旨在调整参数λ、v以查看模拟结果的变化
# 修改脚本前谨记如下内容须格外修正注意
# 1，输出文件路径
# 2，文件各列对应元素
# 3，传入执行文件中的函数


from multiprocessing import Pool, cpu_count
import numpy as np
import os
import time
import sys

sys.path.append("../")
from Simulation_control_3 import simulation
import csv

# import pandas as pd

seed = sys.argv[1]
save_path ='result26'
scope_ = sys.argv[2]
id = 0


def gen_parameter():
    N = 2601
    arr = np.zeros((6, N))
    # alpha
    arr[0] = np.linspace(-2, 2, 51).tolist() * 51
    arr.sort(axis=1)
    # mobility
    arr[2] = np.linspace(0, 1.5, 51).tolist() * 51
    # threshold
    arr[1,] = 0.5
    arr[3] = id
    # 合并三个数组
    arr[4] = seed
    arr[5] = np.arange(2601)
    return arr


def onsimulation(a, t, v, sign, seed, id):
    n, scope,max_time = 1000, int(scope_), 400
    result = simulation(alpha=a, mobility=v, threshold_=t, n=n, scope=scope, seed=seed,max_time = max_time)
    with open(f'../simulation_result/{save_path}_20.csv', 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(result[0] + [a, t, v, sign, seed, id, scope])
    with open(f'../simulation_result/{save_path}_100.csv', 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(result[1] + [a, t, v, sign+1, seed, id+1, scope])
    with open(f'../simulation_result/{save_path}_400.csv', 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(result[2] + [a, t, v, sign+2, seed, id+2, scope])

if __name__ == '__main__':
    start = time.time()
    p = Pool()
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f'Starting computing, start_time = {start_time}')
    array = gen_parameter()
    zip_args = list(zip(array[0], array[1], array[2], array[3], array[4], array[5]))
    p.starmap(onsimulation, zip_args)
    print('等待所有子进程完成。')
    p.close()
    p.join()
    end = time.time()
    print("总共用时{}小时".format(round((end - start) / 3600, 3)))
