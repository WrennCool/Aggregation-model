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
from Simulation_control_2 import simulation
import csv
# import pandas as pd

seed = sys.argv[1]
save_path = sys.argv[2]
id = 2

def gen_parameter():

    N = 2601
    arr = np.zeros((6, N))
    # alpha
    arr[0] = np.linspace(-3,3,51).tolist() * 51
    arr.sort(axis=1)
    # mobility
    arr[2] =  np.linspace(0,2,51) .tolist() * 51
    # threshold
    arr = np.hstack((arr,arr,arr,arr))
    arr[1,:2601],arr[1,2601:2601*2],arr[1,2601*2:2601*3] ,arr[1,2601*3:2601*4]= 0.1,0.3,0.5,0.8
    arr[3] = id
    # 合并三个数组
    arr[4] = seed
    arr[5] = np.arange(2601*4)
    return arr


def onsimulation(a,t,v,sign,seed,id):
    N, scope  = 1000, 15
    result = simulation(alpha = a,mobility=v,threshold_=t,N = N,scope = scope,seed = seed)
    with open(f'../simulation_result/{save_path}', 'a+') as f:
        writer = csv.writer(f)
        writer.writerow(list(result) + [a,t,v,sign,seed,id,scope])


if __name__ =='__main__':
    start = time.time()
    p = Pool()
    start_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f'Starting computing, start_time = {start_time}')
    arr = gen_parameter()
    zip_args = list(zip(arr[0],arr[1],arr[2],arr[3],arr[4],arr[5]))
    p.starmap_async(onsimulation, zip_args)
    print('等待所有子进程完成。')
    p.close()
    p.join()
    end = time.time()
    print("总共用时{}小时".format((end - start)/3600))
