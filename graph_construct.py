
import os
import torch
import numpy as np
import scipy.sparse as sp
from scipy.special import comb
from scipy.sparse import csr_matrix, coo_matrix

if not os.path.exists('./data'):
    os.makedirs('./data')


def graph_construct(j):
    # 读取gossipcop_topic_i.txt文件
    topic_path = f'./ntm/results/gossipcop_topic_{j}.txt'
    with open(topic_path, 'r') as f:
        docs_topic = f.readlines()
        num = len(docs_topic)
        # print('doc长度：', num)
        no_value = []
        for i in range(num):
            no_value.append([i, float(docs_topic[i].strip())])
        # print(no_value[:5])
        # no_value根据docs_topic的值进行排序
        no_value.sort(key=lambda x: float(x[1]), reverse=True)
        # print(no_value[:5])

        # 对no_value的相邻两个值进行乘积，并保存到data、row、col中
        data = []
        row = []
        col = []
        for i in range(num-1):
            data.append(no_value[i][1] * no_value[i+1][1])
            row.append(no_value[i][0])
            col.append(no_value[i+1][0])
        print(data[:5])
        print(row[:5])
        print(col[:5])
        sparse_matrix = csr_matrix((data, (row, col)), shape=(num, num))
        # print(sparse_matrix)
        print(f'==========topic{j}边的数量：', sparse_matrix.count_nonzero())
        # 保存稀疏矩阵到.npz中
        sp.save_npz(f'./data/sparse_matrix_topic_{j}.npz', sparse_matrix)

for i in range(16):
    graph_construct(i)







