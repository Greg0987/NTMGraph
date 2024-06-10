
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
from topic_ware_graph import topic_ware
from scipy.sparse import load_npz, coo_matrix
from utils import scipysp_to_pytorchsp
import argparse
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import os

def train(args):
    writer = SummaryWriter('results')

    # 设置固定种子
    if args.seed:
        seed = args.seed
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    if not args.sample_method:
        # ****************** 抽取方法（一）
        # labels列表中，0-3783为fake，3784-15728为real
        # 从labels中随机选取10个fake和10个real，作为train集，剩下的作为test集
        # 选取的10个fake和10个real的索引分别为train_fake_index和train_real_index
        train_fake_index = np.random.choice(range(3784), 80, replace=False)
        train_real_index = np.random.choice(range(3784, 15729), 80, replace=False)
        print('训练集虚假新闻序号：',train_fake_index)
        print('训练集真实新闻序号：',train_real_index)
        test_fake_index = [i for i in range(3784) if i not in train_fake_index]
        test_real_index = [i for i in range(3784, 15729) if i not in train_real_index]
        test_fake_index = torch.tensor(test_fake_index)
        test_real_index = torch.tensor(test_real_index)

        len_fake = len(test_fake_index)  # 3774
        len_real = len(test_real_index)  # 11935
    else:
        # ****************** 抽取方法（二）
        # 每个主题下，选取5个fake和5个real，作为train集，剩下的作为test集
        # 即共有16*10=160个为train集，剩下的为test集
        """
        16个主题的新闻数量：总数，虚假数，真实数
        [[0, 1188, 315, 873],
         [1, 1037, 134, 903],
         [2, 755, 587, 168],
         [3, 944, 263, 681],
         [4, 1401, 262, 1139],
         [5, 1093, 665, 428],
         [6, 897, 72, 825],
         [7, 879, 56, 823],
         [8, 1046, 91, 955],
         [9, 831, 141, 690],
         [10, 796, 63, 733],
         [11, 884, 418, 466],
         [12, 1188, 283, 905],
         [13, 874, 235, 639],
         [14, 730, 58, 672],
         [15, 1186, 141, 1045]]
         """
        # 读取新闻主题
        with open('./ntm/results/gossipcop_doc_topics.txt', 'r', encoding='utf-8') as f:
            doc_topics = f.readlines()
        doc_topics = [int(topic) for topic in doc_topics]
        topics_indexs = [[[], []] for _ in range(16)]   # 16个主题，每个主题下有两个列表，分别存储fake和real的索引
        for i in range(3784):
            topics_indexs[doc_topics[i]][0].append(i)
        for i in range(3784, 15729):
            topics_indexs[doc_topics[i]][1].append(i)
        # 选取train集
        train_fake_index = []
        train_real_index = []
        for i in topics_indexs:
            # print(len(i[0]), len(i[1]))
            train_fake_index.extend(np.random.choice(i[0], 5, replace=False))
            train_real_index.extend(np.random.choice(i[1], 5, replace=False))
        print(len(train_fake_index), '训练集虚假新闻序号：', train_fake_index)
        print(len(train_real_index), '训练集真实新闻序号：', train_real_index)

        test_fake_index = [i for i in range(3784) if i not in train_fake_index]
        test_real_index = [i for i in range(3784, 15729) if i not in train_real_index]
        test_fake_index = torch.tensor(test_fake_index)
        test_real_index = torch.tensor(test_real_index)
        len_fake = len(test_fake_index)
        len_real = len(test_real_index)
        print('测试集新闻数量：', len_fake, len_real)

        if args.topic_acc:
            # 将test_indexs按topics_indexs格式存储
            test_indexs = topics_indexs.copy()
            for i in range(16):
                test_indexs[i][0] = [j for j in test_indexs[i][0] if j in test_fake_index]
                test_indexs[i][1] = [j for j in test_indexs[i][1] if j in test_real_index]
                print(len(test_indexs[i][0]), len(test_indexs[i][1]))


    model = topic_ware(args.n_topic, args.n_topic, args.n_h, args.dropout)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dp_model = torch.nn.DataParallel(model, device_ids=args.gpu)
    model = dp_model.to(device)

    # 从npz中读取adj
    adj_list = []
    for i in range(args.n_topic):
        # 读取.npz文件
        sparse_matrix_csr = load_npz(f'./data/sparse_matrix_topic_{i}.npz')
        # 转换为coo_matrix
        sparse_matrix_coo = sparse_matrix_csr.tocoo()
        # 转换为torch格式
        adj = scipysp_to_pytorchsp(sparse_matrix_coo)
        adj = adj.to_dense()
        # 输入graphsage需要对adj对角线置为1
        adj.fill_diagonal_(1)
        adj = F.normalize(adj, p=1, dim=1)
        adj = adj.to(device)
        adj_list.append(adj)
        # print(adj.shape)
    # 读取features，格式为npy
    features = np.load('./ntm/results/gossipcop_embed.npy')
    features = torch.tensor(features)
    features = features.to(device)
    print(features.shape)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()   # 交叉熵损失函数
    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)   # Adam优化器


    scores= []
    scores_ = []
    for epoch in tqdm(range(args.epochs+1)):
        # ******************训练
        model.train()
        optimiser.zero_grad()
        output = model(adj_list, features)
        # 计算交叉熵损失
        # fake标签的损失
        loss_fake = criterion(output[train_fake_index], torch.zeros(len(train_fake_index), dtype=torch.long).to(device))
        # real标签的损失
        loss_real = criterion(output[train_real_index], torch.ones(len(train_real_index), dtype=torch.long).to(device))
        loss = loss_fake + loss_real
        writer.add_scalar('train/loss_ALL', loss.item(), epoch)
        writer.add_scalar('train/loss_fake', loss_fake.item(), epoch)
        writer.add_scalar('train/loss_real', loss_real.item(), epoch)
        loss.backward()
        optimiser.step()

        # ****************测试
        model.eval()
        output = model(adj_list, features)
        pred = output.argmax(dim=1)
        # 计算test中fake的准确率
        acc_fake = (pred[test_fake_index] == 0).sum().item() / len_fake
        # 计算test中real的准确率
        acc_real = (pred[test_real_index] == 1).sum().item() / len_real
        # 计算test的准确率
        acc = (pred[torch.cat((test_fake_index, test_real_index))] ==
               torch.cat([torch.zeros(len_fake, dtype=torch.long).to(device), torch.ones(len_real, dtype=torch.long).to(device)])
               ).sum().item() / (len_fake + len_real)
        scores.append([epoch, acc, acc_fake, acc_real])
        writer.add_scalars('test/accuracy',
                           {'All': acc, 'Fake': acc_fake, 'Real': acc_real},
                           epoch)
        writer.add_scalar('test/acc_all', acc, epoch)

        if args.topic_acc:
            # 计算每个topic下的准确率
            for i in range(16):
                fake_index = test_indexs[i][0]
                real_index = test_indexs[i][1]
                fake_index = torch.tensor(fake_index)
                real_index = torch.tensor(real_index)
                len_fake_ = len(fake_index)
                len_real_ = len(real_index)
                acc_fake_ = (pred[fake_index] == 0).sum().item() / len_fake_
                acc_real_ = (pred[real_index] == 1).sum().item() / len_real_
                acc_ = (pred[torch.cat((fake_index, real_index))] ==
                       torch.cat([torch.zeros(len_fake_, dtype=torch.long).to(device), torch.ones(len_real_, dtype=torch.long).to(device)])
                       ).sum().item() / (len_fake_ + len_real_)
                scores_.append([epoch, i, acc_, acc_fake_, acc_real_])
                writer.add_scalars(f'test/accuracy_topic_{i}',
                                   {'All': acc_, 'Fake': acc_fake_, 'Real': acc_real_},
                                   epoch)


        # 每隔100轮打印一下
        if epoch % args.every_epoch == 0:
            # 对scores进行排序，以acc为准
            # scores = sorted(scores, key=lambda x: x[1], reverse=True)
            print(f'Epoch {epoch} Loss: {loss.item()}')
            print(f'Epoch {epoch} Test Accuracy: {scores[-1][1]:.4f}, '
                  f'Fake Accuracy: {scores[-1][2]:.4f}, Real Accuracy: {scores[-1][3]:.4f}')
            if args.topic_acc:
                for i in range(16):
                    print(f'Epoch {epoch} Test Accuracy Topic {i}: {scores_[-16+i][2]:.4f}, '
                          f'Fake Accuracy: {scores_[-16+i][3]:.4f}, Real Accuracy: {scores_[-16+i][4]:.4f}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_method', type=int, default=1, help='0: random sample, 1: topic sample')
    parser.add_argument('--topic_acc', action='store_true', help='whether to calculate topic accuracy')

    parser.add_argument('--n_topic', type=int, default=16)
    parser.add_argument('--n_h', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--every_epoch', type=int, default=10)

    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--wd', type=float, default=0)
    parser.add_argument('--gpu', nargs='+', type=list, default=[0])
    parser.add_argument('--seed', type=int, default=0, help='default random')  # 2021
    args = parser.parse_args()
    print(args)
    train(args)