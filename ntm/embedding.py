import os
import re
import torch
import pickle
import argparse
import logging
import time
import pprint

from GSM import GSM
from utils import *
from dataset import DocDataset
from multiprocessing import cpu_count
from torch.utils.data import Dataset, DataLoader

# from torch.utils.data import Dataset,DataLoader

parser=argparse.ArgumentParser('GSM topic model')
parser.add_argument('--taskname', type=str, default='gossipcop', help='Taskname e.g cnews10k')
parser.add_argument('--no_below', type=int, default=5, help='The lower bound of count for words to keep, e.g 10')
parser.add_argument('--no_above', type=float, default=0.005,
                    help='The ratio of upper bound of count for words to keep, e.g 0.3')
parser.add_argument('--num_epochs', type=int, default=1000,
                    help='Number of iterations (set to 100 as default, but 1000+ is recommended.)')
parser.add_argument('--n_topic', type=int, default=16, help='Num of topics')
parser.add_argument('--bkpt_continue', type=bool, default=True,
                    help='Whether to load a trained model as initialization and continue training.')
parser.add_argument('--use_tfidf', type=bool, default=False, help='Whether to use the tfidf feature for the BOW input')
parser.add_argument('--rebuild', action='store_true',
                    help='Whether to rebuild the corpus, such as tokenization, build dict etc.(default False)')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size (default=512)')
parser.add_argument('--criterion', type=str, default='cross_entropy',
                    help='The criterion to calculate the loss, e.g cross_entropy, bce_softmax, bce_sigmoid')
parser.add_argument('--auto_adj', action='store_true',
                    help='To adjust the no_above ratio automatically (default:rm top 20)')
parser.add_argument('--ckpt', type=str, default="./ckpt/GSM_gossipcop_tp16_2024-06-07-18-09_ep1000.ckpt", help='Checkpoint path')
parser.add_argument('--lang', type=str, default="en", help='Language of the dataset')

args=parser.parse_args()

def main():
    global args
    taskname=args.taskname
    no_below=args.no_below
    no_above=args.no_above
    num_epochs=args.num_epochs
    n_topic=args.n_topic
    n_cpu=cpu_count() - 2 if cpu_count() > 2 else 2
    bkpt_continue=args.bkpt_continue
    use_tfidf=args.use_tfidf
    rebuild=args.rebuild
    batch_size=args.batch_size
    criterion=args.criterion
    auto_adj=args.auto_adj
    ckpt=args.ckpt
    lang=args.lang

    if not os.path.exists(f'./ntm/results/{taskname}'):
        os.makedirs(f'./ntm/results/{taskname}')

    device = torch.device('cuda')
    docSet = DocDataset(taskname, lang=lang, no_below=no_below, no_above=no_above, rebuild=rebuild, use_tfidf=False)

    voc_size=docSet.vocabsize
    print('voc size:', voc_size)
    print('doc size:', len(docSet.docs))

    if ckpt:
        param = {}
        checkpoint=torch.load(ckpt)
        param.update({"device": device})
        param.update(checkpoint["param"])
        model=GSM(**param)
        model.train(train_data=docSet, batch_size=batch_size, test_data=docSet, num_epochs=num_epochs, log_every=10,
                    beta=1.0, criterion=criterion, ckpt=checkpoint)
    else:
        raise Exception("Please provide a checkpoint for training")

    # model.evaluate(test_data=docSet)
    save_name=f'./ntm/ckpt/GSM_{taskname}_tp{n_topic}_{time.strftime("%Y-%m-%d-%H-%M", time.localtime())}.ckpt'

    data_loader = DataLoader(docSet, batch_size=512, shuffle=False, num_workers=4,
                             collate_fn=docSet.collate_fn)



    embed_lst = []
    with torch.no_grad():
        for data_batch in data_loader:
            txts, bows = data_batch
            bows = bows.to(device)
            # 分词过的文档、字典，以及是否经过softmax
            # docSet.docs / txts
            embed = model.inference_by_bow(bows)
            # embed = model.inference(docSet.docs, docSet.dictionary, normalize=True)
            embed_lst.append(embed)
        embed_lst=np.array(embed_lst, dtype=object)
        embed_lst=np.concatenate(embed_lst, axis=0)
    # print(embed_lst.shape)
    # print(embed_lst[:5])

    # 获取每个doc的主题
    doc_topics = np.argmax(embed_lst, axis=1)
    # 写出doc对应的主题
    with open(f'./results/{taskname}_doc_topics.txt', 'w', encoding='utf-8') as f:
        for i in range(len(docSet.docs)):
            f.write(f'{doc_topics[i]}\n')
    # 计算每个主题的doc数量
    topic_count = {}
    for i in range(len(doc_topics)):
        if doc_topics[i] not in topic_count:
            topic_count[doc_topics[i]] = 1
        else:
            topic_count[doc_topics[i]] += 1
    # 计算每个主题的虚假新闻数量
    fake_count = {}
    for i in range(3784):
        if doc_topics[i] not in fake_count:
            fake_count[doc_topics[i]] = 1
        else:
            fake_count[doc_topics[i]] += 1
    # 计算每个主题的真实新闻数量
    real_count = {}
    for i in range(3784, 15729):
        if doc_topics[i] not in real_count:
            real_count[doc_topics[i]] = 1
        else:
            real_count[doc_topics[i]] += 1
    # 合并上述数量，[主题，总数，虚假数，真实数]，打印
    topic_info = []
    for i in range(n_topic):
        topic_info.append([i, topic_count[i], fake_count[i], real_count[i]])
    pprint.pprint(topic_info)



    # 写出docs在每个主题上的分布
    for i in range(n_topic):
        with open(f'./results/{taskname}_topic_{i}.txt', 'w', encoding='utf-8') as f:
            for j in range(len(docSet.docs)):
                f.write(f'{embed_lst[j][i]}\n')

    # 检查长度
    with open(f'./results/{taskname}_topic_0.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        print('主题0的长度：', len(lines))


    # 写出docs的embed
    # with open(f'./results/{taskname}_embed.txt', 'w', encoding='utf-8') as f:
    #     for i in range(len(docSet.docs)):
    #         f.write(f'{embed_lst[i]}\n')

    # 写出docs的embed，用npy格式保存
    np.save(f'./results/{taskname}_embed.npy', embed_lst)



if __name__ == "__main__":
    main()