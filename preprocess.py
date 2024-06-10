# 读取json文件
import json
import os
import numpy as npq
import torch

file_path = os.path.join(os.path.dirname(__file__), 'gossipcop_v3_keep_data_in_proper_length.json')
with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)


texts = []
ids = []
labels = []
for key in data.keys():
    texts.append(data[key]['text'])
    ids.append(data[key]['id'])
    labels.append(data[key]['label'])
# print(len(labels))    # 15729

# 计算labels中fake和real的数量
fake_count = 0
real_count = 0
for label in labels:
    if label == 'fake':
        fake_count += 1
    else:
        real_count += 1
print(fake_count)    # 3784
print(real_count)    # 11945

print(len(texts))    # 15729
# 将texts输出到txt文件中
with open('ntm/data/gossipcop_lines.txt', 'w', encoding='utf-8') as f:
    for text in texts:
        # 去除text中的换行符，用空格代替
        text = text.replace('\n', ' ')
        f.write(text + '\n')
# 将labels输出到txt文件中，fake为0，real为1
with open('ntm/data/gossipcop_label.txt', 'w', encoding='utf-8') as f:
    for label in labels:
        if label == 'fake':
            f.write('0\n')
        else:
            f.write('1\n')


# 读取txt文件
with open('ntm/data/gossipcop_lines.txt', 'r', encoding='utf-8') as f:
     lines = f.readlines()
print('新闻数量：', len(lines))    # 15729