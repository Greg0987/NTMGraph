import torch
import torch.nn as nn
import torch.nn.functional as F

class topic_ware(nn.Module):
    def __init__(self, n_topic, n_in, n_h, dropout):
        super(topic_ware, self).__init__()
        self.n_topic = n_topic
        # 对于每个topic，都有两个图卷积层
        # 二分类任务，输出维度为2，即fake和real
        for i in range(n_topic):
            setattr(self, f'conv1_{i}', GraphSageLayer(n_in, n_h, 1, F.relu, dropout))
            setattr(self, f'conv2_{i}', GraphSageLayer(n_h, n_h, 1, None, dropout))
            setattr(self, f'fc_{i}', nn.Linear(n_h, 2))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, adj_list, features):
        # 共有16个topic，即16个adj
        # 对于每个topic，都有两个图卷积层
        outputs = []
        for i in range(self.n_topic):
            conv1 = getattr(self, f'conv1_{i}')
            conv2 = getattr(self, f'conv2_{i}')
            adj = adj_list[i]
            h = conv1(adj, features)
            h = conv2(adj, h)
            # 各自注意力
            fc = getattr(self, f'fc_{i}')
            linear_output = fc(h)
            attention_weights = self.softmax(linear_output)
            output = attention_weights * linear_output
            outputs.append(output)
        # 对outputs求和并归一化
        outputs = torch.stack(outputs, dim=0)
        outputs = outputs.sum(dim=0)
        outputs = self.softmax(outputs)
        return outputs


class GraphSageLayer(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, activation, dropout, bias=True):
        super(GraphSageLayer, self).__init__()
        self.fc = nn.Linear(input_dim * 2, output_dim, bias=bias)
        self.activation = activation
        if dropout:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        self.init_params()

    def init_params(self):
        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0.0)

    def forward(self, adj, h):
        neighbor = adj @ h
        mean_neigh = neighbor / (adj.sum(1, keepdim=True) + 1e-7)

        h = torch.cat([h, mean_neigh], dim=-1)
        h = self.fc(h)
        if self.activation:
            h = self.activation(h)
        if self.dropout:
            h = self.dropout(h)
        return h
