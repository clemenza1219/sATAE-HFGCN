import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, DynamicEdgeConv
import torch.nn as nn


# 定义简单的GCN模型
class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.conv4 = GCNConv(hidden_dim, hidden_dim)
        self.conv5 = GCNConv(hidden_dim, hidden_dim)
        self.conv6 = GCNConv(hidden_dim, hidden_dim)
        self.conv7 = GCNConv(hidden_dim, num_classes)


    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        x1 = self.conv1(x, edge_index, edge_weight)
        x1 = F.relu(x1)

        x2 = self.conv2(x1, edge_index, edge_weight)
        x2 = F.relu(x2)

        x3 = self.conv3(x2, edge_index, edge_weight)
        x3 = F.relu(x3)

        out = x1 + x3

        # x4 = self.conv4(x3+x1, edge_index, edge_weight)
        # x4 = F.relu(x4)

        # x5 = self.conv5(x4, edge_index, edge_weight)
        # x5 = F.relu(x5)
        #
        # x6 = self.conv6(x1+x3+x5, edge_index, edge_weight)
        # x6 = F.relu(x6)

        # x7 = self.conv7(x4, edge_index, edge_weight)
        # x7 = F.relu(x7)

        return F.log_softmax(x3, dim=1)


class D_GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, k, dropout):
        super(D_GCN, self).__init__()
        # self.conv1 = GCNConv(num_features, hidden_dim)
        self.edge_conv1 = DynamicEdgeConv(nn.Sequential(
            nn.Linear(2 * hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ), k=k)

        self.edge_conv2 = DynamicEdgeConv(nn.Sequential(
            nn.Linear(2 * hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ), k=k)

        self.edge_conv3 = DynamicEdgeConv(nn.Sequential(
            nn.Linear(2 * hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ), k=k)


        self.node_conv1 = GCNConv(num_features, hidden_dim)
        self.node_conv2 = GCNConv(hidden_dim, hidden_dim)
        self.node_conv3 = GCNConv(hidden_dim, hidden_dim)

        self.do = nn.Dropout(dropout)
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
        # print(x.shape)
        n1 = self.node_conv1(x, edge_index, edge_weight) # torch.Size([136, 256])
        n1 = self.tanh(n1)
        n1 = self.do(n1)
        # print(n1.shape)
        # n1_norm = torch.norm(n1, p=2, dim=1, keepdim=True)
        # print(n1_norm.shape)
        e1 = self.edge_conv1(n1)  # torch.Size([136, 256])
        # print(e1.shape)
        # e1_n1 = torch.mul(n1_norm, e1)
        # l2-norm
        e1_n1 = torch.norm(e1 + n1, p=2, dim=1, keepdim=True)
        # print(e1_n1.shape)
        # e1_n1 = e1 + n1

        n2 = self.node_conv2(n1, edge_index, edge_weight)
        n2 = self.tanh(n2)
        n2 = self.do(n2)
        n2_norm = torch.norm(n2, p=2, dim=1, keepdim=True)
        e2 = self.edge_conv2(n2)
        # e2_n2 = torch.mul(n2_norm, e2)
        e2_n2 = e2 + n2

        n3 = self.node_conv3(n2, edge_index, edge_weight)
        n3 = self.tanh(n3)
        n3 = self.do(n3)
        n3_norm = torch.norm(n3, p=2, dim=1, keepdim=True)
        e3 = self.edge_conv3(n3)
        # e3_n3 = torch.mul(n3_norm, e3)
        e3_n3 = e3 + n3

        # fusion stage
        x1 = torch.mul(e1_n1, e2_n2)
        x2 = torch.mul(x1, e3_n3)

        # print('最终输出形状:', x2.shape)

        return F.log_softmax(x2, dim=1)


# 定义三层GCN模型
class ThreeLayerGCN(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(ThreeLayerGCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.conv3 = GCNConv(hidden_dim2, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.relu(self.conv1(x, edge_index))
        x = self.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        return x