import numpy as np
import torch
import random
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_scipy_sparse_matrix
from scipy.sparse import csr_matrix
from torch_sparse import SparseTensor
from model import GCN, D_GCN, ThreeLayerGCN
import os
import time
import logging
from sklearn.metrics import f1_score, precision_score, recall_score


# 设置第一个日志记录器
logger1 = logging.getLogger('logger1')
logger1.setLevel(logging.INFO)
file_handler1 = logging.FileHandler('training_soz_model.log')
formatter1 = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler1.setFormatter(formatter1)
logger1.addHandler(file_handler1)

# 设置第二个日志记录器
logger2 = logging.getLogger('logger2')
logger2.setLevel(logging.INFO)
file_handler2 = logging.FileHandler('soz_model_performance.log')
formatter2 = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
file_handler2.setFormatter(formatter2)
logger2.addHandler(file_handler2)

def get_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

get_random_seed(5)
start_time = time.time()

# names = ['shiqiuxia', 'taojiaqing', 'wangqiyong', 'wuchunrong', 'wushengjiang']
# names = ['dengyongmiao', 'huangsonghua', 'lailimei', 'shuhuanhuan', 'tianfengyuan','wangjinyong', 'wuyuhan', 'yangchen', 'zhangyuming', 'zhengminglong']
# names = ['wudong', 'guoxiaoyan']
names = ['lailimei']
# names = ['shuhuanhuan']
# names = ['tianfengyuan','wangjinyong', 'wuyuhan', 'yangchen', 'zhangyuming', 'zhengminglong']

base_dir = f'./onset_wake_sleep/'
regions = ['HIP']

# 训练模型
def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data)

    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

# 验证模型
def test(data):
    model.eval()
    logits, accs = model(data), []
    loss_fn = torch.nn.CrossEntropyLoss()

    for i, mask in enumerate([data.train_mask, data.val_mask, data.test_mask]):
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()
        accs.append(acc)
        # 只计算测试集上的 Precision、Recall 和 F1-score
        if i == 2:  # data.test_mask
            test_loss = loss_fn(logits[mask], data.y[mask]).item()
            pred_numpy = pred.cpu().numpy()
            true_numpy = data.y[mask].cpu().numpy()
            precision = precision_score(true_numpy, pred_numpy, average='macro', zero_division=0)
            recall = recall_score(true_numpy, pred_numpy, average='macro', zero_division=0)
            f1 = f1_score(true_numpy, pred_numpy, average='macro', zero_division=0)

    train_acc, val_acc, test_acc = accs
    return train_acc, test_loss, val_acc, test_acc, precision, recall, f1


device = torch.device('cuda:6' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# 运行训练和测试
person_best = []
for name in names:
    data_list = []

    # data = np.load(f'{base_dir}/{name}/best_att_encoded_output.npy')
    # data = np.load(f'{base_dir}/{name}/shared_best_att_encoded_output.npy')
    data = np.load(f'/home/phd-yan.huachao/Project/Xiehedata/SOZ_classification/'
                   f'onset_wake_sleep/{name}'
                   f'/shared_best_att_encoded_output.npy')
    print(data.shape)
    print(base_dir)
    # print(data.shape)
    # 选择频带
    # 0-5 对应五个频带
    band = 'all'
    data = data[:, :, :, :]
    # data = data[:, :, band, :]

    print(data.shape)

    label = np.loadtxt(f'{base_dir}/{name}/{name}_true_node_label.csv')
    # print(label.shape)
    num_site, state, bands, time_point = data.shape
    # num_site, state, time_point = data.shape
    nodes_feature = data.reshape(num_site, -1)  # site * 3 * 6 * 8 144
    print(nodes_feature.shape)

    # 转为Tensor
    x = torch.tensor(nodes_feature, dtype=torch.float)
    y = torch.tensor(label, dtype=torch.long)

    # 设置训练、验证和测试掩码
    num_nodes = num_site
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    # 假设训练集为前 10%，验证集为中间 20%，测试集为后 70%
    train_mask[:int(0.1 * num_nodes)] = True
    val_mask[int(0.1 * num_nodes):int(0.3 * num_nodes)] = True
    test_mask[int(0.3 * num_nodes):] = True

    # 构建图数据对象
    data_obj = Data(x=x, y=y, train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)

    # 将数据对象添加到列表
    data_list.append(data_obj)
    data = data_list[0].to(device)

    num_epochs = 400
    region_acc = []
    for region in regions:
        # 加载adj
        adj = np.load(f'{base_dir}{name}/{name}_{region}_adj.npy')
        # print(adj.shape)
        # 将邻接矩阵转换为稀疏矩阵形式
        edge_index, edge_weight = from_scipy_sparse_matrix(csr_matrix(adj))
        # edge_index, edge_weight = dense_to_sparse(torch.tensor(adj))
        # edge_index = edge_index.to(dtype=torch.int64)
        # edge_weight = torch.as_tensor(edge_weight, dtype=torch.float)

        # edge_index, edge_weight = dense_to_sparse(torch.tensor(adj))
        adj = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_weight)

        data.edge_index = adj.to(device)
        # edge_weight = edge_weight.double().to(device)
        # print(edge_weight)
        save_train_acc, save_val_acc, save_test_acc, save_precision, save_recall, save_f1 = [], [], [], [], [], []

        # model = GCN(data_list[0].num_features, hidden_dim=256,
        #             num_classes=int(data_list[0].y.max().item()) + 1).to(device)
        model = D_GCN(data_list[0].num_features, hidden_dim=256, num_classes=2, k=5, dropout=0.2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
        train_loss, val_loss = [], []
        for epoch in range(num_epochs):
            loss = train(data)
            train_acc, loss_fn, val_acc, test_acc, test_precision, test_recall, test_f1 = test(data)

            save_train_acc.append(train_acc), save_val_acc.append(val_acc), save_test_acc.append(test_acc),
            save_precision.append(test_precision), save_recall.append(test_recall), save_f1.append(test_f1)
            train_loss.append(loss), val_loss.append(loss_fn)
            logger1.info(f'{name}, {region}, select_band: {band}, Epoch [{epoch + 1}/{num_epochs}], Loss: {loss:.4f}, '
                         f'Train Acc: {train_acc:.4f}, Val Loss: {loss_fn:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, '
                         f'Test Precision: {test_precision:.4f}, Test Recall: {test_recall:.4f}, Test F1 Score: {test_f1:.4f}')

            print(f'{name}, {region}, select_band: {band}, Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Acc: {val_acc:.4f}, Val Loss: {loss_fn:.4f}, Test Acc: {test_acc:.4f}, Test Precision: {test_precision:.4f}, '
                  f'Test Recall: {test_recall:.4f}, Test F1 Score: {test_f1:.4f}')

        max_test_acc = max(save_test_acc)
        max_precision = max(save_precision)
        max_recall = max(save_recall)
        max_f1 = max(save_f1)

        region_acc.append((max_test_acc, max_precision, max_recall, max_f1))
        logger2.info(
            f'{name}, {region}, select_band: {band}, Max Test Acc: {max_test_acc:.4f}, Max Precision: {max_precision:.4f}, '
            f'Max Recall: {max_recall:.4f}, Max F1 Score: {max_f1:.4f}')

        with open('loss_record.txt', 'w') as f:
            for i, (tr, val) in enumerate(zip(train_loss, val_loss)):
                f.write(f"{tr:.4f}, {val:.4f}\n")
    person_best.append(region_acc)

print(person_best)
# 计算所有人的均值
mean_values = np.mean(person_best, axis=0)
print(mean_values)
end_time = time.time()
print("Elapsed time: {:.2f} seconds".format(end_time - start_time))
