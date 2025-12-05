import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pickle
import random
import logging
from autoencoder import Autoencoder, Att_Autoencoder


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

class sEEGDataset(Dataset):
    def __init__(self, data, label1, label2):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.label1 = torch.tensor(label1, dtype=torch.long)
        self.label2 = torch.tensor(label2, dtype=torch.long)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.label1[idx], self.label2[idx]

# def load_data_with_dataloader(names, batch_size):
#     all_data = []
#     all_label1 = []
#     all_label2 = []
#
#     num_classes = len(names)  
#     for label1, name in enumerate(names, start=1):
#         # print(label1)
#         data = np.load(f'./onset_wake_sleep/{name}/6bands_power_data.npy')
#         num_channels = data.shape[0]
#         all_data.append(data)
#
#         labels = np.full((num_channels,), label1)  
#         all_label1.append(labels)
#
#         labels2 = np.arange(1, num_channels + 1) 
#         all_label2.append(labels2)
#
#     combined_data = np.concatenate(all_data, axis=0)
#     combined_labels = np.concatenate(all_label1, axis=0)
#     combined_labels2 = np.concatenate(all_label2, axis=0)
#
#     output_file_name = 'all_concatenated_data'
#     np.save(output_file_name, combined_data)
#     print(f'Saved combined data to {output_file_name}')
#
#     dataset = sEEGDataset(combined_data, combined_labels, combined_labels2)
#     save_dataset(dataset, 'sEEGDataset.pkl')
#
#     return dataset

def save_dataset(dataset, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"Dataset saved to {filename}")

def load_dataset(filename):
    with open(filename, 'rb') as f:
        dataset = pickle.load(f)
    print(f"Dataset loaded from {filename}")
    return dataset


device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
logging.basicConfig(filename='auto_training.log', level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

set_seed(2)
names1 = ['wangqiyong', 'wuchunrong', 'wushengjiang', 'wudong', 'guoxiaoyan', 'linlifang']
names2 = ['dengyongmiao', 'fangzekai', 'huangsonghua', 'lailimei', 'shuhuanhuan', 'tianfengyuan']
names3 = ['wangjinyong', 'huangnuoxin','wuyuhan', 'yangchen', 'zhangyuming', 'zhengminglong', 'shiqiuxia', 'taojiaqing']
names = names1 + names2 + names3

#
# output_file_name = 'all_concatenated_data'
# load_filename = f'./{output_file_name}.npy'
# data = np.load(load_filename)  # (3021, 3, 6, 30000)
# print(data)
# print(data.shape)

batch_size = 32
num_epochs = 50
input_dim = 30000
dim_coef = 5
#
model = Att_Autoencoder(dim=input_dim, dim_coef=dim_coef)
# model = Autoencoder(dim=input_dim, dim_coef=dim_coef)
model = model.to(device)
criterion = nn.MSELoss() 
optimizer = optim.Adam(model.parameters(), lr=0.002)

loaded_dataset = load_dataset('sEEGDataset.pkl')
dataloader = DataLoader(loaded_dataset, batch_size=32, shuffle=True)

data_len = len(loaded_dataset)

best_loss = float('inf')
best_model_state = None  
best_encoded_output = None  

losses = []
num_iter = data_len // batch_size
for epoch in range(num_epochs):
    epoch_loss = 0
    record_encoder_data = []
    record_label = []
    record_label2 = []
    for data, label, label2 in tqdm(dataloader, desc="Training", unit="batch", leave=False):
        data = data.to(device)
        # data = data.reshape()
        # print(label, label2)

        out, encoded_output = model(data)
        out = out.to(device)

        loss = criterion(out, data)
        epoch_loss = epoch_loss + loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        record_encoder_data.append(encoded_output.detach().cpu().numpy())
        record_label.append(label.detach().cpu().numpy())
        record_label2.append(label2.detach().cpu().numpy())

    avg_loss = epoch_loss / num_iter
    losses.append(avg_loss)
    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss}')
    logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss}')

    if avg_loss < best_loss:
        best_loss = avg_loss
        best_encoded_output = np.concatenate(record_encoder_data, axis=0)
        best_label = np.concatenate(record_label, axis=0)
        best_label2 = np.concatenate(record_label2, axis=0)

        np.savez(f'./shared_best_att_encoded_output.npz', encoded_output=best_encoded_output,
                 label=best_label, label2=best_label2)

print(f'Best model saved with loss {best_loss:.8f}')

# # np.save(f'./onset_wake_sleep/{name}/auto_encoded_output.npy', best_encoded_output)
# # np.save(f'./onset_wake_sleep/{name}/best_att_encoded_output.npy', best_encoded_output)
# np.save(f'./onset_wake_sleep/{name}/shared_best_att_encoded_output.npy', best_encoded_output)
# print(f"Data saved to:./onset_wake_sleep/{name}/shared_best_att_encoded_output.npy")


