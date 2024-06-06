import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def load_data(data_file):
    # path = os.path.dirname(os.path.realpath(__file__)) + '/data/' + data_file
    path = 'data/' + data_file
    df = pd.read_csv(path, encoding='gbk')
    # columns = df.columns
    # df.fillna(df.mean(), inplace=True)
    # MAX = np.max(df[columns[1]])
    # MIN = np.min(df[columns[1]])
    # df[columns[1]] = (df[columns[1]] - MIN) / (MAX - MIN)
    return df


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return len(self.data)


def data_cut(batch_size, window_len, data_file, step_size, input_features):  # B为batch_size，默认30
    print('数据切分开始...')
    step_size = step_size - 1
    data = load_data(data_file)  # 载入数据，data为所有数据
    #标签所在列
    label_nu = 3
    spei = data[data.columns[label_nu]]
    # spei = data['spei']
    spei = spei.tolist()
    data = data.values.tolist()
    seq = []
    for i in range(len(data) - window_len - step_size):  # 外层循环 （总数据-窗口长度） 次，获取所有的数据和标签    ******-1
        train_seq = []
        train_label = []
        for j in range(i, i + window_len):  # 内层循环  （窗口长度）次，获取一条数据序列
            x = []
            #特征选择
            if input_features == 1:
                x.append(data[j][label_nu])
            elif input_features == 2:
                x.append(data[j][1])
                x.append(data[j][2])
            elif input_features == 3:
                x.append(data[j][1])
                x.append(data[j][2])
                x.append(data[j][3])
            elif input_features == 4:
                x.append(data[j][1])
            # else:
            #     x.append(data[j][3])
            train_seq.append(x)
        train_label.append(spei[i + window_len + step_size])  # ***********+1
        train_seq = torch.FloatTensor(train_seq)
        train_label = torch.FloatTensor(train_label).view(-1)
        seq.append((train_seq, train_label))  # seq 每个元素为元组（序列，标签）
    print('seq:', len(seq))
    data_train = seq[0:int(len(seq) * 0.8)]  # 训练集
    # data_test = seq[int(len(seq) * 0.8):len(seq)]  # 测试集
    data_test = seq[-120:]  # 测试集
    print("data_train长度:", len(data_train))
    print("data_test长度:", len(data_test))
    train = MyDataset(data_train)
    test = MyDataset(data_test)
    data_train = DataLoader(dataset=train, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    data_test = DataLoader(dataset=test, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=True)
    print('数据切分结束...')

    return data_train, data_test


def get_mape(y, pred):
    return np.mean(np.abs((y - pred) / y))


def get_mse(y, pred):
    return np.mean(np.square(y - pred))


def get_rmse(y, pred):
    return np.sqrt(np.mean(np.square(y - pred)))


def get_mae(y, pred):
    return np.mean(np.abs(y - pred))


def get_nse(y, pred):
    return 1 - (np.sum(np.square(y - pred)) / np.sum(np.square(y - np.mean(y))))
