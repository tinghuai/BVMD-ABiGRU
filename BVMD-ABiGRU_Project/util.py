import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from itertools import chain
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from models import  GRU, BiGRU
from data_process import data_cut, device, get_mape, get_mae, get_mse, get_nse, setup_seed, get_rmse
import pandas as pd

setup_seed(20)




def train(args, path):
    data_train, data_test = data_cut(batch_size=args.batch_size, window_len=args.window_len, data_file=args.data_file,
                                     step_size=args.step_size,
                                     input_features=args.input_features)
    if args.bidirectional:
        model = BiGRU(args.input_size, args.hidden_size, args.num_layers, args.output_size,
                       batch_size=args.batch_size).to(device)
    else:
        model = GRU(args.input_size, args.hidden_size, args.num_layers, args.output_size,
                     batch_size=args.batch_size).to(device)

    loss_function = nn.MSELoss().to(device)
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=0.9, weight_decay=args.weight_decay)
    # training
    loss = 0
    print('训练开始...')
    for i in range(args.epochs):
        for (seq, label) in data_train:
            seq = seq.to(device)
            label = label.to(device)
            y_pred = model(seq)
            loss = loss_function(y_pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('epoch', i, ':', loss.item())
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    print('训练结束...')
    torch.save(state, path)


def test(args, path):
    data_train, data_test = data_cut(batch_size=args.batch_size, window_len=args.window_len, data_file=args.data_file,
                                     step_size=args.step_size,
                                     input_features=args.input_features)
    pred = []
    y = []
    if args.bidirectional:
        model = BiGRU(args.input_size, args.hidden_size, args.num_layers, args.output_size,
                       batch_size=args.batch_size).to(device)
    else:
        model = GRU(args.input_size, args.hidden_size, args.num_layers, args.output_size,
                     batch_size=args.batch_size).to(device)
    model.load_state_dict(torch.load(path)['model'])
    model.eval()
    print('预测...')
    for (seq, target) in data_test:
        target = list(chain.from_iterable(target.data.tolist()))
        y.extend(target)
        seq = seq.to(device)
        with torch.no_grad():
            y_pred = model(seq)
            y_pred = list(chain.from_iterable(y_pred.data.tolist()))
            pred.extend(y_pred)

    y, pred = np.array([y]), np.array([pred])

    # 输出结果保存到CSV
    spei_real = pd.DataFrame(y.T)
    spei_pred = pd.DataFrame(pred.T)


    merged_df = pd.concat([spei_real, spei_pred], axis=1)
    merged_df.to_csv('./data/spei_xxx.csv', mode='w', index=False, encoding='gbk')
    print('nse:', get_nse(y, pred))
    print('mae:', get_mae(y, pred))
    print('rmse:', get_rmse(y, pred))

    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 15
    fig_size[1] = 5
    plt.rcParams["figure.figsize"] = fig_size

    result_len = len(y.T)
    x = [i for i in range(1, result_len + 1)]
    # 横坐标
    x = np.linspace(np.min(x), np.max(x), result_len)
    plt.plot(x, y.T, c='green', marker='*', ms=1, alpha=0.75, label='true')
    plt.plot(x, pred.T, c='red', marker='o', ms=1, alpha=0.75, label='pred')
    plt.title('spei-12')
    plt.grid(axis='y')
    plt.ylim(-2.5, 2.5)
    plt.tick_params(axis='both', direction='in')
    plt.legend()
    plt.show()

  
