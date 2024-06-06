from util import train, test
import argparse
import torch


def ms_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, default='data.csv', help='input data')
    parser.add_argument('--input_features', type=int, default=1, help='input features')  # 输入特征选择



    parser.add_argument('--input_size', type=int, default=1, help='input dimension')  # 输入维度
    parser.add_argument('--step_size', type=int, default=1, help='step size')  # 预见期
    parser.add_argument('--optimizer', type=str, default='adam', help='type of optimizer')  # 优化器选择 adam  sgd
    parser.add_argument('--bidirectional', type=bool, default=True, help='LSTM direction')  # True   False
    parser.add_argument('--window_len', type=int, default=24, help='window len')  # 窗口长度
    parser.add_argument('--epochs', type=int, default=100, help='input dimension')  # 训练循环次数
    parser.add_argument('--output_size', type=int, default=1, help='output dimension')  # 输出维度
    parser.add_argument('--hidden_size', type=int, default=64, help='hidden size') #隐藏单元个数
    parser.add_argument('--num_layers', type=int, default=1, help='num layers') #层数
    parser.add_argument('--lr', type=float, default=0.05, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=30, help='batch size')
    parser.add_argument('--device', default=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')

    return parser.parse_args()


GRU_PATH = './model/demo.pkl'
args = ms_args_parser()

train(args, GRU_PATH)
test(args, GRU_PATH)
