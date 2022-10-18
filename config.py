# coding=utf-8
import argparse
from utils import read_json
import os


class ArgConfig:
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def initialize(self):
        # 路径和名称
        self.parser.add_argument('--model_name', default='bert', help='model name from [bert/electra/albert]')
        self.parser.add_argument('--data_name', default='CNER', help='data name')
        self.parser.add_argument('--bert_dir', default='./model_hub/chinese-bert-wwm-ext/')
        self.parser.add_argument('--data_dir', default='./data/')
        self.parser.add_argument('--label_name', default='labels')
        self.parser.add_argument('--tag2id_name', default='nor_ent2id')

        self.parser.add_argument('--save_model_name', default='best.pt', help='保存模型文件的名称')
        self.parser.add_argument('--save_model_dir', default='./checkpoints/', help='保存模型文件的路径')
        self.parser.add_argument('--save_result_dir', default='./result/', help='保存预测结果文件夹')

        self.parser.add_argument('--log_save_name', default='log.log', help='保存日志文件的名称')
        self.parser.add_argument('--log_dir', default='./logs/', help='保存日志文件的路径')

        # 模型超参数
        self.parser.add_argument('--num_layers', default=1, type=int, help='lstm层数大小')
        self.parser.add_argument('--lstm_hidden', default=128, type=int, help='lstm隐藏层大小')
        self.parser.add_argument('--dropout', default=0.3, type=float, help='dropout rate')
        self.parser.add_argument('--max_seq_len', default=256, type=int)
        self.parser.add_argument('--dropout_prob', default=0.1, type=float, help='drop out probability')

        self.parser.add_argument('--lr', default=3e-5, type=float, help='bert学习率')
        self.parser.add_argument('--crf_lr', default=3e-2, type=float, help='条件随机场学习率')
        self.parser.add_argument('--other_lr', default=3e-4, type=float, help='bi-lstm和多层感知机学习率')
        self.parser.add_argument('--warmup_proportion', default=0.1, type=float)
        self.parser.add_argument('--weight_decay', default=0.01, type=float)
        self.parser.add_argument('--adam_epsilon', default=1e-8, type=float)
        self.parser.add_argument('--max_grad_norm', default=1, type=float, help='max grad clip')

        # 模型选择参数
        self.parser.add_argument('--use_crf', type=bool, default=True, help='是否使用CRF')
        self.parser.add_argument('--use_lstm', type=bool, default=False, help='是否使用LSTM')

        # 训练参数
        self.parser.add_argument('--do_train', default=True, type=bool, help='是否训练')
        self.parser.add_argument('--do_test', default=True, type=bool, help='是否测试')
        self.parser.add_argument('--epochs', default=10, type=int, help='epoch数量')
        self.parser.add_argument('--batch_size', default=32, type=int, help='epoch数量')
        self.parser.add_argument('--log_interval', default=1, type=int, help='每多少步记录一次训练日志')
        self.parser.add_argument('--valid_step', default=500, type=int, help='每多少步记录一次验证日志')
        self.parser.add_argument('--n_gpu', default=1, type=int, help='设定 GPU device 数量 ')

        con = vars(self.parser.parse_args())
        con['data_dir'] = os.path.join(con['data_dir'], con['data_name'], 'generate_data')
        con['save_model_dir'] = os.path.join(con['save_model_dir'],  con['data_name'])
        con['save_result_dir'] = os.path.join(con['save_result_dir'], con['data_name'])
        con['log_dir'] = os.path.join(con['log_dir'], con['data_name'])
        # con['resume'] = os.path.join(con['save_model_dir'], 'best.pt')
        con['resume'] = None
        con['labels'] = read_json(con['data_dir'], 'labels')
        con['id2tag'] = {int(v): k for k, v in read_json(con['data_dir'], 'nor_ent2id').items()}
        con['num_tags'] = len(con['id2tag'])
        return con


if __name__ == '__main__':
    arg = ArgConfig()
    config = arg.initialize()


