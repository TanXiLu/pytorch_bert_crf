# coding=utf-8
import torch
from config import ArgConfig
import pprint
from transformers import BertTokenizer
from utils import prepare_device, set_seed
from data_loader import get_loader
from trainer import Trainer
from model import BertNerModel


def run(config):
    set_seed(seed=42)
    tokenizer = BertTokenizer.from_pretrained(config['bert_dir'])
    train_loader = get_loader(config, prefix='train', tokenizer=tokenizer)
    dev_loader = get_loader(config, prefix='dev', tokenizer=tokenizer)
    test_loader = get_loader(config, prefix='test', tokenizer=tokenizer)
    # 模型
    device, device_ids = prepare_device(config['n_gpu'])
    model = BertNerModel(config)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    trainer = Trainer(model, device, config, train_loader, dev_loader, test_loader)
    # 训练
    if config["do_train"]:
        trainer.train()
    # 测试
    if config["do_test"]:
        if not config["do_train"]:
            assert config["resume"] is not None, 'make sure resume is not None'
        report = trainer.evaluate()
        print(report)


if __name__ == '__main__':
    arg = ArgConfig()
    conf = arg.initialize()
    pprint.pprint(conf)
    run(conf)
