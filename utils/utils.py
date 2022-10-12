# coding=utf-8
import json
import os
import torch
import random
import logging
import numpy as np
from transformers import AdamW, get_linear_schedule_with_warmup


def set_seed(seed=42):
    """
    设置随机种子，保证实验可重现
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_device(n_gpu_use):
    """
    setup GPU device if available, get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There's no GPU available on this machine, "
              "training will be performed on CPU.")
        n_gpu = 0
    if n_gpu_use > n_gpu:
        print("Warning: The number of GPU configured to use is {}."
              "but only {} are available on this machine".format(n_gpu_use, n_gpu))
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def save_json(data_dir, data, desc):
    """
    保存数据格式为json
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    with open(os.path.join(data_dir, '{}.json'.format(desc)), 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def read_json(data_dir, desc):
    """
    读取数据格式为json
    """
    with open(os.path.join(data_dir, "{}.json".format(desc)), 'r', encoding="utf-8") as f:
        data = json.load(f)
    return data


def set_logger(log_path):
    """
    配置日誌
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 由于每调用一次set_logger函数，就会创建一个handler
    # 会造成重复打印的问题，因此需要判断root logger中是否已有该handler
    if not any(handler.__class__ == logging.FileHandler for handler in logger.handlers):
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(handler.__class__ == logging.StreamHandler for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        logger.addHandler(stream_handler)
    return logger


def build_optimizer_and_scheduler(args, model, t_total):
    module = (
        model.module if hasattr(model, "module") else model
    )

    # 差分学习率
    no_decay = ["bias", "LayerNorm.weight"]
    model_param = list(module.named_parameters())

    bert_param_optimizer = []
    crf_param_optimizer = []
    other_param_optimizer = []

    for name, para in model_param:
        space = name.split('.')
        # print(name)
        if space[0] == 'bert_module':
            bert_param_optimizer.append((name, para))
        elif space[0] == 'crf':
            crf_param_optimizer.append((name, para))
        else:
            other_param_optimizer.append((name, para))

    optimizer_grouped_parameters = [
        # bert other module
        {"params": [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args['weight_decay'], 'lr': args['lr']},
        {"params": [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args['lr']},

        # crf模块
        {"params": [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args['weight_decay'], 'lr': args['crf_lr']},
        {"params": [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args['other_lr']},

        # 其他模块，差分学习率
        {"params": [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
         "weight_decay": args['weight_decay'], 'lr': args['other_lr']},
        {"params": [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0, 'lr': args['other_lr']},
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args['lr'], eps=args['adam_epsilon'], no_deprecation_warning=True)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(args['warmup_proportion'] * t_total), num_training_steps=t_total
    )

    return optimizer, scheduler
