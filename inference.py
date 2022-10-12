# coding=utf-8
import torch
from config import ArgConfig
import pprint
from transformers import BertTokenizer
from utils import prepare_device, decode
from model import BertNerModel


def predict(model, tokenizer, text, device, config):
    model.eval()
    tokens = list(text)
    with torch.no_grad():
        inputs = tokenizer.encode_plus(
            text=tokens,
            max_length=config['max_seq_len'],
            padding="max_length",
            truncation="longest_first",
            return_token_type_ids=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        inputs['token_ids'] = inputs.pop('input_ids')
        inputs['token_type_ids'] = inputs.pop('token_type_ids')
        inputs['attention_masks'] = inputs.pop('attention_mask')
        inputs['labels'] = None
        inputs['attention_masks'] = torch.tensor(inputs['attention_masks'], dtype=torch.uint8)

        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.to(device)
        logits = model(inputs)
        if config["use_crf"] is True:
            output = logits
        else:
            output = torch.argmax(logits, dim=2).cpu().numpy().tolist()
        pre_entities = decode(output[0], text, config["id2tag"])
        pprint.pprint({"文本": text, "预测实体": pre_entities})
        return pre_entities


if __name__ == '__main__':
    arg = ArgConfig()
    conf = arg.initialize()
    pprint.pprint(conf)
    tokenizer = BertTokenizer.from_pretrained(conf['bert_dir'])
    # 模型
    device, device_ids = prepare_device(conf['n_gpu'])
    model = BertNerModel(conf)
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    checkpoint = torch.load(conf["resume"])
    model.load_state_dict(checkpoint['state_dict'])
    text = '1963年出生，工科学士，高级工程师，北京物资学院客座副教授。'
    predict(model, tokenizer, text, device, conf)
