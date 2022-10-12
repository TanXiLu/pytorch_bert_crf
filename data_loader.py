# coding=utf-8
import json
import os
import torch
from utils import read_json
from torch.utils.data import Dataset, DataLoader


class CustomDataSet(Dataset):
    def __init__(self, xpath):
        self.data = self.load_data(xpath)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]

    @staticmethod
    def load_data(xpath):
        examples = []
        with open(xpath, 'r', encoding='utf-8') as f:
            raw_examples = json.load(f)

        for example in raw_examples:
            if len(example["labels"]) != 0 and example["text"] != '':
                examples.append(example)
        return examples


class Collate(object):
    def __init__(self, tokenizer, max_len, tag2id, labels):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.tag2id = tag2id
        self.labels = labels

    def collate_fn(self, batch):

        batch_texts = []
        batch_entity_info = []

        batch_labels = []
        batch_token_ids = []
        batch_attention_mask = []
        batch_token_type_ids = []

        for i, example in enumerate(batch):
            text = example["text"]
            entities = example["labels"]
            tokens = list(text)
            label_ids = [0] * len(tokens)

            entity_info = {x: [] for x in self.labels}
            for entity in entities:
                entity_type = entity[1]
                start = entity[2]
                end = entity[3] - 1
                entity_name = entity[4]
                assert text[start: end+1] == entity_name, f'{text}, [{start}, {end}]'

                entity_info[entity_type].append((entity_name, start, end))

                if start == end:
                    label_ids[start] = self.tag2id['S-' + entity_type]
                else:
                    label_ids[start] = self.tag2id['B-' + entity_type]
                    label_ids[end] = self.tag2id['E-' + entity_type]
                    for k in range(start + 1, end):
                        label_ids[k] = self.tag2id["I-" + entity_type]

            if len(label_ids) > self.max_len - 2:
                label_ids = label_ids[: self.max_len - 2]
            # 给句子首[CLS]和尾[SEP]增加标签"O"的id
            label_ids = [self.tag2id["O"]] + label_ids + [self.tag2id["O"]]
            # pad
            if len(label_ids) < self.max_len:
                pad_length = self.max_len - len(label_ids)
                label_ids = label_ids + [self.tag2id["O"]] * pad_length

            assert len(label_ids) == self.max_len, f'{len(label_ids)}'

            output = self.tokenizer.encode_plus(
                text=tokens,
                max_length=self.max_len,
                padding="max_length",
                truncation="longest_first",
                return_token_type_ids=True,
                return_attention_mask=True
            )
            token_ids = output["input_ids"]
            token_type_ids = output["token_type_ids"]
            attention_mask = output["attention_mask"]

            batch_token_ids.append(token_ids)
            batch_attention_mask.append(attention_mask)
            batch_token_type_ids.append(token_type_ids)
            batch_labels.append(label_ids)
            batch_texts.append(text)
            batch_entity_info.append(entity_info)

        batch_token_ids = torch.tensor(batch_token_ids, dtype=torch.long)
        attention_mask = torch.tensor(batch_attention_mask, dtype=torch.uint8)
        token_type_ids = torch.tensor(batch_token_type_ids, dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)

        batch_data = {
            "token_ids": batch_token_ids,
            "attention_masks": attention_mask,
            "token_type_ids": token_type_ids,
            "labels": batch_labels,
            "texts": batch_texts,
            "entity_info": batch_entity_info
        }
        return batch_data


def get_loader(config, prefix, tokenizer):
    max_len = config['max_seq_len']
    tag2id = read_json(config['data_dir'], config['tag2id_name'])
    labels = read_json(config['data_dir'], config['label_name'])
    collate = Collate(tokenizer, max_len, tag2id, labels)
    filename = os.path.join(config['data_dir'], '{}.json'.format(prefix))
    dataset = CustomDataSet(filename)
    data_loader = DataLoader(dataset,
                             batch_size=config['batch_size'],
                             shuffle=True,
                             collate_fn=collate.collate_fn)
    return data_loader
