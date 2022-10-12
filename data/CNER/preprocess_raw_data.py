# coding=utf-8

import os
import json


def preprocess(input_path, save_path, mode):
    """
    数据格式化、标签、标签类型
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    texts = []
    entities = []
    #
    words = []
    entity_tmp = []
    entities_tmp = []

    labels = set()
    k = 0
    # find sentence、 entities in sentence、entity types
    with open(input_path, 'r', encoding="utf-8") as fp:
        for line in fp:
            line_list = line.strip().split(" ")
            if len(line_list) == 2:
                word = line_list[0]
                label = line_list[1]
                label_type = label.split('-')[-1]
                words.append(word)

                if 'B-' in label:
                    start = k
                    entity_tmp.append(word)
                elif 'M-' in label:
                    entity_tmp.append(word)
                elif 'E-' in label:
                    end = k+1
                    entity_tmp.append(word)
                    if ("".join(entity_tmp), label_type) not in entities_tmp:
                        entities_tmp.append(("".join(entity_tmp), label_type, start, end))
                    labels.add(label_type)
                    entity_tmp = []

                if "S-" in label:
                    entity_tmp.append(word)
                    if ("".join(entity_tmp), label_type) not in entities_tmp:
                        entities_tmp.append(("".join(entity_tmp), label_type, k, k+1))
                    entity_tmp = []
                    labels.add(label_type)

                k += 1

            else:
                entity_tmp = []
                texts.append("".join(words))
                entities_tmp = sorted(entities_tmp, key=lambda x: (x[-2], x[-1]))
                entities.append(entities_tmp)

                words = []
                entities_tmp = []
                k = 0
    i = 0
    result = []
    for text, text_entities in zip(texts, entities):
        tmp = {
            "id": i,
            "text": text,
            "labels": []
        }
        if text_entities:
            for j, entity in enumerate(text_entities):
                tmp['labels'].append(["T{}".format(str(j)), entity[1], entity[2], entity[3], entity[0]])
        result.append(tmp)
        i += 1

    # write file
    save_file = os.path.join(save_path, mode + '.json')
    with open(save_file, 'w', encoding="utf-8") as fw:
        fw.write(json.dumps(result, ensure_ascii=False, indent=4))

    if mode == 'train':
        label_path = os.path.join(save_path, "labels.json")
        with open(label_path, 'w', encoding="utf-8") as fw:
            fw.write(json.dumps(list(labels), ensure_ascii=False, indent=4))
        #
        tmp_labels = ["O"]
        for label in labels:
            tmp_labels.append("B-" + label)
            tmp_labels.append("I-" + label)
            tmp_labels.append("E-" + label)
            tmp_labels.append("S-" + label)
        label2id = {}
        for k, v in enumerate(tmp_labels):
            label2id[v] = k
        nor_label_path = os.path.join(save_path, 'nor_ent2id.json')
        with open(nor_label_path, 'w', encoding="utf-8") as fw:
            fw.write(json.dumps(label2id, ensure_ascii=False, indent=4))


if __name__ == '__main__':
    preprocess('./raw_data/train.char.bmes', './generate_data/', mode='train')
    preprocess('./raw_data/test.char.bmes', './generate_data/', mode='test')
    preprocess('./raw_data/dev.char.bmes', './generate_data/', mode='dev')
















