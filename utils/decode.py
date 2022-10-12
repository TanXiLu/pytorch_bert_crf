# coding=utf-8


def decode(tokens, text, id2tag):
    tokens = tokens[1: len(text)+1]
    predict_entities = {}
    index_ = 0
    while index_ < len(tokens):
        if tokens[index_] == 0:
            index_ += 1
            continue
        else:
            token_label = id2tag[tokens[index_]].split('-')

        if token_label[0].startswith("S"):
            token_type = token_label[1]
            tmp_ent = text[index_]

            if token_type not in predict_entities:
                predict_entities[token_type] = [(tmp_ent, index_)]
            else:
                predict_entities[token_type].append((tmp_ent, index_))

            index_ += 1
        elif token_label[0].startswith("B"):
            token_type = token_label[1]
            start_index = index_

            index_ += 1
            while index_ < len(tokens):
                if tokens[index_] == 0:
                    break
                else:
                    temp_token_label = id2tag[tokens[index_]].split('-')

                if temp_token_label[0].startswith("I") and token_type == temp_token_label[1]:
                    index_ += 1
                elif temp_token_label[0].startswith("E") and token_type == temp_token_label[1]:
                    end_index = index_
                    index_ += 1

                    tmp_ent = text[start_index: end_index + 1]
                    if token_type not in predict_entities:
                        predict_entities[token_type] = [(tmp_ent, start_index)]
                    else:
                        predict_entities[token_type].append((tmp_ent, start_index))

                    break
                else:
                    break
        else:
            index_ += 1
    return predict_entities

