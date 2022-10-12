# coding=utf-8
import numpy as np
from utils import decode


def calculate_metric(gold, predict):
    """
    计算 tp fp fn
    """
    tp, fp, fn = 0, 0, 0
    for entity_predict in predict:
        flag = 0
        for entity_gt in gold:
            if entity_predict[0] == entity_gt[0] and entity_predict[1] == entity_gt[1]:
                flag = 1
                tp += 1
                break
        if flag == 0:
            fp += 1
    fn = len(gold) - tp
    return np.array([tp, fp, fn])


def metric(decode_tokens, texts, callback_entities, id2tag, labels, report_results=False):
    total_count = [0 for _ in range(len(labels))]
    role_metric = np.zeros([len(labels), 3])
    predict_entities_list = []
    for tokens, text, callback_entity in zip(decode_tokens, texts, callback_entities):
        predict_entities = decode(tokens, text, id2tag)
        predict_entities_list.append(predict_entities)
        tmp_metric = np.zeros([len(labels), 3])
        for i, _type in enumerate(labels):
            if _type not in predict_entities:
                predict_entities[_type] = []
            total_count[i] += len(callback_entity[_type])
            tmp_metric[i] += calculate_metric(callback_entity[_type], predict_entities[_type])

        role_metric += tmp_metric
    mirco_metrics = np.sum(role_metric, axis=0)
    p, r, f1 = get_p_r_f(mirco_metrics[0],  mirco_metrics[1], mirco_metrics[2])
    if report_results:
        report = classification_report(role_metric, labels, total_count)
        return report, p, r, f1, predict_entities_list
    return p, r, f1


def classification_report(metrics_matrix, label_list, total_count, digits=2):
    name_width = max([len(label) for label in label_list])
    last_line_heading = 'micro-f1'
    width = max(name_width, len(last_line_heading), digits)

    headers = ["precision", "recall", "f1-score", "support"]
    head_fmt = u'{:>{width}s} ' + u' {:>9}' * len(headers)
    report = head_fmt.format(u'', *headers, width=width)
    report += u'\n\n'

    row_fmt = u'{:>{width}s} ' + u' {:>9.{digits}f}' * 3 + u' {:>9}\n'

    ps, rs, f1s, s = [], [], [], []
    idx = 0
    for label, label_matrix in zip(label_list, metrics_matrix):
        type_name = label
        p, r, f1 = get_p_r_f(label_matrix[0], label_matrix[1], label_matrix[2])
        nb_true = total_count[idx]
        report += row_fmt.format(*[type_name, p, r, f1, nb_true], width=width, digits=digits)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
        s.append(nb_true)
        idx += 1

    report += u'\n'
    mirco_metrics = np.sum(metrics_matrix, axis=0)
    mirco_metrics = get_p_r_f(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2])
    # compute averages
    print('precision:{:.4f} recall:{:.4f} micro_f1:{:.4f}'.format(mirco_metrics[0], mirco_metrics[1], mirco_metrics[2]))
    report += row_fmt.format(last_line_heading,
                             mirco_metrics[0],
                             mirco_metrics[1],
                             mirco_metrics[2],
                             np.sum(s),
                             width=width, digits=digits)

    return report


def get_p_r_f(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp != 0 else 0
    r = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * p * r / (p + r) if p + r != 0 else 0
    return np.array([p, r, f1])
