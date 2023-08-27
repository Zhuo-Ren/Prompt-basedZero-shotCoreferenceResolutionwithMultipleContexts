# 标准库
import re
import os
import _pickle as cPickle
# from typing import Dict, List, Tuple, Union
import logging
import ast
import shutil
import pandas as pd
import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
)
import csv
# 本地库
pass


METRICS = {
    "acc": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "auc": roc_auc_score,
}


def get_cmp_or_csv_files(input_dir, with_cmp):
    """
    输入指定文件夹。定位文件夹中的csv文件（不迭代遍历）。
    如果with_cmp为True，则还会检测是否每个csv文件都对应了c_mp文件。如果有任一一个csv但没有对应的c_mp文件，则raise RuntimeError。

    example::

            >>> input_path = 'E:\\ProgramCode\\WhatGPTKnowsAboutWhoIsWho\\WhatGPTKnowsAboutWhoIsWho-main\\Models2\\data\\3.pred'
            >>> get_cmp_or_csv_files(input_path, with_cmp=False)
            [
                "E:\\ProgramCode\\WhatGPTKnowsAboutWhoIsWho\\WhatGPTKnowsAboutWhoIsWho-main\\Models2\\data\\3.pred/['36_ecb'](strategy3)_ChatGPT3.5(b1t0)_0shot_t13SAU_noSample(r1)",
                "E:\\ProgramCode\\WhatGPTKnowsAboutWhoIsWho\\WhatGPTKnowsAboutWhoIsWho-main\\Models2\\data\\3.pred/['36_ecb'](strategy3)_ChatGPT3.5(b1t0)_0shot_t16DAM_noSample(r1)",
                "E:\\ProgramCode\\WhatGPTKnowsAboutWhoIsWho\\WhatGPTKnowsAboutWhoIsWho-main\\Models2\\data\\3.pred/['36_ecb'](strategy3)_ground_truth_model(none)_0shot_t16DAM_noSample(r1)"
            ]

    :param input_dir:
    :param with_cmp: 如果with_cmp为True，则还会检测是否每个csv文件都对应了c_mp文件。
    :return: list of str. 指定路径下的CSV文件列表（不含.csv这个后缀）。
    """
    # 获取路径下的所有csv文件的名字
    dir_or_file_list = os.listdir(input_dir)
    csv_file_list = []
    for dir_or_file in dir_or_file_list:
        if len(dir_or_file) <= 4:
            continue
        elif dir_or_file[-4:] != ".csv":
            continue
        cur_path = input_dir + "/" + dir_or_file
        if os.path.isfile(cur_path):
            csv_file_list.append(cur_path)
    del dir_or_file_list, dir_or_file, cur_path
    # csv文件和c_mp文件是配对的，检查是否每个csv都有配对的c_mp
    experiment_list = []
    for csv_file_path in csv_file_list:
        file_name = csv_file_path[:-4]
        if with_cmp:
            cmp_file_path = f"{file_name}.c_mp"
            if not os.path.exists(cmp_file_path):
                raise RuntimeError(f"没有找到配对的c_mp文件：{cmp_file_path}")
            else:
                experiment_list.append(file_name)
            del cmp_file_path
        else:
            experiment_list.append(file_name)
        del csv_file_path
    #
    return experiment_list


def get_prob(pred, count):
    if count == 0:
        # If the model didn't produce any yes or no, it means the model is totally uncertain about the result
        return np.nan
    else:
        return pred / count


def get_metrics_scores(
    groundtruth_column, pred_column,
    metrics,
    threshold=0.5,
):
    """
    根据实验结果，计算性能指标。

    return = {
        "prompt 1": {
            "acc": 0.723,
            "precision": 0.531,
            "recall": 0.613,
            "f1": 0.551,
            "auc": 0.333
        },
        "prompt 2": {},
        "prompt 3": {},
        "prompt 4": {},
        "prompt 5": {},
        "prompt mean": {},
        "prompt majority": {}
    }

    :param df: df is a pandas DataFrame which has columns includeing "prompt 1", "prompt 2", "prompt 3", "prompt 4", "prompt 5", "prompt majority", "prompt mean"
    """
    cur_prompt_metrics = {}
    for cur_metirc_name, cur_metirc_scorer in metrics.items():
        if len(groundtruth_column) != 0:
            cur_prompt_metrics[cur_metirc_name] = cur_metirc_scorer(groundtruth_column, pred_column)
        else:
            cur_prompt_metrics[cur_metirc_name] = -1
    return cur_prompt_metrics


def get_experiment_settings(csv_file_path):
    """
    给定文件名，抽取配置信息

    example::

        r = get_experiment_settings(csv_file_path: 'E:\\ProgramCode\\WhatGPTKnowsAboutWhoIsWho\\WhatGPTKnowsAboutWhoIsWho-main\\Models2\\data\\3.pred/[\'36_ecb\', \'36_ecbplus\'](strategy3)_ChatGPT3.5_t14DAM_s0_b1_noSample.csv')
        # {'data': "['36_ecb', '36_ecbplus'](strategy3)", 'model': 'ChatGPT3.5', 'temperature': '14DAM', 'prefix_num': '0', 'beam_size': '1', 'sample': 'noSample'}

    :param csv_file_path: 文件路径
    :return: 抽取的配置信息
    """
    csv_file_name = os.path.basename(csv_file_path)
    groups = re.match(r'(\[[\S ]*\]\(strategy[0-4]\))_([\S]*)\(([\S]*)\)_([0-9])shot_t([^_]*)_(\S*)\(r([0-9])\).[\S]*', csv_file_name).groups()
    #
    r = {}
    r["data"] = groups[0]
    r["model_name"] = groups[1]
    r["model_config"] = groups[2]
    r["prefix_num"] = groups[3]
    r["template"] = groups[4]
    r["sample"] = groups[5]
    r["repeat"] = groups[6]
    #
    return r


def save_mention_pair_scores_into_csv_in_list_format(experiments_scores, output_path, suffix="scores_mp_list.csv"):
    file_path = os.path.join(output_path, suffix)
    csvfile = open(file_path, mode="w", newline='', encoding='utf-8')
    header = [
        'data',
        'model_name', 'model_config',
        'prefix_num', 'template',
        'sample', 'repeat',
        'valid', 'acc', 'auc', 'p', 'r', 'F1'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=header)
    writer.writeheader()
    for cur_experiment_score in experiments_scores:
        writer.writerow({
            'data': cur_experiment_score['data'],
            'model_name': cur_experiment_score['model_name'],
            'model_config': cur_experiment_score['model_config'],
            'prefix_num': cur_experiment_score['prefix_num'],
            'template': cur_experiment_score['template'],
            'sample': cur_experiment_score['sample'],
            'repeat': cur_experiment_score['repeat'],
            'valid': cur_experiment_score['valid'],
            'acc': cur_experiment_score['acc'],
            'auc': cur_experiment_score['auc'],
            'p': cur_experiment_score['precision'],
            'r': cur_experiment_score['recall'],
            'F1': cur_experiment_score['f1']
        })
    print(f"OUTPUT: {suffix}输出到{file_path}")


def save_mention_pair_scores_into_csv_in_table_format(experiments_scores, output_path, suffix="scores_mp_table.csv"):
    result = {}
    #
    all_data = set()
    all_template = set()
    all_setting = set()
    for cur_experiment_score in experiments_scores:
        data = cur_experiment_score['data']
        model_name = cur_experiment_score['model_name']
        model_config = cur_experiment_score['model_config']
        template = cur_experiment_score['template']
        prefix_num = cur_experiment_score['prefix_num']
        sample = cur_experiment_score['sample']
        repeat = cur_experiment_score['repeat']
        setting = f"{model_name}({model_config})_{prefix_num}shot_{sample}(r{repeat})"
        valid = f"{round(cur_experiment_score['valid']*100, 1):.1f}"
        acc = f"{round(cur_experiment_score['acc']*100, 1):.1f}"
        auc = f"{round(cur_experiment_score['auc']*100, 1):.1f}"
        p = f"{round(cur_experiment_score['precision']*100, 1):.1f}"
        r = f"{round(cur_experiment_score['recall']*100, 1):.1f}"
        f1 = f"{round(cur_experiment_score['f1']*100, 1):.1f}"
        #
        all_data.add(data)
        all_setting.add(setting)
        all_template.add(template)
        #
        if data not in result:
            result[data] = {}
        if template not in result[data]:
            result[data][template] = {}
        if setting not in result[data][template]:
            result[data][template][setting] = ""
        #
        result[data][template][setting] = f"v:{valid}%;acc:{acc}%;auc:{auc}%;p:{p}%;r:{r}%;f1:{f1}%"
    #
    all_data = list(all_data)
    all_setting = list(all_setting)
    all_template = list(all_template)
    #
    file_path = os.path.join(output_path, suffix)
    csvfile = open(file_path, mode="w", newline='', encoding='utf-8')
    header = ['template'] + all_setting
    writer = csv.DictWriter(csvfile, fieldnames=header)
    writer.writeheader()
    for cur_data in result.keys():
        # 不同的data新开一个表，用单独的一行作为分隔
        writer.writerow({'template': cur_data})
        for cur_template in result[cur_data]:
            # 写一行数据
            row = {'template': cur_template}
            for cur_model_setting in all_setting:
                if cur_model_setting in result[cur_data][cur_template]:
                    row[cur_model_setting] = result[cur_data][cur_template][cur_model_setting]
                else:
                    row[cur_model_setting] = '-'
            writer.writerow(row)
    print(f"OUTPUT: {suffix}输出到{file_path}")


def mp_scorer_csv(csv_path, template_name):
    # 2.1. 读取csv
    df = pd.read_csv(csv_path)
    df = df[df["wd/cd"] == "wd"]
    #
    s = mp_scorer_df(df, template_name)
    return s


def mp_scorer_df(df, template_name):
    # 2.2. 准备工作
    df["true_num"] = df[template_name].apply(
        lambda x: ast.literal_eval(x)[0]
    )
    df["valid_num"] = df[template_name].apply(
        lambda x: ast.literal_eval(x)[1]
    )
    df["repeat_num"] = df[template_name].apply(
        lambda x: ast.literal_eval(x)[1]
    )
    def get_result(x):
        label = x["label"]
        true_num = x["true_num"]
        validated_num = x["valid_num"]
        if validated_num == 0:
            return np.nan
        else:
            true_prob = true_num / validated_num
            if true_prob >= 0.5:  # 做对了
                return label
            else:  # 做错了
                return 1 - label  # label是1就返回0，label是0就返回1
    df["result"] = df[["label", "true_num", "valid_num"]].apply(
        get_result,
        axis=1
    )
    # 3.1. 计算合法结果占比
    valid_num_all = df["valid_num"].sum()
    repeat_num_all = df["repeat_num"].sum()
    percentage_of_validated_result = (valid_num_all / repeat_num_all) if len(df) != 0 else -1
    # 3.2. 计算性能指标
    performance_scores = get_metrics_scores(
        df["label"], df["result"], metrics=METRICS
    )
    # 4. 记录
    scores = {}
    scores["valid"] = percentage_of_validated_result
    scores.update(performance_scores)
    #
    return scores


def main():
    # config
    config_dict = {
        "csv_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\data\3.pred",
        "output_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\output",
    }
    # 临时代码，用于清空输出路径
    shutil.rmtree(config_dict["output_path"])
    # output dir
    if not os.path.exists(config_dict["output_path"]):
        print(f"make output dir: {config_dict['output_path']}")
        os.makedirs(config_dict["output_path"])
    elif len(os.listdir(config_dict["output_path"])) > 0:
        input("output dir is not empty, press ENTER to continue.")
    # logging
    logging.basicConfig(
        handlers=[logging.FileHandler(
            filename=os.path.join(config_dict["output_path"], "log.txt"),
            encoding='utf-8',
            mode='w'
        )],
        # 使用fileHandler,日志文件在输出路径中(test_log.txt)
        # 配置日志级别
        level=logging.INFO
    )
    print(f"log saved in {os.path.join(config_dict['output_path'], 'log.txt')}")
    logging.info(f"log saved in {os.path.join(config_dict['output_path'], 'log.txt')}")
    # save this file itself
    shutil.copy(os.path.abspath(__file__), config_dict["output_path"])
    #
    pd.options.display.float_format = "{:,.2f}".format

    #
    experiment_path_list = get_cmp_or_csv_files(config_dict["csv_path"], with_cmp=False)

    # 1. mention pair部分
    experiments_scores = []
    for cur_experiment_path in experiment_path_list:
        cur_csv_path = f"{cur_experiment_path}.csv"
        # 打分
        mention_pair_scores = mp_socrer(csv_path=cur_csv_path)
        # 保存
        shutil.copy(cur_csv_path, config_dict["output_path"])
        # 分数整合
        experiments_scores.append(mention_pair_scores)
    save_mention_pair_scores_into_csv_in_list_format(experiments_scores, output_path=config_dict["output_path"])
    save_mention_pair_scores_into_csv_in_table_format(experiments_scores, output_path=config_dict["output_path"])


if __name__ == '__main__':
    main()
