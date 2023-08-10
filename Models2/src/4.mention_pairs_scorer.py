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


# config
config_dict = {
    "csv_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\data\3.pred",
    "output_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\output",
}

# 临时代码，用于清空输出路径
for file in os.listdir(config_dict["output_path"]):
    os.remove(os.path.join(config_dict["output_path"], file))

# logging
logging.basicConfig(
    # 使用fileHandler,日志文件在输出路径中(test_log.txt)
    filename=os.path.join(config_dict["output_path"], "log.txt"),
    filemode="w",
    # 配置日志级别
    level=logging.INFO
)
print(f"log saved in {os.path.join(config_dict['output_path'], 'log.txt')}")

# output dir
if not os.path.exists(config_dict["output_path"]):
    print(f"make output dir: {config_dict['output_path']}")
    os.makedirs(config_dict["output_path"])
elif len(os.listdir(config_dict["output_path"])) > 1:  # 大于1是因为上边配置logging的时候就建立的log.txt这个文件
    input("output dir is not empty, press ENTER to continue.")

# save this file itself
shutil.copy(os.path.abspath(__file__), config_dict["output_path"])


METRICS = {
    "acc": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "auc": roc_auc_score,
}


pd.options.display.float_format = "{:,.2f}".format


def get_prob(pred, count):
    if count == 0:
        # If the model didn't produce any yes or no, it means the model is totally uncertain about the result
        return np.nan
    else:
        return pred / count


def get_metrics_scores(
    groundtruth_column, pred_column,
    metrics=METRICS,
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
        cur_prompt_metrics[cur_metirc_name] = cur_metirc_scorer(groundtruth_column, pred_column)
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
    groups = re.match(r'(\[[\S ]*\]\(strategy[0-4]\))_([\S]*)\(([\S]*)\)_([0-9])shot_t([^_]*)_(\S*)\(r([0-9])\).csv', csv_file_name).groups()
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


def save_into_csv_in_list_format(experiments_scores):
    file_path = config_dict['output_path'] + "/performance_list.csv"
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
    print(f"结果输出到{file_path}")


def save_into_csv_in_table_format(experiments_scores):
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
        valid = f"{round(cur_experiment_score['valid'], 2):.2f}"
        acc = f"{round(cur_experiment_score['acc'], 2):.2f}"
        auc = f"{round(cur_experiment_score['auc'], 2):.2f}"
        p = f"{round(cur_experiment_score['precision'], 2):.2f}"
        r = f"{round(cur_experiment_score['recall'], 2):.2f}"
        f1 = f"{round(cur_experiment_score['f1'], 2):.2f}"
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
        result[data][template][setting] = f"v{valid}acc{acc}auc{auc}p{p}r{r}f1{f1}"
    #
    all_data = list(all_data)
    all_setting = list(all_setting)
    all_template = list(all_template)
    #
    file_path = config_dict['output_path'] + "/performance_table.csv"
    csvfile = open(file_path, mode="w", newline='', encoding='utf-8')
    header = ['template'] + all_setting
    writer = csv.DictWriter(csvfile, fieldnames=header)
    writer.writeheader()
    for cur_data in result.keys():
        # 不同的data新开一个表，用单独的一行作为分隔
        writer.writerow({'template': cur_data})
        for cur_template in result[data]:
            # 写一行数据
            row = {'template': cur_template}
            for cur_model_setting in all_setting:
                if cur_model_setting in result[data][cur_template]:
                    row[cur_model_setting] = result[data][cur_template][cur_model_setting]
                else:
                    row[cur_model_setting] = '-'
            writer.writerow(row)
    print(f"结果输出到{file_path}")


def get_score_from_csv(input_path):
    # 获取config_dict['csv_path']下的所有csv文件作为输入
    dir_or_file_list = os.listdir(input_path)
    csv_file_list = []
    for dir_or_file in dir_or_file_list:
        if len(dir_or_file) <= 4:
            continue
        elif dir_or_file[-4:] != ".csv":
            continue
        cur_path = input_path + "/" + dir_or_file
        if os.path.isfile(cur_path):
            csv_file_list.append(cur_path)
    # 遍历所有csv文件
    experiments_scores = []
    for cur_csv_path in csv_file_list:
        # 1. 抽取配置
        experiment_settings = get_experiment_settings(cur_csv_path)
        # 2. 读取csv
        cur_df = pd.read_csv(cur_csv_path)
        # 3.1. 计算合法结果占比
        cur_result = cur_df[experiment_settings["template"]]
        validated_num = cur_result.apply(
            lambda x: ast.literal_eval(x)[1]
        ).sum()
        repeat_num = cur_result.apply(
            lambda x: ast.literal_eval(x)[1]
        ).sum()
        Percentage_of_validated_result = validated_num / repeat_num
        # 3.2. 计算性能指标
        cur_prob = cur_result.apply(
            lambda x: int(get_prob(ast.literal_eval(x)[0], ast.literal_eval(x)[1]) >= 0.5)
        )
        performance_scores = get_metrics_scores(
            cur_df["label"], cur_prob
        )
        # 4. 记录
        cur_prompt_info = experiment_settings
        cur_prompt_info["valid"] = Percentage_of_validated_result
        cur_prompt_info.update(performance_scores)
        #
        experiments_scores.append(cur_prompt_info)
    return experiments_scores


def main():
    #
    experiments_scores = get_score_from_csv(config_dict['csv_path'])
    # save into csv
    save_into_csv_in_list_format(experiments_scores)
    save_into_csv_in_table_format(experiments_scores)


if __name__ == '__main__':
    main()
