"""
!pip install transformers
Version 1：这是Evaluation.ipynb的.py版本
Version 2：在上个版本的基础上删除了下半截的评估代码，只保留了输出F1性能表格的部分。
Version 3：Version 2 是给定配置，代码对应的找实验结果。这个版本改为到指定文件夹去读取其下所有实验结果。优化了评估方式。
"""
import warnings
import ast
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
import torch
import os
import csv
import re

warnings.filterwarnings("ignore")

# ####################################################################
# Utils for Charts ###################################################
# ####################################################################

# 配置路径
local_path = "/root/WhatGPTKnowsAboutWhoIsWho-main"
local_path = "E:/ProgramCode/WhatGPTKnowsAboutWhoIsWho/WhatGPTKnowsAboutWhoIsWho-main"
root_path = local_path
input_path = f"{root_path}/Results"  # 读取之前各模型的实验结果
results_path = f"{root_path}/Results/Charts_and_plots/final_plots"  # 生成图表。所以是放在charts_and_plots文件夹下。


# 配置模型类型
MODEL_LABELS = {  # 总共可选的model: labels
    # "multi-sieves": "Multi-pass Sieve",
    # "e2e": "E2E",
    # "Streamlining": "Streamlining",
    "GPT2_gold_mentions": "GPT2",
    # "GPT_NEO-125M_gold_mentions": "GPT-NEO",
    # "GPT2_WSC": "WSC",
}
models = [  # 总共可选的model
    # "multi-sieves",
    # "e2e",
    # "Streamlining",
    "GPT2_gold_mentions",
    # "GPT_NEO-125M_gold_mentions",
    # "GPT2_WSC",
]
labels = [  # 总共可选的labels
    # "Multi-pass Sieve",
    # "E2E",
    # "Streamlining",
    "GPT2",
    # "GPT-NEO",
    # "GPT2_WSC",
]


# ####################################################################
# Utils for Charts ###################################################
# ####################################################################
pd.options.display.float_format = "{:,.2f}".format


def write_table(df, output_file, output_dir):
    path = os.path.join(output_dir, output_file)
    df.to_latex(path, float_format="%.2f")


# ####################################################################
# Utils for Data Processing ##########################################
# ####################################################################

def get_prob(pred, count):
    if count == 0:
        # If the model didn't produce any yes or no, it means the model is totally uncertain about the result
        return np.nan
    else:
        return pred / count


def process_prediction_results(
    path, threshold=0.5, reverse=False
):
    """
    从csv文件中读取信息，计算最终结果。

    - 对每个样例（pred， count）。例如["Prompt 1"]列某个样例预测结果是(3,5)。
        - 基于每个prompt计算概率
            - ["prompt_prob 1"]列中概率为0.60（如果count为0则记为np.nan）
            - ["prompt 1"]列中是1（概率大于等于0.5是1，否则是0）
        - 基于所有prompt计算众数：["prompt 1"]列到["prompt 5"]列中哪个值出现最多就记为哪个
        - 基于所有prompt计算均值：["prompt 1"]列到["prompt 5"]列中五个值的均值。
    - 删除了["text"]列和["mention pair"]列
    """
    df = pd.read_csv(path)
    df = df[df.label != "label"]  # get rid of header rows
    df.label = df.label.apply(lambda x: int(float(x)))

    # 添加并计算["prompt_prob n"]列
    for i in range(1, 6):
        df[f"prompt_prob {i}"] = df[f"Prompt {i}"].apply(
            lambda x: get_prob(ast.literal_eval(x)[0], ast.literal_eval(x)[1])
        )
        if reverse:
            df["prompt_prob {}".format(i)] = 1 - df["prompt_prob {}".format(i)]
    # 添加并计算["prompt n"]列
    for i in range(1, 6):
        df[f"prompt {i}"] = df[f"prompt_prob {i}"].apply(
            lambda x: int(x >= threshold)
        )
    # 添加并计算众数列["prompt majority"]列
    majority_columns = [f"prompt {i}" for i in range(1, 6)]
    df["prompt majority"] = df[majority_columns].mode(axis=1)[0]
    # 添加并计算均值列["prompt mean"]列
    avg_columns = [f"prompt_prob {i}" for i in range(1, 6)]
    df["prompt mean prob"] = df[avg_columns].mean(axis=1)
    df["prompt mean"] = df["prompt mean prob"].apply(
        lambda x: int(x >= threshold)
    )
    # 删除了["text"]列和["mention pair"]列
    df = df.drop(columns=["text", "mention pair"])
    return df


# ####################################################################
# # Define Metrics ###################################################
# ####################################################################

METRICS = {
    "acc": accuracy_score,
    "precision": precision_score,
    "recall": recall_score,
    "f1": f1_score,
    "auc": roc_auc_score,
}


def get_metrics_scores(
    df,
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
    groundtruth = df["label"].values
    metrics_of_all_prompt_types = {}
    prompt_types = ["prompt 1", "prompt 2", "prompt 3", "prompt 4", "prompt 5", "prompt majority", "prompt mean"]
    for cur_prompt_type in prompt_types:
        cur_prompt = df[cur_prompt_type].values
        cur_prompt_metrics = {}
        for cur_metirc_name, cur_metirc_scorer in metrics.items():
            cur_prompt_metrics[cur_metirc_name] = cur_metirc_scorer(groundtruth, cur_prompt)
        metrics_of_all_prompt_types[cur_prompt_type] = cur_prompt_metrics
    return metrics_of_all_prompt_types


# ####################################################################
# General Evaluation #################################################
# ####################################################################
def get_experiment_settings(csv_file_path):
    csv_file_name = os.path.basename(csv_file_path)
    groups = re.match(r'([\S]*)_gold_mentions_([\S]*)_shot([\d]{1,2})([\S]*)_beam([\d])_([\S]{2})Sample\(t([\d\.])+r([\d])\).csv', csv_file_name).groups()
    #
    r = {}
    r["model"] = groups[0]
    r["data"] = groups[1]
    r["prefix_num"] = groups[2]
    r["prefix_source"] = groups[3][1:-1] if groups[3] != "" else ""
    r["beam_size"] = groups[4]
    r["sample"] = groups[5]
    r["temperature"] = groups[6]
    r["repeat"] = groups[7]
    #
    return r


def get_score_for_models(input_path):
    # 获取input_path下的所有csv文件作为输入
    dir_or_file_list = os.listdir(input_path)
    csv_file_list = []
    for dir_or_file in dir_or_file_list:
        if len(dir_or_file) <= 4:
            continue
        if dir_or_file[-4:] != ".csv":
            continue
        cur_path = input_path + "/" + dir_or_file
        if os.path.isfile(cur_path):
            csv_file_list.append(cur_path)
    # 遍历所有csv文件
    prompt_info_list = []
    for cur_csv_path in csv_file_list:
        #
        experiment_settings = get_experiment_settings(cur_csv_path)
        #
        experiment_results = process_prediction_results(cur_csv_path)
        experiment_scores = get_metrics_scores(experiment_results)
        #
        for cur_prompt_name, cur_prompt_scores in experiment_scores.items():
            cur_prompt_info = {"prompt_type": cur_prompt_name}
            cur_prompt_info.update(cur_prompt_scores)
            cur_prompt_info.update(experiment_settings)
            prompt_info_list.append(cur_prompt_info)
    return prompt_info_list


prompt_info_list = get_score_for_models(input_path=input_path)
# =============================================================================
file_path = results_path + "/performance_all.csv"
csvfile = open(file_path, mode="w", newline='', encoding='utf-8')
header = [
    'data',
    'model',
    'prefix_num', 'prefix_source',
    'beam_size',
    'sample', 'temperature', 'repeat',
    'prompt_type',
    'valid',
    'acc', 'auc', 'p', 'r', 'F1'
]
writer = csv.DictWriter(csvfile, fieldnames=header)
writer.writeheader()
for cur_prompt_info in prompt_info_list:
    writer.writerow({
        'data': cur_prompt_info['data'],
        'model': cur_prompt_info['model'],
        'prefix_num': cur_prompt_info['prefix_num'],
        'prefix_source': cur_prompt_info['prefix_source'],
        'beam_size': cur_prompt_info['beam_size'],
        'sample': cur_prompt_info['sample'],
        'temperature': cur_prompt_info['temperature'],
        'repeat': cur_prompt_info['repeat'],
        'prompt_type': cur_prompt_info['prompt_type'],
        'valid': '100%',
        'acc': cur_prompt_info['acc'],
        'auc': cur_prompt_info['auc'],
        'p': cur_prompt_info['precision'],
        'r': cur_prompt_info['recall'],
        'F1': cur_prompt_info['f1']
    })
print(f"结果输出到{file_path}")
# =============================================================================
result = {}
for cur_prompt_info in prompt_info_list:
    model = cur_prompt_info['model']
    prompt_type = cur_prompt_info['prompt_type']
    prefix_num = cur_prompt_info['prefix_num']
    sample = cur_prompt_info['sample']
    f1 = f"{round(cur_prompt_info['f1'], 4):.4f}"
    if model not in result:
        result[model] = {}
    if prompt_type not in result[model]:
        result[model][prompt_type] = {}
    if prefix_num not in result[model][prompt_type]:
        result[model][prompt_type][prefix_num] = ""
    cur_value = result[model][prompt_type][prefix_num]
    if sample == "do":
        cur_value = f1 + cur_value
    elif sample == "no":
        cur_value = cur_value + f" [{f1}]"
    else:
        raise RuntimeError("invalid value of sample")
    result[model][prompt_type][prefix_num] = cur_value
file_path = results_path + "/performance_model.csv"
csvfile = open(file_path, mode="w", newline='', encoding='utf-8')
header = ['prompt_type', '0-shot', '2-shot', '4-shot', '10-shot']
writer = csv.DictWriter(csvfile, fieldnames=header)
writer.writeheader()
for model in result.keys():
    writer.writerow({'prompt_type': model})
    for prompt_type in result[model]:
        row = {
            'prompt_type': prompt_type,
            '0-shot': '-' if '0' not in result[model][prompt_type] else result[model][prompt_type]['0'],
            '2-shot': '-' if '2' not in result[model][prompt_type] else result[model][prompt_type]['2'],
            '4-shot': '-' if '4' not in result[model][prompt_type] else result[model][prompt_type]['4'],
            '10-shot': '-' if '10' not in result[model][prompt_type] else result[model][prompt_type]['10']
        }
        writer.writerow(row)
print(f"结果输出到{file_path}")
