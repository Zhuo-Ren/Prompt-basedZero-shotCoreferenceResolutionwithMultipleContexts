"""
!pip install transformers
Version 1：这是Evaluation.ipynb的.py版本
Version 2：在上个版本的基础上删除了下半截的评估代码，只保留了输出F1性能表格的部分。
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

warnings.filterwarnings("ignore")

# ####################################################################
# Utils for Charts ###################################################
# ####################################################################

# 配置路径
local_path = "/root/WhatGPTKnowsAboutWhoIsWho-main"
local_path = "D:/WhatGPTKnowsAboutWhoIsWho-main"
root_path = local_path
input_path = f"{root_path}/Results"  # 读取之前各模型的实验结果
results_path = f"{root_path}/Results/Charts_and_plots/final_plots"  # 生成图表。所以是放在charts_and_plots文件夹下。

# 配置GPU
"We need to use gpu to compute mention similarity by BERT"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device:{device}")

# 配置数据
data_config = "dev"

# 配置模型类型
MODEL_LABELS = {  # 总共可选的model: labels
    # "multi-sieves": "Multi-pass Sieve",
    # "e2e": "E2E",
    # "Streamlining": "Streamlining",
    "GPT2_gold_mentions": "GPT2"
    # "GPT_NEO-125M_gold_mentions": "GPT-NEO"
    # "GPT2_WSC": "WSC"
}
models = [  # 总共可选的model
    # "multi-sieves",
    # "e2e",
    # "Streamlining",
    "GPT2_gold_mentions"
    # "GPT_NEO-125M_gold_mentions"
    # "GPT2_WSC"
]
labels = [  # 总共可选的labels
    # "Multi-pass Sieve",
    # "E2E",
    # "Streamlining",
    "GPT2",
    # "GPT-NEO"
    # "GPT2_WSC"
]

# 配置模型参数
n_shot = 2  # 配置统计数据时，针对GPT-based模型，统计几shot的结果（之前分别按0/2/4/10shot做的实验）
do_sample = False
num_beams = 1
temperature = 1

# 实验参数
repeat_n = 5
prefix_source = "ecb"


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
    model_name, threshold=0.5, reverse=False
):
    dir_name = MODEL_LABELS[model_name]
    if model_name == "GPT2_WSC":
        path = os.path.join(
            input_path, f"{dir_name}/{model_name}_{n_shot}-shots_5-repeats.csv"
        )
    else:
        if do_sample:
            path = input_path + "/" + f"{dir_name}/{model_name}_{data_config}_{n_shot}shots_{prefix_source}Prefix_{num_beams}beams_doSimple_{temperature}temperature_{repeat_n}repeats.csv"
        else:
            path = input_path + "/" + f"{dir_name}/{model_name}_{data_config}_{n_shot}shots_{prefix_source}Prefix_{num_beams}beams_noSimple.csv"

    df = pd.read_csv(path)
    df = df[df.label != "label"]  # get rid of header rows
    df.label = df.label.apply(lambda x: int(float(x)))

    # Adds in the predicted probability for each of the prompts, so how many yes (left number) from total (right number)
    for i in range(1, 6):
        df["pred_prob {}".format(i)] = df["Prompt {}".format(i)].apply(
            lambda x: get_prob(ast.literal_eval(x)[0], ast.literal_eval(x)[1])
        )
        if reverse:
            df["pred_prob {}".format(i)] = 1 - df["pred_prob {}".format(i)]

    for i in range(1, 6):
        # Adds in boolean yes or no
        df["pred {}".format(i)] = df["pred_prob {}".format(i)].apply(
            lambda x: int(x >= threshold)
        )

    majority_columns = [f"pred {i}" for i in range(1, 6)]
    df["pred majority"] = df[majority_columns].mode(axis=1)[0]

    avg_columns = [f"pred_prob {i}" for i in range(1, 6)]
    df["pred mean"] = df[avg_columns].mean(axis=1)

    df = df.drop(columns=["text", "mention pair"])
    return df


def process_prediction_results_nonGPT(model_name, threshold=0.5):
    if model_name == "multi-sieves":
        path = os.path.join(input_path, "Multi-pass-Sieve/pairwise_result.csv")
        df = pd.read_csv(path)
        df.columns = [
            "pair",
            "label",
            "pred",
            "mention pair",
            "sent idx",
            "sentence",
            "sent filter",
            "doc name",
        ]
    elif model_name == "e2e":
        path = os.path.join(input_path, "e2e-coref/pairwise_result.csv")
        df = pd.read_csv(
            path,
            header=None,
        )
        df.columns = [
            "pair",
            "label",
            "mention pair",
            "sent idx",
            "sentence",
            "sent filter",
            "pred",
            "doc name",
        ]

    elif model_name == "Streamlining":
        path = os.path.join(input_path, "Streamlining/pairs_with_mentions.csv")
        df = pd.read_csv(path)

    else:
        raise Exception("Invalid model name. Options are 'multi-sieves' or 'e2e'.")

    df = df[df.label != "label"]  # get rid of header rows
    df.label = df.label.apply(lambda x: int(float(x)))

    if model_name == "Streamlining":
        df["pred mean"] = 1 * (df["pred_proba"] >= threshold)
        df["pred majority"] = 1 * (df["pred_proba"] >= threshold)  # df['pred_proba']
    else:
        # to make it easier for us to use later scoring functions
        df.pred = df.pred.apply(lambda x: int(float(x)))
        df["pred mean"] = df.pred
        df["pred majority"] = df.pred

    # df = df.drop(columns = ['mention pair',"sent idx","sent filter"])
    return df[["label", "pred mean", "pred majority"]]


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
    aggregate_type="majority",
    threshold=0.5,
):
    true = df["label"].values
    if aggregate_type == "majority":
        pred = df["pred majority"].values
    elif aggregate_type == "mean":
        pred = (df["pred mean"] >= threshold).astype(int).values
    else:
        raise Exception("Invalid aggregate type.")
    scores = {}
    for name, scorer in metrics.items():
        scores[name] = scorer(true, pred)

    return scores


# ####################################################################
# General Evaluation #################################################
# ####################################################################

def get_score_for_models(model_name, aggregate_type="mean"):
    if model_name in ["multi-sieves", "e2e", "Streamlining"]:
        df = process_prediction_results_nonGPT(model_name)
    elif model_name in ["GPT2_gold_mentions", "GPT_NEO-125M_gold_mentions", "GPT2_WSC"]:
        df = process_prediction_results(model_name)
    # elif model_name in ['Streamlining']:
    #     df = process_prediction_results_streamlining()
    else:
        raise Exception("Invalid model name.")
    scores = get_metrics_scores(df)
    return scores


models_performance = {}
for model in models:
    scores = get_score_for_models(model, aggregate_type="mean")
    models_performance[model] = scores

#
file_path = results_path + f"/{data_config}_models_performance_all.csv"
csvfile = open(file_path, mode="a", newline='', encoding='utf-8')
header = ['data', 'model', 'template', 'prefix-source', 'prefix-num', 'do_sample', 'num_beam', 'temperature', 'valid', 'acc', 'auc', 'p', 'r', 'F1']
writer = csv.DictWriter(csvfile, fieldnames=header)
#
writer.writeheader()
for cur_model in models_performance:
    writer.writerow({
        'data': ''.join(data_config),
        'model': cur_model,
        'template': models_performance[cur_model]['template'],
        'prefix-source': models_performance[cur_model]['prefix-source'],
        'prefix-num': models_performance[cur_model]['prefix-num'],
        'do_sample': do_sample,
        'num_beam': num_beams,
        'temperature': temperature,
        'valid': '100%',
        'acc': models_performance[cur_model]['acc'],
        'auc': models_performance[cur_model]['auc'],
        'p': models_performance[cur_model]['precision'],
        'r': models_performance[cur_model]['recall'],
        'F1': models_performance[cur_model]['f1']
    })
print(f"结果输出到{file_path}")
