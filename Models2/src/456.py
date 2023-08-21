# 标准库
import re
import os
import _pickle as cPickle
from typing import Dict, List, Tuple, Union
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

from classes import Corpus, Topic, Document, Sentence, Token, Mention, EventMention, EntityMention, MentionData
# 本地库
from th4_mention_pairs_scorer import get_cmp_or_csv_files, get_score_from_csv, save_mention_pair_scores_into_csv_in_list_format, save_mention_pair_scores_into_csv_in_table_format, get_experiment_settings
from th5_clustering import adapter_of_mention_pairs, clustering, remove_unselected_mention, check_whether_all_mentions_are_clustered
from th6_clustering_scorer import coreference_scorer, save_clustering_scores_into_csv_in_list_format, save_clustering_scores_into_csv_in_table_format


def main():
    # config
    config_dict = {
        "input_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\data\3.pred",
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
    experiment_path_list = get_cmp_or_csv_files(config_dict["input_path"], with_cmp=False)

    # 1. mention pair部分
    experiments_scores = []
    for cur_experiment_path in experiment_path_list:
        print(f"\n{'='*(19+len(os.path.basename(cur_experiment_path)))}\n"
              f"===Mention Pair:{os.path.basename(cur_experiment_path)}===\n"
              f"{'='*(19+len(os.path.basename(cur_experiment_path)))}")
        logging.info(f"\n{'='*(19+len(os.path.basename(cur_experiment_path)))}\n"
              f"===Mention Pair:{os.path.basename(cur_experiment_path)}===\n"
              f"{'='*(19+len(os.path.basename(cur_experiment_path)))}")
        #
        cur_csv_path = f"{cur_experiment_path}.csv"
        # 保存
        shutil.copy(cur_csv_path, config_dict["output_path"])
        # 抽取配置
        settings = get_experiment_settings(cur_csv_path)
        # 打分
        scores = get_score_from_csv(csv_path=cur_csv_path)
        # 保存分数
        path = os.path.join(config_dict['output_path'], f"{os.path.basename(cur_experiment_path)}.mp.scores")
        with open(path, 'w', encoding="utf8") as f:
            f.writelines([f"{k}: {v}\n" for k, v in scores.items()])
        print(f"OUTPUT: mention pairs scores saved in {path}")
        logging.info(f"OUTPUT: mention pairs scores saved in {path}")
        # 分数整合
        info = {}
        info.update(settings)
        info.update(scores)
        experiments_scores.append(info)
    print(f"\n{'='*(19+len(os.path.basename(cur_experiment_path)))}\n"
          f"===Mention Pair:整合===\n"
          f"{'='*(19+len(os.path.basename(cur_experiment_path)))}")
    logging.info(f"\n{'='*(19+len(os.path.basename(cur_experiment_path)))}\n"
          f"===Mention Pair:整合===\n"
          f"{'='*(19+len(os.path.basename(cur_experiment_path)))}")
    save_mention_pair_scores_into_csv_in_list_format(experiments_scores, output_path=config_dict["output_path"])
    save_mention_pair_scores_into_csv_in_table_format(experiments_scores, output_path=config_dict["output_path"])
    # 2.cd coref部分
    experiments_scores = []
    for cur_experiment_path in experiment_path_list:
        print(f"\n{'='*(19+len(os.path.basename(cur_experiment_path)))}\n"
              f"===CD Clustering:{os.path.basename(cur_experiment_path)}===\n"
              f"{'='*(19+len(os.path.basename(cur_experiment_path)))}")
        logging.info(f"\n{'='*(19+len(os.path.basename(cur_experiment_path)))}\n"
                     f"===CD Clustering:{os.path.basename(cur_experiment_path)}===\n"
                     f"{'='*(19+len(os.path.basename(cur_experiment_path)))}")
        #
        cur_cmp_path = f"{cur_experiment_path}.c_mp"
        # 抽取配置
        settings = get_experiment_settings(cur_cmp_path)
        # 读取
        with open(cur_cmp_path, 'rb') as f:
            corpus, mention_pairs = cPickle.load(f)
        # cd聚类
        for cur_topic_id, cur_topic_mp in mention_pairs.items():
            adapter_of_mention_pairs(cur_topic_mp)
            clustering(cur_topic_id, cur_topic_mp)
        remove_unselected_mention(corpus)
        check_whether_all_mentions_are_clustered(corpus)
        # 保存聚类结果
        path = os.path.join(config_dict['output_path'], f"{os.path.basename(cur_experiment_path)}.clustering1.clustered_corpus")
        with open(path, 'wb') as f:
            cPickle.dump(corpus, f)
        print(f"OUTPUT: corpus and clustering result under cur config saved in {path}")
        logging.info(f"OUTPUT: corpus and clustering result under cur config saved in {path}")
        # 打分
        prefix = f"{os.path.basename(cur_experiment_path)}.clustering1"
        scores = coreference_scorer(corpus, config_dict["output_path"], output_prefix=prefix)
        # 保存
        path = os.path.join(config_dict["output_path"], f"{prefix}.scores")
        with open(path, 'w', encoding="utf8") as f:
            f.writelines([f"{k}: {v}\n" for k, v in scores.items()])
        print(f"OUTPUT: clustering scores saved in {path}")
        logging.info(f"OUTPUT: clustering scores saved in {path}")
        # 分数整合
        info = {}
        info.update(settings)
        info.update(scores)
        experiments_scores.append(info)
    print(f"\n{'='*(19+len(os.path.basename(cur_experiment_path)))}\n"
          f"===CD Coref:整合===\n"
          f"{'='*(19+len(os.path.basename(cur_experiment_path)))}")
    logging.info(f"\n{'='*(19+len(os.path.basename(cur_experiment_path)))}\n"
                 f"===CD Coref:整合===\n"
                 f"{'='*(19+len(os.path.basename(cur_experiment_path)))}")
    save_clustering_scores_into_csv_in_list_format(experiments_scores, output_path=config_dict["output_path"])
    save_clustering_scores_into_csv_in_table_format(experiments_scores, output_path=config_dict["output_path"])

if __name__ == '__main__':
    main()
