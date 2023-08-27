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
from th4_mention_pairs_scorer import get_cmp_or_csv_files, mp_scorer_df, save_mention_pair_scores_into_csv_in_list_format, save_mention_pair_scores_into_csv_in_table_format, get_experiment_settings
from th5_clustering import adapter_of_mention_pairs, cd_clustering, wd_clustering, remove_unselected_mention, check_whether_all_mentions_are_clustered
from th6_clustering_scorer import coreference_scorer, save_clustering_scores_into_csv_in_list_format, save_clustering_scores_into_csv_in_table_format


def mp_target(experiment_path_list, config_dict, target_type):
    experiments_scores = []
    for cur_experiment_path in experiment_path_list:
        print(f"==={target_type} Mention Pair:{os.path.basename(cur_experiment_path)}===")
        logging.info(f"==={target_type} Mention Pair:{os.path.basename(cur_experiment_path)}===")
        # 读取csv
        cur_csv_path = f"{cur_experiment_path}.csv"
        df = pd.read_csv(cur_csv_path)
        # 备份csv
        shutil.copy(cur_csv_path, config_dict["output_path"])
        # 根据任务类型，只保留特定部分mp
        if target_type == "2s":
            df = df[df["seq"] <= 1]
        elif target_type == "wd-":
            df = df[(df["wd/cd"] == "wd") & ~df["seq"].isin([0, 1])]
        elif target_type == "wd+":
            df = df[df["wd/cd"] == "wd"]
        elif target_type == "cd-":
            df = df[df["wd/cd"] == "cd"]
        elif target_type == "cd+":
            df = df[df["wd/cd"].isin(["wd", "cd"])]
        else:
            raise RuntimeError
        # 抽取配置
        settings = get_experiment_settings(cur_csv_path)
        # 打分
        scores = mp_scorer_df(df, settings["template"])
        scores.update(settings)
        # 保存分数
        path = os.path.join(config_dict['output_path'], f"{os.path.basename(cur_experiment_path)}.{target_type}_mp.scores")
        with open(path, 'w', encoding="utf8") as f:
            f.writelines([f"{k}: {v}\n" for k, v in scores.items()])
        print(f"OUTPUT: {target_type} mention pairs scores saved in {path}")
        logging.info(f"OUTPUT: {target_type} mention pairs scores saved in {path}")
        # 分数整合
        info = {}
        info.update(settings)
        info.update(scores)
        experiments_scores.append(info)
    print(f"==={target_type} Mention Pair:整合===")
    logging.info(f"==={target_type} Mention Pair:整合===")
    save_mention_pair_scores_into_csv_in_list_format(
        experiments_scores,
        output_path=config_dict["output_path"], suffix=f"scores_{target_type}_mp_list.csv")
    save_mention_pair_scores_into_csv_in_table_format(
        experiments_scores,
        output_path=config_dict["output_path"], suffix=f"scores_{target_type}_mp_table.csv")
    print("")
    logging.info("")


def wd_coref(experiment_path_list, config_dict):
    """
    给定experiment_path_list::

      experiment_path_list = [
          'E:\\ProgramCode\\WhatGPTKnowsAboutWhoIsWho\\WhatGPTKnowsAboutWhoIsWho-main\\Models2\\data\\3.pred\\11-13/[\'36_ecb\'](strategy3)_ChatGPT3.5(b1t0)_0shot_t13SAU_noSample(r1)'，
          ...
      ]

    本函数把其中的每个experiment_path当做一次实验，读入对应路径下的.c_mp文件(corpus和mention pair list)，并对每个topic分别做wd聚类并评分。每个实验的结果分别保存到config_dict["output _path"]指定的输出路径。最后再汇总每个实验的结果，输出2个汇总结果。

    :param experiment_path_list:
    :param config_dict:
    :return: no return.结果保存到config_dict["output _path"]指定的输出路径。
    """
    experiments_scores = []
    for cur_experiment_path in experiment_path_list:
        print(f"\n{'=' * (19 + len(os.path.basename(cur_experiment_path)))}\n"
              f"===CD Clustering:{os.path.basename(cur_experiment_path)}===\n"
              f"{'=' * (19 + len(os.path.basename(cur_experiment_path)))}")
        logging.info(f"\n{'=' * (19 + len(os.path.basename(cur_experiment_path)))}\n"
                     f"===CD Clustering:{os.path.basename(cur_experiment_path)}===\n"
                     f"{'=' * (19 + len(os.path.basename(cur_experiment_path)))}")
        #
        cur_cmp_path = f"{cur_experiment_path}.c_mp"
        # 抽取配置
        settings = get_experiment_settings(cur_cmp_path)
        # 读取
        with open(cur_cmp_path, 'rb') as f:
            corpus, mention_pairs = cPickle.load(f)
        # 聚类（无论wd还是cd聚类，都是每个topic分开做的，因为没有跨topic共指的现象。）
        for cur_topic_id, cur_topic_mp in mention_pairs.items():
            adapter_of_mention_pairs(cur_topic_mp)
            # 计算当前topic的prefix。
            n = int(re.search("([0-9]*)_ecb", cur_topic_id).groups()[0])
            p = 1 if "plus" in cur_topic_id else 0
            prefix = n * 100000000 + p * 1000000
            """当前topic的prefix。比如36_ecb的prefix就是3600000000,36_ecbplus的prefix就是3601000000"""
            #
            wd_clustering(prefix=prefix, mention_pairs_list=cur_topic_mp, strategy=2, statistic_dict_path=config_dict["statistic_dict_path"])
            #
            del cur_topic_id, cur_topic_mp, n, p, prefix
        remove_unselected_mention(corpus)
        check_whether_all_mentions_are_clustered(corpus)
        # 保存聚类结果
        path = os.path.join(config_dict['output_path'],
                            f"{os.path.basename(cur_experiment_path)}.wd_clustering.clustered_corpus")
        with open(path, 'wb') as f:
            cPickle.dump(corpus, f)
        print(f"OUTPUT: corpus and wd clustering result under cur config saved in {path}")
        logging.info(f"OUTPUT: corpus and wd clustering result under cur config saved in {path}")
        # 打分
        for cur_topic_id, cur_topic in corpus.topics.items():
            for cur_doc_id, cur_doc in cur_topic.docs.items():
                for cur_sent_id, cur_sent in cur_doc.sentences.items():
                    for cur_mention in cur_sent.gold_entity_mentions + cur_sent.gold_event_mentions:
                        cur_mention.gold_tag = f"{cur_doc_id}-{cur_mention.gold_tag}"
                        del cur_mention
                    del cur_sent_id
                del cur_doc_id, cur_doc
            del cur_topic_id, cur_topic
        prefix = f"{os.path.basename(cur_experiment_path)}.wd_clustering"
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
    print(f"\n===================\n"
          f"===WD Coref:整合===\n"
          f"===================")
    logging.info(f"\n===================\n"
                 f"===WD Coref:整合===\n"
                 f"===================")
    save_clustering_scores_into_csv_in_list_format(experiments_scores,
                                                   output_path=config_dict["output_path"],
                                                   suffix="scores_wd_clustering_list.csv")
    save_clustering_scores_into_csv_in_table_format(experiments_scores,
                                                    output_path=config_dict["output_path"],
                                                    suffix="scores_wd_clustering_table.csv")


def cd_coref(experiment_path_list, config_dict):
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
            #
            n = int(re.search("([0-9]*)_ecb", cur_topic_id).groups()[0])
            p = 1 if "plus" in cur_topic_id else 0
            prefix = n*100000000 + p*1000000
            #
            cd_clustering(prefix=prefix, mention_pairs_list=cur_topic_mp)
        remove_unselected_mention(corpus)
        check_whether_all_mentions_are_clustered(corpus)
        # 保存聚类结果
        path = os.path.join(config_dict['output_path'], f"{os.path.basename(cur_experiment_path)}.cd_clustering.clustered_corpus")
        with open(path, 'wb') as f:
            cPickle.dump(corpus, f)
        print(f"OUTPUT: corpus and cd clustering result under cur config saved in {path}")
        logging.info(f"OUTPUT: corpus and cd clustering result under cur config saved in {path}")
        # 打分
        prefix = f"{os.path.basename(cur_experiment_path)}.cd_clustering"
        scores = coreference_scorer(corpus, config_dict["output_path"], output_prefix=prefix)
        # 保存
        path = os.path.join(config_dict["output_path"], f"{prefix}.scores")
        with open(path, 'w', encoding="utf8") as f:
            f.writelines([f"{k}: {v}\n" for k, v in scores.items()])
        print(f"OUTPUT: cd clustering scores saved in {path}")
        logging.info(f"OUTPUT: cd clustering scores saved in {path}")
        # 分数整合
        info = {}
        info.update(settings)
        info.update(scores)
        experiments_scores.append(info)
    print(f"\n===================\n"
          f"===CD Coref:整合===\n"
          f"===================")
    logging.info(f"\n===================\n"
                 f"===CD Coref:整合===\n"
                 f"===================")
    save_clustering_scores_into_csv_in_list_format(experiments_scores,
                                                   output_path=config_dict["output_path"],
                                                   suffix="scores_cd_clustering_list.csv")
    save_clustering_scores_into_csv_in_table_format(experiments_scores,
                                                    output_path=config_dict["output_path"],
                                                    suffix="scores_cd_clustering_table.csv")


def main():
    # config
    config_dict = {
        "input_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\data\3.pred",
        "output_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\output",
        "statistic_dict_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\data\statistics\statistic_dict.pkl"
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
    print(f"log saved in {os.path.join(config_dict['output_path'], 'log.txt')}\n")
    logging.info(f"log saved in {os.path.join(config_dict['output_path'], 'log.txt')}\n")
    # save this file itself
    shutil.copy(os.path.abspath(__file__), config_dict["output_path"])
    #
    pd.options.display.float_format = "{:,.2f}".format

    #
    experiment_path_list = get_cmp_or_csv_files(config_dict["input_path"], with_cmp=False)

    #
    mp_target(experiment_path_list=experiment_path_list, config_dict=config_dict, target_type="2s")
    mp_target(experiment_path_list=experiment_path_list, config_dict=config_dict, target_type="wd-")
    mp_target(experiment_path_list=experiment_path_list, config_dict=config_dict, target_type="wd+")
    mp_target(experiment_path_list=experiment_path_list, config_dict=config_dict, target_type="cd-")
    mp_target(experiment_path_list=experiment_path_list, config_dict=config_dict, target_type="cd+")
    # wd_coref(experiment_path_list=experiment_path_list, config_dict=config_dict)
    # cd_coref(experiment_path_list=experiment_path_list, config_dict=config_dict)


if __name__ == '__main__':
    main()
