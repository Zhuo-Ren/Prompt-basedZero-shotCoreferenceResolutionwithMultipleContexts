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
    """
    给定csv文件，对指定部分做mention pairs打分。

    experiment_path_list就是一系列csv文件路径组成的列表（这里的路径没有.csv后缀）::

        experiment_path_list = [
            "some_path/['36_ecb'](strategy3)_ChatGPT3.5(b1t0)_0shot_t17MAU_noSample(r1)",
            "some_path/['36_ecb'](strategy3)_ChatGPT3.5(b1t0)_0shot_t18MAU_noSample(r1)",
        ]

    读取csv后，里边是一系列mention pair以及模型对其的预测结果。

    target_type对csv中的mention pair取子集：
        * target_type = "2s"就是只选取mention 1和mention 2处于同一句或前后句的mention pair。
        * target_type = "wd-"就是只选取文档内mention pair，但刨去了"2s"的部分
        * target_type = "wd+"就是只选取文档内mention pair（什么都不刨去）
        * target_type = "cd-"就是选取所有mention pair，但刨去了"wd+"的部分
        * target_type = "cd+"就是选取所有mention pair（什么都不刨去）

    graph::

        |__2s__|__wd-__|__cd-__|
        |XXXXXX|XXXXXXX|XXXXXXX|   ← all the mention pairs in csv file
        |_____wd+______|       |
        |__________cd+_________|

    :param experiment_path_list: A list of path of CSV file. (path with out .csv suffix)
    :param config_dict:
    :param target_type: Take which part of mention pairs in CSV file.
    :return: No return. 对experiment_path_list中的每个csv，程序保存一个打分结果。对所有csv，再保存两个总结的打分结果表。
    """
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


def wd_coref(experiment_path_list, output_path, strategy, threshold=0, statistic_dict_path="仅strategy2需要此参数"):
    """
    给定experiment_path_list, 本函数把其中的每个experiment_path当做一次实验，读入对应路径下的.c_mp文件(corpus和mention pair list)，并对每个topic分别做wd聚类并评分。每个实验的结果分别保存到config_dict["output _path"]指定的输出路径。最后再汇总每个实验的结果，输出2个汇总结果。

    :param experiment_path_list: A list like [
        r"some_path\['36_ecb'](strategy3)_ChatGPT3.5(b1t0)_0shot_t13SAU_noSample(r1)",
        r"some_path\['36_ecb'](strategy3)_ChatGPT3.5(b1t0)_0shot_t25DAU_noSample(r1)"
      ]
    :param strategy: 不同的聚类算法
    :param statistic_dict_path: strategy 2 基于统计数据进行聚类，这个路径指向统计数据。
    :param threshold: strategy 2 中，得分大于等于此阈值的mp才被认为是共指的。
    :return: no return.结果保存到config_dict["output _path"]指定的输出路径。
    """
    experiments_scores = []
    for cur_experiment_path in experiment_path_list:
        print(f"\n{'=' * (19 + len(os.path.basename(cur_experiment_path)))}\n"
              f"===WD Clustering:{os.path.basename(cur_experiment_path)}===\n"
              f"{'=' * (19 + len(os.path.basename(cur_experiment_path)))}")
        logging.info(f"\n{'=' * (19 + len(os.path.basename(cur_experiment_path)))}\n"
                     f"===WD Clustering:{os.path.basename(cur_experiment_path)}===\n"
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
            wd_clustering(prefix=prefix, mention_pairs_list=cur_topic_mp, strategy=strategy, statistic_dict_path=statistic_dict_path, threshold=threshold)
            #
            del cur_topic_id, cur_topic_mp, n, p, prefix
        remove_unselected_mention(corpus)
        check_whether_all_mentions_are_clustered(corpus)
        # 保存聚类结果
        path = os.path.join(output_path, f"{os.path.basename(cur_experiment_path)}.wd_clustering.clustered_corpus")
        with open(path, 'wb') as f:
            cPickle.dump(corpus, f)
        print(f"OUTPUT: corpus and wd clustering result under cur config saved in {path}")
        logging.info(f"OUTPUT: corpus and wd clustering result under cur config saved in {path}")
        # 对聚类结果打分
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
        scores = coreference_scorer(corpus, output_path, output_prefix=prefix)
        # 保存
        path = os.path.join(output_path, f"{prefix}.scores")
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
                                                   output_path=output_path,
                                                   suffix="scores_wd_clustering_list.csv")
    save_clustering_scores_into_csv_in_table_format(experiments_scores,
                                                    output_path=output_path,
                                                    suffix="scores_wd_clustering_table.csv")
    return experiments_scores


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


def get_best_threshold_based_on_cluster(experiment_path_list, config_dict, min_th, max_th):
    """
    注意，这个函数只针对experiment_path_list中的第一个实验进行分析。

    :param experiment_path_list:
    :param config_dict:
    :return:
    """
    delta = 0.0001
    if config_dict["clustering_strategy"] not in [2]:
        raise RuntimeError("这个聚类算法不需要指定阈值")
    e_f1_log = []
    v_f1_log = []
    a_f1_log = []
    cur_threshold = min_th
    while 1:
        r = wd_coref(experiment_path_list=experiment_path_list, output_path=config_dict["output_path"],
                     strategy=config_dict["clustering_strategy"],
                     statistic_dict_path=config_dict["statistic_dict_path"], threshold=cur_threshold)

        e_f1_log.append([r[0]["entity_conll_f1"], cur_threshold])
        v_f1_log.append([r[0]["event_conll_f1"], cur_threshold])
        a_f1_log.append([r[0]["all_conll_f1"], cur_threshold])
        if cur_threshold > max_th:
            break
        else:
            cur_threshold += delta
    del cur_threshold

    return e_f1_log, v_f1_log, a_f1_log


def main(config_dict):
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

    # mp
    # mp_target(experiment_path_list=experiment_path_list, config_dict=config_dict, target_type="2s")
    # mp_target(experiment_path_list=experiment_path_list, config_dict=config_dict, target_type="wd-")
    # mp_target(experiment_path_list=experiment_path_list, config_dict=config_dict, target_type="wd+")
    # mp_target(experiment_path_list=experiment_path_list, config_dict=config_dict, target_type="cd-")
    # mp_target(experiment_path_list=experiment_path_list, config_dict=config_dict, target_type="cd+")

    # get best wd threshold
    # wd_coref(experiment_path_list=experiment_path_list, output_path=config_dict["output_path"],
    #              strategy=config_dict["clustering_strategy"], threshold=1,
    #              statistic_dict_path=config_dict["statistic_dict_path"])
    #
    # min_th, max_th = 0.0007, 0.0270  # t13-25-13 mp_best=0.0158
    # min_th, max_th = 0.0007, 0.0270  # t13-25-25 mp_best=0.0159
    # min_th, max_th = 0.0017, 0.0270  # t13 mp_best=0.0105
    # min_th, max_th = 0.0007, 0.0270  # t17 mp_best=0.0117
    # min_th, max_th = 0.0007, 0.0307  # t25 mp_best=0.0199
    # e_f1_log, v_f1_log, a_f1_log = get_best_threshold_based_on_cluster(
    #     experiment_path_list=experiment_path_list, config_dict=config_dict,
    #     min_th=min_th, max_th=max_th
    # )
    # max_e_f1 = max([i[0] for i in e_f1_log])
    # max_v_f1 = max([i[0] for i in v_f1_log])
    # max_a_f1 = max([i[0] for i in a_f1_log])
    # best_e_th = [i[1] for i in e_f1_log if i[0] == max_e_f1]
    # best_v_th = [i[1] for i in v_f1_log if i[0] == max_v_f1]
    # best_a_th = [i[1] for i in a_f1_log if i[0] == max_a_f1]
    # best_ev_th = sorted(list(set(best_e_th) & set(best_v_th)))
    # best_a_th = sorted(best_a_th)
    # #
    # ev_th_path = os.path.join(config_dict["output_path"], f"{os.path.basename(experiment_path_list[0])}.wd_clustering.th_ev")
    # f = open(ev_th_path, 'w')
    # f.write(f"best e f1:{max_e_f1}\n")
    # f.write(f"best v f1:{max_v_f1}\n")
    # f.write(f"best_th: {best_ev_th}\n")
    # f.write(f"e_f1: {e_f1_log}\n")
    # f.write(f"v_f1: {v_f1_log}\n")
    # f.close()
    # #
    # a_th_path = os.path.join(config_dict["output_path"], f"{os.path.basename(experiment_path_list[0])}.wd_clustering.th_a")
    # f = open(a_th_path, 'w')
    # f.write(f"best a f1:{max_a_f1}\n")
    # f.write(f"best_a_th: {best_a_th}\n")
    # f.write(f"a_f1: {a_f1_log}\n")


    # wd clustering
    # t_13-25-13 th = 0.0097
    # t_13-25-25 th = 0.0031999999999999984
    # t_13 th = 0.0094
    # t_17 th = 0.0110
    # t_25 th = 0.0184
    # r = wd_coref(experiment_path_list=experiment_path_list, output_path=config_dict["output_path"],
    #              strategy=config_dict["clustering_strategy"], threshold=0.0097,
    #              statistic_dict_path=config_dict["statistic_dict_path"])

    # cd clustering
    # cd_coref(experiment_path_list=experiment_path_list, config_dict=config_dict)


if __name__ == '__main__':
    # config
    config_dict = {
        "input_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\data\temp",
        "output_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\output",
        "clustering_strategy": 2,
        "statistic_dict_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\data\statistics\statistic_dict.pkl"
    }
    main(config_dict)
