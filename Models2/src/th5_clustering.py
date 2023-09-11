# 标准库
import re
import os
import shutil
import _pickle as cPickle
# from typing import Dict, List, Tuple, Union
import logging
import time
# 本地库
from classes import Corpus, Topic, Document, Sentence, Token, Mention, EventMention, EntityMention, MentionData
from template import templates_list


def cd_clustering(prefix, mention_pairs_list):
    """
    Give mention_pairs_list, get the clusters.
    聚类算法1：直接cd聚类


    :param prefix: The prefix of cluster_id. 本函数对簇进行编号：0,1,2,...
      但是不同topic会分别执行本函数，这样cluster_id就重了。所以给cluster_id加一个前缀。
      变成36ecbplus0, 36ecbplus1, ...这样。
    :param mention_pairs_list:
    :return: no return. cd_coref_chain of mention obj in mention_pairs_list is changed.
      This also lead to the change of cd_coref_chain of mention obj in Corpus obj.
    """
    # 1. Create a dictionary to represent the coreferred
    coreference_dict = {}
    """
    {
        "mention1_id": {a set of mentions that co-referred with mention1},
        "mention2_id": {a set of mentions that co-referred with mention2},
    }
    """
    for m1, m2, is_coreferred in mention_pairs_list:
        m1_id = f"{m1.doc_id}-{m1.sent_id}-{m1.start_offset}-{m1.end_offset}"
        m2_id = f"{m2.doc_id}-{m2.sent_id}-{m2.start_offset}-{m2.end_offset}"
        #
        if m1_id not in coreference_dict:
            coreference_dict[m1_id] = []
        if m2_id not in coreference_dict:
            coreference_dict[m2_id] = []
        #
        if (m1_id != m2_id) and (is_coreferred is True):
            if m2 not in coreference_dict[m1_id]:
                coreference_dict[m1_id].append(m2)
            if m1 not in coreference_dict[m2_id]:
                coreference_dict[m2_id].append(m1)
        del m1, m2, is_coreferred, m1_id, m2_id

    # 2. 计算得分
    for cur_mention_pair in mention_pairs_list:
        # 抽取
        m1 = cur_mention_pair[0]
        m2 = cur_mention_pair[1]
        m1_id = f"{m1.doc_id}-{m1.sent_id}-{m1.start_offset}-{m1.end_offset}"
        m2_id = f"{m2.doc_id}-{m2.sent_id}-{m2.start_offset}-{m2.end_offset}"
        is_coreferred = cur_mention_pair[2]
        # 准备
        positive_count = 0
        negative_count = 0
        # 算分
        for cur_mention_id in coreference_dict.keys():
            if cur_mention_id in [m1_id, m2_id]:
                continue
            corefer_with_mention1 = (m1 in coreference_dict[cur_mention_id])
            corefer_with_mention2 = (m2 in coreference_dict[cur_mention_id])
            if corefer_with_mention1 and corefer_with_mention2:
                positive_count += 1
            elif (not corefer_with_mention1) and (not corefer_with_mention2):
                pass
            else:
                negative_count += 1
            del corefer_with_mention1, corefer_with_mention2
        if is_coreferred is True:
            positive_count += 2
        elif is_coreferred is False:
            negative_count += 2
        elif is_coreferred is None:  # None 就是模型没输出有效结果
            pass
        else:
            raise RuntimeError("invalid value of is_coreferred")
        # 记录分儿
        cur_mention_pair.append(positive_count - negative_count)
        #
        del cur_mention_pair, m1, m2, m1_id, m2_id, is_coreferred, positive_count, negative_count
    # 3. 排序
    sorted_mention_pairs_list = sorted(mention_pairs_list, key=lambda x: x[3])
    # 4. 凝聚
    clusters = {}
    """簇。 cluster_id: [mention1, mention2]"""
    clustered_mention = {}
    """已经参与聚类的mention。 mention_id: 此mention所属簇的cluster_id"""
    cluster_id = 0
    while 1:
        if len(sorted_mention_pairs_list) == 0:
            break
        #
        cur_mention_pair = sorted_mention_pairs_list.pop()
        m1 = cur_mention_pair[0]
        m2 = cur_mention_pair[1]
        score = cur_mention_pair[3]
        m1_id = f"{m1.doc_id}-{m1.sent_id}-{m1.start_offset}-{m1.end_offset}"
        m2_id = f"{m2.doc_id}-{m2.sent_id}-{m2.start_offset}-{m2.end_offset}"
        #
        mention1_clustered = m1_id in clustered_mention.keys()
        mention2_clustered = m2_id in clustered_mention.keys()
        #
        if score < 1:
            if not mention1_clustered:
                cluster_id += 1
                clustered_mention[m1_id] = cluster_id
                clusters[cluster_id] = [m1]
            if not mention2_clustered:
                cluster_id += 1
                clustered_mention[m2_id] = cluster_id
                clusters[cluster_id] = [m2]
        else:
            if mention1_clustered and (not mention2_clustered):
                mention1_cluster_id = clustered_mention[m1_id]
                clustered_mention[m2_id] = mention1_cluster_id
                clusters[mention1_cluster_id].append(m2)
            elif (not mention1_clustered) and mention2_clustered:
                mention2_cluster_id = clustered_mention[m2_id]
                clustered_mention[m1_id] = mention2_cluster_id
                clusters[mention2_cluster_id].append(m1)
            elif (not mention1_clustered) and (not mention2_clustered):
                cluster_id += 1
                clustered_mention[m1_id] = cluster_id
                clustered_mention[m2_id] = cluster_id
                clusters[cluster_id] = [m1, m2]
            elif mention1_clustered and mention2_clustered:
                mention1_cluster_id = clustered_mention[m1_id]
                mention2_cluster_id = clustered_mention[m2_id]
                if mention1_cluster_id == mention2_cluster_id:
                    pass  # m1和m2已经是同一个簇中了，什么都不用做
                else:
                    # 合并两个簇
                    cluster_id += 1
                    new_cluster = clusters[mention1_cluster_id] + clusters[mention2_cluster_id]
                    del clusters[mention1_cluster_id]
                    del clusters[mention2_cluster_id]
                    clusters[cluster_id] = new_cluster
                    for cur_mention in new_cluster:
                        cur_mention_id = f"{cur_mention.doc_id}-{cur_mention.sent_id}-{cur_mention.start_offset}-{cur_mention.end_offset}"
                        clustered_mention[cur_mention_id] = cluster_id
            # End of if else 4种情况
        # End of if score < 1 ... else
    # 5. 保存
    for cur_cluster_id, cur_cluster in clusters.items():
        for cur_mention in cur_cluster:
            cur_mention.cd_coref_chain = prefix + cur_cluster_id


def wd_clustering_1(prefix, mention_pairs_list):
    """
    给定一个topic下的mention pair list，做wd聚类。

    :param prefix: The prefix of cluster_id. 本函数对簇进行编号：0,1,2,...
      但是不同topic会分别执行本函数，这样cluster_id就重了。所以给cluster_id加一个前缀。
      比如36_ecb这个topic下的cluster 1在加前缀后就变成了3600000001；37_ecbplus这个topic下的cluster 1加前缀后变成了3701000001.这就区分开了。
    :param mention_pairs_list: [[mention_obj_1, mention_obj_2, 预测值], ...]预测值是模型预测的结果，Ture是共指，False是不共指，None是没预测出来。
    :param strategy: 采用哪一种算法来实现wd clustering。1是给定权重，2是统计权重。
    :param statistic_dict_path: 如果使用统计权重，则从这里读取统计信息。
    :return: no return. cd_coref_chain of mention obj in mention_pairs_list is changed.
      This also lead to the change of cd_coref_chain of mention obj in Corpus obj.
    """

    # 1. Create a dictionary to represent the coreferred
    coreference_dict = {}
    """
    {
        "mention1_id": {a set of mentions that co-referred with mention1},
        "mention2_id": {a set of mentions that co-referred with mention2},
    }
    """
    for m1, m2, is_coreferred in mention_pairs_list:
        # 因为是wd共指，所以只看wd的mention pair
        if m1.doc_id != m2.doc_id:
            continue
        #
        m1_id = f"{m1.doc_id}-{m1.sent_id}-{m1.start_offset}-{m1.end_offset}"
        m2_id = f"{m2.doc_id}-{m2.sent_id}-{m2.start_offset}-{m2.end_offset}"
        #
        if m1_id not in coreference_dict:
            coreference_dict[m1_id] = []
        if m2_id not in coreference_dict:
            coreference_dict[m2_id] = []
        #
        if (m1_id != m2_id) and (is_coreferred is True):
            if m2 not in coreference_dict[m1_id]:
                coreference_dict[m1_id].append(m2)
            if m1 not in coreference_dict[m2_id]:
                coreference_dict[m2_id].append(m1)
        del m1, m2, is_coreferred, m1_id, m2_id

    # 2. 计算得分
    for cur_mention_pair in mention_pairs_list:
        # 抽取
        m1 = cur_mention_pair[0]
        m2 = cur_mention_pair[1]
        m1_id = f"{m1.doc_id}-{m1.sent_id}-{m1.start_offset}-{m1.end_offset}"
        m2_id = f"{m2.doc_id}-{m2.sent_id}-{m2.start_offset}-{m2.end_offset}"
        is_coreferred = cur_mention_pair[2]
        # 准备
        positive_count = 0
        negative_count = 0
        # 算分

        for cur_mention_id in coreference_dict.keys():
            if cur_mention_id in [m1_id, m2_id]:
                continue
            corefer_with_mention1 = (m1 in coreference_dict[cur_mention_id])
            corefer_with_mention2 = (m2 in coreference_dict[cur_mention_id])
            if corefer_with_mention1 and corefer_with_mention2:
                positive_count += 1
            elif (not corefer_with_mention1) and (not corefer_with_mention2):
                pass
            else:
                negative_count += 1
            del corefer_with_mention1, corefer_with_mention2
        if is_coreferred is True:
            positive_count += 2
        elif is_coreferred is False:
            negative_count += 2
        elif is_coreferred is None:  # None 就是模型没输出有效结果
            pass
        else:
            raise RuntimeError("invalid value of is_coreferred")
        s = positive_count - negative_count
        # 记录分儿
        cur_mention_pair.append(s)
        #
        del cur_mention_pair, m1, m2, m1_id, m2_id, is_coreferred, positive_count, negative_count, s
    # 3. 排序
    t = [mp for mp in mention_pairs_list if mp[0].doc_id == mp[1].doc_id]  # 只要wd的mp
    sorted_mention_pairs_list = sorted(t, key=lambda x: x[3])
    # 4. 凝聚
    clusters = {}
    """簇。 cluster_id: [mention1, mention2]"""
    clustered_mention = {}
    """已经参与聚类的mention。 mention_id: 此mention所属簇的cluster_id"""
    cluster_id = 0
    while 1:
        if len(sorted_mention_pairs_list) == 0:
            break
        #
        cur_mention_pair = sorted_mention_pairs_list.pop()
        m1 = cur_mention_pair[0]
        m2 = cur_mention_pair[1]
        score = cur_mention_pair[3]
        m1_id = f"{m1.doc_id}-{m1.sent_id}-{m1.start_offset}-{m1.end_offset}"
        m2_id = f"{m2.doc_id}-{m2.sent_id}-{m2.start_offset}-{m2.end_offset}"
        #
        mention1_clustered = m1_id in clustered_mention.keys()
        mention2_clustered = m2_id in clustered_mention.keys()
        #
        if score < 1:
            if not mention1_clustered:
                cluster_id += 1
                clustered_mention[m1_id] = cluster_id
                clusters[cluster_id] = [m1]
            if not mention2_clustered:
                cluster_id += 1
                clustered_mention[m2_id] = cluster_id
                clusters[cluster_id] = [m2]
        else:
            if mention1_clustered and (not mention2_clustered):
                mention1_cluster_id = clustered_mention[m1_id]
                clustered_mention[m2_id] = mention1_cluster_id
                # # 检测是否wd
                # for m in clusters[mention1_cluster_id]:
                #     if m.doc_id != m2.doc_id:
                #         print(1)
                clusters[mention1_cluster_id].append(m2)
            elif (not mention1_clustered) and mention2_clustered:
                mention2_cluster_id = clustered_mention[m2_id]
                clustered_mention[m1_id] = mention2_cluster_id
                # # 检测是否wd
                # for m in clusters[mention2_cluster_id]:
                #     if m.doc_id != m2.doc_id:
                #         print(1)
                clusters[mention2_cluster_id].append(m1)
            elif (not mention1_clustered) and (not mention2_clustered):
                cluster_id += 1
                clustered_mention[m1_id] = cluster_id
                clustered_mention[m2_id] = cluster_id
                # # 检测是否wd
                # if m1.doc_id != m2.doc_id:
                #     print(1)
                clusters[cluster_id] = [m1, m2]
            elif mention1_clustered and mention2_clustered:
                mention1_cluster_id = clustered_mention[m1_id]
                mention2_cluster_id = clustered_mention[m2_id]
                if mention1_cluster_id == mention2_cluster_id:
                    pass  # m1和m2已经是同一个簇中了，什么都不用做
                else:
                    # 合并两个簇
                    cluster_id += 1
                    new_cluster = clusters[mention1_cluster_id] + clusters[mention2_cluster_id]
                    del clusters[mention1_cluster_id]
                    del clusters[mention2_cluster_id]
                    clusters[cluster_id] = new_cluster
                    for cur_mention in new_cluster:
                        cur_mention_id = f"{cur_mention.doc_id}-{cur_mention.sent_id}-{cur_mention.start_offset}-{cur_mention.end_offset}"
                        clustered_mention[cur_mention_id] = cluster_id
            # End of if else 4种情况
        # End of if score < 1 ... else
    # 5. 检测是否wd
    for cur_cluster_id, cur_cluster in clusters.items():
        doc_of_cur_cluster = None
        for cur_mention in cur_cluster:
            if doc_of_cur_cluster is None:
                doc_of_cur_cluster = cur_mention.doc_id
            else:
                if doc_of_cur_cluster != cur_mention.doc_id:
                    raise RuntimeError
    # 6. 保存
    for cur_cluster_id, cur_cluster in clusters.items():
        for cur_mention in cur_cluster:
            cur_mention.cd_coref_chain = prefix + cur_cluster_id


def get_wdmp(mp_list):
    wdmp = {}
    for cur_wdmp in mp_list:
        m1 = cur_wdmp[0]
        m2 = cur_wdmp[1]
        if m1.doc_id != m2.doc_id:
            continue
        else:
            if m1.doc_id not in wdmp.keys():
                wdmp[m1.doc_id] = []
            wdmp[m1.doc_id].append(cur_wdmp)
        del m1, m2, cur_wdmp
    return wdmp


def get_coref_dict(cur_doc_wdmp, pred_or_truth):
    cur_doc_coref_dict = {}
    for m1, m2, is_coref in cur_doc_wdmp:
        if pred_or_truth == "pred":
            pass
        elif pred_or_truth == "truth":
            is_coref = (m1.gold_tag == m2.gold_tag)
        else:
            raise  RuntimeError("非法的pred_or_truth值")
        # 这里是重复检查：因为cur_doc_wdmp应该已经是只包含wd的mp了，所以如果有非wd的mp，就报错
        if m1.doc_id != m2.doc_id:
            raise RuntimeError("这里不应该遇到非wd的mp")
        #
        m1_id = f"{m1.doc_id}-{m1.sent_id}-{m1.start_offset}-{m1.end_offset}"
        m2_id = f"{m2.doc_id}-{m2.sent_id}-{m2.start_offset}-{m2.end_offset}"
        #
        if m1_id not in cur_doc_coref_dict:
            cur_doc_coref_dict[m1_id] = []
        if m2_id not in cur_doc_coref_dict:
            cur_doc_coref_dict[m2_id] = []
        #
        if (m1_id != m2_id) and (is_coref is True):
            if m2 not in cur_doc_coref_dict[m1_id]:
                cur_doc_coref_dict[m1_id].append(m2)
            if m1 not in cur_doc_coref_dict[m2_id]:
                cur_doc_coref_dict[m2_id].append(m1)
        del m1, m2, is_coref, m1_id, m2_id
    return cur_doc_coref_dict


def get_mp_score_strategy_2(cur_wdmp, cur_doc_m_list, cur_doc_pred_coref_dict, sd):
    # 抽取
    m1 = cur_wdmp[0]
    m2 = cur_wdmp[1]
    m1_id = f"{m1.doc_id}-{m1.sent_id}-{m1.start_offset}-{m1.end_offset}"
    m2_id = f"{m2.doc_id}-{m2.sent_id}-{m2.start_offset}-{m2.end_offset}"
    is_coref = cur_wdmp[2]
    # 算分
    Cab = 1 if (m1.gold_tag == m2.gold_tag) else 0
    l0 = 1 if is_coref else 0
    li_count = [0, 0, 0]  # 分别是li=0的个数，li=1的个数，li=2的个数
    for mi in cur_doc_m_list:
        if mi in [m1, m2]:
            del mi
            continue
        else:  # m1 m2 mi组成一组
            li = 0
            if mi in cur_doc_pred_coref_dict[m1_id]:
                li += 1
            if mi in cur_doc_pred_coref_dict[m2_id]:
                li += 1
            #
            li_count[li] += 1
            #
            del mi, li
    s = 1
    s = s * sd[f"P(li=0|l0={l0},Cab=1)"]**li_count[0]
    s = s * sd[f"P(li=1|l0={l0},Cab=1)"]**li_count[1]
    s = s * sd[f"P(li=2|l0={l0},Cab=1)"]**li_count[2]
    s = s * sd[f"P(l0={l0}|Cab=1)"]
    s = s * sd[f"P(Cab=1)"]
    s = s / sd[f"P(li=0|l0={l0})"]**li_count[0]
    s = s / sd[f"P(li=1|l0={l0})"]**li_count[1]
    s = s / sd[f"P(li=2|l0={l0})"]**li_count[2]
    s = s / sd[f"P(l0={l0})"]
    #
    return s, Cab, l0, li_count


def get_mp_score_strategy_3(cur_wdmp, cur_doc_m_list, cur_doc_pred_coref_dict):
    # 抽取
    m1 = cur_wdmp[0]
    m2 = cur_wdmp[1]
    m1_id = f"{m1.doc_id}-{m1.sent_id}-{m1.start_offset}-{m1.end_offset}"
    m2_id = f"{m2.doc_id}-{m2.sent_id}-{m2.start_offset}-{m2.end_offset}"
    is_coref = cur_wdmp[2]
    # 算分
    Cab = 1 if (m1.gold_tag == m2.gold_tag) else 0
    l0 = 1 if is_coref else 0
    li_count = [0, 0, 0]  # 分别是li=0的个数，li=1的个数，li=2的个数
    for mi in cur_doc_m_list:
        if mi in [m1, m2]:
            del mi
            continue
        else:  # m1 m2 mi组成一组
            li = 0
            if mi in cur_doc_pred_coref_dict[m1_id]:
                li += 1
            if mi in cur_doc_pred_coref_dict[m2_id]:
                li += 1
            #
            li_count[li] += 1
            #
            del mi, li
    s = 0
    p = 0.8
    # 直接证据
    s += p if l0 == 1 else -1*p
    # 间接证据
    s += -1*p*p*li_count[1]
    s += p*p*li_count[2]
    #
    if Cab == 0:
        print(f"预测{'正确' if Cab == l0 else '错误'}，打分{'正确' if s < 0 else '错误'}")
        logging.info(f"预测{'正确' if Cab == l0 else '错误'}，打分{'正确' if s < 0 else '错误'}")
    elif Cab == 1:
        print(f"预测{'正确' if Cab == l0 else '错误'}，打分{'正确' if s >= 0 else '错误'}")
        logging.info(f"预测{'正确' if Cab == l0 else '错误'}，打分{'正确' if s >= 0 else '错误'}")
    return s, Cab, l0, li_count


def agglomerate(sorted_cur_doc_mp, cluster_id=0, threshold=0):
    cur_doc_clusters = {}
    """簇。 cluster_id: [mention1, mention2]"""
    cur_doc_clustered_mention = {}
    """已经参与聚类的mention。 mention_id: 此mention所属簇的cluster_id"""
    while 1:
        if len(sorted_cur_doc_mp) == 0:
            break
        #
        cur_mention_pair = sorted_cur_doc_mp.pop()
        m1 = cur_mention_pair[0]
        m2 = cur_mention_pair[1]
        score = cur_mention_pair[3]
        m1_id = f"{m1.doc_id}-{m1.sent_id}-{m1.start_offset}-{m1.end_offset}"
        m2_id = f"{m2.doc_id}-{m2.sent_id}-{m2.start_offset}-{m2.end_offset}"
        #
        mention1_clustered = m1_id in cur_doc_clustered_mention.keys()
        mention2_clustered = m2_id in cur_doc_clustered_mention.keys()
        #
        if score < threshold:
            if not mention1_clustered:
                cluster_id += 1
                cur_doc_clustered_mention[m1_id] = cluster_id
                cur_doc_clusters[cluster_id] = [m1]
            if not mention2_clustered:
                cluster_id += 1
                cur_doc_clustered_mention[m2_id] = cluster_id
                cur_doc_clusters[cluster_id] = [m2]
        else:
            if mention1_clustered and (not mention2_clustered):
                mention1_cluster_id = cur_doc_clustered_mention[m1_id]
                cur_doc_clustered_mention[m2_id] = mention1_cluster_id
                cur_doc_clusters[mention1_cluster_id].append(m2)
            elif (not mention1_clustered) and mention2_clustered:
                mention2_cluster_id = cur_doc_clustered_mention[m2_id]
                cur_doc_clustered_mention[m1_id] = mention2_cluster_id
                cur_doc_clusters[mention2_cluster_id].append(m1)
            elif (not mention1_clustered) and (not mention2_clustered):
                cluster_id += 1
                cur_doc_clustered_mention[m1_id] = cluster_id
                cur_doc_clustered_mention[m2_id] = cluster_id
                cur_doc_clusters[cluster_id] = [m1, m2]
            elif mention1_clustered and mention2_clustered:
                mention1_cluster_id = cur_doc_clustered_mention[m1_id]
                mention2_cluster_id = cur_doc_clustered_mention[m2_id]
                if mention1_cluster_id == mention2_cluster_id:
                    pass  # m1和m2已经是同一个簇中了，什么都不用做
                else:
                    # 合并两个簇
                    cluster_id += 1
                    new_cluster = cur_doc_clusters[mention1_cluster_id] + cur_doc_clusters[mention2_cluster_id]
                    del cur_doc_clusters[mention1_cluster_id]
                    del cur_doc_clusters[mention2_cluster_id]
                    cur_doc_clusters[cluster_id] = new_cluster
                    for cur_mention in new_cluster:
                        cur_mention_id = f"{cur_mention.doc_id}-{cur_mention.sent_id}-{cur_mention.start_offset}-{cur_mention.end_offset}"
                        cur_doc_clustered_mention[cur_mention_id] = cluster_id
                        del cur_mention
                    del new_cluster
            "End of if else 4种情况"
        del cur_mention_pair, m1, m2, m1_id, m2_id, score, mention1_clustered, mention2_clustered
        "End of if score < 1 ... else"
    "End of while 1(迭代聚类)"
    #
    return cur_doc_clusters, cluster_id


def get_best_threshold_based_on_mp(scores_statistic):
    scores_statistic[0] = sorted(scores_statistic[0])
    scores_statistic[1] = sorted(scores_statistic[1])
    print(f"temp[0]长度：{len(scores_statistic[0])}")
    print(f"temp[1]长度：{len(scores_statistic[1])}")
    #
    def get_error_num(sorted_scores, th):
        error_num = [0, 0]
        for i in sorted_scores[1]:
            if i[0] < th:
                error_num[1] += 1
            else:
                break
        for i in sorted(sorted_scores[0], reverse=True):
            if i[0] >= th:
                error_num[0] += 1
            else:
                break
        return error_num
    #
    max_0 = scores_statistic[0][-1][0]
    min_1 = scores_statistic[1][0][0]
    print(f"max_0 - min_1: {max_0} - {min_1}")
    cross_num = [0, 0]
    cross_num[0] = get_error_num(scores_statistic, min_1)[0]
    cross_num[1] = get_error_num(scores_statistic, max_0)[1]
    print(f"交叉部分共{sum(cross_num)}个：{cross_num}")
    #
    best_log = None
    for i in scores_statistic[1]:
        th = i[0]
        error_num = get_error_num(scores_statistic, th)
        TN = len(scores_statistic[0]) - error_num[0]
        FP = error_num[0]
        FN = error_num[1]
        TP = len(scores_statistic[1]) - error_num[1]
        p = TP/(TP+FP)
        r = TP/(TP+FN)
        f1 = 2*r*p/(p+r)
        error_sum = sum(error_num)
        if best_log is None:
            best_log = {}
            best_log[error_sum] = [(th, error_num, r, p, f1)]
        elif error_sum == list(best_log.keys())[0]:
            best_log[error_sum].append(th)
        elif error_sum < list(best_log.keys())[0]:
            best_log = {}
            best_log[error_sum] = [(th, error_num, r, p, f1)]
    #
    print(f"best_log(error_sum: [(阈值， [FP, FN], r, p, f1)]) = {best_log}")


def wd_clustering_2(prefix, mp_list, statistic_dict_path, threshold=0.02):  # 0.015 0.0078
    """
    给定一个topic下的mention pair list，做wd聚类。

    :param prefix: The prefix of cluster_id. 本函数对簇进行编号：0,1,2,...
      但是不同topic会分别执行本函数，这样cluster_id就重了。所以给cluster_id加一个前缀。
      比如36_ecb这个topic下的cluster 1在加前缀后就变成了3600000001；37_ecbplus这个topic下的cluster 1加前缀后变成了3701000001.这就区分开了。
    :param mp_list: [[mention_obj_1, mention_obj_2, 预测值], ...]预测值是模型预测的结果，Ture是共指，False是不共指，None是没预测出来。
    :param statistic_dict_path: 如果使用统计权重，则从这里读取统计信息。
    :param threshold: 对得分大于等于此值的mp进行聚类。小于此值的mp，即使预测为True，也认为是预测错了，不聚类。
    :return: no return. cd_coref_chain of mention obj in mention_pairs_list is changed.
      This also lead to the change of cd_coref_chain of mention obj in Corpus obj.
    """
    # 1.1. 从mp_list中抽取wd的mention pair并按文件组织。
    wdmp = get_wdmp(mp_list)
    """within doc mention pair"""
    del mp_list
    # 1.2. 读取聚类算法所需权重
    with open(statistic_dict_path, 'rb') as f:
        statistic_dict = cPickle.load(f)
        sd = statistic_dict["36_ecb"]["all"]  # TODO: 现在是写死的，以后得改
        """statistic dict, 存放基于历史数据得出的统计结果"""
    del statistic_dict
    # 1.3. 遍历每一个文档，做文档内聚类
    scores_statistic = [[], []]
    """
    第一个list是真实共指的mp信息组成的list，第二个item是真实不共指的mp信息组成的list::
    
        [
            [  # 真实共指的mp们
                [当前mp的得分, 当前mp的预测, 其他信息],
                [当前mp的得分, 当前mp的预测, 其他信息],
                ...
            ],
            [  # 真实不共指的mp们
                [当前mp的得分, 当前mp的预测, 其他信息],
                [当前mp的得分, 当前mp的预测, 其他信息],
                ...
            ]
        ]
    """
    #
    clusters = {}
    """簇。 cluster_id: [mention1, mention2]"""
    cluster_id = 0
    """cluster id"""
    #
    for cur_doc_id, cur_doc_wdmp in wdmp.items():
        # -1.1. 准备cur_doc_m_list
        cur_doc_m_list = []
        """[m1, m2, ...]"""
        for cur_wdmp in cur_doc_wdmp:
            m1 = cur_wdmp[0]
            m2 = cur_wdmp[1]
            #
            if m1 not in cur_doc_m_list:
                cur_doc_m_list.append(m1)
            if m2 not in cur_doc_m_list:
                cur_doc_m_list.append(m2)
            del m1, m2, cur_wdmp
        # -1.2. 准备cur_doc_pred_coref_dict
        cur_doc_pred_coref_dict = get_coref_dict(cur_doc_wdmp, pred_or_truth="pred")
        """统计当前文档中预测的共指信息
        {
            "mention1_id": {a set of mentions that co-referred with mention1},
            "mention2_id": {a set of mentions that co-referred with mention2},
        }
        """
        # -1.3. 准备cur_doc_truth_coref_dict
        cur_doc_truth_coref_dict = get_coref_dict(cur_doc_wdmp, pred_or_truth="truth")
        """统计当前文档中真实的共指信息
        {
            "mention1_id": {a set of mentions that co-referred with mention1},
            "mention2_id": {a set of mentions that co-referred with mention2},
        }
        """
        # -2. 计算得分
        for cur_wdmp in cur_doc_wdmp:
            # 抽取
            m1 = cur_wdmp[0]
            m2 = cur_wdmp[1]
            m1_id = f"{m1.doc_id}-{m1.sent_id}-{m1.start_offset}-{m1.end_offset}"
            m2_id = f"{m2.doc_id}-{m2.sent_id}-{m2.start_offset}-{m2.end_offset}"
            #
            s, Cab, l0, li_count = get_mp_score_strategy_2(cur_wdmp, cur_doc_m_list, cur_doc_pred_coref_dict, sd=sd)
            #
            scores_statistic[Cab].append([s, l0, li_count, len(cur_doc_truth_coref_dict[m1_id]), len(cur_doc_truth_coref_dict[m2_id])])
            # 记录分儿
            cur_wdmp.append(s)
            #
            del cur_wdmp, m1, m2, m1_id, m2_id, s, Cab, l0, li_count
        # -3. 排序
        sorted_cur_doc_mp = sorted(cur_doc_wdmp, key=lambda x: x[3])
        # -4. 凝聚
        cur_doc_clusters, cluster_id = agglomerate(sorted_cur_doc_mp,  cluster_id=cluster_id, threshold=threshold)
        clusters.update(cur_doc_clusters)
        del cur_doc_clusters
        # -5. 清空过期变量
        del cur_doc_pred_coref_dict, cur_doc_truth_coref_dict
        del cur_doc_m_list
        del sorted_cur_doc_mp
        del cur_doc_id, cur_doc_wdmp
    "End of for cur_doc_id, cur_doc_wdmp in wdmp.items()"
    del cluster_id
    # 1.4. 检测是否wd
    for cur_cluster_id, cur_cluster in clusters.items():
        doc_of_cur_cluster = None
        for cur_mention in cur_cluster:
            if doc_of_cur_cluster is None:
                doc_of_cur_cluster = cur_mention.doc_id
            else:
                if doc_of_cur_cluster != cur_mention.doc_id:
                    raise RuntimeError
        del cur_cluster_id, cur_cluster, doc_of_cur_cluster, cur_mention
    # 1.5. 保存
    for cur_cluster_id, cur_cluster in clusters.items():
        for cur_mention in cur_cluster:
            cur_mention.cd_coref_chain = prefix + cur_cluster_id
    del cur_cluster_id, cur_cluster, cur_mention
    # 1.6
    get_best_threshold_based_on_mp(scores_statistic)


def wd_clustering_3(prefix, mp_list, threshold=0.0):  # 0.015 0.0078
    """
    给定一个topic下的mention pair list，做wd聚类。

    :param prefix: The prefix of cluster_id. 本函数对簇进行编号：0,1,2,...
      但是不同topic会分别执行本函数，这样cluster_id就重了。所以给cluster_id加一个前缀。
      比如36_ecb这个topic下的cluster 1在加前缀后就变成了3600000001；37_ecbplus这个topic下的cluster 1加前缀后变成了3701000001.这就区分开了。
    :param mp_list: [[mention_obj_1, mention_obj_2, 预测值], ...]预测值是模型预测的结果，Ture是共指，False是不共指，None是没预测出来。
    :param statistic_dict_path: 如果使用统计权重，则从这里读取统计信息。
    :param threshold: 对得分大于等于此值的mp进行聚类。小于此值的mp，即使预测为True，也认为是预测错了，不聚类。
    :return: no return. cd_coref_chain of mention obj in mention_pairs_list is changed.
      This also lead to the change of cd_coref_chain of mention obj in Corpus obj.
    """
    # 1.1. 从mp_list中抽取wd的mention pair并按文件组织。
    wdmp = get_wdmp(mp_list)
    """within doc mention pair"""
    del mp_list
    # 1.2. 读取聚类算法所需权重
    pass  # 无需权重
    # 1.3. 遍历每一个文档，做文档内聚类
    scores_statistic = [[], []]
    """
    第一个list是真实共指的mp信息组成的list，第二个item是真实不共指的mp信息组成的list::
    
        [
            [  # 真实共指的mp们
                [当前mp的得分, 当前mp的预测, 其他信息],
                [当前mp的得分, 当前mp的预测, 其他信息],
                ...
            ],
            [  # 真实不共指的mp们
                [当前mp的得分, 当前mp的预测, 其他信息],
                [当前mp的得分, 当前mp的预测, 其他信息],
                ...
            ]
        ]
    """
    #
    clusters = {}
    """簇。 cluster_id: [mention1, mention2]"""
    cluster_id = 0
    """cluster id"""
    #
    for cur_doc_id, cur_doc_wdmp in wdmp.items():
        # -1.1. 准备cur_doc_m_list
        cur_doc_m_list = []
        """[m1, m2, ...]"""
        for cur_wdmp in cur_doc_wdmp:
            m1 = cur_wdmp[0]
            m2 = cur_wdmp[1]
            #
            if m1 not in cur_doc_m_list:
                cur_doc_m_list.append(m1)
            if m2 not in cur_doc_m_list:
                cur_doc_m_list.append(m2)
            del m1, m2, cur_wdmp
        # -1.2. 准备cur_doc_pred_coref_dict
        cur_doc_pred_coref_dict = get_coref_dict(cur_doc_wdmp, pred_or_truth="pred")
        """统计当前文档中预测的共指信息
        {
            "mention1_id": {a set of mentions that co-referred with mention1},
            "mention2_id": {a set of mentions that co-referred with mention2},
        }
        """
        # -1.3. 准备cur_doc_truth_coref_dict
        cur_doc_truth_coref_dict = get_coref_dict(cur_doc_wdmp, pred_or_truth="truth")
        """统计当前文档中真实的共指信息
        {
            "mention1_id": {a set of mentions that co-referred with mention1},
            "mention2_id": {a set of mentions that co-referred with mention2},
        }
        """
        # -2. 计算得分
        for cur_wdmp in cur_doc_wdmp:
            # 抽取
            m1 = cur_wdmp[0]
            m2 = cur_wdmp[1]
            m1_id = f"{m1.doc_id}-{m1.sent_id}-{m1.start_offset}-{m1.end_offset}"
            m2_id = f"{m2.doc_id}-{m2.sent_id}-{m2.start_offset}-{m2.end_offset}"
            #
            s, Cab, l0, li_count = get_mp_score_strategy_3(cur_wdmp, cur_doc_m_list, cur_doc_pred_coref_dict)
            #
            scores_statistic[Cab].append([s, l0, li_count, len(cur_doc_truth_coref_dict[m1_id]), len(cur_doc_truth_coref_dict[m2_id])])
            # 记录分儿
            cur_wdmp.append(s)
            #
            del cur_wdmp, m1, m2, m1_id, m2_id, s, Cab, l0, li_count
        # -3. 排序
        sorted_cur_doc_mp = sorted(cur_doc_wdmp, key=lambda x: x[3])
        # -4. 凝聚
        cur_doc_clusters, cluster_id = agglomerate(sorted_cur_doc_mp,  cluster_id=cluster_id, threshold=threshold)
        clusters.update(cur_doc_clusters)
        del cur_doc_clusters
        # -5. 清空过期变量
        del cur_doc_pred_coref_dict, cur_doc_truth_coref_dict
        del cur_doc_m_list
        del sorted_cur_doc_mp
        del cur_doc_id, cur_doc_wdmp
    "End of for cur_doc_id, cur_doc_wdmp in wdmp.items()"
    del cluster_id
    # 1.4. 检测是否wd
    for cur_cluster_id, cur_cluster in clusters.items():
        doc_of_cur_cluster = None
        for cur_mention in cur_cluster:
            if doc_of_cur_cluster is None:
                doc_of_cur_cluster = cur_mention.doc_id
            else:
                if doc_of_cur_cluster != cur_mention.doc_id:
                    raise RuntimeError
        del cur_cluster_id, cur_cluster, doc_of_cur_cluster, cur_mention
    # 1.5. 保存
    for cur_cluster_id, cur_cluster in clusters.items():
        for cur_mention in cur_cluster:
            cur_mention.cd_coref_chain = prefix + cur_cluster_id
    del cur_cluster_id, cur_cluster, cur_mention
    # 1.6
    get_best_threshold_based_on_mp(scores_statistic)


def wd_clustering(prefix, mention_pairs_list, strategy, statistic_dict_path="", threshold=""):
    """
    给定一个topic下的mention pair list，做wd聚类。

    :param prefix: The prefix of cluster_id. 本函数对簇进行编号：0,1,2,...
      但是不同topic会分别执行本函数，这样cluster_id就重了。所以给cluster_id加一个前缀。
      比如36_ecb这个topic下的cluster 1在加前缀后就变成了3600000001；37_ecbplus这个topic下的cluster 1加前缀后变成了3701000001.这就区分开了。
    :param mention_pairs_list: [[mention_obj_1, mention_obj_2, 预测值], ...]预测值是模型预测的结果，Ture是共指，False是不共指，None是没预测出来。
    :param strategy: 支持多种聚类算法，这是指定使用哪个算法。
    :param statistic_dict_path: 如果使用统计权重聚类算法（strategy=2），则从这里读取统计信息。
    :return: no return. cd_coref_chain of mention obj in mention_pairs_list is changed.
      This also lead to the change of cd_coref_chain of mention obj in Corpus obj.
    """
    if strategy == 1:
        return wd_clustering_1(prefix, mention_pairs_list)
    elif strategy == 2:
        return wd_clustering_2(prefix, mention_pairs_list, statistic_dict_path=statistic_dict_path, threshold=threshold)
    elif strategy == 3:
        return wd_clustering_3(prefix, mention_pairs_list)
    else:
        raise RuntimeError("无效的strategy")


def adapter_of_mention_pairs(mention_pairs):
    """
    输入的mention_pairs长这样：[[mention_obj_1, mention_obj_2, [true_num, valid_num, all_num]],...].
    这里true_num是预测对了几次，valid_num是有几次预测返回了有效结果（也就是yes或no）。

    本函数处理一下，把[true_num， valid_num, all_num]改为预测值：
      * 如果true_num/valid_num大于0.5就算预测对了。
        * 如果mention_obj_1和mention_obj_2真的共指，预测值就是True，否则为False。
        * 反之亦然。
      * 如果true_num/valid_num小于等于0.5就算预测错了。
        * 如果mention_obj_1和mention_obj_2真的共指，预测值就是False，否则为True。
        * 反之亦然。
      * 特殊的，如果valid_num为0，则预测值是None

    处理后的mention_pairs长这样：[[mention_obj_1, mention_obj_2, 预测值],...].

    :param mention_pairs:
    :return: no returen. Parameter mention_pairs is changed.
    """
    for cur_mention_pair in mention_pairs:
        result = cur_mention_pair.pop()
        true_num = result[0]
        valid_num = result[0]
        if valid_num != 0:
            pred_true = True if (true_num / valid_num) > 0.5 else False
        else:
            pred_true = None
        label = (cur_mention_pair[0].gold_tag == cur_mention_pair[1].gold_tag)
        cur_mention_pair.append(label if pred_true else not label)


def remove_unselected_mention(corpus):
    for cur_topic in corpus.topics.values():
        for cur_doc in cur_topic.docs.values():
            for cur_sent in cur_doc.sentences.values():
                if cur_sent.is_selected:
                    pass
                else:
                    cur_sent.gold_entity_mentions = []
                    cur_sent.gold_event_mentions = []


def check_whether_all_mentions_are_clustered(corpus):
    """
    检查是否一个corpus对象中的所有mention都被分到到簇中了。

    检测标准是查看a_mention_obj.cd_coref_chain。
    默认是“-”，如果是“-”就说明这个mention在clustering过程中被漏掉了。

    :param corpus:
    :return: No return. 如果有mention被漏掉了，就print异常。
    """
    for cur_topic_id, cur_topic in corpus.topics.items():
        for cur_doc_id, cur_doc in cur_topic.docs.items():
            for cur_sent_index, cur_sent in cur_doc.sentences.items():
                mentions = []
                mentions += cur_sent.gold_entity_mentions
                mentions += cur_sent.gold_event_mentions
                for cur_mention in mentions:
                    if cur_mention.cd_coref_chain == "-":
                        m_id = f"{cur_mention.doc_id}-{cur_mention.sent_id}-{cur_mention.start_offset}-{cur_mention.end_offset}"
                        print(f"被漏掉的mention：{m_id}")


def main():
    # config
    config_dict = {
        "corpus_and_mention_pairs_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\data\3.pred\10groundtruth\['36_ecb'](strategy3)_ground_truth_model(none)_0shot_t16DAM_noSample(r1).c_mp",
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

    # 读取
    with open(config_dict["corpus_and_mention_pairs_path"], 'rb') as f:
        corpus, mention_pairs = cPickle.load(f)
    # 聚类
    for cur_topic_id, cur_topic_mp in mention_pairs.items():
        adapter_of_mention_pairs(cur_topic_mp)
        cd_clustering(cur_topic_id, cur_topic_mp)
    remove_unselected_mention(corpus)
    check_whether_all_mentions_are_clustered(corpus)
    # 保存
    file_name = os.path.basename(config_dict["corpus_and_mention_pairs_path"])[:-5]
    path = os.path.join(config_dict['output_path'], f"{file_name}.c")
    with open(path, 'wb') as f:
        cPickle.dump(corpus, f)
    print(f"OUTPUT: corpus and clustering result under cur config saved in {path}")
    logging.info(f"OUTPUT: corpus and clustering result under cur config saved in {path}")


if __name__ == '__main__':
    main()
