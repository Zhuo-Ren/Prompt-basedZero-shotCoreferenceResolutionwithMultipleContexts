# 标准库
import re
import os
import shutil
import _pickle as cPickle
# from typing import Dict, List, Tuple, Union
import logging
import time
# 本地库
from classes import Corpus, Topic, Document, Sentence, Token, EventMention, EntityMention, MentionData
from template import templates_list


# config
config_dict = {
    "corpus_and_mention_pairs_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\data\3.pred\10gt\['36_ecb'](strategy3)_ground_truth_model(none)_0shot_t16DAM_noSample(r1).c_mp",
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


with open(config_dict["corpus_and_mention_pairs_path"], 'rb') as f:
    corpus, mention_pairs = cPickle.load(f)

print(1)


def process_pairs(pair_list):
    # Step 1: Create initial groups based on true pairs
    groups = {}
    for student1, student2, same_group in pair_list:
        if same_group:
            if student1 not in groups and student2 not in groups:
                groups[student1] = [student1, student2]
            elif student1 in groups and student2 not in groups:
                groups[student1].append(student2)
            elif student2 in groups and student1 not in groups:
                groups[student2].append(student1)
            elif student1 in groups and student2 in groups:
                group1 = groups[student1]
                group2 = groups[student2]
                if group1 != group2:
                    group1.extend(group2)
                    for student in group2:
                        groups[student] = group1

    # Step 2: Adjust groups based on num_of_true_pair
    new_groups = []
    for group in groups.values():
        num_of_true_pair = [0] * len(group)
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                pair = [group[i], group[j]]
                pair.sort()
                for p in pair_list:
                    if p[:2] == pair:
                        if p[2]:
                            num_of_true_pair[i] += 1
                            num_of_true_pair[j] += 1

        adjusted_group = [student for i, student in enumerate(group) if num_of_true_pair[i] >= len(group) / 2]
        new_groups.append(adjusted_group)

    return new_groups


def clustering(mention_pairs_list):
    # 1. Create a dictionary to represent the coreferred
    coreference_dict = {}
    """
    {
        "mention1": {a set of mentions that co-referred with mention1},
        "mention2": {a set of mentions that co-referred with mention2},
    }
    """
    for a, b, is_coreferred in mention_pairs_list:
        if a != b and is_coreferred:
            if a not in coreference_dict:
                coreference_dict[a] = set()
            coreference_dict[a].add(b)

            if b not in coreference_dict:
                coreference_dict[b] = set()
            coreference_dict[b].add(a)

    # 2. 计算得分
    for cur_mention_pair in mention_pairs_list:
        # 抽取
        mention1 = cur_mention_pair[0]
        mention2 = cur_mention_pair[1]
        is_coreferred = cur_mention_pair[2]
        # 准备
        positive_count = 0
        negtive_count = 0
        # 算分儿
        for cur_mention in coreference_dict.keys():
            if cur_mention in [mention1, mention2]:
                continue
            corefer_with_mention1 = (mention1 in coreference_dict[cur_mention])
            corefer_with_mention2 = (mention2 in coreference_dict[cur_mention])
            if corefer_with_mention1 and corefer_with_mention2:
                positive_count += 1
            elif (not corefer_with_mention1) and (not corefer_with_mention2):
                pass
            else:
                negtive_count += 1
        if mention1 in coreference_dict[mention2]:
            positive_count += 2
        else:
            negtive_count += 2
        # 记录分儿
        cur_mention_pair.append(positive_count - negtive_count)
    # 3. 排序
    sorted_mention_pairs_list = sorted(mention_pairs_list, key=lambda x: x[3])
    # 4.
    clusters = {}
    clustered_mention = {}
    cluster_id = 0
    while 1:
        cur_mention_pair = sorted_mention_pairs_list.pop()
        mention1 = cur_mention_pair[0]
        mention2 = cur_mention_pair[1]
        score = cur_mention_pair[3]
        #
        if score < 1:
            break
        #
        mention1_clustered = mention1 in clustered_mention.keys()
        mention2_clustered = mention2 in clustered_mention.keys()
        if mention1_clustered and (not mention2_clustered):
            mention1_cluster_id = clustered_mention[mention1]
            clustered_mention[mention2] = mention1_cluster_id
            clusters[mention1_cluster_id].append(mention2)
        elif (not mention1_clustered) and mention2_clustered:
            mention2_cluster_id = clustered_mention[mention2]
            clustered_mention[mention1] = mention2_cluster_id
            clusters[mention2_cluster_id].append(mention1)
        elif (not mention1_clustered) and (not mention2_clustered):
            cluster_id += 1
            clustered_mention[mention1] = cluster_id
            clustered_mention[mention2] = cluster_id
            clusters[cluster_id] = [mention1, mention2]
        elif mention1_clustered and mention2_clustered:
            mention1_cluster_id = clustered_mention[mention1]
            mention2_cluster_id = clustered_mention[mention2]
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
                    clustered_mention[cur_mention] = cluster_id
    return 1


pair_list = [
    ["a", "b", True],
    ["a", "c", True],
    ["b", "c", False],
    ["a", "d", False],
    ["a", "e", True],
    ["b", "d", False],
    ["b", "e", True],
    ["c", "d", True],
    ["c", "e", False],
    ["d", "e", False]
]

# result = process_pairs(pair_list)
result = clustering(pair_list)
print(result)
