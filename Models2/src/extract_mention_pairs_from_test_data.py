# 标准库
import os
import _pickle as cPickle
# from typing import Dict, List, Tuple, Union
import logging
# 本地库
from classes import Corpus, Topic, Document, Sentence, Token, EventMention, EntityMention, MentionData


# config
config_dict = {
    "corpus_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\data\read_corpus\test_data",
    "output_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\output",
    "selected_sentences_only": True,
    "strategy": 0,
    "predicted_topics": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\data\document_clustering\predicted_topics"
}
"""
selected_sentences_only: Bool
    Some sentences are selected in ECB+ corpus.
    True: Only mentions in selected sentences are extracted.
    False: All mentions are extracted.
    这个值一直都是True，False的功能就没实现。放一个配置选项在这里，只是为了强调一下。
strategy: Union[0,1,2,3]
    0: All the following strategies.
    1: sentence level: mention pair in a continuous sentence or two are extracted.
    2: wd: mention pair in the same doc are extracted.
    3: cd-golden: mention pair in the same golden topic are extracted.
    4: cd-pred: mention pair in the same predicted topic are extracted. The only difference between golden topics and the predicted topics are:
       predcted:
        ['38_11ecbplus', '38_1ecb', '38_2ecb', '38_3ecb', '38_3ecbplus', '38_4ecb', '38_4ecbplus', '38_7ecbplus', '38_8ecbplus']
        ['38_10ecbplus', '38_1ecbplus', '38_2ecbplus', '38_5ecbplus', '38_6ecbplus', '38_9ecbplus']
       golden:
        ['38_10ecbplus', '38_11ecbplus', '38_1ecbplus', '38_2ecbplus', '38_3ecbplus', '38_4ecbplus', '38_5ecbplus', '38_6ecbplus', '38_7ecbplus', '38_8ecbplus', '38_9ecbplus']
        ['38_1ecb', '38_2ecb', '38_3ecb', '38_4ecb'] 
predicted_topics: str
    if strategy is 4, this is the path to predicted topics.
"""

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


def strategy_1(corpus):
    mention_pairs = {}
    num_of_mention_pairs = 0
    for topic_id in corpus.topics.keys():
        cur_topic = corpus.topics[topic_id]
        topic_num = int(topic_id.split("_")[0])  # 36_ecb and 36_ecbplus all mapped to the same topic num 36
        if topic_num not in mention_pairs.keys():
            mention_pairs[topic_num] = {}
        for doc_id in cur_topic.docs.keys():
            cur_doc = cur_topic.docs[doc_id]
            if doc_id not in mention_pairs[topic_num].keys():
                mention_pairs[topic_num][doc_id] = []
            # 上一个句子中的mention
            mentions_in_last_sent = []
            for sent_id in cur_doc.sentences.keys():
                cur_sent = cur_doc.sentences[sent_id]
                # 当前句子的mention
                mentions_in_cur_sent = []
                if cur_sent.is_selected:
                    mentions_in_cur_sent += cur_sent.gold_entity_mentions
                    mentions_in_cur_sent += cur_sent.gold_entity_mentions
                # 生成mention pair： 当前句子和上一个句子之间的mention pair
                for mention_i in mentions_in_last_sent:
                    for mention_j in mentions_in_cur_sent:
                        mention_pairs[topic_num][doc_id].append([mention_i, mention_j])
                # 生成mention pair： 当前句子中的mention pair
                for i in range(len(mentions_in_cur_sent)):
                    for j in range(len(mentions_in_cur_sent)):
                        if i < j:
                            mention_i = mentions_in_cur_sent[i]
                            mention_j = mentions_in_cur_sent[j]
                            mention_pairs[topic_num][doc_id].append([mention_i, mention_j])
                # 更新
                mentions_in_last_sent = mentions_in_cur_sent
        # END OF  for doc_id in cur_topic.docs.keys():
        num_of_mention_pairs_in_cur_topic = sum([len(d) for d in mention_pairs[topic_num].values()])
        logging.info(f'strategy 1: topic {topic_id} has {num_of_mention_pairs_in_cur_topic} mention pairs')
        num_of_mention_pairs += num_of_mention_pairs_in_cur_topic
    # END OF for topic_id in corpus.topics.keys():
    logging.info(f'strategy 1: {num_of_mention_pairs} mention pairs')
    print(f'strategy 1: {num_of_mention_pairs} mention pairs')

    # save
    path = os.path.join(config_dict["output_path"], 'strategy_1_corpus_and_mention_pairs')
    with open(path, 'wb') as f:
        cPickle.dump((corpus, mention_pairs), f)
        print(f'strategy 1: (corpus, mention_pairs) saved in {path}')
    return corpus, mention_pairs


def strategy_2(corpus):
    mention_pairs = {}
    num_of_mention_pairs = 0
    for topic_id in corpus.topics.keys():
        cur_topic = corpus.topics[topic_id]
        topic_num = int(topic_id.split("_")[0])  # 36_ecb and 36_ecbplus all mapped to the same topic num 36
        if topic_num not in mention_pairs.keys():
            mention_pairs[topic_num] = {}
        for doc_id in cur_topic.docs.keys():
            cur_doc = cur_topic.docs[doc_id]
            if doc_id not in mention_pairs[topic_num].keys():
                mention_pairs[topic_num][doc_id] = []
            mentions_in_cur_doc = []
            for sent_id in cur_doc.sentences.keys():
                cur_sent = cur_doc.sentences[sent_id]
                if cur_sent.is_selected:
                    mentions_in_cur_doc += cur_sent.gold_entity_mentions
                    mentions_in_cur_doc += cur_sent.gold_entity_mentions
            # 生成mention pair： 当前句子中的mention pair
            for i in range(len(mentions_in_cur_doc)):
                for j in range(len(mentions_in_cur_doc)):
                    if i < j:
                        mention_i = mentions_in_cur_doc[i]
                        mention_j = mentions_in_cur_doc[j]
                        mention_pairs[topic_num][doc_id].append([mention_i, mention_j])
        # END OF  for doc_id in cur_topic.docs.keys():
        num_of_mention_pairs_in_cur_topic = sum([len(d) for d in mention_pairs[topic_num].values()])
        logging.info(f'strategy 2: topic {topic_id} has {num_of_mention_pairs_in_cur_topic} mention pairs')
        num_of_mention_pairs += num_of_mention_pairs_in_cur_topic
    # END OF for topic_id in corpus.topics.keys():
    logging.info(f'strategy 2: {num_of_mention_pairs} mention pairs')
    print(f'strategy 2: {num_of_mention_pairs} mention pairs')

    # save
    path = os.path.join(config_dict["output_path"], 'strategy_2_corpus_and_mention_pairs')
    with open(path, 'wb') as f:
        cPickle.dump((corpus, mention_pairs), f)
        print(f'strategy 2: (corpus, mention_pairs) saved in {path}')
    return corpus, mention_pairs


def strategy_3(corpus):
    mention_pairs = {}
    for topic_id in corpus.topics.keys():
        cur_topic = corpus.topics[topic_id]  # 36_ecb and 36_ecbplus are different topics
        mentions_in_cur_topic = []
        for doc_id in cur_topic.docs.keys():
            cur_doc = cur_topic.docs[doc_id]
            for sent_id in cur_doc.sentences.keys():
                cur_sent = cur_doc.sentences[sent_id]
                if cur_sent.is_selected:
                    mentions_in_cur_topic += cur_sent.gold_entity_mentions
                    mentions_in_cur_topic += cur_sent.gold_event_mentions
        if topic_id not in mention_pairs.keys():
            mention_pairs[topic_id] = []
        for i in range(len(mentions_in_cur_topic)):
            for j in range(len(mentions_in_cur_topic)):
                if i < j:
                    mention_i = mentions_in_cur_topic[i]
                    mention_j = mentions_in_cur_topic[j]
                    mention_pairs[topic_id].append([mention_i, mention_j])
        logging.info(f'strategy 3: topic {topic_id} has {len(mention_pairs[topic_id])} mention pairs')
    # END OF for topic_id in corpus.topics.keys():
    logging.info(f'strategy 3: {sum([len(mention_pairs[cur_topic_num]) for cur_topic_num in mention_pairs.keys()])} mention pairs')
    print(f'strategy 3: {sum([len(mention_pairs[cur_topic_num]) for cur_topic_num in mention_pairs.keys()])} mention pairs')

    # save
    path = os.path.join(config_dict["output_path"], 'strategy_3_corpus_and_mention_pairs')
    with open(path, 'wb') as f:
        cPickle.dump((corpus, mention_pairs), f)
        print(f'strategy 3: (corpus, mention_pairs) saved in {path}')
    return corpus, mention_pairs


# def strategy_4(corpus, predicted_topics):
#     for topic_id in corpus.topics.keys():
#         cur_topic = corpus.topics[topic_id]
#         mentions_in_cur_topic = []
#         for doc_id in cur_topic.docs.keys():
#             cur_doc = cur_topic.docs[doc_id]
#             for sent_id in cur_doc.sentences.keys():
#                 cur_sent = cur_doc.sentences[sent_id]
#     t1 = test_set.topics
#     t2 = topics
#     s1 = set()
#     s2 = set()
#     for cur_t in t1.values():
#         s1.add(str(
#             sorted(cur_t.docs.keys())
#         ))
#     for cur_t in t2.values():
#         s2.add(str(
#             sorted(cur_t.docs.keys())
#         ))
#     cross = (s1 & s2)
#     r1 = s1 - cross
#     r2 = s2 - cross


def main():
    # read corpus
    with open(config_dict["corpus_path"], 'rb') as f:
        corpus = cPickle.load(f)

    # read predicted topics (if using strategy 4)
    if config_dict["strategy"] in [0, 4]:
        logging.info(f"load predicted topic from {config_dict['predicted_topics']}")
        with open(config_dict["predicted_topics"], 'rb') as f:
            predicted_topics = cPickle.load(f)
            ''' 外部算法预测的文档聚类结果
                predicted_topics = [
                    ['45_6ecb', '45_8ecb'],  # 这是一个簇
                    ['43_6ecb',43_4ecb]，  # 这是一个簇
                    ...
                ]
            '''

    #
    if config_dict["strategy"] == 0:
        ml1 = strategy_1(corpus=corpus)
        ml2 = strategy_2(corpus=corpus)
        ml3 = strategy_3(corpus=corpus)
        # strategy_4(corpus=corpus, predicted_topics=predicted_topics)
    elif config_dict["strategy"] == 1:
        c, ml1 = strategy_1(corpus=corpus)
    elif config_dict["strategy"] == 2:
        c, ml2 = strategy_2(corpus=corpus)
    elif config_dict["strategy"] == 3:
        c, ml3 = strategy_3(corpus=corpus)
    # elif config_dict["strategy"] == 4:
    #     strategy_4(corpus=corpus, predicted_topics=predicted_topics)
    print("END")


if __name__ == '__main__':
    main()
