# python -m spacy download en_core_web_sm
# pip install transformers
# pip install datasets
import warnings
import json
from typing import List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import math
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import logging, get_linear_schedule_with_warmup, set_seed
from transformers import pipeline
from transformers import GPT2TokenizerFast, GPT2Model, GPT2LMHeadModel
from transformers import AutoTokenizer, AutoModel
import spacy
from spacy.matcher import Matcher
from spacy.util import filter_spans
from spacy.symbols import ORTH
from spacy.tokenizer import Tokenizer
import random
import pickle
from datasets import load_dataset
import time
import csv


# 配置地址========================================================================
local_path = "/root/WhatGPTKnowsAboutWhoIsWho-main"
'local_path = "D:/WhatGPTKnowsAboutWhoIsWho-main"'
root_path = local_path

# 配置n-shot
n_examples = 4  # n-shot。就是从下边的simple_examples中选前n个用于生成prefix。也就是prefix中包含n个example，也就是n-shot。

# 配置答案选项
answer_choices = ["No", "Yes"]

# 配置模板（模板指的是如何把样例转换为问答的方式，用于基于样例生成prefixes和prompts）
templates = [
    "'[TEXT]' In previous sentences, does '[MENTION2]' refer to '[MENTION1]'? Yes or no?",
    "'[TEXT]' Here, by '[MENTION2]' they mean '[MENTION1]'? Yes or no?",
    "'[TEXT]' Here, does '[MENTION2]' stand for '[MENTION1]'? Yes or no? ",
    "'[TEXT]' In the passage above, can '[MENTION2]' be replaced by '[MENTION1]'? Yes or no?",
    "'[TEXT]' I think '[MENTION2]' means '[MENTION1]'. Yes or no?"
]

# 配置模型
'model_name = "GPT2"'
model_name = "ChatGLM-6B"


# 配置数据
"""这一部分配置在后边"""
data_config = "dev"  # "dev"完整的DevSet "first two docs"只用头两个文档做实现 "a given doc"只针对给定文档做实现
given_doc_name = "2_1ecbplus.xml"  # 当data_config为"a given doc"时，这个变量给出指定文档的名字

#
warnings.filterwarnings('ignore')
# 调整huggingface transformer的logging等级
logging.set_verbosity_error()  # 运行时打开，调试时请注释掉
logging.disable_progress_bar()

#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
#
SPACY_NLP = spacy.load("en_core_web_sm")
SPACY_NLP.tokenizer = Tokenizer(SPACY_NLP.vocab)  # To tokenize just by space
#
log_file = open(local_path + "/Models/log.txt", mode="w", encoding="utf8")


class docDataset:
    def __init__(self, doc_name, text, vocab, clusters=None, gold_mentions=None):
        self.text = text
        self.sentences = text.split("[EOS]")
        self.doc_name = doc_name
        self.gold_mentions = gold_mentions
        """所有entity mention"""
        self.clusters = clusters
        """所有entity cluster"""
        self.window_size = 2
        """构建mention对时，只考虑连续window_size个句子中的mention"""
        self.cluster_map = {}
        """{t_id: 所述cluster的cluster_id} """
        self.vocab = vocab
        self.mentions = {}
        self.context_sents = np.array([[-1, -1]])
        """
        [
            [-1, -1],
            [start_sent_id, end_sent_id],  # 每行对应一个mention 对
            [start_sent_id, end_sent_id],
            ...
        ]
        """
        self.pairs = np.array([[-1, -1]])
        """
        [
            [-1, -1],
            [mention1, mention2],  # 每行对应一个mention对
            [mention1, mention2],
            ...
        ]
        """
        self.labels = np.array([])
        """[1, 0, 0, 1,...] # 没位对应一个mention对的真实共指标签，1是共指，0是不共指。"""
        self.create_cluster_map()
        # self.create_vocab()
        self.create_mention()

    def create_cluster_map(self):
        for id in self.clusters:
            tokens_in_cluster = self.clusters[id]
            for token in tokens_in_cluster:
                if token in self.cluster_map:
                    self.cluster_map[token].add(id)
                else:
                    self.cluster_map[token] = {id}

    def create_mention(self):
        for i in range(len(self.sentences)):
            self.mentions[i] = []

        if self.gold_mentions:
            for m_id in self.gold_mentions:
                mention_info = self.gold_mentions[m_id]
                mention_token_ids = mention_info["tokens_ids"]
                mention_text = " ".join(mention_info["tokens"])
                sent_id = int(mention_info["sentence_id"])  # TODO: need to change for new index, as the original xml starts from 0

                annotation = {"mention": mention_text,
                              "start_token_id": mention_token_ids[0],
                              "end_token_id": mention_token_ids[-1]}
                self.mentions[sent_id].append(annotation)
        else:
            raise Exception("TODO: need to add mentions if there is no gold data")

    def decode_mention(self, mention):
        """获取mention的token id list。

        比如一个mention的start_token_id是3，end_token_id是6，那么就返回[3,4,5,6]

        :param mention: 一个mention信息，至少要含有start_token_id和end_token_id这两个属性。
        :return: mention的token id list
        """
        start_token_id = mention['start_token_id']
        end_token_id = mention['end_token_id']
        decoded_mention = list(range(start_token_id, end_token_id + 1))

        return decoded_mention

    def label_pairs(self, mention1, mention2):
        """ 给定俩mention，输出他们是否共指的真实标签（不是预测的标签）。

        :param mention1: 第一个mention的t_id list
        :param mention2: 第二个mention的t_id list
        :return: 1(共指) or 0(不共指)
        """
        if not self.cluster_map:
            raise Exception("No Label Data")

        cluster1 = []
        cluster2 = []

        for t_id in mention1:
            if t_id in self.cluster_map:
                cluster1.append(self.cluster_map[t_id])

        for t_id in mention2:
            if t_id in self.cluster_map:
                cluster2.append(self.cluster_map[t_id])

        s1 = set().union(*cluster1)
        s2 = set().union(*cluster2)
        if len(s1.intersection(s2)) > 0:
            return 1
        return 0

    def create_mention_pairs(self):
        """抽取当前doc中的所有相距不超过windw_size个句子的mention对及其真实共指标签。

        就是说如果当前doc中的两个mention，
        它们处在在同一个句子中，或连续的window_size个句子中，
        就抽取出来，存放在self的几个属性中。

        :return: 无。
        抽取出来的mention对存放在self。pairs，
        对应的lable存放在self.labels,
        对应的句子id存放在self.context_sents.
        """
        n = len(self.sentences)
        if n == 1:
            self.window_size = 1
            self.mention_pairs_helper(0)
        else:
            for i in range(n - self.window_size):
                self.mention_pairs_helper(i)
        self.pairs = self.pairs[1:, :]
        self.context_sents = self.context_sents[1:, :]

    def mention_pairs_helper(self, start_idx=1):
        """给定起点句，由连续的n个句子组成窗口，获取窗口内所有可能的指称对及其真实共指标签。

        1.当前文档中有n句话：sent_0，sent_1，...
        从中选择连续的self.window_size句话，以sent_i开始，i=start_idx-1。
        例如start_idx=3， window_size=4，那么就选择了sent_2,sent_3,sent_4,sent_5。

        2. 把选中连续句子中的entity mention都取出来，然后构建所有可能的mention对。

        3. 对每个mention对，
           ①把mention对保存得到self.pairs，
           ②把mention对是否共指的真实标签保存到self.labels。
           ③把mention对所在句子的start_sent_id和end_sent_id保存到self.context_sents。

        :param start_idx:
        :return:
        """
        # 获取窗口内的所有mention
        sent_idxs = range(start_idx, start_idx + self.window_size)
        sents_mentions = []
        for i in sent_idxs:
            sents_mentions += self.mentions[i]

        # 遍历窗口内的所有mention对
        for i in range(len(sents_mentions) - 1):
            for j in range(i + 1, len(sents_mentions)):
                """
                遍历所有可能的mention对（不分前后）
                """
                # 记录当前mention对
                mention1, mention2 = (sents_mentions[i], sents_mentions[j])
                self.pairs = np.append(self.pairs, [(mention1, mention2)], axis=0)
                """
                self.pairs = numpyArray[
                    [-1, -1],
                    [m1, m2],  # 一个mention对
                    [m1, m2],  # 有一个mention对
                    ...        # 每次append一个mention对
                ]
                """

                # 记录当前mention对的真实标签
                if self.cluster_map:
                    # 获取mention1和mention2的t_id list
                    decoded_mention1 = self.decode_mention(mention1)
                    decoded_mention2 = self.decode_mention(mention2)
                    # 获取mention1或mention2的真实标签（1：共指，0：共指）
                    label = self.label_pairs(decoded_mention1, decoded_mention2)
                    # 记录当前mention对的真实标签
                    self.labels = np.append(self.labels, label)
                    """
                    self.labels = numpyArray[1, 0, 0, 1,...每次的新标签append到后边]
                    """

                self.context_sents = np.append(self.context_sents, [(start_idx, start_idx + self.window_size - 1)], axis=0)
            # end for
        # end for

    def get_experiment_samples(self):
        """基于每个mention对，生成samples。

        e.g.::

            samples = [
                [
                    [0 1], # mention对所在句子的sent_id
                    'http://www.hindustantimes.com/audio-news-video/AV-NewsX/INS-Sukanya-intercepts-pirate-ships-arms-seized/Article2-768142. aspx  INS Sukanya intercepts pirate ships, arms seized ',
                    [  # tow mention
                        {'mention': 'INS Sukanya', 'start_token_id': 4, 'end_token_id': 5},
                        {'mention': 'ships', 'start_token_id': 8, 'end_token_id': 8}
                    ],
                    0.0  # 指称对共指的真实标签
                ]
            ]

        :return: self.pairs中存储的指称对一一对应为samples格式。
        """
        samples = []
        # for i in tqdm(range(len(self.pairs)), desc="sample", position=1, leave=False, ncols=50):
        for i in range(len(self.pairs)):
            text = self.extract_sents_text(self.context_sents[i])
            pair = self.pairs[i]
            label = self.labels[i]
            samples.append([self.context_sents[i], text, pair, label])
            # time.sleep(0.1)
        return samples

    def extract_sents_text(self, sent_ids):
        sents = ""
        sent_ids = list(set(sent_ids))
        for i in sent_ids:
            sents += self.sentences[i]
        return sents
# end Class


def load_gold_data(local_path):
    root_path = local_path
    # Path to the ecb data
    dir_path = f"{root_path}/Data/ECB+/"
    print(dir_path)
    # load train
    file_path = dir_path + "processed/train_with_new_index.json"
    with open(file_path) as f:
        train = json.load(f)
    print(len(train))
    # load dev
    file_path = dir_path + "processed/dev_with_new_index.json"
    with open(file_path) as f:
        dev = json.load(f)
    print(len(dev))
    #
    return train, dev


# Generate Prompt
def generate_prompt(template, text, mention_pair, text_token="[TEXT]", mention1_token="[MENTION1]", mention2_token="[MENTION2]"):
    template = template.replace(text_token, text)
    template = template.replace(mention1_token, mention_pair[0])
    template = template.replace(mention2_token, mention_pair[1])
    return template


def create_prefix(examples, template, answer_choices):
    """
    基于给定template，把一组例子转换为一组prefix。
    各个例子用列表分开，对应的各个prefix用\\n分开。

    e.g.::

        example = [
            ['Anna told her friends that she was about to go to college.', ['Anna', 'she'], 1],
            ["Eva and Martha didn't want their friend Jenny to feel lonely so they invited her to the party", ['Eva', 'her'], 0]
        ]
        template =  "'[TEXT]' In previous sentences, does '[MENTION2]' refer to '[MENTION1]'? Yes or no?"
        answer_choices = ["No", "Yes"]
        r = create_prefix(example, template, answer_choices)
        r = r"'Anna told her friends that she was about to go to college.' In previous sentences, does 'she' refer to 'Anna'? Yes or no? Yes\\n'Eva and Martha didn't want their friend Jenny to feel lonely so they invited her to the party' In previous sentences, does 'her' refer to 'Eva'? Yes or no? No"


    :param examples: 用于生成prefix的例子，一组n个。
    :param template: 生成prefix的模板template。
    :param answer_choices: 答案的选项。
    :return: 生成的prefix。
    """
    prefix = ""
    for (text, mention_pair, label) in examples:
        label_text = answer_choices[int(label)]
        prefix += generate_prompt(template, text, mention_pair) + " " + label_text + "\n"
    return prefix


#
def parse_superglue(data):
    texts = data["text"]
    pairs = list(zip(data["span1_text"], data["span2_text"]))
    labels = data["label"]
    return texts, pairs, labels


def get_examples_superglue(n, texts, pairs, labels,):
    '''Return examples for prefix.
            Parameters:
                    n (int): total number of examples, expected to be an even number
                    texts (list): list of text
                    pairs (list): list of mention pairs
                    labels (list): list of labels

            Returns:
                    examples (list): list of examples
    '''
    # we want a balanced example set
    n_positives = n_negatives = n // 2
    i = 0
    examples = []
    while (n_positives > 0) or (n_negatives > 0):
        text = texts[i]
        mention_pair = pairs[i]
        label = labels[i]
        if (label == 1) and (n_positives > 0):
            examples.append([text, mention_pair, label])
            n_positives -= 1

        if (label == 0) and (n_negatives > 0):
            examples.append([text, mention_pair, label])
            n_negatives -= 1
        i += 1
    return examples


def extract_sents_text(doc, sent_ids):
    sents = ""
    for i in sent_ids:
        sents += doc.sentences[i]  # + " "
    return sents


def get_example_info(doc, filter, idx):
    text_ids = doc.context_sents[filter]
    text = extract_sents_text(doc, text_ids[idx])
    m1, m2 = doc.pairs[filter][idx]
    mention_pair = [m1["mention"], m2["mention"]]
    label = doc.labels[filter][idx]
    return text, mention_pair, label


def get_examples_ecb(n, train, train_docs):
    # train: 训练集。它是一个dict
    #   key:文件名，例如'20_10ecbplus.xml'
    #   value: 是一个list
    #     [0]: 原文，例如http://secretdubai. blogspot. nl/2005/11/did-earth-move-for-you. html [EOS] 27 NOVEMBER, 2005 [EOS] Did the earth move for you? [EOS] Because it didn't down here at Cell Block G. Dubai experienced a slight "tremor" today, after a more serious earthquake in Southern Iran, resulting in the evacuation of Emirates Towers and a few other scrapers: [EOS]
    #     [1]: 分词列表
    # train_docs: 训练集所有key的列表，即list[train.keys()]
    examples = []
    pos_n = neg_n = n // 2
    for i in range(len(train_docs)):
        if pos_n <= 0:
            return examples
        doc_name = train_docs[i]
        text, toks, mentions, clusters = train[doc_name]
        sample = docDataset(doc_name, text, toks, clusters, mentions)
        sample.create_mention_pairs()

        pos_filter = (sample.labels == 1)
        neg_filter = (sample.labels == 0)

        if np.sum(pos_filter) == 0:
            print("%s has no positive examples" % (doc_name))
            continue
        else:
            pos_text, pos_mention_pair, pos_label = get_example_info(sample, pos_filter, 0)
            examples.append([pos_text, pos_mention_pair, pos_label])

            neg_text, neg_mention_pair, neg_label = get_example_info(sample, neg_filter, 0)
            examples.append([neg_text, neg_mention_pair, neg_label])
            pos_n -= 1


def gpt2_create(prefix):
    ppl = pipeline('text-generation', model='gpt2', return_full_text=False, prefix=prefix, device=torch.cuda.current_device())
    return ppl


def gpt2_pred(ppl, prompt):
    pred = ppl(prompt, max_new_tokens=1, num_return_sequences=1)[0]["generated_text"]
    return pred


def chatglm6b_create(prefix):
    tokenizer = AutoTokenizer.from_pretrained(
        "/root/autodl-tmp/chatglm-6b",
        trust_remote_code=True,
        revision=""
    )
    model = AutoModel.from_pretrained(
        "/root/autodl-tmp/chatglm-6b",
        trust_remote_code=True,
        prefix=prefix,
        revision=""
    ).half().cuda()
    return tokenizer, model


def chatglm6b_pred(tokenizer, model, prompt):
    """pred, history = model.chat(tokenizer, prompt)"""
    inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    outputs = model.generate(inputs, max_new_tokens=1, do_sample=False, num_beams=8)
    outputs = outputs.tolist()[0]
    outputs = outputs[len(inputs[0]):]
    pred = tokenizer.decode(outputs, skip_special_tokens=True)
    return pred


def annotate(data, templates, prefixes, prefixes_source, repeated_n, model_name):
    """ 在data数据上，对generators中的几个模型分别进行repeated_n次测试。结果保存到文件。

    e.g.::

        data = {
            "12_10ecb": [
                # clean_text
                "The Indian navy has captured 23 Somalian pirates. [EOS] The pirates were about to board a vessel in the Gulf of Aden when they were apprehended. [EOS] Last month, India's navy drew criticism after sinking a Thai fishing trawler that had been commandeered hours earlier by pirates, but the navy says it fired in self-defence. [EOS] Somali pirates have become increasingly brazen, and recently seized a Saudi supertanker loaded with e75m worth of crude oil.[EOS]",
                # clean_token['The', 'Indian', 'navy', 'has', 'captured', '23', 'Somalian', 'pirates', '.', ' ', 'The', 'pirates', 'were', 'about', 'to', 'board', 'a', 'vessel', 'in', 'the', 'Gulf', 'of', 'Aden', 'when', 'they', 'were', 'apprehended', '.', ' ', 'Last', 'month', ',', 'India', "'s", 'navy', 'drew', 'criticism', 'after', 'sinking', 'a', 'Thai', 'fishing', 'trawler', 'that', 'had', 'been', 'commandeered', 'hours', 'earlier', 'by', 'pirates', ',', 'but', 'the', 'navy', 'says', 'it', 'fired', 'in', 'self', '-', 'defence', '.', ' ', 'Somali', 'pirates', 'have', 'become', 'increasingly', 'brazen', ',', 'and', 'recently', 'seized', 'a', 'Saudi', 'supertanker', 'loaded', 'with', 'e75', 'm', 'worth', 'of', 'crude', 'oil', '.']
            entity_mentions = { # 注意，仅entity mention
                0: {
                    'doc_id': '12_10ecb.xml', 'topic': 12,
                    'subtopic': '12_1',
                    'm_id': '4', 'sentence_id': 0,
                    'tokens_ids': [2],  # 这里的id等于语料库中的t_id减一
                    'tokens': ['navy'], 'tags': 'NNP', 'lemmas': 'navy',
                    'cluster_id': 17403785688719729,
                    # cluster_id是一个簇的id，有三种情况：
                    # 1. 对跨文档共指的簇，存在<CROSS_DOC_COREF r_id="34175" note="HUM17403785688719729" >这样的标签，cluster_id就是17403785688719729
                    # 2. 对文档内共指的簇，
                    # 3. 对孤立簇，自己编码一个id
                    'cluster_desc': 't12b_navy'
                },
                1: {
                    'doc_id': '12_10ecb.xml', 'topic': 12,
                    'subtopic': '12_1',
                    'm_id': '6', 'sentence_id': 0,
                    'tokens_ids': [7],
                    'tokens': ['pirates'], 'tags': 'NNS', 'lemmas': 'pirate',
                    'cluster_id': 17403498594960941, 'cluster_desc': 't12b_pirates'
                }
            }
            cluster = { # 注意，仅entity cluster
                # key是上边提到的cluster_id，value是上边提到的tokens_ids
                17403498594960941: [7],
                17403785688719729: [2]
            }
            ],
        }
        templates = [ # 5个模板
            "'[TEXT]' In previous sentences, does '[MENTION2]' refer to '[MENTION1]'? Yes or no?",
            "'[TEXT]' Here, by '[MENTION2]' they mean '[MENTION1]'? Yes or no?",
            "'[TEXT]' Here, does '[MENTION2]' stand for '[MENTION1]'? Yes or no? ",
            "'[TEXT]' In the passage above, can '[MENTION2]' be replaced by '[MENTION1]'? Yes or no?",
            "'[TEXT]' I think '[MENTION2]' means '[MENTION1]'. Yes or no?"
        ]
        prefixes = [
            '\'Perennial party girl Tara Reid checked herself into Promises Treatment Center, her rep told People.  "We appreciate your respect to her and her family\'s privacy at this time, " the 33-year-old actress\'s rep Jack Ketsoyan told the magazine exclusively for their Web site. \' In previous sentences, does \'herself\' refer to \'Tara Reid\'? Yes or no? Yes\n\'Perennial party girl Tara Reid checked herself into Promises Treatment Center, her rep told People.  "We appreciate your respect to her and her family\'s privacy at this time, " the 33-year-old actress\'s rep Jack Ketsoyan told the magazine exclusively for their Web site. \' In previous sentences, does \'Promises Treatment Center\' refer to \'Tara Reid\'? Yes or no? No\n',
            '\'Perennial party girl Tara Reid checked herself into Promises Treatment Center, her rep told People.  "We appreciate your respect to her and her family\'s privacy at this time, " the 33-year-old actress\'s rep Jack Ketsoyan told the magazine exclusively for their Web site. \' Here, by \'herself\' they mean \'Tara Reid\'? Yes or no? Yes\n\'Perennial party girl Tara Reid checked herself into Promises Treatment Center, her rep told People.  "We appreciate your respect to her and her family\'s privacy at this time, " the 33-year-old actress\'s rep Jack Ketsoyan told the magazine exclusively for their Web site. \' Here, by \'Promises Treatment Center\' they mean \'Tara Reid\'? Yes or no? No\n',
            '\'Perennial party girl Tara Reid checked herself into Promises Treatment Center, her rep told People.  "We appreciate your respect to her and her family\'s privacy at this time, " the 33-year-old actress\'s rep Jack Ketsoyan told the magazine exclusively for their Web site. \' Here, does \'herself\' stand for \'Tara Reid\'? Yes or no?  Yes\n\'Perennial party girl Tara Reid checked herself into Promises Treatment Center, her rep told People.  "We appreciate your respect to her and her family\'s privacy at this time, " the 33-year-old actress\'s rep Jack Ketsoyan told the magazine exclusively for their Web site. \' Here, does \'Promises Treatment Center\' stand for \'Tara Reid\'? Yes or no?  No\n',
            '\'Perennial party girl Tara Reid checked herself into Promises Treatment Center, her rep told People.  "We appreciate your respect to her and her family\'s privacy at this time, " the 33-year-old actress\'s rep Jack Ketsoyan told the magazine exclusively for their Web site. \' In the passage above, can \'herself\' be replaced by \'Tara Reid\'? Yes or no? Yes\n\'Perennial party girl Tara Reid checked herself into Promises Treatment Center, her rep told People.  "We appreciate your respect to her and her family\'s privacy at this time, " the 33-year-old actress\'s rep Jack Ketsoyan told the magazine exclusively for their Web site. \' In the passage above, can \'Promises Treatment Center\' be replaced by \'Tara Reid\'? Yes or no? No\n',
            '\'Perennial party girl Tara Reid checked herself into Promises Treatment Center, her rep told People.  "We appreciate your respect to her and her family\'s privacy at this time, " the 33-year-old actress\'s rep Jack Ketsoyan told the magazine exclusively for their Web site. \' I think \'herself\' means \'Tara Reid\'. Yes or no? Yes\n\'Perennial party girl Tara Reid checked herself into Promises Treatment Center, her rep told People.  "We appreciate your respect to her and her family\'s privacy at this time, " the 33-year-old actress\'s rep Jack Ketsoyan told the magazine exclusively for their Web site. \' I think \'Promises Treatment Center\' means \'Tara Reid\'. Yes or no? No\n'
        ]
        prefixes_source = "ecb"
        repeated_n = 5  # 模型对一个样例重复试验5次
        model_name = "GPT2"


    :param data: 在这个数据上做测试。格式参见[这里](cl://20230601153023/)。
    :param templates: 几个不同的模板
    :param prefixes: 多个前缀，每个模板对应一个前缀
    :param prefixes_source: 前缀是基于什么数据生成的（不影响实现，仅用于log）
    :param repeated_n: 对每个模型重复实现几次
    :param model_name: 模型类型（CharGLM、GPT2等）
    :return: 无。测试结果保存的文件。
    """
    prompt_column_names = [f"Prompt {i + 1}" for i in range(len(templates))]
    start = time.time()
    result = []
    # 遍历每种模板
    for i, template in enumerate(tqdm(templates, desc="模板", position=0, leave=False, ncols=60)):
        result_index = 0  # 当前要写入result的哪一行
        prefix = prefixes[i]
        # 构建模型=================================================================
        if model_name == "GPT2":
            ppl = gpt2_create(prefix)
        elif model_name == "ChatGLM-6B":
            tokenizer, model = chatglm6b_create(prefix)
        # 遍历每个文件
        for doc_name in tqdm(data, desc="文档", position=1, leave=False, ncols=60):
            if i == 0:
                result.append(prompt_column_names + ["doc_name", "sent_idx", "text", "mention pair", "label"])
            result_index += 1
            # 基于当前文件创建docDataset对象（因为这个对象中实现了许多方法）
            text, tokens, mentions, clusters = data[doc_name]
            doc = docDataset(doc_name, text, tokens, clusters, mentions)
            #
            doc.create_mention_pairs()
            samples = doc.get_experiment_samples()
            log_file.write(f"Doc:{doc_name}=============================\n")
            # 遍历每个样本
            results = []
            for s in tqdm(range(len(samples)), desc="样本", position=2, leave=False, ncols=60):
                # 基于当前template构建当前sample的prompt
                _, text, pair, _ = samples[s]
                mention_pair = [pair[0]["mention"], pair[1]["mention"]]
                prompt = generate_prompt(template, text, mention_pair)
                # 预测n次
                yes_count = 0
                valid_count = 0
                # for j in tqdm(range(repeated_n), desc="重复", position=3, leave=False):
                for j in range(repeated_n):
                    # 使用模型预测当前sample===========================================
                    if model_name == "GPT2":
                        pred = gpt2_pred(ppl, prompt)
                    elif model_name == "ChatGLM-6B":
                        pred = chatglm6b_pred(tokenizer, model, prompt)
                    pred = pred.strip().lower()
                    # 记录这次预测的结果
                    if pred == "yes":  # 如果是yes记一次正例
                        yes_count += 1
                        valid_count += 1
                    elif pred == "no":  # 如果是no记一次反例
                        valid_count += 1
                    else:  # 其他结果不记入
                        # print(pred)
                        log_file.write(f"the {j}_th instance of model with template {i} got a invalid pred '{pred}' at sample {s} \n")
                # 记录当前simple上n次预测的结果
                if i == 0:
                    empty_line = [[0, 0] for _ in range(len(templates))] + [doc_name] + samples[s]
                    result.append(empty_line)
                result[result_index][i][0] += yes_count
                result[result_index][i][1] += valid_count
                result_index += 1
        # 删除模型=======================================================================
        if "tokenizer" in locals():
            del tokenizer
        if "model" in locals():
            del model
        if "ppl" in locals():
            del ppl
        torch.cuda.empty_cache()
    # 保存结果
    output_file = f"{root_path}/Results/{model_name}/{model_name}_gold_mentions_{n_examples}-shots_{prefixes_source}_prefix_{repeated_n}-repeats.csv"
    result_df = pd.DataFrame(result)
    result_df.to_csv(output_file, mode="a", encoding="utf-8", index=False, header=False)
    """
    with open(output_file, 'w', newline='', encoding="utf8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(result)
    """
    print(f"{prefixes_source}_prefix_{repeated_n}-repeats Results saved")
    #
    end = time.time()
    print(f"用时{(end - start)/60}分钟")


# 1. load data
train, dev = load_gold_data(local_path)

# 2. Generate Prefix
if 1:
    # 2.1. Mannually create prefix
    if 1:
        prefixes_simple = []
        simple_examples = [
            ["Anna told her friends that she was about to go to college.", ["Anna", "she"], 1],
            ["Eva and Martha didn't want their friend Jenny to feel lonely so they invited her to the party", ["Eva", "her"], 0],
            ["Paul Allen was born on Jan 21, 1953. Allen attended Lakeside School, where he befriended Bill Gates", ["Paul Allen", "Allen"], 1],
            ["A dog named Teddy ran to his owner Jane. Jane loves her dog.", ["Teddy", "Jane"], 0],
            ["I bought 3 bottles of wine today, when I went to John Doe’s store", ["I", "John Doe"], 0],
            ["Vasco told me yesterday that is his final exam went pretty well. Vasco worked really hard.", ["Vasco", "Vasco"], 1],
            ["Her car was so fast, that it went past the speed limit", ["Her car", "it"], 1],
            ["Some of our colleagues are going to be supportive. These kinds of people will earn our gratitude", ["Some of our colleagues", "our gratitude"], 0],
            ["Barack Obama won the midterm elections, so he was in office for 2 terms", ["Barack Obama", "he"], 1],
            ["Our neighbors dislike the music. If they are angry, the cops will show up soon", ["they", "the cops"], 0]
        ]
        for template in templates:
            prefix = create_prefix(simple_examples[:n_examples], template, answer_choices)
            prefixes_simple.append(prefix)
        print("Mannually create prefix based on template 1: ", prefixes_simple[0])
        """
        根据simple_examples中的前n_examples=2个例子，分别按照5个templates生成prefix。
        共得到10个prefix。
        prefixes_simple = [
        基于第一个templates生成的2个prefix字符串由“\n”连接起来，
        基于第二个templates生成的2个prefix字符串由“\n”连接起来，
        ...
        基于第五个templates生成的2个prefix字符串由“\n”连接起来，
        ]
        """
    
    # 2.2. Create prefix from SuperGLUE
    if 1:
        prefixes_super_glue = []  
        # 因为暂时不用glue的实验数据，所以先注释了。这里留一个空列表。
        """
        super_glue = load_dataset("super_glue", 'wsc.fixed')
        super_glue_train = super_glue["train"]
        super_glue_texts, super_glue_pairs, super_glue_labels = parse_superglue(super_glue_train)
        superglue_examples = get_examples_superglue(n_examples, super_glue_texts, super_glue_pairs, super_glue_labels)
        prefixes_super_glue = []
        for template in templates:
            prefix = create_prefix(superglue_examples, template, answer_choices)
            prefixes_super_glue.append(prefix)
        print("SuperGLUE prefix based on template 1: ", prefixes_super_glue[0])
        """

    # 2.3. Create prefix from ECB+ train
    if 1:
        ecb_examples = get_examples_ecb(n_examples, train, list(train.keys()))
        prefixes_ecb = []
        for template in templates:
            prefix = create_prefix(ecb_examples, template, answer_choices)
            prefixes_ecb.append(prefix)
        print("ECB+ prefix based on template 1: ", prefixes_ecb[0])
        """
        根据ecb_examples中的前n_examples=2个例子，分别按照5个templates生成prefix。
        共得到10个prefix。
        prefixes_simple = [
        基于第一个templates生成的2个prefix字符串由“\n”连接起来，
        基于第二个templates生成的2个prefix字符串由“\n”连接起来，
        ...
        基于第五个templates生成的2个prefix字符串由“\n”连接起来，
        ]
        """

    # 2.4
    """
    # TODO: experiment with different temparature [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    # # prefixes_simple
    # generator = pipeline('text-generation', model='gpt2', return_full_text=False,
    #                       prefix = prefixes_simple[0], device=torch.cuda.current_device(), temperature = 0.1, ) #top_k=0
    # text = "I like Hamilton. That musical is great. It is also Tim's favorite play."
    # mention_pair = ["That musical", "play"]
    # prompt = generate_prompt(templates[0], text, mention_pair)
    # print(prompt)
    # pred = generator(prompt, max_length=1, num_return_sequences=1,)[0]["generated_text"]
    # pred = pred.strip().lower()
    # print(pred)
    """


# 3. Create Generator
# prefix_types = ["simple", "superglue", "ecb"]
# all_prefix = [prefixes_simple, prefixes_super_glue, prefixes_ecb]
# all_generators = {}
# for i, prefixes in enumerate(all_prefix):
#     generators = []
#     for prefix in prefixes:
#         # generator = pipeline('text-generation', model='gpt2', return_full_text=False, prefix=prefix, device=torch.cuda.current_device())
#         generator = {
#             "target": 'text-generation',
#             "model": 'gpt2',  # '/root/autodl-tmp/chatglm-6b',
#             "return_full_text": False,
#             "prefix": prefix,
#             "device": torch.cuda.current_device(),
#             "trust_remote_code": True
#         }
#         # generator = f"基于{prefix_types[i]}语料和第k个template生成的模型"
#         generators.append(generator)
#     all_generators[prefix_types[i]] = generators

# 4. Experiment
if data_config == "dev":
    data = dev
elif data_config == "first two docs":
    dev_keys = list(dev.keys())
    dev_keys = dev_keys[:2]
    data = {key: dev[key] for key in dev_keys}
elif data_config == "a given doc":
    data = {given_doc_name: dev[given_doc_name]}
else:
    raise RuntimeError("'data' got a invalid value")
#
"""prefix_types = ["simple","superglue", "ecb"]"""
# 在data(其实就是ECB+的Dev集)上，对基于prefix_type（其实就是ECB+）语料和5种prefix生成的5个模型分别进行n(5)次测试。
"""annotate(data=dev, prefix_type="ecb", repeated_n=5, generators=all_generators["ecb"], templates=templates)"""
annotate(data=data, templates=templates, prefixes=prefixes_ecb, prefixes_source="ecb", repeated_n=5, model_name=model_name)

#
log_file.close()
