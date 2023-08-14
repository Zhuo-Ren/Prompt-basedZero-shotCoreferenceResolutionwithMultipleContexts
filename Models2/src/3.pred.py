# 标准库
import re
import os
import sys
from tqdm import tqdm
import _pickle as cPickle
# from typing import Dict, List, Tuple, Union
import logging
import time
import datetime
import shutil
import torch
import pandas as pd
# 本地库
from classes import Corpus, Topic, Document, Sentence, Token, EventMention, EntityMention, MentionData
from template import templates_list


global pred_index
pred_index = 0

project_path = ""
if sys.platform.startswith('win'):
    project_path = r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main"
elif sys.platform.startswith('linux'):
    project_path = r"/root/WhatGPTKnowsAboutWhoIsWho-main"
else:
    print('未知的操作系统')
print(f"Using project path: {project_path}")

# config
config_dict = {
    "corpus_and_mention_pairs_path": os.path.join(
        project_path,
        "Models2", "data", "2.extract_mention_pairs_from_test_data",
        "test_data(strategy3).c_mp"),
    "output_path": os.path.join(project_path, "Models2", "output"),
    "models": {
        # "GPT2": {
        #     "prefix": "",
        # },
        # "ChatGLM6B": {
        #     # model
        #     "model_path": "/root/autodl-tmp/chatglm-6b",
        #     "model_config_desc": "b1t0",
        #     "beam": 1,
        #     "temperature": 0,
        #     #
        #     "prefix_num": 0,
        #     "prefix": "",
        #     "do_sample": False,
        #     "repeat": 1,
        # },
        # "ChatGPT3.5": {
        #     "system_message": {
        #         "role": "system",
        #         "content": "You can only answer 'Yes' or 'No'."
        #     },  # or None if you do not use system message
        #     "model_config_desc": "b1t0",
        #     "beam": 1,
        #     "temperature": 0,
        #     #
        #     "prefix_num": 0,
        #     "do_sample": False,
        #     "repeat": 1,
        # },
        "ground_truth_model": {
            "model_config_desc": "none",
            "prefix_num": 0,
            "do_sample": False,
            "repeat": 1,
        }
    },
    "templates": ["16DAM"],
    "data": ["36_ecb"]  # , "36_ecbplus"]  # "all"
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

# error log
file_path = os.path.join(config_dict["output_path"], "err.log.txt")
f = open(file_path, mode="w", encoding="utf8")
config_dict["error_log"] = f

#
# save this file itself
shutil.copy(os.path.abspath(__file__), config_dict["output_path"])


# 配置GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


def get_context(mention1, mention2, context_level="sentence", mark_mention=False, selected_sentence_only=True):
    """

    :param mention1:
    :param mention2:
    :param context_level: "sent" or "doc"
    :param mark_mention:
    :param selected_sentence_only:
    :return:
    """
    #
    corpus = config_dict["corpus"]
    # 指称对应文档中所有句子对象（[<sent obj>, <sent obj>, <sent obj>, <sent obj>, <sent obj>, <sent obj>]）
    mention1_all_sents = corpus.topics[re.sub(r"_[0-9]+", "_", mention1.doc_id)].docs[mention1.doc_id].sentences
    mention2_all_sents = corpus.topics[re.sub(r"_[0-9]+", "_", mention2.doc_id)].docs[mention2.doc_id].sentences
    # 指称对应文档中所有句子编号（[0, 1, 2, 3, 4, 5]）
    mention1_all_sents_indexs = sorted(mention1_all_sents.keys())
    mention2_all_sents_indexs = sorted(mention2_all_sents.keys())
    # 指称对应文档中选中句子编号（[0, 2]）
    mention1_selected_sents_indexs = [i for i in mention1_all_sents_indexs if mention1_all_sents[i].is_selected]
    mention2_selected_sents_indexs = [i for i in mention2_all_sents_indexs if mention2_all_sents[i].is_selected]
    #
    start_tokens = [mention1.tokens[0], mention2.tokens[0]]
    end_tokens = [mention1.tokens[-1], mention2.tokens[-1]]
    # 根据配置计算目标文档、目标句
    if context_level == "doc":
        if mention1.doc_id == mention2.doc_id:
            sent_max = max(mention1.sent_id, mention2.sent_id)
            sent_indexes = mention1_selected_sents_indexs if selected_sentence_only else mention1_all_sents_indexs
            target = [
                [
                    mention1.doc_id,
                    [i for i in sent_indexes if i <= sent_max]
                ]
            ]
        else:
            mention1_sent_indexes = mention1_selected_sents_indexs if selected_sentence_only else mention1_all_sents_indexs
            mention2_sent_indexes = mention2_selected_sents_indexs if selected_sentence_only else mention2_all_sents_indexs
            target = [
                [
                    mention1.doc_id,
                    [i for i in mention1_sent_indexes if i <= mention1.sent_id]
                ],
                [
                    mention2.doc_id,
                    [i for i in mention2_sent_indexes if i <= mention2.sent_id]
                ]
            ]
    elif context_level == "sent":
        if (mention1.doc_id == mention2.doc_id) & (mention1.sent_id == mention2.sent_id):
            target = [
                [
                    mention1.doc_id, [mention1.sent_id]
                ]
            ]
        else:
            target = [
                [
                    mention1.doc_id, [mention1.sent_id]
                ],
                [
                    mention2.doc_id, [mention2.sent_id]
                ]
            ]
    # 根据目标文档、目标句来读取context
    context = ""
    for cur_target in target:
        # 目标文档
        doc_id = cur_target[0]
        # 目标句列表（index）
        sent_indexes = cur_target[1]
        # 目标句列表（obj）
        sents = corpus.topics[re.sub(r"_[0-9]+", "_", doc_id)].docs[doc_id].sentences
        #
        for cur_sent_index in sent_indexes:
            # 目标句（obj）
            cur_sent = sents[cur_sent_index]
            #
            for cur_token in cur_sent.tokens:
                # 加<mention>
                if mark_mention & (id(cur_token) in [id(i) for i in start_tokens]):
                    context += " <mention>"
                # 加空格
                if cur_token.token not in ["'s", ".", "!", "?", ",", ";", ":", ")", "]", "}"]:
                    context += " "
                # 加词
                context += cur_token.token
                # 加</mention>
                if mark_mention & (id(cur_token) in [id(i) for i in end_tokens]):
                    context += " </mention>"
            else:
                # 给标题加个句号
                if cur_sent.tokens[-1].token not in [".", "!", "?", "''"]:
                    config_dict["error_log"].write(f"End of sent error at {doc_id}-sent{cur_sent_index}:{cur_sent.tokens[-1].token}\n")
                    context += "."
        "END OF for cur_sent_index in sent_indexes"
        context += " $$"  # 如果是多个文档，那么之间用此符号分割
    context = context[:-2]
    #
    context = context.strip()
    context = context.replace("`` ", "\"")
    context = context.replace(" ''", "\"")
    # 如果mark mention，则检测是否有2个<mention>
    if mark_mention:
        if len(re.findall("<mention>", context)) != 2:
            context_doc_mark_all = get_context(mention1, mention2, context_level="doc", mark_mention=True, selected_sentence_only=False)
    #
    return context


def make_prompt(template_index, mention1, mention2):
    template = templates_list[template_index]
    #
    mention1_str = mention1.mention_str
    mention2_str = mention2.mention_str
    # context with target mentions un-marked
    """
    context_sent_nomark_all = get_context(mention1, mention2, context_level="sent", mark_mention=False, selected_sentence_only=False)
    context_doc_nomark_all = get_context(mention1, mention2, context_level="doc", mark_mention=False, selected_sentence_only=False)
    context_doc_nomark_selected = get_context(mention1, mention2, context_level="doc", mark_mention=False, selected_sentence_only=True)
    """
    # context with target mentions marked by labels
    # context_sent_mark_all = get_context(mention1, mention2, context_level="sent", mark_mention=True, selected_sentence_only=False)
    context_doc_mark_all = get_context(mention1, mention2, context_level="doc", mark_mention=True, selected_sentence_only=False)
    # context_doc_mark_selected = get_context(mention1, mention2, context_level="doc", mark_mention=True, selected_sentence_only=True)
    #
    # template = template.replace("[S_CONTEXT]", context_sent_nomark_all)
    # template = template.replace("[S_CONTEXT_marked]", context_sent_mark_all)
    # template = template.replace("[D_CONTEXT]", context_doc_nomark_all)
    # template = template.replace("[D_CONTEXT_selected]", context_doc_nomark_selected)
    template = template.replace("[D_CONTEXT_marked]", context_doc_mark_all)
    # template = template.replace("[D_CONTEXT_selected_marked]", context_doc_mark_selected)
    template = template.replace("[MENTION1]", mention1_str)
    template = template.replace("[MENTION2]", mention2_str)
    # template = template.strip()
    return template


def create_model(model_name):
    model_config = config_dict["models"][model_name]
    if model_name == "GPT2":
        from transformers import GPT2Tokenizer, GPT2Model, GPT2LMHeadModel
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2', prefix=model_config["prefix"])
        return tokenizer, model
    elif model_name == "ChatGLM6B":
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained(
            model_config["model_path"],
            trust_remote_code=True,
            revision=""
        )
        model = AutoModel.from_pretrained(
            model_config["model_path"],
            trust_remote_code=True,
            prefix=model_config["prefix"],
            revision=""
        ).half().cuda()
        return tokenizer, model
    elif model_name == "ChatGPT3.5":
        messages = []
        system_message = config_dict["models"][model_name]["system_message"]
        if system_message is not None:
            messages.append(system_message)
        return messages


def process_a_mention_pair(model_name, template_id, mention1, mention2):
    #
    model_config = config_dict["models"][model_name]
    true_num, validated_num, repeat_num = 0, 0, model_config["repeat"]
    ground_truth = (mention1.gold_tag == mention2.gold_tag)
    #
    prompt = make_prompt(template_id, mention1, mention2)
    global pred_index
    pred_index += 1
    prompt_log_file = config_dict["prompt_log_file"]
    prompt_log_file.write(f"{pred_index}##############\n{prompt}\n")
    for repeat_index in range(repeat_num):
        if model_name == "GPT2":
            tokenizer = model_config["model_components"][0]
            model = model_config["model_components"][1]
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(
                **inputs, max_new_tokens=1,
                do_sample=model_config["do_sample"],
                num_beams=model_config["num_beams"],
                temperature=model_config["temperature"]
            )
            outputs = outputs.tolist()[0]
            pred = outputs[len(inputs.input_ids[0]):]
            pred = tokenizer.decode(pred, skip_special_tokens=True)
        elif model_name == "ChatGLM6B":
            tokenizer, model = model_config["model_components"]
            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            outputs = model.generate(inputs, max_new_tokens=1, do_sample=model_config["do_sample"], num_beams=model_config["beam"], temperature=model_config["temperature"])
            outputs = outputs.tolist()[0]
            outputs = outputs[len(inputs[0]):]
            pred = tokenizer.decode(outputs, skip_special_tokens=True)
        elif model_name == "ChatGPT3.5":
            import openai
            #
            messages = model_config["model_components"]
            #
            pred = ""
            cur_time = time.time()
            while True:
                try:
                    openai.api_key = os.getenv("OPENAI_API_KEY")
                    openai.proxy = "127.0.0.1:15732"
                    m = []
                    m = m + messages
                    m.append({
                        "role": "user",
                        "content": prompt
                    })
                    if model_config["do_sample"]:  # do sample
                        r = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=m, temperature=model_config["temperature"], max_tokens=1,timeout=10, request_timeout=10)
                    else:  # no sample
                        r = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=m, temperature=0.0, max_tokens=1, timeout=10, request_timeout=10)
                    pred = r["choices"][0]["message"]["content"]
                    break
                except openai.OpenAIError as e:
                    # 如果是访问过快就等等
                    if re.match(r"Rate limit reached for default-gpt-3.5-turbo in organization org-ooYBZEO3oP4YJO7LGYUbhNjf on requests per min. Limit: 3 / min.", e._message) is not None:
                        print("rate limit", time.time())
                        time.sleep(10)
                    # 如果是bad gateway就重试
                    elif "Bad gateway" in e._message:
                        print("bad gateway", time.time())
                        if (time.time() - cur_time) < 1:
                            input("two error in 1 second")
                            time.sleep(10)
                        cur_time = time.time()
                        pass
                    # 如果是The server is overloaded or not ready yet.就重试
                    elif "The server is overloaded or not ready yet" in e._message:
                        print("server overloaded", time.time())
                        time.sleep(10)
                    # 如果是其他问题就暂停
                    else:
                        print(e._message)
                        """
                        ipt = input("Press Enter to request again... Press 'Code' to debug...")
                        if ipt == "Code":
                            while True:
                                command = input("input your debug command:")
                                o = eval(command)
                                print(o)
                        """
        elif model_name == "ground_truth_model":
            pred = "Yes" if ground_truth else "No"
        #
        pred = pred.strip().lower()
        if pred not in ["yes", "no"]:
            logging.info(f"       prompt {pred_index} 在第{repeat_index}次重复时获得预测{pred}。结果非法。")
        else:
            validated_num += 1
            if ((pred == "yes") & ground_truth):
                true_num += 1
                logging.info(f"       prompt {pred_index} 在第{repeat_index}次重复时获得预测{pred}。结果正确。")
            elif ((pred == "no") & (not ground_truth)):
                true_num += 1
                logging.info(f"       prompt {pred_index} 在第{repeat_index}次重复时获得预测{pred}。结果正确。")
            else:
                logging.info(f"       prompt {pred_index} 在第{repeat_index}次重复时获得预测{pred}。结果错误。")
    return [true_num, validated_num, repeat_num]


def predicate(model_name, template_id, mention_pairs):
    """

    :param model_name:
    :param template_id:
    :param mention_pairs:
    :return: No return. mention_pairs is changed.
    """
    #
    global pred_index
    #
    # get the first value of mention_pairs
    first_value = list(mention_pairs.values())[0]
    # check its type
    if type(first_value) is list:  # mention_pairs是topic-mentionPairs的嵌套结构
        #
        pred_index = 0
        #
        topic_num_all = len(mention_pairs.keys())
        topic_num_cur = 0
        for topic_id, topic_value in mention_pairs.items():
            logging.info(f"TOPIC: {topic_id} ({topic_num_cur}/{topic_num_all}) start ({datetime.datetime.now()})")
            print(f"TOPIC: {topic_id} ({topic_num_cur}/{topic_num_all}) start ({datetime.datetime.now()})")
            for cur_mention_pairs in tqdm(topic_value):
                predicated_result = process_a_mention_pair(model_name, template_id, cur_mention_pairs[0], cur_mention_pairs[1])
                cur_mention_pairs.append(predicated_result)
            logging.info(f"TOPIC: {topic_id} ({topic_num_cur}/{topic_num_all}) end ({datetime.datetime.now()})")
            print(f"TOPIC: {topic_id} ({topic_num_cur}/{topic_num_all}) end ({datetime.datetime.now()})")
            topic_num_cur += 1
    elif type(first_value) is dict:  # mention_pairs是topic-doc-mentionPairs的嵌套结构
        #
        pred_index = 0
        #
        for topic_id, topic_value in mention_pairs.items():
            for doc_id, doc_value in topic_value.items():
                for cur_mention_pairs in doc_value:
                    predicated_result = process_a_mention_pair(model_name, template_id, cur_mention_pairs[0], cur_mention_pairs[1])
                    cur_mention_pairs.append(predicated_result)
    #
    print(f"OUTPUT: prompt log under cur config saved in {config_dict['prompt_log_file_path']}")
    logging.info(f"OUTPUT: prompt log under cur config saved in {config_dict['prompt_log_file_path']}")
    config_dict["prompt_log_file"].close()


def main():
    # 代码对每个model和每个template分别进行实验
    for cur_model_name, cur_model_config in config_dict["models"].items():
        for cur_template_id in config_dict["templates"]:
            # 针对当前model和当前template实验一次
            logging.info(f"============================\nStart {cur_model_name} model with template {cur_template_id} ({datetime.datetime.now()})")
            print(f"============================\nStart {cur_model_name} model with template {cur_template_id} ({datetime.datetime.now()})")
            # read corpus
            with open(config_dict["corpus_and_mention_pairs_path"], 'rb') as f:
                corpus, mention_pairs = cPickle.load(f)
            #
            strategy_id = re.search(r"strategy([0-4])", config_dict["corpus_and_mention_pairs_path"]).groups()[0]
            # select a subset of the whole corpus
            if config_dict["data"] != "all":
                topics_list = list(corpus.topics.keys())  # TODO: 知识点总结。如果不加list()，那么topics_list是会随着corpus的改变而改变的。
                for cur_topic_id in topics_list:
                    if cur_topic_id not in config_dict["data"]:
                        del corpus.topics[cur_topic_id]
                topics_list = list(mention_pairs.keys())
                for cur_topic_id in topics_list:
                    if cur_topic_id not in config_dict["data"]:
                        del mention_pairs[cur_topic_id]
            config_dict["corpus"] = corpus
            config_dict["mention_pairs"] = mention_pairs
            #
            config_dict["file_name"] = f"{config_dict['data']}(strategy{strategy_id})_{cur_model_name}({cur_model_config['model_config_desc']})_{cur_model_config['prefix_num']}shot_t{cur_template_id}_{'doSample' if cur_model_config['do_sample'] else 'noSample'}(r{cur_model_config['repeat']})"
            config_dict["prompt_log_file_path"] = f"{config_dict['output_path']}/{config_dict['file_name']}.promptlog"
            config_dict["prompt_log_file"] = open(config_dict["prompt_log_file_path"], mode="w", encoding="utf8")
            #
            global pred_index
            pred_index = 0
            # 构建模型
            config_dict["models"][cur_model_name]["model_components"] = create_model(cur_model_name)
            # 预测+保存promptlog
            predicate(cur_model_name, cur_template_id, mention_pairs)
            # 保存corpus和mp
            path = os.path.join(config_dict['output_path'], f"{config_dict['file_name']}.c_mp")
            if os.path.exists(path):
                raise RuntimeError("重复的保存")
            with open(path, 'wb') as f:
                cPickle.dump((corpus, mention_pairs), f)
                print(f"OUTPUT: corpus and mention pairs list and pred result under cur config saved in {path}")
                logging.info(f"OUTPUT: corpus and mention pairs list and pred result under cur config saved in {path}")
            # 保存csv
            path = os.path.join(config_dict['output_path'], f"{config_dict['file_name']}.csv")
            csv_list = []
            csv_list.append(["topic", "m1_doc", "m1_sent", "m1_str", "m2_doc", "m2_sent", "m2_str", "wd/cd", "seq", "label", cur_template_id])
            for cur_topic_id in mention_pairs.keys():
                for mention1, mention2, [true_num, validated_num, repeat_num] in mention_pairs[cur_topic_id]:
                    if_wd = (mention1.doc_id == mention2.doc_id)
                    csv_list.append([
                        cur_topic_id,
                        mention1.doc_id, mention1.sent_id, mention1.mention_str,
                        mention2.doc_id, mention2.sent_id, mention2.mention_str,
                        "wd" if if_wd else "cd",
                        abs(mention1.sent_id - mention2.sent_id) if if_wd else None,
                        1 if mention1.gold_tag == mention2.gold_tag else 0,
                        [true_num, validated_num, repeat_num]
                    ])
            csv_df = pd.DataFrame(csv_list)
            csv_df.to_csv(path, mode="a", encoding="utf-8", index=False, header=False)
            print(f"OUTPUT: mention pairs csv under cur config saved in {path}")
            logging.info(f"OUTPUT: mention pairs csv under cur config saved in {path}")
            # 销毁模型（释放内存）
            del config_dict["models"][cur_model_name]["model_components"]
            #
            del config_dict["file_name"]
            del config_dict["prompt_log_file_path"]
            del config_dict["prompt_log_file"]
            del config_dict["mention_pairs"]
            #
            logging.info(f"End {cur_model_name} model with template {cur_template_id} ({datetime.datetime.now()})\n======================================")
            print(f"End {cur_model_name} model with template {cur_template_id} ({datetime.datetime.now()})\n======================================")


if __name__ == '__main__':
    main()
    print("\a")
