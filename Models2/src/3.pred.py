# 标准库
import re
import os
import _pickle as cPickle
# from typing import Dict, List, Tuple, Union
import logging
import time
import shutil
import pandas as pd
# 本地库
from classes import Corpus, Topic, Document, Sentence, Token, EventMention, EntityMention, MentionData
from template import templates_list


# config
config_dict = {
    "corpus_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\data\1.read_corpus\test_data",
    "mention_pairs_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\data\2.extract_mention_pairs_from_test_data\test(strategy3).mp",
    "output_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\output",
    "models": {
        # "GPT2": {
        #     "prefix": "",
        # },
        # "ChatGLM6B": {
        # },
        "ChatGPT3.5": {
            "system_message": {
                "role": "system",
                "content": "You can only answer 'Yes' or 'No'."
            },  # or None if you do not use system message
            "temperature": 0,
            "do_sample": False,
            "repeat": 1,
        },
    },
    "templates": ["14DAM"],
    "data": ["36_ecb","36_ecbplus"]
}

# 临时代码，用于清空输出路径
for file in os.listdir(config_dict["output_path"]):
    os.remove(os.path.join(config_dict["output_path"], file))

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


# save this file itself
shutil.copy(os.path.abspath(__file__), config_dict["output_path"])


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
                    [i for i in sent_indexes if i >= sent_max]
                ]
            ]
        else:
            mention1_sent_indexes = mention1_selected_sents_indexs if selected_sentence_only else mention1_all_sents_indexs
            mention2_sent_indexes = mention2_selected_sents_indexs if selected_sentence_only else mention2_all_sents_indexs
            target = [
                [
                    mention1.doc_id,
                    [i for i in mention1_sent_indexes if i >= mention1.sent_id]
                ],
                [
                    mention2.doc_id,
                    [i for i in mention2_sent_indexes if i >= mention2.sent_id]
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
                if cur_token in start_tokens:
                    context += " <mention>"
                context += " "
                context += cur_token.token
                if cur_token in end_tokens:
                    context += " </mention>"
    #
    context = context.strip()
    return context

def make_prompt(template_index, mention1, mention2):
    template = templates_list[template_index]
    corpus = config_dict["corpus"]
    #
    mention1_str = mention1.mention_str
    mention2_str = mention2.mention_str
    # context with target mentions un-marked
    context_sent_nomark_all = get_context(mention1, mention2, context_level="sent", mark_mention=False, selected_sentence_only=False)
    context_doc_nomark_all = get_context(mention1, mention2, context_level="doc", mark_mention=False, selected_sentence_only=False)
    context_doc_nomark_selected = get_context(mention1, mention2, context_level="doc", mark_mention=False, selected_sentence_only=True)
    # context with target mentions marked by labels
    context_sent_mark_all = get_context(mention1, mention2, context_level="sent", mark_mention=True, selected_sentence_only=False)
    context_doc_mark_all = get_context(mention1, mention2, context_level="doc", mark_mention=True, selected_sentence_only=False)
    context_doc_mark_selected = get_context(mention1, mention2, context_level="doc", mark_mention=True, selected_sentence_only=True)
    #
    template = template.replace("[S_CONTEXT]", context_sent_nomark_all)
    template = template.replace("[S_CONTEXT_marked]", context_sent_mark_all)
    template = template.replace("[D_CONTEXT]", context_doc_nomark_all)
    template = template.replace("[D_CONTEXT_selected]", context_doc_nomark_selected)
    template = template.replace("[D_CONTEXT_marked]", context_doc_mark_all)
    template = template.replace("[D_CONTEXT_selected_marked]", context_doc_mark_selected)
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
            "/root/autodl-tmp/chatglm-6b",
            trust_remote_code=True,
            revision=""
        )
        model = AutoModel.from_pretrained(
            "/root/autodl-tmp/chatglm-6b",
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
    model_config = config_dict["models"][model_name]
    prompt = make_prompt(template_id, mention1, mention2)
    true_num, validated_num, repeat_num = 0, 0, model_config["repeat"]
    ground_truth = (mention1.gold_tag == mention2.gold_tag)
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
            pred = True
        elif model_name == "ChatGPT3.5":
            import openai
            #
            messages = model_config["model_components"]
            #
            global pred_index
            pred_index += 1
            print(f"{pred_index}##############\n{prompt}\n")
            prompt_log_file = config_dict["prompt_log_file"]
            prompt_log_file.write(f"{pred_index}##############\n{prompt}\n")
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
                    # if model_config["do_sample"]:  # do sample
                    #     r = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=m, temperature=model_config["temperature"], max_tokens=1)
                    # else:  # no sample
                    #     r = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=m, temperature=0.0, max_tokens=1)
                    # pred = r["choices"][0]["message"]["content"]
                    pred = "Yes"
                    pred = True
                    validated_num += 1
                    if pred == ground_truth:
                        true_num += 1
                    print("success", time.time())
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
                        cur_time = time.time()
                        pass
                    # 如果是The server is overloaded or not ready yet.就重试
                    elif "The server is overloaded or not ready yet" in e._message:
                        print("server overloaded", time.time())
                        time.sleep(1)
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
    if type(first_value) is list: # mention_pairs是topic-mentionPairs的嵌套结构
        #
        pred_index = 0
        #
        for topic_id, topic_value in mention_pairs.items():
            for cur_mention_pairs in topic_value:
                predicated_result = process_a_mention_pair(model_name, template_id, cur_mention_pairs[0], cur_mention_pairs[1])
                cur_mention_pairs.append(predicated_result)
    elif type(first_value) is dict: # mention_pairs是topic-doc-mentionPairs的嵌套结构
        #
        pred_index = 0
        #
        for topic_id, topic_value in mention_pairs.items():
            for doc_id, doc_value in topic_value.items():
                for cur_mention_pairs in doc_value:
                    predicated_result = process_a_mention_pair(model_name, template_id, cur_mention_pairs[0], cur_mention_pairs[1])
                    cur_mention_pairs.append(predicated_result)
    #
    print("\n")


def main():
    # 代码对每个model和每个template分别进行实验
    for cur_model_name, cur_model_config in config_dict["models"].items():
        for cur_template_id in config_dict["templates"]:
            # 针对当前model和当前template实验一次
            logging.info(f"Start {cur_model_name} model with template {cur_template_id}============================")
            print(f"Start {cur_model_name} model with template {cur_template_id}============================")
            # read corpus
            with open(config_dict["corpus_path"], 'rb') as f:
                corpus = cPickle.load(f)
            with open(config_dict["mention_pairs_path"], 'rb') as f:
                mention_pairs = cPickle.load(f)
            #
            strategy_id = re.search("test\(strategy([0-4])\).mp", config_dict["mention_pairs_path"]).groups()[0]
            # select a subset of the whole corpus
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
            config_dict["file_name"] = f"{config_dict['data']}(strategy{strategy_id})_{cur_model_name}_t{cur_template_id}_s0_b1_noSample"
            config_dict["corpus_path"] = f"{config_dict['output_path']}/{config_dict['file_name']}.pkl"
            config_dict["prompt_log_file_path"] = f"{config_dict['output_path']}/{config_dict['file_name']}.promptlog"
            config_dict["prompt_log_file"] = open(config_dict["prompt_log_file_path"], mode="w", encoding="utf8")
            #
            pred_index = 0
            # 构建模型
            config_dict["models"][cur_model_name]["model_components"] = create_model(cur_model_name)
            # 预测
            predicate(cur_model_name, cur_template_id, mention_pairs)
            # 保存
            corpus_path = f"{config_dict['output_path']}\{config_dict['data']}.corpus"
            if not os.path.exists(corpus_path):
                with open(corpus_path, 'wb') as f:
                    cPickle.dump(corpus, f)
                    print(f"corpus obj of {config_dict['data']} saved in {corpus_path}")
            os.path.exists(corpus_path)
            # 保存
            mention_pairs_path = f"{config_dict['output_path']}\{config_dict['file_name']}.mp"
            if os.path.exists(mention_pairs_path):
                raise RuntimeError("重复的保存")
            with open(mention_pairs_path, 'wb') as f:
                cPickle.dump(mention_pairs, f)
                print(f"mention pairs list and pred result under cur config saved in {corpus_path}")
            # 保存
            mention_pairs_path = f"{config_dict['output_path']}\{config_dict['file_name']}.csv"
            csv_list = []
            csv_list.append(["topic", "m1_doc", "m1_sent", "m1_str", "m2_doc", "m2_sent", "m2_str", "wd/cd", "seq", "corefer?", cur_template_id])
            for cur_topic_id in mention_pairs.keys():
                for mention1, mention2, [true_num, validated_num, repeat_num] in mention_pairs[cur_topic_id]:
                    if_wd = (mention1.doc_id == mention2.doc_id)
                    csv_list.append([
                        cur_topic_id,
                        mention1.doc_id, mention1.sent_id, mention1.mention_str,
                        mention2.doc_id, mention2.sent_id, mention2.mention_str,
                        "wd" if if_wd else "cd",
                        abs(mention1.sent_id - mention2.sent_id) if if_wd else None,
                        mention1.gold_tag == mention2.gold_tag,
                        [true_num, validated_num, repeat_num]
                    ])
            csv_df = pd.DataFrame(csv_list)
            csv_df.to_csv(mention_pairs_path, mode="a", encoding="utf-8", index=False, header=False)
            # 销毁模型（释放内存）
            del config_dict["models"][cur_model_name]["model_components"]
            #
            del config_dict["file_name"]
            del config_dict["corpus_path"]
            del config_dict["prompt_log_file_path"]
            del config_dict["prompt_log_file"]
            #
            del config_dict["corpus"]
            del config_dict["mention_pairs"]

if __name__ == '__main__':
    main()
