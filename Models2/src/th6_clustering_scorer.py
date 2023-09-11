import logging
import csv
import shutil
import _pickle as cPickle
import os
import re
import subprocess
from src.shared.classes import Corpus, Topic, Document, Sentence, Mention, EventMention, EntityMention, Token, Srl_info, Cluster


def write_mention_based_cd_clusters(corpus, is_event, is_gold, out_file):
    """
    给定corpus对象，这个对象中应该已经保存好了共指消解结果（真值结果存在mention的gold_tag属性，预测结果存在mention的cd_coref_chain属性），此方法把预测结果以特定顺序、CoNLL2012格式输出。

    用途：先把真实结果输出（data/gold/cybulska_gold/CD_test_entity_mention_based.key_conll或data/gold/cybulska_gold/CD_test_event_mention_based.key_conll），再把预测结果输出（CD_test_entity_mention_based.response_conll或CD_test_event_mention_based.response_conll），最后对比这俩文件即可计算性能。

    This function writes the cross-document (CD) predicted clusters to a file (in a CoNLL format) in a mention based manner, means that each token represents a mention and its coreference chain id is marked in a parenthesis.

    mention顺序：
        * topic按照先ecb后ecbplus，先小后大的顺序排序：['36_ecb', '37_ecb', '38_ecb', '39_ecb', '40_ecb', '41_ecb', '42_ecb', '43_ecb', '44_ecb', '45_ecb', '36_ecbplus', '37_ecbplus', '38_ecbplus', '39_ecbplus', '40_ecbplus', '41_ecbplus', '42_ecbplus', '43_ecbplus', '44_ecbplus', '45_ecbplus']
        * doc按照显小后大的顺序排序：['40_10ecb', '40_1ecb', '40_2ecb', '40_3ecb', '40_4ecb', '40_5ecb', '40_6ecb', '40_7ecb', '40_8ecb', '40_9ecb']
        * sent按照先小后大的顺序排序：[0, 1]
        * gold mention按照start_offset先小后大的顺序排列
        * 记录mention的cd_coref_chain

    输出格式：
        #begin document (ECB+/ecbplus_all); part 000
        ECB+/ecbplus_all	(1)
        ECB+/ecbplus_all	(2)
        ECB+/ecbplus_all	(3)
        ECB+/ecbplus_all	(4)
        ECB+/ecbplus_all	(5)
        ECB+/ecbplus_all	(1)
        ECB+/ecbplus_all	(2)
        ECB+/ecbplus_all	(3)
        ECB+/ecbplus_all	(1)
        #end document
        每一行是一个mention，括号中是它属于哪个簇。簇号相同的就是共指的，比如上例中的mention1和mention6.

    在Cybulska的设置中，共指消解使用的是真实mention（即没有mention detection，共指消解结果中涉及的mention和真实mention是一模一样的）
    Used in Cybulska setup, when gold mentions are used during evaluation and there is no need
    to match predicted mention with a gold one.

    :param corpus: A Corpus object, contains the documents of each split, grouped by topics.
    :param out_file: filename of the CoNLL output file
    :param is_event: whether to write event or entity mentions
    :param is_gold: whether to write a gold-standard file (key) which contains the gold clusters
      （mention对象的gold_tag属性）or to write a system file (response) that contains the predicted clusters（mention对象的cd_coref_chain属性）.
    """
    out_coref = open(out_file, 'w')
    # 1 开头
    generic = 'ECB+/ecbplus_all'
    out_coref.write("#begin document (" + generic + "); part 000" + '\n')

    # 2 遍历
    cd_coref_chain_to_id = {}
    cd_coref_chain_to_id_counter = 0

    # 2.1 排序并遍历topics
    ecb_topics = {}
    ecbplus_topics = {}
    for topic_id, topic in corpus.topics.items():
        if 'plus' in topic_id:
            ecbplus_topics[topic_id] = topic
        else:
            ecb_topics[topic_id] = topic
    topic_keys = sorted(ecb_topics.keys()) + sorted(ecbplus_topics.keys())
    for topic_id in topic_keys:
        curr_topic = corpus.topics[topic_id]
        # 2.2 排序并遍历docs
        for doc_id in sorted(curr_topic.docs.keys()):
            curr_doc = curr_topic.docs[doc_id]
            # 2.3 排序并遍历sents
            for sent_id in sorted(curr_doc.sentences.keys()):
                curr_sent = curr_doc.sentences[sent_id]
                # 2.4 排序并遍历mentions
                if is_event is True:
                    mentions = curr_sent.gold_event_mentions
                elif is_event is False:
                    mentions = curr_sent.gold_entity_mentions
                elif is_event is None:
                    mentions = curr_sent.gold_entity_mentions + curr_sent.gold_event_mentions
                mentions.sort(key=lambda x: x.start_offset, reverse=True)
                for mention in mentions:
                    # map the gold coref tags to unique ids
                    if is_gold:  # creating the key files(这个就没用，因为函数调用中写死的is_gold=False)
                        if mention.gold_tag not in cd_coref_chain_to_id:
                            cd_coref_chain_to_id_counter += 1
                            cd_coref_chain_to_id[mention.gold_tag] = cd_coref_chain_to_id_counter
                        coref_chain = cd_coref_chain_to_id[mention.gold_tag]
                    else:  # writing the clusters at test time (response files)
                        coref_chain = mention.cd_coref_chain
                    out_coref.write('{}\t({})\n'.format(generic,coref_chain))

    # 3 结尾
    out_coref.write('#end document\n')
    out_coref.close()


def read_conll_f1(filename):
    '''
    This function reads the results of the CoNLL scorer , extracts the F1 measures of the MUS,
    B-cubed and the CEAF-e and calculates CoNLL F1 score.
    :param filename: a file stores the scorer's results.
    :return: the CoNLL F1
    '''
    r_list = []
    p_list = []
    f1_list = []
    with open(filename, "r") as ins:
        for line in ins:
            new_line = line.strip()
            if new_line.find('F1:') != -1:
                find = re.findall("[0-9.]*%", new_line)
                r_list.append(float(find[0][:-1]))
                p_list.append(float(find[1][:-1]))
                f1_list.append(float(find[2][:-1]))

    return {
        "muc_p": p_list[1],
        "muc_r": r_list[1],
        "muc_f1": f1_list[1],
        "bcubed_p": p_list[3],
        "bcubed_r": r_list[3],
        "bcubed_f1": f1_list[3],
        "ceafe_p": p_list[7],
        "ceafe_r": r_list[7],
        "ceafe_f1": f1_list[7],
        "conll_f1": (f1_list[1] + f1_list[3] + f1_list[7])/float(3)
    }


def save_clustering_scores_into_csv_in_list_format(experiments_scores, output_path, suffix="scores_clustering_list.csv"):
    file_path = os.path.join(output_path, suffix)
    csvfile = open(file_path, mode="w", newline='', encoding='utf-8')
    header = [
        'data',
        'model_name', 'model_config',
        'prefix_num', 'template',
        'sample', 'repeat',
        'E_MUC_r', 'E_MUC_p', 'E_MUC_F1',
        'E_Bcubed_r', 'E_Bcubed_p', 'E_Bcubed_F1',
        'E_CEAFe_r', 'E_CEAFe_p', 'E_CEAFe_F1',
        'E_CoNLL_F1',
        'V_MUC_r', 'V_MUC_p', 'V_MUC_F1',
        'V_Bcubed_r', 'V_Bcubed_p', 'V_Bcubed_F1',
        'V_CEAFe_r', 'V_CEAFe_p', 'V_CEAFe_F1',
        'V_CoNLL_F1',
        'A_MUC_r', 'A_MUC_p', 'A_MUC_F1',
        'A_Bcubed_r', 'A_Bcubed_p', 'A_Bcubed_F1',
        'A_CEAFe_r', 'A_CEAFe_p', 'A_CEAFe_F1',
        'A_CoNLL_F1',
    ]
    writer = csv.DictWriter(csvfile, fieldnames=header)
    writer.writeheader()
    for cur_experiment_score in experiments_scores:
        writer.writerow({
            'data': cur_experiment_score['data'],
            'model_name': cur_experiment_score['model_name'],
            'model_config': cur_experiment_score['model_config'],
            'prefix_num': cur_experiment_score['prefix_num'],
            'template': cur_experiment_score['template'],
            'sample': cur_experiment_score['sample'],
            'repeat': cur_experiment_score['repeat'],
            #
            'E_MUC_r': cur_experiment_score['entity_muc_r'],
            'E_MUC_p': cur_experiment_score['entity_muc_p'],
            'E_MUC_F1': cur_experiment_score['entity_muc_f1'],
            'E_Bcubed_r': cur_experiment_score['entity_bcubed_r'],
            'E_Bcubed_p': cur_experiment_score['entity_bcubed_p'],
            'E_Bcubed_F1': cur_experiment_score['entity_bcubed_f1'],
            'E_CEAFe_r': cur_experiment_score['entity_ceafe_r'],
            'E_CEAFe_p': cur_experiment_score['entity_ceafe_p'],
            'E_CEAFe_F1': cur_experiment_score['entity_ceafe_f1'],
            'E_CoNLL_F1': cur_experiment_score['entity_conll_f1'],
            #
            'V_MUC_r': cur_experiment_score['event_muc_r'],
            'V_MUC_p': cur_experiment_score['event_muc_p'],
            'V_MUC_F1': cur_experiment_score['event_muc_f1'],
            'V_Bcubed_r': cur_experiment_score['event_bcubed_r'],
            'V_Bcubed_p': cur_experiment_score['event_bcubed_p'],
            'V_Bcubed_F1': cur_experiment_score['event_bcubed_f1'],
            'V_CEAFe_r': cur_experiment_score['event_ceafe_r'],
            'V_CEAFe_p': cur_experiment_score['event_ceafe_p'],
            'V_CEAFe_F1': cur_experiment_score['event_ceafe_f1'],
            'V_CoNLL_F1': cur_experiment_score['event_conll_f1'],
            #
            'A_MUC_r': cur_experiment_score['all_muc_r'],
            'A_MUC_p': cur_experiment_score['all_muc_p'],
            'A_MUC_F1': cur_experiment_score['all_muc_f1'],
            'A_Bcubed_r': cur_experiment_score['all_bcubed_r'],
            'A_Bcubed_p': cur_experiment_score['all_bcubed_p'],
            'A_Bcubed_F1': cur_experiment_score['all_bcubed_f1'],
            'A_CEAFe_r': cur_experiment_score['all_ceafe_r'],
            'A_CEAFe_p': cur_experiment_score['all_ceafe_p'],
            'A_CEAFe_F1': cur_experiment_score['all_ceafe_f1'],
            'A_CoNLL_F1': cur_experiment_score['all_conll_f1'],
        })
    print(f"OUTPUT: {suffix}输出到{file_path}")


def save_clustering_scores_into_csv_in_table_format(experiments_scores, output_path, suffix="scores_clustering_table.csv"):
    result = {}
    #
    all_data = set()
    all_template = set()
    all_setting = set()
    for cur_experiment_score in experiments_scores:
        data = cur_experiment_score['data']
        model_name = cur_experiment_score['model_name']
        model_config = cur_experiment_score['model_config']
        template = cur_experiment_score['template']
        prefix_num = cur_experiment_score['prefix_num']
        sample = cur_experiment_score['sample']
        repeat = cur_experiment_score['repeat']
        setting = f"{model_name}({model_config})_{prefix_num}shot_{sample}(r{repeat})"
        e_muc_f1 = f"{round(cur_experiment_score['entity_muc_f1'], 2):.2f}"
        e_bcubed_f1 = f"{round(cur_experiment_score['entity_bcubed_f1'], 2):.2f}"
        e_ceafe_f1 = f"{round(cur_experiment_score['entity_ceafe_f1'], 2):.2f}"
        e_conll_f1 = f"{round(cur_experiment_score['entity_conll_f1'], 2):.2f}"
        v_muc_f1 = f"{round(cur_experiment_score['event_muc_f1'], 2):.2f}"
        v_bcubed_f1 = f"{round(cur_experiment_score['event_bcubed_f1'], 2):.2f}"
        v_ceafe_f1 = f"{round(cur_experiment_score['event_ceafe_f1'], 2):.2f}"
        v_conll_f1 = f"{round(cur_experiment_score['event_conll_f1'], 2):.2f}"
        #
        all_data.add(data)
        all_setting.add(setting)
        all_template.add(template)
        #
        if data not in result:
            result[data] = {}
        if template not in result[data]:
            result[data][template] = {}
        if setting not in result[data][template]:
            result[data][template][setting] = ""
        #
        result[data][template][setting] = f"EMUC{e_muc_f1};EBcubed{e_bcubed_f1};ECEAFe{e_ceafe_f1};ECoNLL{e_conll_f1};VMUC{v_muc_f1};VBcubed{v_bcubed_f1};VCEAFe{v_ceafe_f1};VCoNLL{v_conll_f1};"
    #
    all_data = list(all_data)
    all_setting = list(all_setting)
    all_template = list(all_template)
    #
    file_path = os.path.join(output_path, suffix)
    csvfile = open(file_path, mode="w", newline='', encoding='utf-8')
    header = ['template'] + all_setting
    writer = csv.DictWriter(csvfile, fieldnames=header)
    writer.writeheader()
    for cur_data in result.keys():
        # 不同的data新开一个表，用单独的一行作为分隔
        writer.writerow({'template': cur_data})
        for cur_template in result[cur_data]:
            # 写一行数据
            row = {'template': cur_template}
            for cur_model_setting in all_setting:
                if cur_model_setting in result[cur_data][cur_template]:
                    row[cur_model_setting] = result[cur_data][cur_template][cur_model_setting]
                else:
                    row[cur_model_setting] = '-'
            writer.writerow(row)
    print(f"OUTPUT: {suffix}输出到{file_path}")


def coreference_scorer(corpus, output_path, output_prefix=""):
    #
    gold_event_path = os.path.join(output_path, f'{output_prefix}.event.key_conll')
    gold_entity_path = os.path.join(output_path, f'{output_prefix}.entity.key_conll')
    gold_all_path = os.path.join(output_path, f'{output_prefix}.all.key_conll')
    pred_event_path = os.path.join(output_path, f'{output_prefix}.event.response_conll')
    pred_entity_path = os.path.join(output_path, f'{output_prefix}.entity.response_conll')
    pred_all_path = os.path.join(output_path, f'{output_prefix}.all.response_conll')
    event_conll_path = os.path.join(output_path, f'{output_prefix}.event.scores.txt')
    entity_conll_path = os.path.join(output_path, f'{output_prefix}.entity.scores.txt')
    all_conll_path = os.path.join(output_path, f'{output_prefix}.all.scores.txt')
    # 1 保存真值
    logging.info('Creating mention-based mentions key file')
    write_mention_based_cd_clusters(corpus, is_event=True, is_gold=True, out_file=gold_event_path)
    write_mention_based_cd_clusters(corpus, is_event=False, is_gold=True, out_file=gold_entity_path)
    write_mention_based_cd_clusters(corpus, is_event=None, is_gold=True, out_file=gold_all_path)
    # 2 保存预测值
    logging.info('Creating mention-based mentions response file')
    write_mention_based_cd_clusters(corpus, is_event=True, is_gold=False, out_file=pred_event_path)
    write_mention_based_cd_clusters(corpus, is_event=False, is_gold=False, out_file=pred_entity_path)
    write_mention_based_cd_clusters(corpus, is_event=None, is_gold=False, out_file=pred_all_path)
    # 3 计算性能
    # 3.1 生成命令
    logging.info('Calc metrics')
    event_scorer_command = (f'perl scorer/scorer.pl all {gold_event_path} {pred_event_path} none > {event_conll_path} \n')
    """
    perl scorer / scorer.pl all E:\ProgramCode\Barhom\Barhom2019My\event_entity_coref_ecb_plus\data\gold\cybulska_gold\CD_test_entity_mention_based.key_conll output\CD_test_entity_mention_based.response_conll none > 输出路径
    """
    entity_scorer_command = (f'perl scorer/scorer.pl all {gold_entity_path} {pred_entity_path} none > {entity_conll_path} \n')
    all_scorer_command = (f'perl scorer/scorer.pl all {gold_all_path} {pred_all_path} none > {all_conll_path} \n')
    # 3.2 执行命令
    processes = [subprocess.Popen(event_scorer_command, shell=True),
                 subprocess.Popen(entity_scorer_command, shell=True),
                 subprocess.Popen(all_scorer_command, shell=True)
                 ]
    while processes:
        status = processes[0].poll()
        if status is not None:
            processes.pop(0)
    logging.info('Running scorers has been done.')
    # 4 精简结果
    # 4.1 读取perl的打分结果
    event_scores = read_conll_f1(event_conll_path)
    entity_scores = read_conll_f1(entity_conll_path)
    all_scores = read_conll_f1(all_conll_path)
    scores = {}
    scores.update({f"entity_{k}": v for k, v in entity_scores.items()})
    scores.update({f"event_{k}": v for k, v in event_scores.items()})
    scores.update({f"all_{k}": v for k, v in all_scores.items()})
    #
    return scores


def main():
    # config
    config_dict = {
        "corpus_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\data\5.clustering\['36_ecb'](strategy3)_ground_truth_model(none)_0shot_t16DAM_noSample(r1).c",
        "output_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\output"
    }
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


    # 读取corpus
    logging.info('Loading corpus data...')
    with open(config_dict["corpus_path"], 'rb') as f:
        corpus = cPickle.load(f)
    logging.info('Corpus data have been loaded.')

    # 打分
    prefix = os.path.basename(config_dict["corpus_path"])[:-2] + ".clustering1"
    scores = coreference_scorer(corpus, config_dict["output_path"], output_prefix=prefix)

    # 保存
    path = os.path.join(config_dict["output_path"], f"{prefix}.scores")
    with open(path, 'w', encoding="utf8") as f:
        f.writelines([f"{k}: {v}\n" for k, v in scores.items()])
    print(f"OUTPUT: clustering scores saved in {path}")
    logging.info(f"OUTPUT: clustering scores saved in {path}")


if __name__ == '__main__':
    main()
