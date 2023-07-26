import logging
import _pickle as cPickle
import os
import subprocess
from src.shared.classes import Corpus, Topic, Document, Sentence, Mention, EventMention, EntityMention, Token, Srl_info, Cluster


logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

corpus_path = r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models\event_entity_coref_ecb_plus\scorer\test_data.after"
output_path = r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models\event_entity_coref_ecb_plus\scorer\output"


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
                mentions = curr_sent.gold_event_mentions if is_event else curr_sent.gold_entity_mentions
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
    f1_list = []
    with open(filename, "r") as ins:
        for line in ins:
            new_line = line.strip()
            if new_line.find('F1:') != -1:
                f1_list.append(float(new_line.split(': ')[-1][:-1]))

    muc_f1 = f1_list[1]
    bcued_f1 = f1_list[3]
    ceafe_f1 = f1_list[7]

    return (muc_f1 + bcued_f1 + ceafe_f1)/float(3)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    # 读取corpus
    logger.info('Loading corpus data...')
    with open(corpus_path, 'rb') as f:
        corpus = cPickle.load(f)
    logger.info('Corpus data have been loaded.')
    #
    gold_event_path = os.path.join(output_path, 'CD_test_event_mention_based.key_conll')
    gold_entity_path = os.path.join(output_path, 'CD_test_entity_mention_based.key_conll')
    pred_event_path = os.path.join(output_path, 'CD_test_event_mention_based.response_conll')
    pred_entity_path = os.path.join(output_path, 'CD_test_entity_mention_based.response_conll')
    event_conll_path = os.path.join(output_path, 'event_scorer_cd_out.txt')
    entity_conll_path = os.path.join(output_path, 'entity_scorer_cd_out.txt')
    # 1 保存真值
    logger.info('Creating mention-based mentions key file')
    write_mention_based_cd_clusters(corpus, is_event=True, is_gold=True, out_file=gold_event_path)
    write_mention_based_cd_clusters(corpus, is_event=False, is_gold=True, out_file=gold_entity_path)
    # 2 保存预测值
    logger.info('Creating mention-based mentions response file')
    write_mention_based_cd_clusters(corpus, is_event=True, is_gold=False, out_file=pred_event_path)
    write_mention_based_cd_clusters(corpus, is_event=False, is_gold=False, out_file=pred_entity_path)
    # 3 计算性能
    # 3.1 生成命令
    logger.info('Calc metrics')
    event_scorer_command = (f'perl scorer/scorer.pl all {gold_event_path} {pred_event_path} none > {event_conll_path} \n')
    """
    perl scorer / scorer.pl all E:\ProgramCode\Barhom\Barhom2019My\event_entity_coref_ecb_plus\data\gold\cybulska_gold\CD_test_entity_mention_based.key_conll output\CD_test_entity_mention_based.response_conll none > 输出路径
    """
    entity_scorer_command = (f'perl scorer/scorer.pl all {gold_entity_path} {pred_entity_path} none > {entity_conll_path} \n')
    # 3.2 执行命令
    processes = [subprocess.Popen(event_scorer_command, shell=True),
                 subprocess.Popen(entity_scorer_command, shell=True)]
    while processes:
        status = processes[0].poll()
        if status is not None:
            processes.pop(0)
    logger.info('Running scorers has been done.')
    # # 4 精简结果
    # # 4.1 读取perl的打分结果
    # event_f1 = read_conll_f1(event_conll_path)
    # entity_f1 = read_conll_f1(entity_conll_path)
    # # 4.2 保存精简的打分结果
    # scores_file = open(os.path.join(output_path, 'conll_f1_scores.txt'), 'w', encoding="utf8")
    # scores_file.write('Event CoNLL F1: {}\n'.format(event_f1))
    # scores_file.write('Entity CoNLL F1: {}\n'.format(entity_f1))
    # scores_file.close()
