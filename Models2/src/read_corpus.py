# 标准库
import os
import sys
import json
import torch
import _pickle as cPickle
from typing import Dict, List, Tuple, Union
import logging
import shutil
# 三方库
import spacy
# 本地库
from classes import Corpus, Topic, Document, Sentence, Token, EventMention, EntityMention, MentionData


# config
config_dict = {
    "ecb_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\data\raw\ECBplus",
    "output_dir": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\output",
    "data_setup": 2,
    "selected_sentences_file": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\data\raw\ECBplus_coreference_sentences.csv",
    "read_selected_sentences_only": False
}
"""
data_setup: Union[1, 2]
    如何把ECB+数据集中的topics分为train、dev、test三部分。
    if data_setup == 1:  # Yang setup
        train_topics = range(1, 23)
        dev_topics = range(23, 26)
        test_topics = list(set(topic_list) - set(train_topics))
    else:  # Cybulska setup
        dev_topics = [2, 5, 12, 18, 21, 23, 34, 35]
        train_topics = [i for i in range(1, 36) if i not in dev_topics]  # train topics 1-35 , test topics 36-45
        test_topics = list(set(topic_list) - set(train_topics) - set(dev_topics))
read_selected_sentences_only: Bool
    Some sentences are selected in ECB+ corpus.
    True: Only the selected sentences are extracted.
    False: All sentences are extracted.
"""

# logging
logging.basicConfig(
    # 使用fileHandler,日志文件在输出路径中(test_log.txt)
    filename=os.path.join(config_dict["output_dir"], "log.txt"),
    filemode="w",
    # 配置日志级别
    level=logging.INFO
)

# output dir
if not os.path.exists(config_dict["output_dir"]):
    print(f"make output dir: {config_dict['output_dir']}")
    os.makedirs(config_dict["output_dir"])
elif len(os.listdir(config_dict["output_dir"])) > 1:  # 大于1是因为上边配置logging的时候就建立的log.txt这个文件
    input(f"output dir is not empty, press ENTER to continue.")

# spacy
nlp = spacy.load('en_core_web_sm')

# save this file itself
shutil.copy(os.path.abspath(__file__), config_dict["output_dir"])

def load_ECB_plus(token_info_list: list) -> Dict[str, Document]:
    r"""
    This function gets the intermediate data  ECB_Train/test/dev_corpus.text and load it into a dict of Document obj.

    Note: load_ECB_plus 不是说只load 源自X_Xecbplus.xml的数据。
    这里ECB_plus指的是整个语料库。
    参数指定的ECB_Train/Test/Dev_corpus.txt文件中包含整个语料库的信息。

    Example of the text file::

        1_10ecb	0	0	Perennial	-
        1_10ecb	0	1	party	-
        1_10ecb	0	2	girl	-
        1_10ecb	0	3	Tara	HUM16236184328979740
        1_10ecb	0	4	Reid	HUM16236184328979740

    Example of the return::

        {
            '1_10ecb': Document obj,
            '1_11ecb': Document obj,
            ...
        }

    In detail, each Document obj in return dict includes follow info::

        Document_obj
            Document_obj.sentences -> Sentence_obj
        Sentence_obj
            Sentence_obj.tokens -> Token_obj
        Token_obj

    :param processed_ecb_file: The path of the ECB_Train/test/dev_corpus.text.
        e.g. "data/interim/cybulska_setup/ECB_Train_corpus.txt".
    :return: A dictionary of document objects, which represents the documents in the split.
    """
    doc_changed = True
    sent_changed = True
    docs = {}
    last_doc_id = None
    last_sent_id = None

    for line in token_info_list:
        stripped_line = line.strip()  # 去掉多余的空格
        try:
            # if  line != "\n"
            if stripped_line:
                doc_id, sent_id, token_num, word, coref_chain = stripped_line.split('\t')
                doc_id = doc_id.replace('.xml', '')  # 这句是废话，因为doc_id中都没有“.xml”。
            # if line == "\n"
            else:
                pass
        except:
            # There may be a exception because some special line like:
            # '2_5ecbplus\t0\t9\tAwards\t\tACT16239369414744113'
            # There are 5 \t and you will get 6 elements, instead of 5 elements, after split('\t').
            # The '\t\t' makes a unexpected empty element.
            # So, you need to filter out the empty element.
            row = stripped_line.split('\t')
            clean_row = []
            for item in row:
                # append the normal element
                if item:
                    clean_row.append(item)
                # filter out the empty element
                else:
                    pass
            doc_id, sent_id, token_num, word, coref_chain = clean_row
            doc_id = doc_id.replace('.xml', '')  # 这句是废话，因为doc_id中都没有“.xml”。

        if stripped_line:
            sent_id = int(sent_id)

            # test the change of doc and sent
            if last_doc_id is None:
                last_doc_id = doc_id
            elif last_doc_id != doc_id:
                doc_changed = True
                sent_changed = True
            if last_sent_id is None:
                last_sent_id = sent_id
            elif last_sent_id != sent_id:
                sent_changed = True

            # new Document
            if doc_changed:
                new_doc = Document(doc_id)
                docs[doc_id] = new_doc
                doc_changed = False
                last_doc_id = doc_id

            # new Sentence
            if sent_changed:
                new_sent = Sentence(sent_id)
                new_doc.add_sentence(sent_id, new_sent)
                sent_changed = False
                last_sent_id = sent_id

            # new Token
            new_tok = Token(token_num, word, '-')
            new_sent.add_token(new_tok)

    return docs


def order_docs_by_topics(docs: Dict[str, Document]) -> Corpus:
    """
    Gets list of document objects and returns a Corpus object.
    The Corpus object contains Document objects which are ordered by their gold
    topics

    The returned Corpus obj has structure likes::

        普通变量是docs本来就有的信息，
        尖括号中的变量是本函数运行后添加的信息。
        <Courpus_obj>
            <Corpus_obj.topics> -> Topic_obj
        <Topic_obj>
            <Topic_obj.docs> -> Document_obj
        Document_obj
            Document_obj.sentences -> Sentence_obj
            gold/pred_event/entity_mentions -> Mention obj
        Sentence_obj
            Sentence_obj.tokens -> Token_obj
        Mention_obj
            cd/wd_coref_chain -> Coref_chain_str
            doc_id -> Document_obj
            sent_id -> Sentence_obj
            tokens -> Token_obj
        Token_obj
            gold_event/entity_cd/wd_coref_chain -> Coref_chain_str

    :param docs: dict of document objects
    :return: Corpus object
    """
    corpus = Corpus()
    for doc_id, doc in docs.items():
        topic_id, doc_no = doc_id.split('_')
        if 'ecbplus' in doc_no:
            topic_id = topic_id + '_' +'ecbplus'
        else:
            topic_id = topic_id + '_' +'ecb'
        if topic_id not in corpus.topics:
            topic = Topic(topic_id)
            corpus.add_topic(topic_id, topic)
        topic = corpus.topics[topic_id]
        topic.add_doc(doc_id, doc)
    return corpus


def load_mentions_from_json(mentions: list, docs: Dict[str, Document], is_event: bool, is_gold_mentions) -> None:
    """
    This function extract mention info from a given json file and add the
    mention info to param *docs*. The *docs* param has a structure as shown below::

        普通变量是docs本来就有的信息，
        尖括号中的变量是本函数运行后添加的信息。
        Document_obj
            Document_obj.sentences -> Sentence_obj
            <gold/pred_event/entity_mentions> -> Mention obj
        Sentence_obj
            Sentence_obj.tokens -> Token_obj
        <Mention_obj>
            <cd/wd_coref_chain> -> Coref_chain_str
            <doc_id> -> Document_obj
            <sent_id> -> Sentence_obj
            <tokens> -> Token_obj
        Token_obj
            <gold_event/entity_cd/wd_coref_chain> -> Coref_chain_str

    * This function has a Sanity check. Check whether mention in json file and the corresponding
      mention in docs has same token. If not this function stop and err info will be printed.

    * The json file is like (a list of **mention dict**)::

        [
            {
                "coref_chain": "HUM16284637796168708",
                "doc_id": "1_10ecb",
                "is_continuous": true,
                "is_singleton": false,
                "mention_type": "HUM",
                "score": -1.0,
                "sent_id": 0,
                "tokens_number": [
                    13
                ],
                "tokens_str": "rep"
            },
            {
                "coref_chain": "HUM16236184328979740",
                "doc_id": "1_10ecb",
                "is_continuous": true,
                "is_singleton": false,
                "mention_type": "HUM",
                "score": -1.0,
                "sent_id": 0,
                "tokens_number": [
                    3,
                    4
                ],
                "tokens_str": "Tara Reid"
            },


    :param mentions_json_file: path to the JSON file that contains the mentions.
        The json file has a content like that of
        ECB_Dev/Test/Train_Entity/Event_gold_mentions.json.

    :param docs: { 'XX_XXecb': a src.shared.classes.Document Obj }

    :param is_event: a boolean indicates whether the mention in json file is event or entity
     mentions.

    :param is_gold_mentions: a boolean indicates whether the mention in json file is gold or
     predicted mentions.
    """
    for mention_info in mentions:
        doc_id = mention_info.doc_id.replace('.xml', '')  # 这是废话，因为doc_id里都没有‘.xml’
        sent_id = mention_info.sent_id
        tokens_numbers = mention_info.tokens_number
        mention_type = mention_info.mention_type
        is_singleton = mention_info.is_singleton
        is_continuous = mention_info.is_continuous
        score = mention_info.score
        mention_str = mention_info.tokens_str
        if mention_str is None:
            print('Err: mention str is None:', mention_info)
        coref_chain = mention_info.coref_chain
        head_text, head_lemma = ("暂无", "暂无")  # find_head(mention_str)

        """Sanity check
        Check whether mention in json file and the corresponding Mention obj in 
        docs has same tokens. 
        """
        # Find the tokens of corresponding mention in docs.
        try:
            token_objects = docs[doc_id].get_sentences()[sent_id].find_mention_tokens(tokens_numbers)
        except:
            print('error when looking for mention tokens')
            print('doc id {} sent id {}'.format(doc_id, sent_id))
            print('token numbers - {}'.format(str(tokens_numbers)))
            print('mention string {}'.format(mention_str))
            print('sentence - {}'.format(docs[doc_id].get_sentences()[sent_id].get_raw_sentence()))
            raise  # stop the script
        if not token_objects:
            # Never hit this if condition. token_objects = []?.
            print('Can not find tokens of a mention - {} {} {}'.format(doc_id, sent_id,tokens_numbers))
        # whether the tokens are same
            pass

        # 1. add coref chain to Token.
        if is_gold_mentions:
            for token in token_objects:
                if is_event:
                    token.gold_event_coref_chain.append(coref_chain)
                else:
                    token.gold_entity_coref_chain.append(coref_chain)

        # 2. Create Mention
        if is_event:
            mention = EventMention(doc_id, sent_id, tokens_numbers, token_objects, mention_str, head_text, head_lemma, is_singleton, is_continuous, coref_chain)
        else:
            mention = EntityMention(doc_id, sent_id, tokens_numbers, token_objects, mention_str, head_text, head_lemma, is_singleton, is_continuous, coref_chain, mention_type)
        mention.is_person = mention_info.is_person
        mention.probability = score
        # a confidence score for predicted mentions (if used), gold mentions prob is setted to 1.0 in the json file.

        # 3. add Mention to Sentence
        if is_gold_mentions:
            docs[doc_id].get_sentences()[sent_id].add_gold_mention(mention, is_event)
        else:
            docs[doc_id].get_sentences()[sent_id].add_predicted_mention(
                mention, is_event,
                relaxed_match=config_dict["relaxed_match_with_gold_mention"])


def load_gold_mentions(docs: Dict[str, Document], entity_mentions: list, event_mentions: list) -> None:
    """
    This function loads given event and entity mentions as gold mention.
    No return. This function add the mention info into *docs*, instead of output
    a return value.

    Example of *docs*::
        {'XX_XXecb': Document Obj, ... }

    Example of *events_json* and *entities_json*::
        "data/interim/cybulska_setup/ECB_Train_Event_gold_mentions.json"

    :param docs: A dict of Document objects of train/text/dev set.
    :param events_json:  Path to the JSON file which contains the gold event
    mentions of a specific split - train/dev/test
    :param entities_json: Path to the JSON file which contains the gold entity
    mentions of a specific split - train/dev/test
    """
    load_mentions_from_json(event_mentions, docs, is_event=True, is_gold_mentions=True)
    load_mentions_from_json(entity_mentions, docs, is_event=False, is_gold_mentions=True)


def load_predicted_mentions(docs: Dict[str, Document], events_json: str, entities_json: str) -> None:
    """
    This function loads given event and entity mentions as predicted mention.
    No return. This function add the mention info into *docs*, instead of output
    a return value.

    Example of *docs*::
        {'XX_XXecb': Document Obj, ... }

    Example of *events_json* and *entities_json*::
        "data/interim/cybulska_setup/ECB_Train_Event_pred_mentions.json"

    :param docs: A dict of document objects of train/text/dev set.
    :param events_json:  Path to the JSON file which contains the predicted event
    mentions of a specific split - train/dev/test
    :param entities_json: Path to the JSON file which contains the predicted entity
    mentions of a specific split - train/dev/test
    """
    load_mentions_from_json(events_json, docs, is_event=True, is_gold_mentions=False)
    load_mentions_from_json(entities_json, docs, is_event=False, is_gold_mentions=False)


def load_predicted_data(docs: dict, pred_events_json: str, pred_entities_json: str):
    '''
    This function loads the predicted mentions and stored them within their
    suitable document objects (suitable for loading the test data)

    :param docs: dictionary that contains the document objects
    :param pred_events_json: path to the JSON file contains predicted event mentions
    :param pred_entities_json: path to the JSON file contains predicted entities mentions
    :return:
    '''
    logging.info('Loading predicted mentions...')
    load_predicted_mentions(docs, pred_events_json, pred_entities_json)


def find_head(mention_str: str) -> Tuple[str, str]:
    """
    This function find the root in dependency parsing of param *mention_str*.

    The head of the root is itself. The dependency type of the root is 'ROOT'.
    Based on those feature, we can find the root. For example::
        >>> import spacy
        >>> nlp = spacy.load('en')
        >>> text = "The yellow dog eat shite."
        >>> doc = nlp(text)
        >>> [(i, i.head) for i in doc]
        [(The, dog), (yellow, dog), (dog, eat), (eat, eat), (shite, eat), (., eat)]

    Usually, a mention or a sentence has only one root, as the example above shows.
    However, if your *mention_str* is long and complex, there can be more roots. For example::
        >>> text = "The yellow dog eat shite, but the white cat eat fish."
        >>> doc = nlp(text)
        >>> [(i, i.head) for i in doc]
        [(The, dog), (yellow, dog), (dog, eat), (eat, eat), (shite, eat), (,, eat), (but, eat), (the, cat), (white, cat), (cat, eat), (eat, eat), (fish, eat), (., eat)]
        >>> [(i, i.dep_) for i in doc]
        [(The, 'det'), (yellow, 'amod'), (dog, 'nsubj'), (eat, 'ROOT'), (shite, 'dobj'), (,, 'punct'), (but, 'cc'), (the, 'det'), (white, 'amod'), (cat, 'nsubj'), (eat, 'conj'), (fish, 'dobj'), (., 'punct')]
    The two 'eat' are root.
    In this case, this function find only the fist root.
    But, this function is not designed for this case. The param *mention_str* should be a real shour mention string which has only one root.

    After find the root, this function returns (text_of_root, lemma_of_root).
    Specially, lemma of pronone is '-PRON-', for example the 'you' in 'spaCy is designed to help you do real work'.
    In this case, this function returns (text_of_root, lower_case_of_root)

    :param mention_str: A mention string.
    :return: (text_of_root, lemma_of_root) or (text_of_root, lower_case_of_root)
    """
    mention = nlp(mention_str)
    for token in mention:
        # The token whose head is itself is the root
        if token.head == token:  # if token.dep_ == 'ROOT'
            if token.lemma_ == u'-PRON-':
                return token.text, token.text.lower()
            return token.text, token.lemma_


def have_string_match(mention,arg_str ,arg_start, arg_end):
    '''
    This function checks whether a given entity mention has a string match (strict or relaxed)
    with a span of an extracted argument
    :param mention: a candidate entity mention
    :param arg_str: the argument's text
    :param arg_start: the start index of the argument's span
    :param arg_end: the end index of the argument's span
    :return: True if there is a string match (strict or relaxed) between the entity mention
    and the extracted argument's span, and false otherwise
    '''
    if mention.mention_str == arg_str and mention.start_offset == arg_start:  # exact string match + same start index
        return True
    if mention.mention_str == arg_str:  # exact string match
        return True
    if mention.start_offset >= arg_start and mention.end_offset <= arg_end:  # the argument span contains the mention span
        return True
    if arg_start >= mention.start_offset and arg_end <= mention.end_offset:  # the mention span contains the mention span
        return True
    if len(set(mention.tokens_numbers).intersection(set(range(arg_start,arg_end + 1)))) > 0: # intersection between the mention's tokens and the argument's tokens
        return True
    return False


def add_arg_to_event(entity: EntityMention, event: EventMention, rel_name: str):
    '''
    Adds the entity mention as an argument (in a specific role) of an event mention and also adds the
    event mention as predicate (in a specific role) of the entity mention

    :param entity: an entity mention object
    :param event: an event mention object
    :param rel_name: the specific role
    '''
    if rel_name == 'A0':
        event.arg0 = (entity.mention_str, entity.mention_id)
        entity.add_predicate((event.mention_str, event.mention_id), 'A0')
    elif rel_name == 'A1':
        event.arg1 = (entity.mention_str, entity.mention_id)
        entity.add_predicate((event.mention_str, event.mention_id), 'A1')
    elif rel_name == 'AM-TMP':
        event.amtmp = (entity.mention_str, entity.mention_id)
        entity.add_predicate((event.mention_str, event.mention_id), 'AM-TMP')
    elif rel_name == 'AM-LOC':
        event.amloc = (entity.mention_str, entity.mention_id)
        entity.add_predicate((event.mention_str, event.mention_id), 'AM-LOC')


def find_argument(rel_name, rel_tokens, matched_event, sent_entities, sent_obj, is_gold, srl_obj):
    '''
    This function matches between an argument of an event mention and an entity mention.

    :param rel_name: the specific role of the argument
    :param rel_tokens: the argument's tokens
    :param matched_event: the event mention
    :param sent_entities: a entity mentions exist in the event's sentence.
    :param sent_obj: the object represents the sentence
    :param is_gold: whether the argument need to be matched with a gold mention or not
    :param srl_obj: an object represents the extracted SRL argument.
    :return: True if the extracted SRL argument was matched with an entity mention.
    '''
    arg_start_ix = rel_tokens[0]
    if len(rel_tokens) > 1:
        arg_end_ix = rel_tokens[1]
    else:
        arg_end_ix = rel_tokens[0]

    if arg_end_ix >= len(sent_obj.get_tokens()):
        print('argument bound mismatch with sentence length')
        print('arg start index - {}'.format(arg_start_ix))
        print('arg end index - {}'.format(arg_end_ix))
        print('sentence length - {}'.format(len(sent_obj.get_tokens())))
        print('raw sentence: {}'.format(sent_obj.get_raw_sentence()))
        print('matched event: {}'.format(str(matched_event)))
        print('srl obj - {}'.format(str(srl_obj)))

    arg_str, arg_tokens = sent_obj.fetch_mention_string(arg_start_ix, arg_end_ix)

    entity_found = False
    matched_entity = None
    for entity in sent_entities:
        if have_string_match(entity, arg_str, arg_start_ix, arg_end_ix):
            if rel_name == 'AM-TMP' and entity.mention_type != 'TIM':
                continue
            if rel_name == 'AM-LOC' and entity.mention_type != 'LOC':
                continue
            entity_found = True
            matched_entity = entity
            break
    if entity_found:
        add_arg_to_event(matched_entity, matched_event, rel_name)
        if is_gold:
            return True
        else:
            if matched_entity.gold_mention_id is not None:
                return True
            else:
                return False
    else:
        return False


def match_entity_with_srl_argument(sent_entities, matched_event ,srl_arg,rel_name, is_gold):
    '''
    This function matches between an argument of an event mention and an entity mention.
    Designed to handle the output of Allen NLP SRL system
    :param sent_entities: the entity mentions in the event's sentence
    :param matched_event: the event mention
    :param srl_arg: the extracted argument
    :param rel_name: the role name
    :param is_gold: whether to match the argument with gold entity mention or with predicted entity mention
    :return:
    '''
    found_entity = False
    matched_entity = None
    for entity in sent_entities:
        if srl_arg.ecb_tok_ids == entity.tokens_numbers or \
                srl_arg.text == entity.mention_str or \
                srl_arg.text in entity.mention_str or \
                entity.mention_str in srl_arg.text:
            if rel_name == 'AM-TMP' and entity.mention_type != 'TIM':
                continue
            if rel_name == 'AM-LOC' and entity.mention_type != 'LOC':
                continue
            found_entity = True
            matched_entity = entity

        if found_entity:
            break

    if found_entity:
        add_arg_to_event(matched_entity, matched_event, rel_name)
        if is_gold:
            return True
        else:
            if matched_entity.gold_mention_id is not None:
                return True
            else:
                return False
    else:
        return False


def find_topic_gold_clusters(topic):
    '''
    Finds the gold clusters of a specific topic
    :param topic: a topic object
    :return: a mapping of coref chain to gold cluster (for a specific topic) and the topic's mentions
    '''
    event_mentions = []
    entity_mentions = []
    # event_gold_tag_to_cluster = defaultdict(list)
    # entity_gold_tag_to_cluster = defaultdict(list)

    event_gold_tag_to_cluster = {}
    entity_gold_tag_to_cluster = {}

    for doc_id, doc in topic.docs.items():
        for sent_id, sent in doc.sentences.items():
            event_mentions.extend(sent.gold_event_mentions)
            entity_mentions.extend(sent.gold_entity_mentions)

    for event in event_mentions:
        if event.gold_tag != '-':
            if event.gold_tag not in event_gold_tag_to_cluster:
                event_gold_tag_to_cluster[event.gold_tag] = []
            event_gold_tag_to_cluster[event.gold_tag].append(event)
    for entity in entity_mentions:
        if entity.gold_tag != '-':
            if entity.gold_tag not in entity_gold_tag_to_cluster:
                entity_gold_tag_to_cluster[entity.gold_tag] = []
            entity_gold_tag_to_cluster[entity.gold_tag].append(entity)

    return event_gold_tag_to_cluster, entity_gold_tag_to_cluster, event_mentions, entity_mentions


def write_dataset_statistics(split_name: str, dataset: dict, check_predicted):
    '''
    Prints the split statistics.

    :param split_name: the split name (a string)
    :param dataset: an object represents the split
    :param check_predicted: whether to print statistics of predicted mentions too
    '''
    docs_count = 0
    sent_count = 0
    event_mentions_count = 0
    entity_mentions_count = 0
    event_chains_count = 0
    entity_chains_count = 0
    topics_count = len(dataset.topics.keys())
    predicted_events_count = 0
    predicted_entities_count = 0
    matched_predicted_event_count = 0
    matched_predicted_entity_count = 0


    for topic_id, topic in dataset.topics.items():
        event_gold_tag_to_cluster, entity_gold_tag_to_cluster, \
        event_mentions, entity_mentions = find_topic_gold_clusters(topic)

        docs_count += len(topic.docs.keys())
        sent_count += sum([len(doc.sentences.keys()) for doc_id, doc in topic.docs.items()])
        event_mentions_count += len(event_mentions)
        entity_mentions_count += len(entity_mentions)

        entity_chains = set()
        event_chains = set()

        for mention in entity_mentions:
            entity_chains.add(mention.gold_tag)

        for mention in event_mentions:
            event_chains.add(mention.gold_tag)

        # event_chains_count += len(set(event_gold_tag_to_cluster.keys()))
        # entity_chains_count += len(set(entity_gold_tag_to_cluster.keys()))

        event_chains_count += len(event_chains)
        entity_chains_count += len(entity_chains)

        if check_predicted:
            for doc_id, doc in topic.docs.items():
                for sent_id, sent in doc.sentences.items():
                    pred_events = sent.pred_event_mentions
                    pred_entities = sent.pred_entity_mentions

                    predicted_events_count += len(pred_events)
                    predicted_entities_count += len(pred_entities)

                    for pred_event in pred_events:
                        if pred_event.has_compatible_mention:
                            matched_predicted_event_count += 1

                    for pred_entity in pred_entities:
                        if pred_entity.has_compatible_mention:
                            matched_predicted_entity_count += 1

    with open(os.path.join(config_dict["output_dir"], '{}_statistics.txt'.format(split_name)), 'w') as f:
        f.write('Number of topics - {}\n'.format(topics_count))
        f.write('Number of documents - {}\n'.format(docs_count))
        f.write('Number of sentences - {}\n'.format(sent_count))
        f.write('Number of event mentions - {}\n'.format(event_mentions_count))
        f.write('Number of entity mentions - {}\n'.format(entity_mentions_count))

        if check_predicted:
            f.write('Number of predicted event mentions  - {}\n'.format(predicted_events_count))
            f.write('Number of predicted entity mentions - {}\n'.format(predicted_entities_count))
            f.write('Number of predicted event mentions that match gold mentions- '
                    '{} ({}%)\n'.format(matched_predicted_event_count,
                                        (matched_predicted_event_count/float(event_mentions_count)) *100 ))
            f.write('Number of predicted entity mentions that match gold mentions- '
                    '{} ({}%)\n'.format(matched_predicted_entity_count,
                                        (matched_predicted_entity_count / float(entity_mentions_count)) * 100))


def obj_dict(obj):
    obj_d = obj.__dict__
    obj_d = stringify_keys(obj_d)
    return obj_d


def stringify_keys(d):
    """Convert a dict's keys to strings if they are not."""
    for key in d.keys():

        # check inner dict
        if isinstance(d[key], dict):
            value = stringify_keys(d[key])
        else:
            value = d[key]

        # convert nonstring to string if needed
        if not isinstance(key, str):
            try:
                d[str(key)] = value
            except Exception:
                try:
                    d[repr(key)] = value
                except Exception:
                    pass

            # delete old key
            del d[key]
    return d


def set_elmo_embed_to_mention(mention, sent_embeddings):
    '''
    Sets the ELMo embeddings of a mention
    :param mention: event/entity mention object
    :param sent_embeddings: the embedding for each word in the sentence produced by ELMo model
    :return:
    '''
    head_index = mention.get_head_index()
    head_embeddings = sent_embeddings[int(head_index)]
    mention.head_elmo_embeddings = torch.from_numpy(head_embeddings)


def set_elmo_embeddings_to_mentions(elmo_embedder, sentence, set_pred_mentions):
    '''
     Sets the ELMo embeddings for all the mentions in the sentence
    :param elmo_embedder: a wrapper object for ELMo model of Allen NLP
    :param sentence: a sentence object
    '''
    avg_sent_embeddings = elmo_embedder.get_elmo_avg(sentence)
    event_mentions = sentence.gold_event_mentions
    entity_mentions = sentence.gold_entity_mentions

    for event in event_mentions:
        set_elmo_embed_to_mention(event,avg_sent_embeddings)

    for entity in entity_mentions:
        set_elmo_embed_to_mention(entity, avg_sent_embeddings)

    # Set the contextualized vector also for predicted mentions
    if set_pred_mentions:
        event_mentions = sentence.pred_event_mentions
        entity_mentions = sentence.pred_entity_mentions

        for event in event_mentions:
            set_elmo_embed_to_mention(event, avg_sent_embeddings)  # set the head contextualized vector

        for entity in entity_mentions:
            set_elmo_embed_to_mention(entity, avg_sent_embeddings)  # set the head contextualized vector


def read_selected_sentences(file_path: str) -> dict:
    """
    A CSV file is released with ECB+ corpus which list all "selected sentence" (the labeled sentences). This file
    contains the IDs of 1840 sentences which were manually reviewed and checked for correctness. The ECB+ creators
    recommend to use this subset of the dataset.
    the content in this .csv file is like:
        1 10ecbplus 1  \n
        1 10ecbplus 3  \n
        1 11ecbplus 1  .

    This function reads this file (given by param *file_path*), and returns a dictionary contains those sentences IDs.

    example1:
        >>> file_path = 'data\\raw\\ECBplus_coreference_sentences.csv'.
        >>> read_selected_sentences(file_path)
        {'1_10ecbplus.xml':[1,3],'1_11ecbplus.xml':[1]}

    :param file_path: path to the CSV file
    :return: a dictionary, where a key is an XML filename (i.e. ECB+ document) and the value is a list contains all
        the sentences IDs that were selected from that XML filename.

    """
    import csv
    xml_to_sent_dict = {}
    with open(file_path, 'rt') as csv_file:
        reader = csv.reader(csv_file, delimiter=',')
        # reader.next()
        next(reader)
        for line in reader:
            xml_filename = '{}_{}.xml'.format(line[0], line[1])
            sent_id = int(line[2])

            if xml_filename not in xml_to_sent_dict:
                xml_to_sent_dict[xml_filename] = []
            xml_to_sent_dict[xml_filename].append(sent_id)

    return xml_to_sent_dict


def m_tag_to_type(tag: str) -> str:
    """
    example::

        'ACT' = type_to_type_abbr('ACTION_REPORTING')
        'UNK' = type_to_type_abbr('UNKNOWN_INSTANCE_TAG')

    :param tag:  m_tag.
    :return: m_type.
    """
    if tag == 'UNKNOWN_INSTANCE_TAG':
        return 'UNK'
    if 'ACTION' in tag:
        return 'ACT'
    if 'LOC' in tag:
        return 'LOC'
    if 'NON' in tag:
        return 'NON'
    if 'HUMAN' in tag:
        return 'HUM'
    if 'TIME' in tag:
        return 'TIM'
    else:
        print('unknown tag:', tag)


def i_id_to_i_type(i_id: str) -> str:
    '''
    example::
        i_id_to_i_type('ACT150798320') -> 'ACT'

    :param i_id: i_id or anything that like a i_id( e.g. note property in cd r， instance_id
        property in cd tm).
    :return: type
    '''
    if 'ACT' in i_id or 'NEG' in i_id:
        return 'ACT'
    if 'LOC' in i_id:
        return 'LOC'
    if 'NON' in i_id:
        return 'NON'
    if 'HUM' in i_id or 'CON' in i_id:
        return 'HUM'
    if 'TIM' in i_id:
        return 'TIM'
    if 'UNK' in i_id:
        return 'UNK'


def sentencindex_to_bodysentenceindex(doc_id: str, sentenceindex: int, parse_all: bool) -> int:
    '''
    in xml file, a news text = [newsUrl] + [newsTime] + newsBody

    In X_Xecb.xml, news body starts with sentence 0. So sm_bodysentenceindex = sm_sentenceindex.
    A sm_bodysentenceindex is different from sm_sentenceindex in 2 condition:
        - 1st: Only in X_Xecbplus.xml, there is a newsUrl which is the sentence 0. So, newsBody
          starts with sentence 1.
        - 2nd: Only in 9_3ecbplus.xml and 9_4ecbplus.xml, there is a newsTime which is the sentence
          1. So, newsBody starts with sentence 2.
    '''
    bodysentenceindex = sentenceindex
    # 1st condition: there is url sentence.
    if 'plus' in doc_id:
        if int(bodysentenceindex) > 0:
            bodysentenceindex -= 1
    # 2nd condition: there is time sentence.
    if parse_all and (doc_id == '9_3ecbplus' or doc_id == '9_4ecbplus'):
        if bodysentenceindex > 0:
            bodysentenceindex -= 1
    return bodysentenceindex


def calc_split_statistics(dataset_split, split_name, statistics_file_name):
    '''
    This function calculates and saves the statistics of a split (train/dev/test) into a file.
    :param dataset_split: a list that contains all the mention objects in the split
    :param split_name: the split name (a string)
    :param statistics_file_name: a filename for the statistics file
    '''
    event_mentions_count = 0
    human_mentions_count = 0
    non_human_mentions_count = 0
    loc_mentions_count = 0
    time_mentions_count = 0
    non_continuous_mentions_count = 0
    unk_coref_mentions_count = 0
    coref_chains_dict = {}

    for mention_obj in dataset_split:
        mention_type = mention_obj.mention_type
        if 'ACT' in mention_type or 'NEG' in mention_type:
            event_mentions_count += 1
        elif 'NON' in mention_type:
            non_human_mentions_count += 1
        elif 'HUM' in mention_type:
            human_mentions_count += 1
        elif 'LOC' in mention_type:
            loc_mentions_count += 1
        elif 'TIM' in mention_type:
            time_mentions_count += 1
        else:
            print(mention_type)

        is_continuous = mention_obj.is_continuous
        if not is_continuous:
            non_continuous_mentions_count += 1

        coref_chain = mention_obj.coref_chain
        if 'UNK' in coref_chain:
            unk_coref_mentions_count += 1
        if coref_chain not in coref_chains_dict:
            coref_chains_dict[coref_chain] = 1
    with open(statistics_file_name, 'a') as f:

        f.write('{} statistics\n'.format(split_name))
        f.write('-------------------------\n')
        f.write('Number of event mentions - {}\n'.format(event_mentions_count))
        f.write('Number of human participants mentions - {}\n'.format(human_mentions_count))
        f.write('Number of non-human participants mentions - {}\n'.format(non_human_mentions_count))
        f.write('Number of location mentions - {}\n'.format(loc_mentions_count))
        f.write('Number of time mentions - {}\n'.format(time_mentions_count))
        f.write('Total number of mentions - {}\n'.format(len(dataset_split)))

        f.write('Number of non-continuous mentions - {}\n'.format(non_continuous_mentions_count))
        f.write('Number of mentions with coref id = UNK - {}\n'.format(unk_coref_mentions_count))
        f.write('Number of coref chains = {}\n'.format(len(coref_chains_dict)))
        f.write('\n')


# def save_gold_mention_statistics(train_extracted_mentions, dev_extracted_mentions,
#                                  test_extracted_mentions):
#     '''
#     This function calculates and saves the statistics of each split (train/dev/test) into a file.
#     :param train_extracted_mentions: a list that contains all the mention objects in the train split
#     :param dev_extracted_mentions: a list that contains all the mention objects in the dev split
#     :param test_extracted_mentions: a list that contains all the mention objects in the test split
#     '''
#     logging.info('Calculate mention statistics...')
#
#     all_data_mentions = train_extracted_mentions + dev_extracted_mentions + test_extracted_mentions
#     filename = 'mention_stats.txt'
#     calc_split_statistics(train_extracted_mentions, 'Train set',
#                           os.path.join(args.output_dir, filename))
#
#     calc_split_statistics(dev_extracted_mentions, 'Dev set',
#                           os.path.join(args.output_dir, filename))
#
#     calc_split_statistics(test_extracted_mentions, 'Test set',
#                           os.path.join(args.output_dir, filename))
#
#     calc_split_statistics(all_data_mentions, 'Total',
#                           os.path.join(args.output_dir, filename))
#
#     logging.info('Save mention statistics...')





def read_ecb_plus_doc(selected_sent_list: List[int],
                      doc_name: str,
                      doc_id: str,
                      extracted_tokens: List[str],
                      extracted_mentions: List[Union[None, MentionData]],
                      parse_all: bool,
                      load_singletons: bool):
    """
    Read xml file of ecb+ corpora, extract t info and sm info, and save it.

    ---------------
     t info
    ---------------
    t info are saved into the text file given by *output_file_obj*.

    - A t will be saved when the following statement is True::

        parse_all or (t_sentenceindex in selected_sent_list)

    - In the file, one line corresponds to one t.
      There is one black line after the end token of a sentence.
      And there are 5 columns:

      - the 1st column: *doc_id* parameter of this function.
      - the 2nd column: t_bodysentenceindex. For detail , see sentencindex_to_bodysentenceindex
      - the 3rd column: t_id.
      - the 4th column: t_text, and if t_text is ' ' or '\t', it will be replaced by a '-'.
      - The 5th column: the id of the i that this t refer to.

        - '-', if this t isn't in a mention( so it do not refer to any i).
        - wd_i_id( For detail, see wd_i_info.), if this t refer to a wd i.
        - cd_i_id( For detail, see cd_i_info.), if this t refer to a cd i.
        - sg_i_id( For detail, see sg_i_info.), if this t refer to a sg i and load singletons;
        - '-', if this t refer to a sg i and do not load singletons;

      example::

            14_5ecb	2	37	being	-
            14_5ecb	2	38	treated	ACT17478359686333015
            14_5ecb	2	39	as	-
            14_5ecb	2	40	potentially	-
            14_5ecb	2	41	suspicious	-
            14_5ecb	2	42	.	-
            14_5ecb	2	43	''	-

            14_6ecb	0	0	Residents	Singleton_HUM_14_14_6ecb
            14_6ecb	0	1	evacuated	Singleton_ACT_15_14_6ecb
            14_6ecb	0	2	from	-
            14_6ecb	0	3	their	INTRA_UNK_33690_14_6ecb
            14_6ecb	0	4	homes	INTRA_UNK_33690_14_6ecb
            14_6ecb	0	5	after	-
            14_6ecb	0	6	a	-
            14_6ecb	0	7	huge	-
            14_6ecb	0	8	fire	ACT17478306085573007

    ---------------
     sm info
    ---------------
    sm info: are saved into parameter *extracted_mentions* which is a shallow copied object.

    - A sm will be saved when the following statement is True::

        parse_all or (sm_sentenceindex in selected_sent_list)

    - sm info is saved as a *src.mention_data.MentionData* objects, which has the following property：

      - doc_id = doc_id
      - cur_t_sentenceindex = cur_sm_bodysentenceindex
      - tokens_numbers = cur_sm_tokenindex
      - tokens_str = cur_sm_string
      - coref_chain = cur_i_id
      - mention_type = cur_sm_type,
      - is_continuous = is_continuous
      - is_singleton = is_singleton
      - score = float(-1)

    :param selected_sent_list:
        A list of the sentenceindex( rather than a bodysentenceindex) of
        'selected sentence'. This parameter is activated only when parse_all=
        False, and then extract info from only those 'selected sentences'.
        e.g. [0, 3]
    :param doc_name:
        Path of the xml file. The xml file should be encoded with utf-8.
        e.g. 'data/raw/ECBplus\\2\\2_10ecbplus.xml' or 'data/raw/ECBplus\\1\\1_10ecb.xml'
    :param doc_id:
        Document ID of the xml file, in form of  {topic id}_{file
        name}{ecb/ecbplus type}).
        e.g. '1_10ecb' or '1_10ecbplus'.
    :param extracted_tokens:
        Each t info, which is represented by a string, is appended to this list.
    :param extracted_mentions:
        Each sm info, which is represented by src.data.mention_data.MentionData
        object, is appended to this list.
    :param parse_all:
        From which sentence to extract information
        True, extract info from all sentences in xml file as in Yang setup;
        False, extract info from only the selected sentences as in Cybulska setup.
            The selected sentences is given by parameter selected_sent_list.
    :param load_singletons:
        A boolean variable indicates whether to read singleton mentions as in
        Cybulska setup or whether to ignore them as in Yang setup.
    """
    ecb_file = open(doc_name, 'r', encoding='UTF-8')

    import xml.etree.ElementTree as ET  # for the parse of xml file.
    tree = ET.parse(ecb_file)
    root = tree.getroot()

    t_info: dict[str, dict[str, str]] = {}
    """
    {t_id: t_info_dict}
        - t_id: id of a t, **All t(selected or not) are included.** The 1st t in whole
          doc has a t_id = 1.

          - Value: 't_id' attr of this t label.
          - Type: str 

        - t_info_dict['t_text']: text of this t.

          - Value: text of this t label, e.g. 'has'.
          - Type: str.

        - t_info_dict['t_sentenceindex']:  index of this sentence, the 1st sentence in
          whole doc is index 0. 

          - Value: 'sentence' attr of this t label, e.g. "3".
          - Type: str.

        - t_info_dict['t_bodysentenceindex']:  index of this sentence, the 1st sentence in
          news body is index 0. For detail about sentenceindex and bodysentenceindex, see
          *sentencindex_to_bodysentenceindex*.

          - Value: calculated by *sentencindex_to_bodysentenceindex(sentencindex)*, e.g. "1"
          - Type: str

        - t_info_dict['t_tokenindex']: index of this t, the 1st t in cur sentence is
          index 0.

          - Value: 'number' attr of this t, e.g. "0".
          - Type: str.

        - t_info_dict['i_id']: id of the corresponding i.

          - wd_i_id( For detail, see wd_i_info.) for a wd i. 
          - cd_i_id( For detail, see cd_i_info.) for a cd i. 
          - sg_i_id( For detail, see sg_i_info.) for a sg i, if load singletons; 
            None, if not load singletons.

        - t_info_dict['i_desc']: description of the corresponding i.

          - wd_i_desc for a wd i. For detail, see wd_i_info.
          - cd_i_desc for a cd i. For detail, see cd_i_info.
          - 'padding' for a sg i, if load singletons; None, if not load singleton.

    """

    sm_info: dict[str, str] = {}
    """
    {sm_id: sm_tag}
        - sm_id: id of a sm. 
          **All(selected and not; sg, wd and cd) sm are included.**

          - Value: 'm_id' attr of this sm label.

        - sm_tag: 

          -Value: tag of this sm label.
    """

    tm_info: dict[str, tuple[str, str]] = {}
    """
    {'tm_id':(tm_desc, i_id_or_tag)}
        - tm_id: id of a tm. **All( selected and not) cd and wd tm are included.**

          - Value: 'm_id' attr of this tm label. 

        - tm_desc: describe of this tm.

          - Value: 'TAG_DESCRIPTOR' attr of the tm label. 

        - i_id_or_tag: info of the corresponding i:
          for a cd tm, we have the i_id of the corresponding i; 
          for a wd tm, we have the i_type of the corresponding i.

          - Value:

            - for CD tm, it is the 'instance_id' attr of this tm label.
            - for WD tm, it is the tag of this tm label.

    """

    mapped_sm_id: [str] = []
    """
    [sm_id]
        - sm_id: id of a sm which refer to a tm. 
          All(seleted and not; CD and WD sm, except SG sm) sm which refer to a tm are
          included.
    """

    sm_id_to_t_id: dict[str, list[str]] = {}
    """
    {sm_id: sm_tidlist}
        - sm_id: id of a sm. **All sm included( selected and not; sg sm(if loaded), wd sm and
          cd sm).**
        - sm_tidlist:[t_id,t_id,...]. t_id is the id of all(in cur doc) the t that
          refer to this sm.
    """

    sm_id_to_i_id: dict[str, str] = {}
    """
    {sm_id: i_id}
        - sm_id: id of a sm. **All sm included( selected or not; cd sm, wd sm and 
          sg sm).**
        - i_id: 

          - a wd_i_id(for detail, see wd_i_info) for a wd i. 
          - a cd_i_id(for detail, see cd_i_info) for a cd i.
          - a sg_i_if(for detail, see sg_i_info) for a sg i, if load_singletons=True.
    """

    wd_i_info: dict[str, tuple[list[str], str]] = {}
    """
    {wd_i_id:(t_id_list, wd_i_desc)}
        - wd_i_id: represented in the form of 'INTRA_{wd_i_type}_{rId}_{docId}'.
            **All wd_i included( selected or not).**

          - wd_i_type: A wd_i has many tm and sm, each of them has a type. wd_i_type
            equals to the type if all the types are same; otherwise, a accordance strategy
            is need:

            - tm_type strategy: set wd_i_type as wd_tm_type without accordance check
            - Barhom2019 strategy: 
              this value is 'UNK' if the type of tm is 'UNKNOWN_INSTANCE_TAG', 
              otherwise, the value equals to the type of the first sm.

          - rId: A wd_i has only one wd_r, rId is the 'r_id' attr of this wd_r
          - docId:

        - t_id_list:[t_id, t_id, ...], t_id of all(in cur doc) the t that refer to this wd_i.
        - wd_i_desc: description string of this wd_i.
          The value equals to 'TAG_DESCRIPTOR' attr of the tm of this wd_i.
    """

    cd_i_info: dict[str, tuple[list[str], str]] = {}
    """
    {'cd_i_id': (cd_i_tidlist, cd_i_desc)}
        - cd_i_id: if the 'instance_id ' attr of corresponding tm and the 'note'
          attr of the corresponding r is equal, cd_i_id is this value. otherwise, a 
          accordance strategy is need:

        - cd_i_tidlist:[t_id, t_id, ...], t_id of all(in cur doc) the t that refer to
          this cd_i.
        - cd_i_desc: description string of this cd_i.
          The value equals to 'TAG_DESCRIPTOR' attr of the tm of this cd_i.
    """

    sg_i_info: dict[str, tuple[list[str], str]] = {}
    """
    **there is no sg_i in ecb+, only when load_singletons=True, we make a sg_i for every
    sg sm.**

    {'sg_i_id': (t_id_list, sg_i_desc)}
        - sg_i_id: in the form of 'Singleton_{sg_i_type}_{sg_sm_id}_{doc_id}'

          - sg_i_type: equals to the type of sg_sm.( A sg_i has only one sg_sm, no 
            accordance check needed)
          - sg_sm_id: id of the sg_sm.( A sg_i has only one sg_sm, no accordance check 
            needed)
          - doc_id: id of the doc

        - t_id_list:[t_id, t_id, ...], t_id of all(in cur doc) the t that refer to
          this sg_i. But it is None because we do not need this, the varable sg_i_tidlist
          here is to keep the same structer as cd_i_info and wd_i_info.
        - sg_i_desc: description string of this cd_i.But it is always 'padding' because
          there is no description for a sg i in ecb+.
    """

    # 1.extract info from <Markables>...</Markables> tag
    """
    iterate through every tag in <Markables>...</Markables>
    there are 4 kinds of conditions: token anchor and 3 kinds of mention: 
        (1)token anchor,
        (2)source mention,
        (3)CD target mention,
        (4)WD target mention:    
    e.g.::
        <Markables>
        (2) <XXX m_id="48" note="byCROMER" >
        (1)     <token_anchor t_id="19"/>
        (1)     <token_anchor t_id="20"/>
            </XXX>
        (3) <TIME_DURATION m_id="52" RELATED_TO="" TAG_DESCRIPTOR="t26_decades" instance_id="TIM18440826675897964" />
        (3) <ACTION_OCCURRENCE m_id="51" RELATED_TO="" TAG_DESCRIPTOR="t26_died" instance_id="ACT18440577880137709" />
        (4) <UNKNOWN_INSTANCE_TAG m_id="17" RELATED_TO="" TAG_DESCRIPTOR="" />"
        (4) <HUMAN_PART_PER m_id="40" RELATED_TO="" TAG_DESCRIPTOR="spokesman" />
        </Markables>
    """
    cur_m_id = ''
    for cur_tag in root.find('Markables').iter():
        if cur_tag.tag == 'Markables':
            continue
        # for condition (1)
        elif cur_tag.tag == 'token_anchor':
            sm_id_to_t_id[cur_m_id].append(cur_tag.attrib['t_id'])
        # for condition (2)(3)(4)
        else:
            cur_m_id = cur_tag.attrib['m_id']
            # for condition (3)(4), it is a tm
            if 'TAG_DESCRIPTOR' in cur_tag.attrib:
                # for condition (3), it is a CD tm
                if 'instance_id' in cur_tag.attrib:
                    tm_info[cur_m_id] = (
                        cur_tag.attrib['TAG_DESCRIPTOR'],
                        cur_tag.attrib['instance_id']
                    )
                # for condition (4), it is a WD tm
                else:
                    tm_info[cur_m_id] = (
                        cur_tag.attrib['TAG_DESCRIPTOR'],
                        cur_tag.tag
                    )
            # for condition (2), it is a source mention
            else:
                sm_id_to_t_id[cur_m_id] = []
                sm_info[cur_m_id] = cur_tag.tag

    # 2.extract info from <Relations><INTRA_DOC_COREF>...</INTRA_DOC_COREF></Relations>
    """
    iterate through every tag in <Markables><INTRA_DOC_COREF>...</INTRA_DOC_COREF></Markables>
    there are 3 kinds of conditions marked as (i1)(i2)(i3)
        <Relations>
        (i1) <INTRA_DOC_COREF r_id="37615" >
        (i2)    <source m_id="24" />
        (i2)    <source m_id="25" />
        (i3)    <target m_id="17" />
             </INTRA_DOC_COREF>
             ...
        </Relations>
    填写了:wd_i_info完成, mapped_sm_id填了wd_sm的部分，sm_id_to_i_id填了wd_sm的部分
    """
    cur_wd_i_id = ''
    cur_wd_i_tidlist = []
    for cur_wd_r in root.find('Relations').findall('INTRA_DOC_COREF'):
        for cur_label in cur_wd_r.iter():
            # for condition (i1), cur r is a WD r
            if cur_label.tag == 'INTRA_DOC_COREF':
                # two strategy for setting the cur_wd_i_type, leave the accordance check for later.
                strategy = 'Barhom2019 strategy'
                if strategy == 'Barhom2019 strategy':
                    cur_wd_tm_tag = tm_info[cur_wd_r.find('target').get('m_id')][1]
                    # this value is 'UNK' if the type of tm is 'UNKNOWN_INSTANCE_TAG'
                    if cur_wd_tm_tag == 'UNKNOWN_INSTANCE_TAG':
                        cur_wd_i_type = m_tag_to_type(cur_wd_tm_tag)
                    # otherwise, the value equals to the type of the first sm.
                    else:
                        cur_1th_wd_sm_tag = sm_info[cur_wd_r.find('source').get('m_id')]
                        cur_1th_wd_sm_type = m_tag_to_type(cur_1th_wd_sm_tag)
                        cur_wd_i_type = cur_1th_wd_sm_type
                    # cid of a WD coref is represented by 'INTRA_tagAttr_rId_docId'
                elif strategy == 'tm_type strategy':
                    # tm_type strategy:
                    # set wd_i_type as wd_tm_type immediately
                    cur_wd_tm_tag = tm_info[cur_wd_r.find('target').get('m_id')][1]
                    cur_wd_tm_type = m_tag_to_type(cur_wd_tm_tag)
                    cur_wd_i_type = cur_wd_tm_type
                elif strategy == '一致性检查':
                    pass  # 检查tm和所有sm的type，如果一致，才作为wd_i_type
                cur_wd_i_id = 'INTRA_{}_{}_{}'.format(cur_wd_i_type, cur_label.attrib['r_id'], doc_id)
                wd_i_info[cur_wd_i_id] = ()
            # for condition (i2), it is the sm in cur r
            elif cur_label.tag == 'source':
                cur_wd_i_tidlist += (sm_id_to_t_id[cur_label.attrib['m_id']])
                mapped_sm_id.append(cur_label.attrib['m_id'])
                sm_id_to_i_id[cur_label.attrib['m_id']] = cur_wd_i_id
            # for condition (i3), it is the tm in cur r
            elif cur_label.tag == 'target':
                wd_i_info[cur_wd_i_id] = (cur_wd_i_tidlist, tm_info[cur_label.attrib['m_id']][0])
                # end of iteration of cur relation, clear variable for iteration of the next relation.
                cur_wd_i_tidlist = []

    # 3. extract info from <Relations><CROSS_DOC_COREF>...</CROSS_DOC_COREF></Relations>
    """
    iterate through every tag in <Markables><CROSS_DOC_COREF>...</CROSS_DOC_COREF></Markables>
    there are 3 kinds of conditions marked as (c1)(c2)(c3)
        <Relations>
        (c1) <CROSS_DOC_COREF r_id="37623" note="ACT16235311629112331" >
        (c2)      <source m_id="36" />
        (c2)      <source m_id="37" />
        (c3)      <target m_id="49" />
              </CROSS_DOC_COREF>
        </Relations>
    填写了:cd_i_info完成, mapped_sm_id填了cd_sm的部分，sm_id_to_i_id填了cd_sm的部分
    """
    cur_cd_i_id = ''
    cur_cd_i_tidlist = []
    for cross_doc_coref in root.find('Relations').findall('CROSS_DOC_COREF'):
        for cur_label in cross_doc_coref.iter():
            # for condition (c1), cur r is CD r
            if cur_label.tag == 'CROSS_DOC_COREF':
                # set the cd_i_id as r_note immediately, leave the accordance check for later.
                cur_cd_r_note = cur_label.attrib['note']
                cur_cd_i_id = cur_cd_r_note
                cd_i_info[cur_cd_i_id] = ()
            # for condition (c2), it is the sm in cur r
            elif cur_label.tag == 'source':
                cur_cd_i_tidlist += (sm_id_to_t_id[cur_label.attrib['m_id']])
                mapped_sm_id.append(cur_label.attrib['m_id'])
                sm_id_to_i_id[cur_label.attrib['m_id']] = cur_cd_i_id
            # for condition (c3), it is the tm in cur r
            else:
                cd_i_info[cur_cd_i_id] = (
                    cur_cd_i_tidlist,
                    tm_info[cur_label.attrib['m_id']][0]
                )
                # end of iteration of cur relation, clear variable for iteration of the next relation.
                cur_cd_i_tidlist = []

    # 4. extract info from <token>...</token>
    """
    填写了:t_info基本完成，只是对属于sg_i的token，其i_id和i_desc全为None，没有根据load_singletons做判断。
    """
    for cur_t in root.findall('token'):
        t_info[cur_t.attrib['t_id']] = {
            't_text': cur_t.text,
            't_sentenceindex': cur_t.attrib['sentence'],
            't_tokenindex': cur_t.attrib['number'],
            'i_id': None,
            'i_desc': None
        }
    for cur_cd_i_id in cd_i_info:
        for cur_t_id in cd_i_info[cur_cd_i_id][0]:
            t_info[cur_t_id]['i_id'] = cur_cd_i_id
            t_info[cur_t_id]['i_desc'] = cd_i_info[cur_cd_i_id][1]
    for cur_wd_i_id in wd_i_info:
        for cur_t_id in wd_i_info[cur_wd_i_id][0]:
            t_info[cur_t_id]['i_id'] = cur_wd_i_id
            t_info[cur_t_id]['i_desc'] = wd_i_info[cur_wd_i_id][1]

    # 5. Load singletons if required
    """
    if load_singletons:
        填写了:cd_i_info完成； mapped_sm_id填了cd_sm的部分；
        填写了：sm_id_to_i_id填了sg_sm的部分,支持sm_id_to_i_id填完了；
        填写了：t_info中属于sg_i的token的i_id和i_desc，至此t_info填完了。
    """
    if load_singletons:
        # find the sg sm(singleton source mention is sm that isn't refer to any cd tm or wd tm)
        for mid in sm_id_to_t_id:
            if mid not in mapped_sm_id:
                # sm in sm_id_to_t_id include sg, cd and wd sm.
                # sm in mapped_sm_id include cd and wd sm.
                # so, the sg sm are the difference between them.
                # iterate through every sg sm
                cur_sg_sm_id = mid

                # (1. create instance id for each singleton mention
                cur_sg_sm_tag = sm_info[cur_sg_sm_id]
                cur_sg_sm_type = m_tag_to_type(cur_sg_sm_tag)
                cur_sg_i_type = cur_sg_sm_type  # a sg_i has only one sg_sm, no accordance check needed.
                cur_sg_i_id = 'Singleton_{}_{}_{}'.format(cur_sg_i_type, cur_sg_sm_id, doc_id)
                sg_i_info[cur_sg_i_id] = (sm_id_to_t_id[cur_sg_sm_id], "")
                # (2. updated sm_id_to_i_id
                # this mention is related to the singleton instance, so this mention appears
                # in this singleton coref, so this mention can be listed in sm_id_to_i_id
                sm_id_to_i_id[cur_sg_sm_id] = cur_sg_i_id
                # (3. updated tokens
                # the token of this mention had it rel_id property as None, after this mention
                # is related to the singleton instance, the rel_id property should save the info
                # of the singleton instance.
                unmapped_tids = sm_id_to_t_id[cur_sg_sm_id]
                for cur_t_id in unmapped_tids:
                    if t_info[cur_t_id]['i_id'] is None:
                        t_info[cur_t_id]['i_id'] = cur_sg_i_id
                        t_info[cur_t_id]['i_desc'] = 'padding'

    # 6. sm info is saved into *extracted_mentions*
    for cur_sm_id in sm_id_to_t_id:
        '''
        check if cur mention need to be save.
        if user want to parse all sentences, then process cur mention
        if user want to parse only the selected sentences, and cur mention is
            selected，then process cur mention
        '''
        cur_sm_sentenceindex = int(t_info[
                                       sm_id_to_t_id[cur_sm_id][0]
                                   ]['t_sentenceindex'])
        if not (parse_all or (cur_sm_sentenceindex in selected_sent_list)):
            continue

        # (1
        cur_sm_tag = sm_info[cur_sm_id]
        cur_sm_type = m_tag_to_type(cur_sm_tag)
        cur_sm_is_person = True if cur_sm_tag == "HUMAN_PART_PER" else False

        # (2
        cur_i_id = sm_id_to_i_id[cur_sm_id]
        cur_i_type = i_id_to_i_type(cur_i_id)

        """
        # the 2 types above should be same
        # if cur_sm_type != cur_i_type:
        #     print('err: diff types in same coref: {}'.format(cur_i_id))
        #     print('  type_attr_of_cur_c: {}'.format(cur_i_type))
        #     print('  type_attr_of_cur_m: {}'.format(cur_sm_type))
        """

        # (3
        cur_sm_tokenindex = []  # One sm has some t, t has *number* attr( token index), this is a list of every *number*
        cur_sm_textlist = []
        tids = sm_id_to_t_id[cur_sm_id]
        for cur_t_id in tids:
            cur_t = t_info[cur_t_id]
            if int(cur_t['t_tokenindex']) not in cur_sm_tokenindex:
                cur_sm_tokenindex.append(int(cur_t['t_tokenindex']))
                cur_sm_textlist.append(cur_t['t_text'])  # .encode('ascii', 'ignore')修改了这里
        cur_sm_string = ' '.join(cur_sm_textlist)

        # (4
        cur_sm_bodysentenceindex = sentencindex_to_bodysentenceindex(
            doc_id, cur_sm_sentenceindex, parse_all
        )

        # (5
        is_continuous = True if cur_sm_tokenindex == list(
            range(cur_sm_tokenindex[0], cur_sm_tokenindex[-1] + 1)) else False

        # (6
        is_singleton = True if 'Singleton' in cur_i_id else False

        # create mention obj based on the above info, and add it to extracted_mentions
        mention_obj = MentionData(doc_id,
                                  cur_sm_bodysentenceindex,
                                  cur_sm_tokenindex,
                                  cur_sm_string,
                                  cur_i_id,
                                  cur_sm_type,
                                  is_continuous=is_continuous,
                                  is_singleton=is_singleton,
                                  score=float(-1))
        mention_obj.is_person = cur_sm_is_person
        extracted_mentions.append(mention_obj)

    # 7. t info is saved into text file
    prev_t_bodysentenceindex = None  # bodysentenceindex of previous sentence
    for cur_t_id in t_info:
        cur_t = t_info[cur_t_id]
        cur_t_sentenceindex = int(cur_t['t_sentenceindex'])

        if not parse_all and cur_t_sentenceindex not in selected_sent_list:
            continue

        cur_t_tokenindex = int(cur_t['t_tokenindex'])
        cur_t_text = cur_t['t_text'] if (cur_t['t_text'] != '' and cur_t['t_text'] != '\t') else '-'
        cur_t_iid = cur_t['i_id']
        cur_t_bodysentenceindex = sentencindex_to_bodysentenceindex(
            doc_id, cur_t_sentenceindex, parse_all
        )

        # write into output file: if go to next sentence, go to next line
        if prev_t_bodysentenceindex is None or prev_t_bodysentenceindex != cur_t_bodysentenceindex:
            extracted_tokens.append("")
            prev_t_bodysentenceindex = cur_t_bodysentenceindex
        # write into output file: token info
        s = doc_id \
            + '\t' + str(cur_t_bodysentenceindex) \
            + '\t' + str(cur_t_tokenindex) \
            + '\t' + cur_t_text \
            + '\t' + (cur_t_iid if cur_t_iid is not None else '-')
        extracted_tokens.append(s)


def obj_dict(obj):
    return obj.__dict__


def read_corpora(file_to_selected_sentence: dict, parse_all: bool,
                 load_singletons: bool, data_setup: int):
    """
    read ecb+ corpora and extract info with given config.

    :param file_to_selected_sentence: This parameter is effective only when parse_all=False, and this
        parameter shows which sentence in a xml file will be extracted.
        e.g. {'1_10ecbplus.xml':[1,3],'1_11ecbplus.xml':[1]}
    :param parse_all: a boolean variable indicates whether to read all the ECB+ corpus as in
        Yang setup or whether to filter the sentences according to a selected sentences list
        as in Cybulska setup.
    :param load_singletons:  boolean variable indicates whether to read singleton mentions as in
        Cybulska setup or whether to ignore them as in Yang setup.
    :param data_setup: the variable indicates the strategy of corpus split, which topics are for dev
        set, which topics are test set, and which topics are for train set)
        - 1 for Yang and Choubey setup
        - 2 for Cybulska Kenyon-Dean setup (recommended).
    """
    # 1. topic_list
    topic_str_list = os.listdir(config_dict["ecb_path"])  # ['1', '10', '11', ...]
    topic_list = [int(d) for d in topic_str_list]  # [1, 10, 11, ...]
    """topic list, e.g. [1, 10, 11, ...]"""

    # 2. split train/dec/test topic
    if data_setup == 1:  # Yang setup
        train_topics = range(1, 23)
        dev_topics = range(23, 26)
        test_topics = list(set(topic_list) - set(train_topics))
    else:  # Cybulska setup
        dev_topics = [2, 5, 12, 18, 21, 23, 34, 35]
        train_topics = [i for i in range(1, 36) if i not in dev_topics]  # train topics 1-35 , test topics 36-45
        test_topics = list(set(topic_list) - set(train_topics) - set(dev_topics))
    print('train_topics:', train_topics)
    print('dev_topics:', dev_topics)
    print('test_topics:', test_topics)

    # 3. split train/dec/test file
    # classify all the ecb/ecb+ docs into train/dev/test set in sorted order.
    """
    The sort is like::
        1_10ecb, 1_11ecb, 1_12ecb, 1_13ecb, 1_14ecb, 1_15ecb, 1_17ecb, 1_18ecb, 1_19ecb, 1_1ecb, 1_2ecb, 1_3ecb, 1_4ecb, 1_5ecb, 1_6ecb, 1_7ecb, 1_8ecb, 1_9ecb, 
        3_1ecb, 3_2ecb, 3_3ecb, 3_4ecb, 3_5ecb, 3_6ecb, 3_7ecb, 3_8ecb, 3_9ecb, 
        4_10ecb, 4_11ecb, 4_12ecb, 4_13ecb, 4_14ecb, 4_1ecb, 4_2ecb, 4_3ecb, 4_4ecb, 4_5ecb, 4_6ecb, 4_7ecb, 4_8ecb, 4_9ecb, 
        ...
    可见这排序是失败的，1_1ecb排在1_19ecb后边，这叫什么事？
    """
    train_ecb_files_sorted = []
    r""" [([0, 3], 'data/raw/ECBplus\\1\\1_10ecb.xml', '1_10ecb'), ...] """
    dev_ecb_files_sorted = []
    r""" [([0], 'data/raw/ECBplus\\2\\2_11ecb.xml', '2_11ecb'), ...] """
    test_ecb_files_sorted = []
    r""" [([0, 1], 'data/raw/ECBplus\\36\\36_1ecb.xml', '36_1ecb'), ...] """
    train_ecb_plus_files_sorted = []
    r""" [([1, 3], 'data/raw/ECBplus\\1\\1_10ecbplus.xml', '1_10ecbplus'), ...] """
    dev_ecb_plus_files_sorted = []
    r""" [([1, 3, 4], 'data/raw/ECBplus\\2\\2_10ecbplus.xml', '2_10ecbplus'), ...] """
    test_ecb_plus_files_sorted = []
    r"""[([1, 11], 'data/raw/ECBplus\\36\\36_10ecbplus.xml', '36_10ecbplus'), ...] """
    # traverse the topics, and fill the above list
    for topic in sorted(topic_list):
        topic_id = str(topic)
        file_list = os.listdir(os.path.join(config_dict["ecb_path"], topic_id))
        """['1_10ecb.xml', '1_10ecbplus.xml', '1_11ecb.xml', ... ]"""
        # classify the file_list into ecb_file_list and ecb_plus_file_list
        ecb_file_list = []
        ecb_plus_file_list = []
        for file_name in file_list:
            if 'plus' in file_name:
                ecb_plus_file_list.append(file_name)
            else:
                ecb_file_list.append(file_name)
        # sort the two file_list
        ecb_file_list = sorted(ecb_file_list)  # 没卵用，sorted完后还跟之前一毛一样
        ecb_plus_file_list = sorted(ecb_plus_file_list)
        # traverse the ecb docs in cur topic, add info to train/test/dev_ecb_files_sorted list
        for ecb_file in ecb_file_list:
            # if user want to parse all sentences, then process cur doc
            # if user want to parse only the selected sentences, and cur doc includes at
            #   least 1 selected sentence, then process cur doc
            if parse_all or (ecb_file in file_to_selected_sentence):
                # get the relative path of xml file of cur doc
                xml_file_path = os.path.join(os.path.join(config_dict["ecb_path"], topic_id), ecb_file)
                # get the selected sentence id in cur doc
                if parse_all:
                    selected_sentences = None
                else:
                    selected_sentences = file_to_selected_sentence[ecb_file]
                # classify cur doc into train/dev/test
                if topic in train_topics:
                    train_ecb_files_sorted.append((selected_sentences, xml_file_path,
                                                   ecb_file.replace('.xml', '')))
                elif topic in dev_topics:
                    dev_ecb_files_sorted.append((selected_sentences, xml_file_path,
                                                 ecb_file.replace('.xml', '')))
                else:
                    test_ecb_files_sorted.append((selected_sentences, xml_file_path,
                                                  ecb_file.replace('.xml', '')))
        # traverse the ecb+ docs in cur topic, add info to train/test/dev_ecb_plus_files_sorted list
        for ecb_file in ecb_plus_file_list:
            if parse_all or ecb_file in file_to_selected_sentence:
                xml_file_path = os.path.join(os.path.join(config_dict["ecb_path"], topic_id), ecb_file)
                if parse_all:
                    selected_sentences = None
                else:
                    selected_sentences = file_to_selected_sentence[ecb_file]
                if topic in train_topics:
                    train_ecb_plus_files_sorted.append((selected_sentences,
                                                        xml_file_path, ecb_file.replace('.xml', '')))
                elif topic in dev_topics:
                    dev_ecb_plus_files_sorted.append((selected_sentences,
                                                      xml_file_path, ecb_file.replace('.xml', '')))
                else:
                    test_ecb_plus_files_sorted.append(
                        (selected_sentences, xml_file_path, ecb_file.replace('.xml', '')))
    # combine the above list
    train_files = train_ecb_files_sorted + train_ecb_plus_files_sorted
    r""" [([0, 3], 'data\\raw\\ECBplus\\1\\1_10ecb.xml', '1_10ecb'), ...] """
    dev_files = dev_ecb_files_sorted + dev_ecb_plus_files_sorted
    r""" [([0], 'data/raw/ECBplus\\2\\2_11ecb.xml', '2_11ecb'), ...] """
    test_files = test_ecb_files_sorted + test_ecb_plus_files_sorted
    r""" [([0, 1], 'data/raw/ECBplus\\36\\36_1ecb.xml', '36_1ecb'), ...] """

    # 4. extract train/dec/test mention/token info
    train_extracted_mentions: List[MentionData] = []
    """
    A sm **in train split** will be saved when the following statement is True::
        parse_all or (sm_sentenceindex in selected_sent_list)

    sm info is saved as a src.data.mention_data.MentionData objects, which has the following property：

    - doc_id = doc_id
    - cur_t_sentenceindex = cur_sm_bodysentenceindex
    - tokens_numbers = cur_sm_tokenindex
    - tokens_str = cur_sm_string
    - coref_chain = cur_i_id
    - mention_type = cur_sm_type,
    - is_continuous = is_continuous
    - is_singleton = is_singleton
    - score = float(-1)
    """
    dev_extracted_mentions: List[MentionData] = []
    """
    A sm **in dev split** will be saved when the following statement is True::
        parse_all or (sm_sentenceindex in selected_sent_list)

    sm info is saved as a src.data.mention_data.MentionData objects, which has the following property：

    - doc_id = doc_id
    - cur_t_sentenceindex = cur_sm_bodysentenceindex
    - tokens_numbers = cur_sm_tokenindex
    - tokens_str = cur_sm_string
    - coref_chain = cur_i_id
    - mention_type = cur_sm_type,
    - is_continuous = is_continuous
    - is_singleton = is_singleton
    - score = float(-1)
    """
    test_extracted_mentions: List[MentionData] = []
    """
    A sm **in test split** will be saved when the following statement is True::
        parse_all or (sm_sentenceindex in selected_sent_list)

    sm info is saved as a src.data.mention_data.MentionData objects, which has the following property：

    - doc_id = doc_id
    - cur_t_sentenceindex = cur_sm_bodysentenceindex
    - tokens_numbers = cur_sm_tokenindex
    - tokens_str = cur_sm_string
    - coref_chain = cur_i_id
    - mention_type = cur_sm_type,
    - is_continuous = is_continuous
    - is_singleton = is_singleton
    - score = float(-1)
    """
    train_extract_tokens = []
    """
    This list includes all tokens (selected or not) in train set.
    Each element corresponds to a token.
    There is a "" element between 2 sentence.
    e.g.::
        [
            '', 
            '1_10ecb\t0\t0\tPerennial\t-', 
            '1_10ecb\t0\t1\tparty\t-', 
            '1_10ecb\t0\t2\tgirl\t-', 
            ...
            '1_10ecb\t0\t16\t.\t-', 
            '', 
            '1_10ecb\t3\t0\tA\t-', 
            '1_10ecb\t3\t1\tfriend\tSingleton_HUM_29_1_10ecb',
            ...
        ]  
    """
    dev_extracted_tokens = []
    """
    This list includes all tokens (selected or not) in dev set.
    Each element corresponds to a token.
    There is a "" element between 2 sentence.
    e.g.::
        [
            '', 
            '1_10ecb\t0\t0\tPerennial\t-', 
            '1_10ecb\t0\t1\tparty\t-', 
            '1_10ecb\t0\t2\tgirl\t-', 
            ...
            '1_10ecb\t0\t16\t.\t-', 
            '', 
            '1_10ecb\t3\t0\tA\t-', 
            '1_10ecb\t3\t1\tfriend\tSingleton_HUM_29_1_10ecb',
            ...
        ]  
    """
    test_extracted_tokens = []
    """
    This list includes all tokens (selected or not) in test set.
    Each element corresponds to a token.
    There is a "" element between 2 sentence.
    e.g.::
        [
            '', 
            '1_10ecb\t0\t0\tPerennial\t-', 
            '1_10ecb\t0\t1\tparty\t-', 
            '1_10ecb\t0\t2\tgirl\t-', 
            ...
            '1_10ecb\t0\t16\t.\t-', 
            '', 
            '1_10ecb\t3\t0\tA\t-', 
            '1_10ecb\t3\t1\tfriend\tSingleton_HUM_29_1_10ecb',
            ...
        ]  
    """

    # extract mention and token info.
    for file in train_files:
        read_ecb_plus_doc(file[0], file[1], file[2], train_extract_tokens, train_extracted_mentions, parse_all,
                          load_singletons)
    for file in dev_files:
        read_ecb_plus_doc(file[0], file[1], file[2], dev_extracted_tokens, dev_extracted_mentions, parse_all,
                          load_singletons)
    for file in test_files:
        read_ecb_plus_doc(file[0], file[1], file[2], test_extracted_tokens, test_extracted_mentions, parse_all,
                          load_singletons)

    # token info is saved in txt file. i.e. ECB_Dev/Train/Test_corpus.txt
    all_mentions = train_extracted_mentions + dev_extracted_mentions + test_extracted_mentions
    return (train_extract_tokens, dev_extracted_tokens, test_extracted_tokens,
            train_extracted_mentions, dev_extracted_mentions, test_extracted_mentions,
            all_mentions)


def split_entity_and_event_mention(mentions_list):
    event_mentions = []
    entity_mentions = []
    for mention_obj in mentions_list:
        mention_type = mention_obj.mention_type
        if 'ACT' in mention_type or 'NEG' in mention_type:
            event_mentions.append(mention_obj)
        else:
            entity_mentions.append(mention_obj)
    return (entity_mentions, event_mentions)


def add_selected_tag(corpus, file_to_selected_sentence):
    for topic_id in corpus.topics.keys():
        cur_topic = corpus.topics[topic_id]
        for doc_id in cur_topic.docs.keys():
            cur_doc = cur_topic.docs[doc_id]
            cur_doc_name = cur_doc.doc_id + '.xml'
            for sent_id in cur_doc.sentences.keys():
                cur_sent = cur_doc.sentences[sent_id]
                if cur_doc_name not in file_to_selected_sentence.keys():
                    cur_sent.is_selected = False
                elif sent_id not in file_to_selected_sentence[cur_doc_name]:
                    cur_sent.is_selected = False
                else:
                    cur_sent.is_selected = True


def main():
    logging.info('Read ECB+ files')

    file_to_selected_sentence = read_selected_sentences(config_dict["selected_sentences_file"])

    (
        train_extract_tokens, dev_extracted_tokens, test_extracted_tokens,
        train_extracted_mentions, dev_extracted_mentions, test_extracted_mentions,
        all_mentions
    ) = read_corpora(
        file_to_selected_sentence=file_to_selected_sentence,
        parse_all=not config_dict["read_selected_sentences_only"],
        load_singletons=True,
        data_setup=2
    )

    logging.info('ECB+ Reading was done.')
    (train_entity_mentions, train_event_mentions) = split_entity_and_event_mention(train_extracted_mentions)
    (dev_entity_mentions, dev_event_mentions) = split_entity_and_event_mention(dev_extracted_mentions)
    (test_entity_mentions, test_event_mentions) = split_entity_and_event_mention(test_extracted_mentions)


    # 1. load and create Document, Sentence and Token objs.
    logging.info('Training data - loading and create Document, Sentence and Token objs')
    training_data: Dict[str, Document] = load_ECB_plus(train_extract_tokens)
    """{'XX_XXecb': Document Obj, ... } """
    logging.info('Dev data - loading and create Document, Sentence and Token objs')
    dev_data: Dict[str, Document] = load_ECB_plus(dev_extracted_tokens)
    """{'XX_XXecb': Document Obj, ... } """
    logging.info('Test data - loading and create Document, Sentence and Token objs')
    test_data: Dict[str, Document] = load_ECB_plus(test_extracted_tokens)
    """{'XX_XXecb': Document Obj, ... } """

    # 2. load and create gold Mention objs
    logging.info('Training data - Loading gold mentions')
    load_gold_mentions(training_data, train_entity_mentions, train_event_mentions)
    logging.info('Dev data - Loading gold mentions')
    load_gold_mentions(dev_data, dev_entity_mentions, dev_event_mentions)
    logging.info('Test data - Loading gold mentions')
    load_gold_mentions(test_data, test_entity_mentions, test_event_mentions)

    # 3. create Corpus objs
    logging.info('Train_set - Createing Corpus and Topic')
    train_set = order_docs_by_topics(training_data)
    logging.info('dev_set - Createing Corpus and Topic')
    dev_set = order_docs_by_topics(dev_data)
    logging.info('test_set - Createing Corpus and Topic')
    test_set = order_docs_by_topics(test_data)

    # 4. add selected tag
    if not config_dict["read_selected_sentences_only"]:
        add_selected_tag(train_set, file_to_selected_sentence)
        add_selected_tag(dev_set, file_to_selected_sentence)
        add_selected_tag(test_set, file_to_selected_sentence)

    # # 5. statistic number of t,d,s,em,vm in each split
    # logging.info('dataset statistic')
    # write_dataset_statistics('train', train_set, check_predicted=False)
    # write_dataset_statistics('dev', dev_set, check_predicted=False)
    # write_dataset_statistics('test', test_set, check_predicted=config_dict["load_predicted_mentions"])

    # 10.
    train_set_path = os.path.join(config_dict["output_dir"], 'training_data')
    with open(train_set_path, 'wb') as f:
        cPickle.dump(train_set, f)
        logging.info(f'train set save in {train_set_path}')
    dev_set_path = os.path.join(config_dict["output_dir"], 'dev_data')
    with open(os.path.join(config_dict["output_dir"], 'dev_data'), 'wb') as f:
        cPickle.dump(dev_set, f)
        logging.info(f'dev set save in {dev_set_path}')
    test_set_path = os.path.join(config_dict["output_dir"], 'test_data')
    with open(os.path.join(config_dict["output_dir"], 'test_data'), 'wb') as f:
        cPickle.dump(test_set, f)
        logging.info(f'test set save in {test_set_path}')

if __name__ == '__main__':
    main()
