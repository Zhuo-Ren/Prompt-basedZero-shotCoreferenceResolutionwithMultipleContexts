#!/usr/bin/env python
# coding: utf-8
# 这是process_with_new_idx.ipynb的.py版本。后经过了少许修改。
# 读取Data/ECB+/original(就是原始ECB+语料库),输出到Data/ECB+/processed/train_with_new_index.json和dev_with_new_index.json
import os, re
from pprint import pprint
import spacy
from functools import partial
import pandas as pd
from tqdm import tqdm
import xml.etree.ElementTree as ET
import spacy
from spacy.lang.en import English
from spacy.tokenizer import Tokenizer
import json
en = English()
tok_en = Tokenizer(en.vocab)


# Load spacy since we need it for token tags and lemmas
nlp = spacy.load('en_core_web_sm', disable=['textcat'])
#
local_path = "D:/WhatGPTKnowsAboutWhoIsWho-main"
root_path = local_path
# Path to the ecb data
ecb_path = f"{root_path}/Data/ECB+"
ecb_input_path = f"{root_path}/Data/ECB+/original"
print(f"ecb_input_path:{ecb_input_path}")


# In[class Doc]:
class ECBMentionsDoc:  # 文档建模

    # Problematic tokens in the dataset
    # From the CDLM repo https://github.com/aviclu/CDLM
    error_tokens = [('31_10ecbplus.xml', 979),
                  ('9_3ecbplus.xml', 30),
                  ('9_4ecbplus.xml', 32)]

    def __init__(self, doc_path, doc_name, topic_id):
        self.doc_path = doc_path  # 文件所在路径
        self.doc_name = doc_name  # 文件名
        self.topic_id = topic_id  # 不知道为啥给了个2
        self.mentions_fields = {}
        """对应ECB+源文件中所有不带RELATED_TO属性的mention
            {
            'doc_id': '1_1ecb.xml', 
            'topic': 2, 
            'subtopic': '1_1',  # 1_1plus、1_2plus这些算subtopic1_1，1_1、1_2这些算subtopic1_0
            'm_id': '27',       # m_id
            'sentence_id': 1,   # sentence
            'tokens_ids': [28], # t_id列表
            'tokens': 'starred', 
            'tags': 'VBD',      # spacy做的POS
            'lemmas': 'star',   # spacy做的lemmas
            'event': True
            }
        """
        self.mention_cluster_info = {}
        """对应ECB+源文件中所有带RELATED_TO属性的mention
        """
        self.relation_source_target = {}
        self.relation_ids = {}
        self.relation_tag = {}
        self.event_singleton_idx = int(1E8)
        self.entity_singleton_idx = int(2E8)
        self.entity_mentions, self.event_mentions  = [], []
        self.clean_event_mentions, self.clean_entity_mentions = [], []
        self.tagged_event_tokens = {}
        self.tagged_entity_tokens = {}
        self.doc_token_texts = {}
        self.b_open, self.b_close = "{", "}"
        self.prev_wrap, self.prev_tag_id = '', ''
        self.tag_is_opened = False
        self.keep_start_url = True
        self.plain_tokens, self.clean_tokens = [], []
        self.plain_text, self.clean_text = '', ''
        self.root = None  # root of ET.parse

    def parse_xml(self):
        # Start parsing
        self.root = ET.parse(self.doc_path).getroot()

        # Set all mention ids from the full document for both event and entity mentions
        self.set_all_marked_mentions()

        # Set all cross doc ids
        self.set_cross_doc_mentions()

        # Creates both arrays containing all event and entity mention info
        self.compute_event_entity_mentions()

        # Parses all the actual tokens from the current document into a dict we can use
        self.set_doc_texts()


    # Loops through each word in the full document and stores the info in a dict like this:
    def set_doc_texts(self):
        '''
        Example text:
        <token t_id="53" sentence="4" number="1">Williams</token>
        <token t_id="54" sentence="4" number="2">,</token>
        <token t_id="55" sentence="4" number="3">the</token>
        <token t_id="56" sentence="4" number="4">swimming</token>
        <token t_id="57" sentence="4" number="5">champion</token>
        <token t_id="58" sentence="4" number="6">turned</token>
        <token t_id="59" sentence="4" number="7">actress</token>
        '''
        prev_sent_id = -1
        for token in self.root.findall('token'):
            token_id = int(token.get('t_id'))
            
            # A few tokens per should not be used
            if (self.doc_name, token_id) not in self.error_tokens:
                # Parse actual token text in the right format
                token_text = token.text.replace('�', '').strip()
                sent_id = int(token.get('sentence'))

                # word_id_sent = token.get('number') # word index per sentence
                token_info = (token_text, sent_id)
                
                # Write data for sentence reconstruction
                if (prev_sent_id > -1) and (sent_id != prev_sent_id):
                    prev_token_info = self.doc_token_texts[prev_token_id]
                    self.doc_token_texts[prev_token_id] = (prev_token_info[0] + " [EOS]", prev_token_info[1])

                self.doc_token_texts[token_id] = token_info
                prev_sent_id = sent_id
                prev_token_id = token_id


    # Maps each mentions to a possible relation is has, meaning singleton or not
    # Then saves the info from the self.mentions_fields for each mention with
    # additional info like if it's cluster or not and the description of the cluster
    # It also splits them into 2 buckets: Event and Entity mentions
    def compute_event_entity_mentions(self):
        
        # Loop through all mentions of the current document
        for m_id, mention in self.mentions_fields.items():

            # For this specific mention check if's a source by checking if it maps to a target
            # Since the dict containts {source_mention_id: target_mention_id}
            target_id = self.relation_source_target.get(m_id, None)

            # If it's just a source_id with no second target_id in it's cluster;
            # then we know that this event or enntity mention has to be a singleton
            if target_id is None: # 如果一个mention没有对应的target_id，那么它是一个孤立mention，例如1_1ecb中的m_id=28的mention
                # cluster_id
                if mention['event']:
                    cluster_id = self.event_singleton_idx
                    self.event_singleton_idx += 1
                else:
                    cluster_id = self.entity_singleton_idx
                    self.entity_singleton_idx += 1
                # cluster_desc
                cluster_desc = ''# cluster_desc =  'Singleton_' + file_name + '_' + m_id
            else: # 如果一个mention有对应的target_id，那么它不是一个孤立mention
                # cluster_id
                if 1:
                    r_id = self.relation_ids[target_id]
                    tag = self.relation_tag[target_id] # E.g. CROSS_DOC_COREF
                    # 区分WD corefer和CD corefer
                    if tag.startswith('INTRA'):  # WD corefer的relation标签是INTRA开头的，例如1_11ecb.xml中的<INTRA_DOC_COREF r_id="37615" >
                        # Entity and event mentions may have the same intra cluster id ,所以要加个后缀加以区分
                        suffix = '1' if mention['event'] else '0'
                        cluster_id = int(r_id + suffix)
                    else:  # CD corefer的relation标签是CROSS开头的，例如1_11ecb.xml中的<CROSS_DOC_COREF r_id="37623" note="ACT16235311629112331" >
                        # Grab the cluster info dict from the mention clusters we created
                        target_cluster_info = self.mention_cluster_info[target_id]
                        # E.g. ACT16236402809085484
                        target_cluster_id_str = target_cluster_info['cluster_id']
                        # We grab all the integers from this string to construct an int we can use
                        cluster_id = int(target_cluster_id_str[3:])
                # cluster_desc
                cluster_desc = self.mention_cluster_info[target_id]['cluster_desc']  # e.g. t4_swimming_skills
            # end of if-else

            # Now that we retrieved the cluster id and description for this mention;
            # We can update the mention dict we create before with this and append;
            # it to the entities correpsonding group -> Event or Entity mention
            mention_info = mention.copy()
            mention_info["cluster_id"] = cluster_id
            mention_info["cluster_desc"] = cluster_desc
            event = mention_info.pop("event")
            if not event:
                self.entity_mentions.append(mention_info)
            else:
                self.event_mentions.append(mention_info)
        # end for

    # Set a dict with cross doc relations, meaning the mention id of the target for this doc
    # with the relation id, which is the crossdoc id, it's saved like this:
    # {target_mention_id: relation_id}
    def set_cross_doc_mentions(self):
        '''
        Example part to parse:
        <CROSS_DOC_COREF r_id="22306" note="ACT16195873839112917">
            <source m_id="28" />
            <source m_id="34" />
            <target m_id="60" />
        </CROSS_DOC_COREF>
        '''

        # Relation -> Cross doc relation
        for relation in self.root.find('Relations'):
            
            # Last element of each cluster is 'target'
            target_mention_id = relation[-1].attrib['m_id']
            
            # All the other elements are of type 'source'
            source_tags = relation[:-1]

            # Set a mapping from coref source id to it's master target
            for source_tag in source_tags:
                source_mention_id = source_tag.attrib['m_id']
                self.relation_source_target[source_mention_id] = target_mention_id

            
            # Save tag 'CROSS_DOC_COREF' 
            self.relation_tag[target_mention_id] = relation.tag

            # Save the target mention id to cross doc id entries
            self.relation_ids[target_mention_id] = relation.attrib['r_id']


    def set_all_marked_mentions(self):
        '''
        Example part to parse:
        <ACTION_ASPECTUAL m_id="53">
            <token_anchor t_id="186"/>
        </ACTION_ASPECTUAL>
        <ACTION_OCCURRENCE m_id="50">
            <token_anchor t_id="179"/>
            <token_anchor t_id="180"/>
            <token_anchor t_id="181"/>
        </ACTION_OCCURRENCE>
        '''

        # Store our results
        subtopic = '0' if 'plus' in self.doc_name else '1'

        for mention in self.root.find('Markables'):
            m_id = mention.attrib['m_id']

            if 'RELATED_TO' not in mention.attrib:

                # ACTION or NEG is an event mention 
                is_event_mention = mention.tag.startswith('ACT') or mention.tag.startswith('NEG')
                
                # Grab all token ids under current Markable tag
                tokens_ids = [int(term.attrib['t_id']) for term in mention]

                if len(tokens_ids) == 0:
                    print(ET.tostring(mention, encoding='unicode'))
                    continue

                # print(is_event_mention, tokens_ids)

                # Indexing our sentences also starts at 0
                token_sent_index = tokens_ids[0]
                sent_id = self.root[token_sent_index].attrib['sentence']  ## bug +1

                # Construct the actual mention text, e.g. "Barack Obama"
                # NOTE: We -1 the token id itself, since they started indexing at 1 and map starts at 0
                mention_word_tokens = ' '.join(list(map(lambda x: self.root[x-1].text, tokens_ids)))

                lemmas, tags = [], []
                for tok in nlp(mention_word_tokens):
                    lemmas.append(tok.lemma_)
                    tags.append(tok.tag_)
                
                self.mentions_fields[m_id] = {
                    "doc_id": self.doc_name,
                    "topic": self.topic_id,
                    "subtopic": self.doc_name.split('_')[0] + '_' + subtopic,
                    "m_id": m_id,
                    "sentence_id" : int(sent_id),
                    "tokens_ids": tokens_ids,
                    "tokens": mention_word_tokens,
                    "tags": ' '.join(tags),
                    "lemmas": ' '.join(lemmas),
                    "event": is_event_mention
                }
            else:
                self.mention_cluster_info[m_id] = {
                    "cluster_id": mention.attrib.get('instance_id', ''),
                    "cluster_desc": mention.attrib['TAG_DESCRIPTOR']
                }
                   
                

    def clear_url(self):
        
        no_space_doc =  self.original_text.replace(" ", "")

        print(no_space_doc)


    # http://www.ws.com/May 2, 2013.. -> http://www.ws.com/ May 2, 2013.. or May 2, 2013..
    def split_url_on_month(self, match):
        matched_url = match.group()
        months = ['Lindsay', 'Former', 'Footage', 'Video', '.html', 'Gunman',
                  'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        # print(matched_url)
        # Minimum amount of characters for an url plus month string
        if len(matched_url) > 10 and any([x in matched_url for x in months]):
            
            # Compile a regex with each month as option
            regexPattern = '|'.join(map(re.escape, months))
            
            # Split the string by 1 of the months, also keeping the matched month itself
            matches = re.split(f"({regexPattern})", matched_url, 1)
            new_url, month = matches[0], matches[1]   
            
            # So we want to keep the original url            
            if self.keep_start_url :
            
                # Return the url with the space in between the month
                return f"{new_url} {month}"
            
            return month

        
        # So we want to keep the original url            
        if self.keep_start_url :
            return matched_url
        
        # We can just skip the url altogether
        return ''            
    

    # https://stackoverflow.com/questions/21948019/python-untokenize-a-sentence
    # https://github.com/commonsense/metanl/blob/master/metanl/token_utils.py
    def create_clean_text(self, text):
        """
        Untokenizing a text undoes the tokenizing operation, restoring
        punctuation and spaces to the places that people expect them to be.
        Ideally, `untokenize(tokenize(text))` should be identical to `text`,
        except for line breaks.
        """
        text = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
        text = text.replace(" ( ", " (").replace(" ) ", ") ").replace(" 's", "'s")
        text = re.sub(r' ([.,:;?!%]+)([ \'`])', r"\1\2", text)
        text = re.sub(r' ([.,:;?!%]+)$', r"\1", text)
        text = text.replace(" n't", "n't").replace("can not", "cannot")
        text = text.replace(" ` ", " '").replace(" -", "-").replace("- ", "-")
        text = text.replace(" ,", ",").replace(' /',  '/').replace('/ ',  '/')
        text = text.replace(" ’ s", "'s").replace("“ ", "“").replace(" ”", "”")
        text = text.replace(" ’ s", "'s").replace("“ ", "“").replace(" ”", "”")
        text = text.replace("www. ", "www.").replace(". com", ".com").replace(" ”", "”")
        text = text.replace(" _ ", "_")
        text = text.replace("p. m.", "p.m.").replace("a. m.", "a.m.")
        text = text.replace("P. M.", "P.M.").replace("A. M.", "A.M.")
        text = text.replace("[EOS]", "")

        # Regex to match even amount of ", because removing trailing or start space;
        # Will also remove any characters before and after quotes start.
        # So we need to match the even amount, see: https://stackoverflow.com/a/53436792/8970591
        # Inspiration for regex: https://stackoverflow.com/questions/14906492/how-can-whitespace-be-trimmed-from-a-regex-capture-group
        quote_regex = '\\"\s?([^\]]*?)\s?\\"'
        text = re.sub(quote_regex, '\"'+r'\1'+'\"' , text)
        
        # Re-replace this exception
        text = text.replace(',"', ', "').replace('."', '. "')

        # A lot of articles start with an url in the as the source it came from
        # So we can optionally get rid of this to get a cleaner text for text generation
        first_url_regex = '^(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])'        
        
        # Also stops match if there is a Capital letter after the last .com, .nl etc
        # first_url_regex = '^(http|ftp|https):\/\/([\w_-]+(?:(?:\.[\w_-]+)+))([a-z0-9_.,@?^=%&:\/~+#-]*[a-z0-9_@?^=%&\/~+#-])'
        pattern = re.compile(first_url_regex)
        
        # Partials can be used to make new derived functions that have some input parameters pre-assigned
        re_sub_callback = partial(self.split_url_on_month)
        
        # Sometime the first word, which is a month, can be captured by our regex
        # So we want to split on this character to we can keep the month
        text = re.sub(pattern, re_sub_callback, text, 1)
        
        tokenize_text = [i.text for i in nlp(text)] 
        
        return text, tokenize_text


    def format_doc_and_mentions(self):
        self.plain_tokens = [el[1][0] for el in self.doc_token_texts.items()] 
        self.plain_text = ' '.join(word for word in self.plain_tokens)
        sents = self.plain_text.split("[EOS]")
        for sent in sents:
            # print(sent)
            clean_text, clean_tokens = self.create_clean_text(sent)
            self.clean_text += clean_text + "[EOS]"
            self.clean_tokens += clean_tokens 
        
        
    
    def clean_and_reindex(self): 
        # Creates a clean version of the text along with the raw one
        self.format_doc_and_mentions()
        self.reindex_all_marked_mentions()

    # You can pass a key function to sorted which
    # returns a tuple containing the two things you wish to sort on
    def sort_by_token_ids(self, mention_item):
        first_token_index = mention_item['tokens_ids'][0]
        return first_token_index


    def get_span_by_ids(self, span_token_ids):
        span_tokens = []
        
        for token_id, (token, _) in self.doc_token_texts.items():
            if token_id in span_token_ids:
                span_tokens.append(token)
                
        return span_tokens

    def reindex_all_marked_mentions(self):
        '''
        Input -> {m_id: all_mention_info}
        {'1': 
            {'doc_id': '1_1ecbplus.xml',
            'event': False,
            'lemmas': 'June 13 , 2013 4 : 59 PM EDT',
            'm_id': '1',
            'sentence_id': 2,
            'subtopic': '1_0',
            'tags': 'NNP CD , CD CD SYM CD NNP NNP',
            'tokens': 'June 13 , 2013 4 : 59 PM EDT',
            'tokens_ids': [45, 46, 47, 48, 49, 50, 51, 52, 53],
            'topic': 2},
        }
        '''
        # Sorts the mention info dict by the first token_id of each mention
        # So the mentions are in sequential order
        entity_mention_info_sorted = sorted(self.entity_mentions, key = self.sort_by_token_ids)
        event_mention_info_sorted = sorted(self.event_mentions, key = self.sort_by_token_ids)

        # pprint(event_mention_info_sorted)
        # New dictionaries to use for updated spans and indices (token_ids) of them 
        self.clean_entity_mentions = self.reindex_mentions(entity_mention_info_sorted)
        # self.clean_event_mentions = self.reindex_mentions(event_mention_info_sorted)
        
    def get_new_token_index(self, token, start_index):
        if token in self.clean_tokens[start_index:]:
            new_token_index = self.clean_tokens.index(token, start_index) # Changed
        else:
            # if token == "m":
            #     return 
            # Could happen that our original mention has made a different split then the next text
            # E.g. 'facility in Malibu, Calif' -> 'facility in Malibu, Calif.'
            new_token_index = -1
            for i, tok_in_sent in enumerate(self.clean_tokens[start_index:]):
                if tok_in_sent[:len(token)] == token:
                    new_token_index = i+start_index
                    break
            if new_token_index == -1:
                raise Exception(f"ERROR ({self.doc_path}) : after index {start_index}, '{token}' is not part of -> '{self.clean_tokens}' ")
        return new_token_index



    def reindex_mentions(self, mention_info_sorted):
                
        new_mentions = {}
        start_index = 0
        
        # pprint(self.mentions_fields)            
        for m_id, mention_info in enumerate(mention_info_sorted):
            # raw_mention_str = mention_info['tokens']
            raw_token_ids = mention_info['tokens_ids']
            raw_mention_tokens = self.get_span_by_ids(raw_token_ids)
            
            if m_id > 0:
                prev_m_id = m_id-1
                prev_m = mention_info_sorted[prev_m_id]
                prev_raw_token_ids = prev_m['tokens_ids']
                # Deal with overlapping mentions, ex., m1=[5,6,7,8], m2=[6,7] 
                if prev_raw_token_ids[-1] >= raw_token_ids[-1]:
                    start_index = new_mentions[prev_m_id]['tokens_ids'][0]
                else:
                    start_index = new_mentions[prev_m_id]['tokens_ids'][-1]

            # Clean and create tokens from raw mention text, 
            # so we can use that to match all the current cleaned tokens
            mention_token_srt, mention_tokens = self.create_clean_text(" ".join(raw_mention_tokens))
            # Get new tokens and new token ids from the current mention
            new_tokens_ids = []
            new_mention_tokens = []
            
            for token in mention_tokens:
                new_token_index = self.get_new_token_index(token, start_index+1)                   
                new_tokens_ids.append(new_token_index)
                new_mention_tokens.append(self.clean_tokens[new_token_index])
                start_index = new_token_index
            
            # Create new update dict with new values
            clean_mention_info = mention_info
            clean_mention_info['tokens'] = new_mention_tokens
            clean_mention_info['tokens_ids'] = new_tokens_ids
            
            new_mentions[m_id] = clean_mention_info
            # print(f"mention {m_id}: {new_tokens_ids};  {raw_token_ids}; {new_mention_tokens}, {raw_mention_tokens}", )         
        return new_mentions
            
    def get_clusters(self, mentions):
        clusters = {}
        for m_id in mentions:
            mention = mentions[m_id]
            cluster_id = mention['cluster_id']
            # Create empty list entry if not existent
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id] += mention["tokens_ids"]

        return clusters

# In[test Doc]
# ## Running the doc-level parser
# We run the doc-level parser on 1 document to inspect it's outputs and if it's running correctly
# Test cell:
file_name = "1_1ecb.xml"
topic = file_name.split("_")[0]
ecb_doc_path = f"{ecb_input_path}/{1}/{file_name}"
print(f"ecb_doc_path:{ecb_doc_path}")
ecb_mention_doc = ECBMentionsDoc(ecb_doc_path, file_name, 2)
ecb_mention_doc.parse_xml()
ecb_mention_doc.clean_and_reindex()
print("Original untokenized text")
print(ecb_mention_doc.plain_text)
print("\n\n Cleaned untokenized text")
print(ecb_mention_doc.clean_text)


# In[class corpus]:
class ECBMentionsDataset:
    skip_list = ["35_2ecbplus.xml", "34_9ecbplus.xml",
                 "18_8ecbplus.xml","21_3ecbplus.xml","2_10ecbplus.xml", # Validation
                 "20_3ecbplus.xml", "33_7ecbplus.xml",
                 "11_2ecbplus.xml", "29_2ecbplus.xml", "6_11ecbplus.xml","6_10ecbplus.xml"
                 ] 
    # skip_list = [
    #     "20_3ecbplus.xml", # Exception for matching '-5' new tokens 'magnitude-5'
    #     "18_3ecbplus.xml", # Exception for matching '1' new tokens 'Up-1'
    #     "11_4ecbplus.xml", # Exception for matching 'city' new tokens 'city-15'
    #     "29_2ecbplus.xml", # Exception for 'success' is not in list
    #     "16_2ecbplus.xml", # something not part of list
    #     "45_3ecb.xml",
    #     "19_15ecb.xml",
    #     "43_9ecbplus.xml",
    #     "43_6ecb.xml",
    #     "43_7ecbplus.xml",
    #     "38_9ecbplus.xml",
    #     "36_10ecbplus.xml",
    #     "41_1ecb.xml",
    #     "40_5ecb.xml",
    #     "21_4ecbplus.xml",
    #     "21_12ecbplus.xml",
    #     "19_2ecbplus.xml",
    #     "30_4ecbplus.xml", 
    #     "24_10ecb.xml",
    #     "24_9ecbplus.xml",
    #     "4_1ecbplus.xml",
    #     "13_13ecbplus.xml", 
    #     "14_5ecbplus.xml",
    #     "14_6ecbplus.xml",
    #     "22_8ecbplus.xml",
    #     "22_7ecbplus.xml",
    #     "22_11ecbplus.xml",
    #     "29_10ecbplus.xml",
    #     "12_3ecbplus.xml",
    #     "2_10ecbplus.xml"
    # ]

    def __init__(self, path, topic_ids):
        """
        语料库对象，包含ECB+中的多个topic。

        :param path: ECB+语料库的路径
        :param topic_ids: 要对ECB+语料库中的哪几个topic进行建模。
        """
        self.path = path
        """
        e.g. 'D:/WhatGPTKnowsAboutWhoIsWho-main/Data/ECB+/original'
        """

        self.selected_topic_ids = topic_ids
        """
        e.g. [2, 5, 12, 18, 21, 23, 34, 35]
        """

        self.docs = {}
        """数据结构"""


    def parse_data(self):
        """读入多个topic。结果存入self.docs

        1. 根据self.path找到ECB+语料库。
        2. 根据self.selected_topic_ids指定的topic，从ECB+语料库中抽取数据。
        3. 抽取的数据存入self.docs

        :return: 无。抽取的数据存入self.docs。
        """
        # if not Path(self.path).exists():
        if not os.path.exists(self.path):
            raise Exception(f"{self.path} does not exists!")

        # 遍历self.selected_topic_ids中指定的topic文件夹（及其下的所有文件）
        topic_count = 0
        for topic_dir in os.scandir(self.path):
            if topic_dir.is_dir() and topic_dir.name.isnumeric():
                topic_id = int(topic_dir.name)
                if topic_id in self.selected_topic_ids:
                    """
                    对每个选定的topic进行处理
                    """
                    topic_count += 1
                    # 核心：处理一个topic
                    self.parse_topic_docs(topic_dir.path, topic_id)
                    # log
                    perc = (topic_count/(len(self.selected_topic_ids ))) * 100
                    print(f"{round(perc, 2)}% Done parsing topic -> \t {topic_id}")
            else:
                print(f"Skipping dir/file '{topic_dir.name}' in topic parsing because not a directory or number like...")

    def parse_topic_docs(self, topic_path, topic_id):
        """
        读入一个topic，抽取一个个doc，结果存入self.docs。
        self.skip_list中的doc不被处理。

        以12_10ecb.xml为例，存入self.docs的信息如下(cl://20230601153023/def)::

            self.docs[ecb_file.name] = [clean_text, clean_tokens, entity_mentions, clusters]
            #
            clean_text = "The Indian navy has captured 23 Somalian pirates. [EOS] The pirates were about to board a vessel in the Gulf of Aden when they were apprehended. [EOS] Last month, India's navy drew criticism after sinking a Thai fishing trawler that had been commandeered hours earlier by pirates, but the navy says it fired in self-defence. [EOS] Somali pirates have become increasingly brazen, and recently seized a Saudi supertanker loaded with e75m worth of crude oil.[EOS]"
            clean_token = ['The', 'Indian', 'navy', 'has', 'captured', '23', 'Somalian', 'pirates', '.', ' ', 'The', 'pirates', 'were', 'about', 'to', 'board', 'a', 'vessel', 'in', 'the', 'Gulf', 'of', 'Aden', 'when', 'they', 'were', 'apprehended', '.', ' ', 'Last', 'month', ',', 'India', "'s", 'navy', 'drew', 'criticism', 'after', 'sinking', 'a', 'Thai', 'fishing', 'trawler', 'that', 'had', 'been', 'commandeered', 'hours', 'earlier', 'by', 'pirates', ',', 'but', 'the', 'navy', 'says', 'it', 'fired', 'in', 'self', '-', 'defence', '.', ' ', 'Somali', 'pirates', 'have', 'become', 'increasingly', 'brazen', ',', 'and', 'recently', 'seized', 'a', 'Saudi', 'supertanker', 'loaded', 'with', 'e75', 'm', 'worth', 'of', 'crude', 'oil', '.']
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


        :param topic_path: 'D:/WhatGPTKnowsAboutWhoIsWho-main/Data/ECB+/original\\12'
        :param topic_id: 12
        :return: 无。抽取的数据存入self.docs。
        """
        # Loop through all the files of the specific topic dir
        for ecb_file in os.scandir(topic_path):
            if ecb_file.is_file() and ecb_file.name not in self.skip_list:
                """
                对此topic下的每一个文件。
                """
                print(f"Parsing -> {ecb_file.name}")
                # Creates in instance of the class that handles all document;
                # level parsing for: Actual doc's word + event and entity mentions
                try:
                    ecb_doc = ECBMentionsDoc(ecb_file.path, ecb_file.name, topic_id)
                    ecb_doc.parse_xml()
                    ecb_doc.clean_and_reindex()

                    clean_text = ecb_doc.clean_text
                    clean_tokens = ecb_doc.clean_tokens
                    entity_mentions = ecb_doc.clean_entity_mentions
                    clusters = ecb_doc.get_clusters(entity_mentions)

                    # Extend this corpus' data from the current document level information
                    self.docs[ecb_file.name] = [clean_text, clean_tokens, entity_mentions, clusters]
                except:
                    print(f"Skip {ecb_file.name}")


# In[划分三大集并保存]:
VALIDATION = [2, 5, 12, 18, 21, 23, 34, 35]
TRAIN = [i for i in range(1, 36) if i not in VALIDATION]
TEST = [i for i in range(36, 46)]

dev = ECBMentionsDataset(ecb_input_path, VALIDATION)
dev.parse_data()
dev_data = dev.docs
print(len(dev_data))
file_path = ecb_path + "/processed/dev_with_new_index.json"
with open(file_path, 'w') as f:
    json.dump(dev_data, f)


train = ECBMentionsDataset(ecb_input_path, TRAIN)
train.parse_data()
train_data = train.docs
print(len(train_data))
file_path = ecb_path + "/processed/train_with_new_index.json"
with open(file_path, 'w') as f:
    json.dump(train_data, f)
