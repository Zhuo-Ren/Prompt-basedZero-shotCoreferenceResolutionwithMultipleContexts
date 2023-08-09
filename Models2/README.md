## read corpus 
* 运行`src/1.read_corpus.py`。
  * 输入：
    * ECB+语料
  * 配置(在`src/read_corpus.py`中的config_dict中配置)
    * ecb_path: str: ECB+语料的路径。
    * output_dir: str: ECB+语料附带的ECBplus_coreference_sentences.csv的路径。
    * data_setup: Union[1, 2]: 如何把ECB+数据集中的topics分为train、dev、test三部分。
      ```
      if data_setup == 1:  # Yang setup
        train_topics = range(1, 23)
        dev_topics = range(23, 26)
        test_topics = list(set(topic_list) - set(train_topics))
      else:  # Cybulska setup
        dev_topics = [2, 5, 12, 18, 21, 23, 34, 35]
        train_topics = [i for i in range(1, 36) if i not in dev_topics]  # train topics 1-35 , test topics 36-45
        test_topics = list(set(topic_list) - set(train_topics) - set(dev_topics))
      ```
    * read_selected_sentences_only: Bool: Some sentences are selected in ECB+ corpus.
      * True: Only the selected sentences are extracted.
      * False: All sentences are extracted.
    * output_dir: str: 输出路径。
  * 输出（之后剪切到`data/read_corpus`中保存）：
    * training_data 以Corpus对象存储的training数据。
    * dev_data 以Corpus对象存储的dev数据。
    * test_data 以Corpus对象存储的test数据。
    * log.txt 日志。
    * read_corpus.py 就是代码本身，用于保存配置。
     
## extract mention pairs 
* 输入(在`src/extract_mention_pair_from_test_data.py`中的config_dict中配置)
    * corpus_path: str: `src/read_corpus.py`输出的test_data文件的路径。
    * predicted_topics: str: 如果不使用真实topic，而是使用其他文档聚类算法预测的topic(比如strategy 4)，那么还要提供预测的topic信息。这个配置给出指向外部文档聚类算法预测得到的topic信息的路径。
    * output_path: str: ECB+语料附带的`ECBplus_coreference_sentences.csv`的路径。
    * selected_sentences_only: Bool: Some sentences are selected in ECB+ corpus.
      * True: Only mentions in selected sentences are extracted.
      * False: All mentions are extracted.
      * 注：这个值一直都是True，False的功能就没实现。放一个配置选项在这里，只是为了强调一下。
    * strategy: Union[0,1,2,3]: 生成指称对的策略。
      * 0: All the following strategies.
      * 1: sentence level: mention pair in a continuous sentence or two are extracted.
      * 2: wd: mention pair in the same doc are extracted. 注意，因为cd（策略3、策略4）的mention pairs其实包含了wd（策略2）的mention
       pairs。所以从成本考虑，如果你既要做cd的实验，又要做wd的实验，那么其实只做cd的实验就够了，wd的实验结果可以从cd实验的结果中抽取得到。
      * 3: cd-golden: mention pair in the same golden topic are extracted.
      * 4: cd-pred: mention pair in the same predicted topic are extracted.
* 运行`src/2.extract_mention_pair_from_test_data.py`
* 输出（之后剪切到`data/extract_mention_pair_from_test_data`中保存）：
    * test_strategy{n}.mp/csv 一个pkl/csv文件，描述使用策略n抽取得到的mention pairs信息，两者内容同源，只是csv更可视化一些。
        * pkl文件类似
          ```python
            # 一种是topic-mentionPairs的2层嵌套结构
            mention_pairs = {
                "36_ecbplus": [
                    [mention_obj, mention_obj],
                    [mention_obj, mention_obj],
                    ...
                ],
                ...
            }
                
            # 另一种是topic-doc-mentionPairs的3层嵌套结构
            mention_pairs = {
                "36_ecbplus": {
                    "36_1ecbplus":[
                        [mention_obj, mention_obj],
                        [mention_obj, mention_obj],
                        ...
                    ],
                    "36_2ecbplus":[
                        [mention_obj, mention_obj],
                        [mention_obj, mention_obj],
                        ...
                    ],
                    ...
                },
                "36_ecb": {
                    ...
                },
                ...
            }
          ```
        * csv文件类似于：
          ```csv
            topic,m1_doc,m1_sent,m1_str,m2_doc,m2_sent,m2_str,wd/cd,seq,label
            36,36_1ecb,0,leaders,36_1ecb,0,in Canada,wd,0,0
            36,36_1ecb,0,leaders,36_1ecb,0,polygamist group,wd,0,0
          ```
        * 观察test_strategy3.csv，筛选其中的wd的mention pair，一共是39252条。正好就是test_strategy2.csv中的那些。所以说strategy3包含了strategy2。
    * log.txt 日志。
    * extract_mention_pairs_from_test_data.py 就是代码本身，用于保存配置。
     
## pred
* 运行`src/3.pred.py`
  * 输入：
    * `src/read_corpus.py`输出的test_data文件。
    * 如果不使用真实topic，而是使用其他文档聚类算法预测的topic，那么还要提供预测的topic信息。
  * 配置(在`src/extract_mention_pair_from_test_data.py`中的config_dict中配置)
    }
    * corpus_path: str: `src/read_corpus.py`输出的test_data文件的路径。
    * predicted_topics: str: 指向外部文档聚类算法预测得到的topic信息的路径。
    * output_path: str: ECB+语料附带的ECBplus_coreference_sentences.csv的路径。
    * selected_sentences_only: Bool: Some sentences are selected in ECB+ corpus.
      * True: Only mentions in selected sentences are extracted.
      * False: All mentions are extracted.
      * 注：这个值一直都是True，False的功能就没实现。放一个配置选项在这里，只是为了强调一下。
    * strategy: Union[0,1,2,3]: 生成指称对的策略。
      * 0: All the following strategies.
      * 1: sentence level: mention pair in a continuous sentence or two are extracted.
      * 2: wd: mention pair in the same doc are extracted. 注意，因为cd（策略3、策略4）的mention pairs其实包含了wd（策略2）的mention
      * pairs。所以从成本考虑，如果你既要做cd的实验，又要做wd的实验，那么其实只做cd的实验就够了，wd的实验结果可以从cd实验的结果中抽取得到。
      * 3: cd-golden: mention pair in the same golden topic are extracted.
      * 4: cd-pred: mention pair in the same predicted topic are extracted.
  * 输出（之后剪切到`data/extract_mention_pair_from_test_data`中保存）：
    * {选中数据}.corpus 一个pkl文件，保存了一个corpus对象。这个corpus对象根据你在`pred.py: config_dict['data']`中的配置,保存完整的测试集数据或其中的选定子集。预测是基于这个子集展开的。
    * {model_name}_{data}_t{template_id}_s0_b1_noSample.mp 一个pkl文件，保存了mention pairs list。这个mention
    *  pairs list使用策略n抽取得到的mention pairs信息。
    * log.txt 日志。
    * pred.py 就是代码本身，用于保存配置。
