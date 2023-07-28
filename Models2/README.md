* 运行`src/read_corpus.py`。
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
* 运行`src/extract_mention_pair_from_test_data.py`
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
      * 2: wd: mention pair in the same doc are extracted.
      * 3: cd-golden: mention pair in the same golden topic are extracted.
      * 4: cd-pred: mention pair in the same predicted topic are extracted.
  * 输出（之后剪切到`data/extract_mention_pair_from_test_data`中保存）：
    * strategy_{n}_corpus_and_mention_pairs 一个pkl文件，保存了一个元组。第一个项是corpus，第二个项是使用策略n抽取得到的mention pairs信息。
    * log.txt 日志。
