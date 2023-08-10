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
     
## extract mention pairs from test data 
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
          ```text
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
* 输入(在`src/extract_mention_pair_from_test_data.py`中的config_dict中配置)：
    * corpus_path: str: `src/1.read_corpus.py`输出的`test_data`文件的路径。
    * mention_pairs_path: 输`src/2.extract_mention_pairs_from_test_data.py`出的`test(strategy{1/2/3/4}).mp`的路径。
    * output_path: str: 输出路径。
    * models: 使用哪几个模型进行试验，以及这几个模型各自的配置。具体请见`src/extract_mention_pair_from_test_data.py: config_dict['models']`。
    * templates: 使用哪几款模板来生成prompt。这里写模板的id。模板id是``src/template.py`中的字典key。
    * data: corpus_path中传入的整个测试集。如果想只跑其中的部分topic，则在这里指定。如果跑整个测试集则设为字符串"all"。
* 运行`src/3.pred.py`
* 输出（之后剪切到`data/extract_mention_pair_from_test_data/{这里随便写，每次试验放在不同的文件夹中}`中保存）：
    * {选中数据}.corpus 一个pkl文件，保存了一个corpus对象。这个corpus对象根据你在`pred.py: config_dict['data']`中的配置,保存完整的测试集数据或其中的选定子集。预测是基于这个子集展开的。
    * {data}_{model_setting}_{prefix_num}shot_t{template_id}_{do_sample}(r{repeat}).{mp/csv/promptlog} 
        * 类似`['36_ecb', '36_ecbplus'](strategy3)_ChatGPT3.5(b1t0)_0shot_t14DAM_noSample(r1).mp`。
        * 看输出可以发现，每次可以输出多个model和多个template。本程序把一个model-template对儿看做一次实验，为每次试验输出一组mp csv promptlog文件。文件名就表名了试验的配置（在什么数据上调用了什么model，跑了基于哪个模板的prompt等等）。比如配置中models给了仨，templates给了俩，那么就是2×3=6次试验，每次试验都有一组mp csv promptlog文件输出。
        * mp是一个pkl文件，在`src/2.extract_mention_pairs_from_test_data.py`输出的`test(strategy{1/2/3/4}.mp`的基础上，增加了模型预测结果。
        * csv是mp文件的可视化输出。
        * promptlog是每个样例的prompt记录。
    * log.txt 日志。
    * 3.pred.py 就是代码本身，用于保存配置。
## mention pairs scorer
* 输入(在`src/4.mention_pairs_scorer.py`中的config_dict中配置)：
    * csv_path: str: `src/3.pred.py`中每个试验都对应输出一个csv文件。你相对那几个试验进行统计，就把哪些csv文件放到一个文件夹下。此程序会非迭代得读取此文件夹下的所有csv文件。这一配置就给出此文件夹的路径。
    * output_path: str: 输出路径。
* 运行`src/4.mention_pairs_scorer.py`。功能是对指称对的预测结果进行性能评估。注意，这里没做聚类，没有形成簇，只是对指称对的预测结果进行性能评估，所以还是采用了传统的p r F1等指标。
* 输出
    * `4.mention_pairs_scorer.py`就是代码本身，用于保存配置。
    * log.txt 日志。
    * performance_list.csv 一个试验一行。
    * performance_table.csv 和performance_list.csv相同的内容，但是按照model为列，template为行的形式组织成表格的形式了。
