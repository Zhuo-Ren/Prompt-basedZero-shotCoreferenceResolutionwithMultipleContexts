我们借用Barhom2019的性能打分代码。
Barhom2019的代码放在Models/event_entity_coref_ecb_plus中
* 借助src/shared/classes.py中定义的Corpus类对数据进行建模。
  * 真实的共指消解结果放在每个mention的gold_tag属性中。
  * 预测的共指消解结果放在每个mention的cd_coref_chain属性中。
* 借助src/data/make_gold_files.py，读入一个Corpus对象，输出其真实的共指消解结果到文件。
* 借助src/all_models/predict_model.py中的main>test_mode>test_models>write_event_coref_results>write_mention_based_cd_clusters，读入一个Corpus对象，输出其预测的共指消解结果到文件。
* 借助src/all_models/predict_model.py中的run_conll_scorer，读入真实结果和预测结果，输出性能。
* 最终抽取上述步骤中的有效代码，整到成为scorer/scorer.py