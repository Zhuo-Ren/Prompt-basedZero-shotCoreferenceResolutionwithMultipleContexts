import shutil
import os
import logging
import _pickle as cPickle
import re
import pandas as pd
import numpy as np


def mixture_csv(ts_path, wd_path, cd_path, output_path):
    ts_csv_path = f"{ts_path}.csv"
    wd_csv_path = f"{wd_path}.csv"
    cd_csv_path = f"{cd_path}.csv"
    #
    ts_df = pd.read_csv(ts_csv_path)
    wd_df = pd.read_csv(wd_csv_path)
    cd_df = pd.read_csv(cd_csv_path)
    #
    ts_rows = ts_df[ts_df["seq"] <= 1]
    wd_rows = wd_df[(wd_df["wd/cd"] == "wd") & (~wd_df["seq"].isin([0, 1]))]
    cd_rows = cd_df[cd_df["wd/cd"] == "cd"]
    # 把最后一列的列名（template id）改为新的template id
    ts_template_id = ts_rows.columns.to_list()[-1]
    wd_template_id = wd_rows.columns.to_list()[-1]
    cd_template_id = cd_rows.columns.to_list()[-1]
    new_id = f"{ts_template_id}-{wd_template_id}-{cd_template_id}"
    ts_rows = ts_rows.rename(columns={ts_template_id: new_id})
    wd_rows = wd_rows.rename(columns={wd_template_id: new_id})
    cd_rows = cd_rows.rename(columns={cd_template_id: new_id})
    del ts_template_id, wd_template_id, cd_template_id, new_id
    #
    r = pd.concat([ts_rows, wd_rows, cd_rows])
    # 检查
    l1 = len(cd_df)
    l2 = len(r)
    if l1 != l2:
        raise RuntimeError("xxxx")
    del l1, l2
    # 保存
    r.to_csv(output_path, mode="a", encoding="utf-8", index=False, header=True)
    print(f"OUTPUT: csv saved in {output_path}")
    logging.info(f"OUTPUT: csv saved in {output_path}")
    #
    return r


def get_mention_id(mention_obj):
    m_id = f"{mention_obj.doc_id}-{mention_obj.sent_id}-{mention_obj.start_offset}-{mention_obj.end_offset}"
    return m_id


def mixture_cmp(ts_path, wd_path, cd_path, output_path):
    ts_cmp_path = f"{ts_path}.c_mp"
    wd_cmp_path = f"{wd_path}.c_mp"
    cd_cmp_path = f"{cd_path}.c_mp"
    #
    with open(ts_cmp_path, 'rb') as f:
        ts_corpus, ts_mention_pairs = cPickle.load(f)
    with open(wd_cmp_path, 'rb') as f:
        wd_corpus, wd_mention_pairs = cPickle.load(f)
    with open(cd_cmp_path, 'rb') as f:
        cd_corpus, cd_mention_pairs = cPickle.load(f)
    # 融合
    for cur_topic_id in ts_mention_pairs.keys():
        # 检查
        if not(len(ts_mention_pairs[cur_topic_id]) == len(wd_mention_pairs[cur_topic_id]) == len(cd_mention_pairs[cur_topic_id])):
            raise RuntimeError("三个corpus对象的结果不一致")
        # 遍历
        for cur_mp_index in range(len(ts_mention_pairs[cur_topic_id])):
            cur_ts_mp = ts_mention_pairs[cur_topic_id][cur_mp_index]
            cur_wd_mp = wd_mention_pairs[cur_topic_id][cur_mp_index]
            cur_cd_mp = cd_mention_pairs[cur_topic_id][cur_mp_index]
            # 检查
            if not (get_mention_id(cur_ts_mp[0]) == get_mention_id(cur_wd_mp[0]) == get_mention_id(cur_cd_mp[0])):
                raise RuntimeError("三个corpus对象的结果不一致")
            if not (get_mention_id(cur_ts_mp[1]) == get_mention_id(cur_wd_mp[1]) == get_mention_id(cur_cd_mp[1])):
                raise RuntimeError("三个corpus对象的结果不一致")
            #
            m1 = cur_ts_mp[0]
            m2 = cur_ts_mp[1]
            # 看看当前mention pair属于哪个类型，是2s还是wd-还是cd-
            cur_mp_type = None
            if m1.doc_id != m2.doc_id:
                cur_mp_type = "cd-"
            elif abs(m1.sent_id - m2.sent_id) in [0, 1]:
                cur_mp_type = "2s"
            else:
                cur_mp_type = "wd-"
            # 融合
            if cur_mp_type == "2s":
                cur_ts_mp[2] = cur_ts_mp[2]
            elif cur_mp_type == "wd-":
                cur_ts_mp[2] = cur_wd_mp[2]
            elif cur_mp_type == "cd-":
                cur_ts_mp[2] = cur_cd_mp[2]
    # 保存
    with open(output_path, 'wb') as f:
        cPickle.dump((ts_corpus, ts_mention_pairs), f)
        print(f"OUTPUT: c_mp saved in {output_path}")
        logging.info(f"OUTPUT: c_mp saved in {output_path}")
    #
    return ts_corpus, ts_mention_pairs

    
def main(config_dict):
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
    # save this file itself
    shutil.copy(os.path.abspath(__file__), config_dict["output_path"])

    # 准备路径
    th1 = re.findall(r"shot_([\S]*)_", config_dict["2s_path"])[0]
    th2 = re.findall(r"shot_([\S]*)_", config_dict["wd-_path"])[0]
    th3 = re.findall(r"shot_([\S]*)_", config_dict["cd-_path"])[0]
    file_name =os.path.basename(config_dict["2s_path"]).replace(th1, f"{th1}-{th2}-{th3}")
    # 合并、保存csv
    csv_output_path = os.path.join(config_dict["output_path"], f"{file_name}.csv")
    df = mixture_csv(ts_path=config_dict["2s_path"], wd_path=config_dict["wd-_path"], cd_path=config_dict["cd-_path"], output_path=csv_output_path)
    # 合并、保存cmp
    cmp_output_path = os.path.join(config_dict["output_path"], f"{file_name}.c_mp")
    cmp = mixture_cmp(ts_path=config_dict["2s_path"], wd_path=config_dict["wd-_path"], cd_path=config_dict["cd-_path"], output_path=cmp_output_path)
    #
    return df, cmp


if __name__ == '__main__':
    # config
    config_dict = {
        "2s_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\data\3.pred\['36_ecb'](strategy3)_ChatGPT3.5(b1t0)_0shot_t13SAU_noSample(r1)",
        "wd-_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\data\3.pred\['36_ecb'](strategy3)_ChatGPT3.5(b1t0)_0shot_t25DAU_noSample(r1)",
        "cd-_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\data\3.pred\['36_ecb'](strategy3)_ChatGPT3.5(b1t0)_0shot_t13SAU_noSample(r1)",
        "output_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\output",
    }
    main(config_dict)
