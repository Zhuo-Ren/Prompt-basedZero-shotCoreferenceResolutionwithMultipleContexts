import os
import _pickle as cPickle


def mp_adaptor(mp):
    """
    读取的mention_pairs是topic-mp的双层结构，这里改为topic-doc-mp的三层结构，并去掉cd mp，只保留wd mp。

    :param mp: 双层结构的mp
    :return: 三层结构的wd mp
    """
    t = {}  # 临时变量，保存修改后的mention pairs
    for cur_topic_id, cur_topic in mp.items():
        if cur_topic_id not in t.keys():
            t[cur_topic_id] = {}
        for cur_mp in cur_topic:
            m1 = cur_mp[0]
            m2 = cur_mp[1]
            if m1.doc_id != m2.doc_id:
                continue
            else:
                if m1.doc_id not in t[cur_topic_id].keys():
                    t[cur_topic_id][m1.doc_id] = []
                t[cur_topic_id][m1.doc_id].append(cur_mp)
            del m1, m2, cur_mp
        del cur_topic_id, cur_topic
    return t


def mp_list_to_mention_list(mp_list):
    """
    输入mention pairs list，整理并返回mention list: [m1, m2]

    :param mp_list:
    :return: mention list
    """
    m_list = []
    for cur_mp in mp_list:
        m1 = cur_mp[0]
        m2 = cur_mp[1]
        #
        if m1 not in m_list:
            m_list.append(m1)
        if m2 not in m_list:
            m_list.append(m1)
    return m_list


def mp_list_to_mp_dict(mp_list):
    """
    输入mention pairs list，整理并返回mention pairs dict. {(m1_id, m2_id): mp_obj}

    :param mp_list:
    :return: mention pairs dict
    """
    mp_dict = {}
    for cur_mp in mp_list:
        m1 = cur_mp[0]
        m2 = cur_mp[1]
        m1_id = f"{m1.doc_id}-{m1.sent_id}-{m1.start_offset}-{m1.end_offset}"
        m2_id = f"{m2.doc_id}-{m2.sent_id}-{m2.start_offset}-{m2.end_offset}"
        if (m1_id, m2_id) not in mp_dict:
            mp_dict[(m1_id, m2_id)] = cur_mp
    return mp_dict


def get_mp_from_mp_dict(mp_dict, key):
    m1_id = key[0]
    m2_id = key[1]
    if (m1_id, m2_id) in mp_dict.keys():
        return mp_dict[(m1_id, m2_id)]
    elif (m2_id, m1_id) in mp_dict.keys():
        return mp_dict[(m2_id, m1_id)]
    else:
        return None


def get_pred_result_from_mp(mp):
    """
    输入mp。mp是一个tuple （mention1_obj, mention2_obj, [true_num, valid_num, all_num]）。

    输出预测结果：如果true_num/valid_num大于0.5就算是作对了。真值共指就返回True，真值不共指就返回False；否则反过来。

    :param mp:
    :return: True共指 or False不共指 or None没预测出来（有效输出为0）
    """
    m1 = mp[0]
    m2 = mp[1]
    tva = mp[2]
    #
    groundtruth = (m1.gold_tag == m2.gold_tag)
    #
    t = tva[0]
    v = tva[1]
    if v == 0:
        return None
    if t/v > 0.5:
        # 作对了
        return groundtruth
    else:
        # 做错了
        return not groundtruth


def main():
    config_dict = {
        "cmp_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\data\3.pred\11-13\['36_ecb'](strategy3)_ChatGPT3.5(b1t0)_0shot_t13SAU_noSample(r1).c_mp",
        "output_path": r"E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\output",
    }
    # read corpus
    with open(config_dict["cmp_path"], 'rb') as f:
        corpus, mention_pairs = cPickle.load(f)
    #
    mention_pairs = mp_adaptor(mention_pairs)
    #
    statistic_dict = {}
    for cur_topic_id, cur_topic in mention_pairs.items():
        if cur_topic_id not in statistic_dict:
            statistic_dict[cur_topic_id] = {}
        for cur_doc_id, cur_doc in cur_topic.items():
            mp_dict = mp_list_to_mp_dict(cur_doc)
            m_list = mp_list_to_mention_list(cur_doc)
            statistic_dict[cur_topic_id][cur_doc_id] = {
                "总数量": 0,
                #
                "TP": 0,
                "TN": 0,
                "FP": 0,
                "FN": 0,
                #
                "li=0,l0=0,Cab=0": 0,
                "li=0,l0=1,Cab=0": 0,
                "li=1,l0=0,Cab=0": 0,
                "li=1,l0=1,Cab=0": 0,
                "li=2,l0=0,Cab=0": 0,
                "li=2,l0=1,Cab=0": 0,
                "li=0,l0=0,Cab=1": 0,
                "li=0,l0=1,Cab=1": 0,
                "li=1,l0=0,Cab=1": 0,
                "li=1,l0=1,Cab=1": 0,
                "li=2,l0=0,Cab=1": 0,
                "li=2,l0=1,Cab=1": 0,
            }
            for cur_mp in cur_doc:
                m1 = cur_mp[0]
                m2 = cur_mp[1]
                m1_id = f"{m1.doc_id}-{m1.sent_id}-{m1.start_offset}-{m1.end_offset}"
                m2_id = f"{m2.doc_id}-{m2.sent_id}-{m2.start_offset}-{m2.end_offset}"
                # Cab
                Cab = 1 if (m1.gold_tag == m2.gold_tag) else 0
                # l_0
                l0 = get_pred_result_from_mp(cur_mp)
                if l0 is None:
                    continue
                else:
                    l0 = 1 if l0 is True else 0
                # l_i
                for mi in m_list:
                    if mi in [m1, m2]:
                        continue
                    #    mi
                    #  /   \
                    # m1 - m2
                    mi_id = f"{mi.doc_id}-{mi.sent_id}-{mi.start_offset}-{mi.end_offset}"
                    #
                    li = 0  # l_i,记录m1-mi-m2的预测情况。这里一共2对关系，有几对被预测为共指，l_i就是几。
                    mp_1_i = get_mp_from_mp_dict(mp_dict, (m1_id, mi_id))
                    mp_2_i = get_mp_from_mp_dict(mp_dict, (m2_id, mi_id))
                    mp_1_i_result = get_pred_result_from_mp(mp_1_i)
                    mp_2_i_result = get_pred_result_from_mp(mp_2_i)
                    if None in [mp_1_i_result, mp_2_i_result]:
                        continue
                    else:
                        if mp_1_i_result is True:
                            li += 1
                        if mp_2_i_result is True:
                            li += 1
                    #
                    statistic_dict[cur_topic_id][cur_doc_id][f'li={li},l0={l0},Cab={Cab}'] += 1
    #
    print(statistic_dict)
    #
    for cur_topic_id, cur_topic in statistic_dict.items():
        statistic_dict[cur_topic_id]["all"] = {
            "总数量": 0,
            #
            "TP": 0,
            "TN": 0,
            "FP": 0,
            "FN": 0,
            #
            "li=0,l0=0,Cab=0": 0,
            "li=0,l0=1,Cab=0": 0,
            "li=1,l0=0,Cab=0": 0,
            "li=1,l0=1,Cab=0": 0,
            "li=2,l0=0,Cab=0": 0,
            "li=2,l0=1,Cab=0": 0,
            "li=0,l0=0,Cab=1": 0,
            "li=0,l0=1,Cab=1": 0,
            "li=1,l0=0,Cab=1": 0,
            "li=1,l0=1,Cab=1": 0,
            "li=2,l0=0,Cab=1": 0,
            "li=2,l0=1,Cab=1": 0,
        }
        for cur_doc_id, cur_doc in cur_topic.items():
            for k, v in cur_doc.items():
                statistic_dict[cur_topic_id]["all"][k] += v
        t = statistic_dict[cur_topic_id]["all"]
        def get_conditional_num(statistic_dict, conditions):
            r = 0
            for k, v in statistic_dict.items():
                flag = True
                for cur_condition in conditions:
                    if cur_condition not in k:
                        flag = False
                if flag:
                    r += v
            return r
        t["P(li=0|l0=0)"] = (get_conditional_num(t, ["li=0", "l0=0"]) / get_conditional_num(t, ["l0=0"]))
        t["P(li=0|l0=1)"] = (get_conditional_num(t, ["li=0", "l0=1"]) / get_conditional_num(t, ["l0=1"]))
        t["P(li=1|l0=0)"] = (get_conditional_num(t, ["li=1", "l0=0"]) / get_conditional_num(t, ["l0=0"]))
        t["P(li=1|l0=1)"] = (get_conditional_num(t, ["li=1", "l0=1"]) / get_conditional_num(t, ["l0=1"]))
        t["P(li=2|l0=0)"] = (get_conditional_num(t, ["li=2", "l0=0"]) / get_conditional_num(t, ["l0=0"]))
        t["P(li=2|l0=1)"] = (get_conditional_num(t, ["li=2", "l0=1"]) / get_conditional_num(t, ["l0=1"]))
        t["P(li=0|l0=0,Cab=0)"] = (get_conditional_num(t, ["li=0,l0=0,Cab=0"]) / get_conditional_num(t, ["l0=0,Cab=0"]))
        t["P(li=0|l0=0,Cab=1)"] = (get_conditional_num(t, ["li=0,l0=0,Cab=1"]) / get_conditional_num(t, ["l0=0,Cab=1"]))
        t["P(li=0|l0=1,Cab=0)"] = (get_conditional_num(t, ["li=0,l0=1,Cab=0"]) / get_conditional_num(t, ["l0=1,Cab=0"]))
        t["P(li=0|l0=1,Cab=1)"] = (get_conditional_num(t, ["li=0,l0=1,Cab=1"]) / get_conditional_num(t, ["l0=1,Cab=1"]))
        t["P(li=1|l0=0,Cab=0)"] = (get_conditional_num(t, ["li=1,l0=0,Cab=0"]) / get_conditional_num(t, ["l0=0,Cab=0"]))
        t["P(li=1|l0=0,Cab=1)"] = (get_conditional_num(t, ["li=1,l0=0,Cab=1"]) / get_conditional_num(t, ["l0=0,Cab=1"]))
        t["P(li=1|l0=1,Cab=0)"] = (get_conditional_num(t, ["li=1,l0=1,Cab=0"]) / get_conditional_num(t, ["l0=1,Cab=0"]))
        t["P(li=1|l0=1,Cab=1)"] = (get_conditional_num(t, ["li=1,l0=1,Cab=1"]) / get_conditional_num(t, ["l0=1,Cab=1"]))
        t["P(li=2|l0=0,Cab=0)"] = (get_conditional_num(t, ["li=2,l0=0,Cab=0"]) / get_conditional_num(t, ["l0=0,Cab=0"]))
        t["P(li=2|l0=0,Cab=1)"] = (get_conditional_num(t, ["li=2,l0=0,Cab=1"]) / get_conditional_num(t, ["l0=0,Cab=1"]))
        t["P(li=2|l0=1,Cab=0)"] = (get_conditional_num(t, ["li=2,l0=1,Cab=0"]) / get_conditional_num(t, ["l0=1,Cab=0"]))
        t["P(li=2|l0=1,Cab=1)"] = (get_conditional_num(t, ["li=2,l0=1,Cab=1"]) / get_conditional_num(t, ["l0=1,Cab=1"]))
        t["P(Cab=1)"] = (get_conditional_num(t, ["Cab=1"]) / get_conditional_num(t, ["="]))
        t["P(Cab=0)"] = (get_conditional_num(t, ["Cab=0"]) / get_conditional_num(t, ["="]))
        t["P(l0=0)"] = (get_conditional_num(t, ["l0=0"]) / get_conditional_num(t, ["="]))
        t["P(l0=1)"] = (get_conditional_num(t, ["l0=0"]) / get_conditional_num(t, ["="]))
        t["P(l0=0|Cab=1)"] = (get_conditional_num(t, ["l0=0,Cab=1"]) / get_conditional_num(t, ["Cab=1"]))
        t["P(l0=1|Cab=1)"] = (get_conditional_num(t, ["l0=1,Cab=1"]) / get_conditional_num(t, ["Cab=1"]))
        t["P(l0=0|Cab=0)"] = (get_conditional_num(t, ["l0=0,Cab=0"]) / get_conditional_num(t, ["Cab=0"]))
        t["P(l0=1|Cab=0)"] = (get_conditional_num(t, ["l0=1,Cab=0"]) / get_conditional_num(t, ["Cab=0"]))
    print(statistic_dict)
    path = os.path.join("E:\ProgramCode\WhatGPTKnowsAboutWhoIsWho\WhatGPTKnowsAboutWhoIsWho-main\Models2\output", "statistic_dict.pkl")
    with open(path, mode="wb") as f:
        cPickle.dump(statistic_dict, f)
        print(f"OUTPUT: statistic dict saved in {path}")


if __name__ == '__main__':
    main()
