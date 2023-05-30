import math
import pickle
import pandas as pd
import numpy as np
from nltk import FreqDist

# ---------------------------------------------主程序------------------------------------
def main(corpus_folder,data_folder,lang_type):
    # 所有选项 [(qid, aid), qhtmlstr, ahtmlstr, soical_terms]
    corpus_qid_allterms = pickle.load(open(corpus_folder+'%s_codemf_corpus_qid_allterms.pickle' % lang_type, 'rb'))
    corpus_qids = set([i[0][0] for i in corpus_qid_allterms])

    # 读取codedb数据
    # [(id, index),[[c]] 代码，[q] 查询, 块长度，标签]
    codeqa_blocks = pickle.load(open(data_folder+'%s_codeqa_qid2index_blocks_labeled.pickle' % lang_type, 'rb'))
    codeqa_qids = set([i[0][0] for i in codeqa_blocks])

    # 社交数据 (qid,soical_terms)
    corpus_qid_features = [(i[0][0], i[4]) for i in corpus_qid_allterms]

    # 社交数据 (qid,[qview,qanswer,qcomment,qfavorite,qscorenum,ccomment,cscorenum])
    qidnum_lis    = [i[0] for i in corpus_qid_features]

    qview_lis     = [i[1][0] for i in corpus_qid_features]
    qanswer_lis   = [i[1][1] for i in corpus_qid_features]
    qcomment_lis  = [i[1][2] for i in corpus_qid_features]
    qfavorite_lis = [i[1][3] for i in corpus_qid_features]
    qscore_lis    = [i[1][4] for i in corpus_qid_features]
    ccomment_lis  = [i[1][5] for i in corpus_qid_features]
    cscore_lis    = [i[1][6] for i in corpus_qid_features]

    # 数据字典
    data_dict = {'qid': qidnum_lis,
                 'qview': qview_lis, 'qanswer': qanswer_lis, 'qcomment': qcomment_lis,'qfavorite': qfavorite_lis,
                 'qscore': qscore_lis, 'ccomment': ccomment_lis, 'cscore': cscore_lis}

    # 索引列表
    column_list = ['qid', 'qview', 'qanswer', 'qcomment', 'qfavorite', 'qscore', 'ccomment','cscore']
    # 数据保存
    attr_data = pd.DataFrame(data_dict, columns=column_list)

    # 缺省值补全
    attr_data[attr_data == -10000] = np.nan  # 将缺省值置换nan  数据爬虫不存在
    attr_data[attr_data == 0] = np.nan  # 将缺省值置换nan  帖子本身没给值

    # 每列去除噪声点,不具备分布的特征需要去除噪点
    remove_keys = ['qview', 'qscore', 'cscore']  # 类别点
    for key in remove_keys:
        key_freq = FreqDist([i for i in attr_data[key].tolist() if not math.isnan(i)])
        freq_num = [i[0] for i in key_freq.most_common()]  # 按频率出现从大到小排序
        key_noise = freq_num[-int(0.01 * len(freq_num)):]  # 出现最小%1比例作为噪声点,分析数据
        key_data = [i if i not in key_noise else np.nan for i in attr_data[key].tolist()]
        key_Serie = pd.Series(key_data, index=attr_data.index)
        attr_data[key] = key_Serie

    # 每列NaN填充平均数
    column_keys = ['qview', 'qanswer', 'qcomment', 'qfavorite', 'qscore', 'ccomment', 'cscore']

    # 列属性依次循环
    for key in column_keys:
        #  填充数据
        attr_data[key] = attr_data[key].fillna(round(attr_data[key].mean()))

    # 数据归一化操作
    data_mn = attr_data.reindex(columns=column_keys).apply(lambda x: (x - x.min()) / (x.max() - x.min()), 0)
    # 相应的值替换
    attr_data[column_keys] = data_mn
    # ---------------------------------------------数据预处理------------------------------------
    # 打分值排序，从大到小排序
    sort_attr = attr_data.sort_values(by='qfavorite', ascending=False)
    # 重建0-K索引值
    sort_attr = sort_attr.reset_index(drop=True)
    # 抽取前项索引值
    sort_attr = sort_attr.reindex(columns=['qid', 'qfavorite'])

    # 保存前项索引值
    sort_attr.to_csv(data_folder + '%s_codemf_corpus_qids_ranked.csv' % lang_type, sep='|', encoding='utf-8')



# 参数配置
corpus_folder = '../token_embed/corpus/'

data_folder ='./corpus/'

sql_type = 'sql'
python_type = 'python'

if __name__ == '__main__':
    # sqlang
    main(corpus_folder,data_folder,sql_type)
    # python
    main(corpus_folder,data_folder,python_type)