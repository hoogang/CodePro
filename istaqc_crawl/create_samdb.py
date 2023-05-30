import pickle
import pandas as pd

def main(lang_type):
    # 读取codemf对qid的排序
    corpus_sorted_qids =  pd.read_csv('./corpus/%s_saihnn_corpus_qids_scored.csv' %lang_type, index_col=0, sep='|', encoding='utf-8')
    corpus_qids  =  corpus_sorted_qids['qid'].tolist()
    corpus_scores = corpus_sorted_qids['score'].tolist()
    corpus_qid2scores = [(i,j) for i,j in zip(corpus_qids,corpus_scores)]

    # 读取hnn标签
    hnn_iids_labels = pickle.load(open('../data_hnn/%s_hnn_iids_labeles.pickle' % lang_type, 'rb'))
    # 取帖子的qid列表
    hnn_qids = set([i[0] for i in hnn_iids_labels])

    # 读取codedb数据
    # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度]
    codedb_blocks = pickle.load(open('./corpus/%s_codedb_qid2index_blocks_unlabeled.pickle' % lang_type, 'rb'))

    codedb_qids = set([i[0][0] for i in  codedb_blocks])
    print('%s语言的原始数据开始的qid长度%s'%(lang_type,len(codedb_qids)))

    codedb_qid2scores = [i for i in corpus_qid2scores if i[0] in codedb_qids]

    codedb_sorted_qids = [i[0] for i in  sorted(codedb_qid2scores,key=lambda x: x[1],reverse=True)]

    #  保留hnn qids
    codedb_filter_sqids = [i for i in  codedb_sorted_qids if i not in hnn_qids]
    print('%s语言的原始数据滤除后qid长度%s'%(lang_type, len(codedb_filter_sqids)))

    # 取前两位数据排名靠前的
    cut_blockqid_intlen   = (len(codedb_filter_sqids)//10000)*10000
    topcut_block_qids = codedb_filter_sqids[:cut_blockqid_intlen]

    remove_block_qids = list(set(codedb_filter_sqids).difference(set(topcut_block_qids)))
    print('%s语言的原始数据去除的qid长度%s'%(lang_type, len(remove_block_qids)))

    samdb_blocks = [i for i in codedb_blocks if i[0][0] not in remove_block_qids]
    samdb_block_qids = set([i[0][0] for i in samdb_blocks])
    print('%s语言的原始数据保留的qid长度%s'%(lang_type, len(samdb_block_qids)))
    print('%s语言的原始数据qid-index的长度为%s'%(lang_type, len(samdb_blocks)))

    # 读取最终的标签
    with open("./corpus/%s_codedb_blocks_labled_by_multiple_iids.txt" % lang_type, "r") as f:
        # 读取list
        qid2index_labels = [eval(i) for i in f.read().splitlines()]
        # 新qid-index的排序
        qid2index_labels = sorted(qid2index_labels, key=lambda x: (x[0], x[1]))

    filter_samdb_blocks= [i for i in samdb_blocks  if i[0] in qid2index_labels]
    print('%s语言的标注数据qid-index的长度为%s' % (lang_type, len(filter_samdb_blocks)))

    # 根据qid-index排序 [(qid,index),query,code]
    samdb_iid_title2codes_labled = [[i[0],i[3][0],i[2][0][0]] for i in filter_samdb_blocks]

    # 统计qid类型
    samdb_labled_qids = set([i[0][0] for i in samdb_iid_title2codes_labled])
    # 计算qid-index
    samdb_labled_iids_count = [(j, len([i for i in samdb_iid_title2codes_labled if i[0][0] == j])) for j in samdb_labled_qids]

    # 拿出单候选数据
    samdb_unlabeled_single_qids = set([i[0] for i in  samdb_labled_iids_count if i[1] ==1])
    print('%s语言分类后的single-qid总长度为%d'%(lang_type, len(samdb_unlabeled_single_qids)))
    samdb_unlabeled_single_iid_title2codes= [i for i in samdb_iid_title2codes_labled  if i[0][0] in samdb_unlabeled_single_qids]
    print('%s语言分类后的single-qid-index总长度为%d'%(lang_type,len(samdb_unlabeled_single_iid_title2codes)))

    # 拿出多候选数据
    samdb_classifier_mutiple_qids = set([i[0] for i in samdb_labled_iids_count  if i[1] != 1])
    print('%s语言分类后的mutiple-qid总长度为%d'%(lang_type,len(samdb_classifier_mutiple_qids)))
    
    # qid-index 重新排序
    samdb_classifier_mutiple_iid_title2codes = []
    # 循环遍历
    for qid in samdb_classifier_mutiple_qids:
        # 拿出指定qid 列表数据
        samdb_iid_title2codes = [i for i in samdb_iid_title2codes_labled if i[0][0] == qid]
        # 根据qid-index排序 [(qid,index),query,code]
        samdb_iid_title2codes = sorted(samdb_iid_title2codes, key=lambda x: (x[0][0], x[0][1]))
        # 拿出qid-coun计数
        count = [i[1] for i in samdb_labled_iids_count if i[0] == qid][0]
        # 重新index排序
        for i, j in zip(range(count), samdb_iid_title2codes):
            samdb_classifier_mutiple_iid_title2codes.append([(qid, i), j[1], j[2]])

    print('%s语言分类后的mutiple-qid-index总长度为%d' % (lang_type, len(samdb_classifier_mutiple_iid_title2codes)))

    samdb_classifier_single_qid_title2codes = [[i[0][0], i[1], i[2]] for i in samdb_unlabeled_single_iid_title2codes]

    # 新qid-index的排序
    samdb_mergeall_single_qid_title2codes = sorted(samdb_classifier_single_qid_title2codes, key=lambda x: x[0])
    # 拿出对应qid
    samdb_mergeall_mutiple_qids = [i[0] for i in samdb_mergeall_single_qid_title2codes]

    print('%s语言合并后的single-qid总长度为%d' % (lang_type, len(samdb_mergeall_mutiple_qids)))

    samdb_single_qid_titles = dict([(i[0], i[1]) for i in samdb_mergeall_single_qid_title2codes])

    pickle.dump(samdb_single_qid_titles,open('../compare_data/samdb_corpus/%s_single_qid_to_title.pickle' % lang_type, 'wb'))

    samdb_single_qid_codes = dict([(i[0], i[2]) for i in samdb_mergeall_single_qid_title2codes])

    pickle.dump(samdb_single_qid_codes, open('../compare_data/samdb_corpus/%s_single_qid_to_code.pickle' % lang_type, 'wb'))

    #############################################多候选#############################################
    samdb_mutiple_qid_titles = dict([(i[0][0], i[1]) for i in samdb_classifier_mutiple_iid_title2codes])

    pickle.dump(samdb_mutiple_qid_titles,open('../compare_data/samdb_corpus/%s_multiple_qid_to_title.pickle' % lang_type, 'wb'))

    samdb_mutiple_iid_codes = dict([(i[0], i[2]) for i in samdb_classifier_mutiple_iid_title2codes])

    pickle.dump(samdb_mutiple_iid_codes,open('../compare_data/samdb_corpus/%s_multiple_iid_to_code.pickle' % lang_type, 'wb'))



sql_type = 'sql'
python_type = 'python'


#sql
'''
sql语言的原始数据开始的qid长度492295
sql语言的原始数据滤除后qid长度490709
sql语言的原始数据去除的qid长度709
sql语言的原始数据保留的qid长度491586
sql语言的原始数据qid-index的长度为804313
sql语言的标注数据qid-index的长度为453089
sql语言分类后的single-qid总长度为345547
sql语言分类后的single-qid-index总长度为345547
sql语言分类后的mutiple-qid总长度为48757
sql语言分类后的mutiple-qid-index总长度为107542
sql语言合并后的single-qid总长度为345547
'''

#python
'''
python语言的原始数据开始的qid长度628920
python语言的原始数据滤除后qid长度626868
python语言的原始数据去除的qid长度6868
python语言的原始数据保留的qid长度622052
python语言的原始数据qid-index的长度为1181405
python语言的标注数据qid-index的长度为584865
python语言分类后的single-qid总长度为417721
python语言分类后的single-qid-index总长度为417721
python语言分类后的mutiple-qid总长度为73171
python语言分类后的mutiple-qid-index总长度为167144
python语言合并后的single-qid总长度为417721
'''






if __name__ == '__main__':
    main(sql_type)
    main(python_type)








