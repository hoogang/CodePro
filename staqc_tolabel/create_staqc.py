import pickle


def main(lang_type):
    # [(id,index), q,c]
    with open('./data/code_solution_labeled_data/filter/%s_how_to_do_it_by_classify_multiple_iid_title&codes.txt'%lang_type,"r") as f:
        # 字符串以\n分割
        staqc_mutiple_iid_title2codes_unlabled= [eval(i) for i in f.read().splitlines()]
        # 新qid-index的排序
        staqc_mutiple_iid_title2codes_unlabled = sorted(staqc_mutiple_iid_title2codes_unlabled, key=lambda x: (x[0][0], x[0][1]))

    # 读取最终的标签
    with open("../final_collection/%s_staqc_labled_by_filter_multiple_iids.txt" % lang_type, "r") as f:
        # 读取list
        qid2index_labels = [eval(i) for i in f.read().splitlines()]
        # 新qid-index的排序
        qid2index_labels = sorted(qid2index_labels, key=lambda x: (x[0], x[1]))

    staqc_mutiple_iid_title2codes_labled = [i for i in staqc_mutiple_iid_title2codes_unlabled if i[0] in qid2index_labels]

    staqc_labled_mutiple_qids = set([i[0] for i in staqc_mutiple_iid_title2codes_labled])

    print('%s语言分类前的mutiple-qid总长度为%d' % (lang_type, len(staqc_labled_mutiple_qids)))

    # 统计qid个数
    staqc_labled_count_qids = set([i[0][0] for i in staqc_mutiple_iid_title2codes_labled])
    # 统计qid个数
    staqc_labled_iid_count = [(j, len([i for i in staqc_mutiple_iid_title2codes_labled if i[0][0] == j])) for j in staqc_labled_count_qids]

    # 单候选的拿出来
    staqc_classify_single_qids = set([i[0] for i in staqc_labled_iid_count  if i[1] == 1])

    print('%s语言分类后的single-qid总长度为%d' % (lang_type, len(staqc_classify_single_qids)))
    print('%s语言分类后的single-qid-index总长度为%d' % (lang_type,len(staqc_classify_single_qids)))

    staqc_classify_single_iid_title2codes = [i for i in staqc_mutiple_iid_title2codes_labled  if i[0][0] in  staqc_classify_single_qids]

    # 多候选的拿出来
    staqc_classify_mutiple_qids = set([i[0] for i in staqc_labled_iid_count if i[1] != 1])

    print('%s语言分类后的mutiple-qid总长度为%d' %(lang_type, len(staqc_classify_mutiple_qids)))

    # qid-index 重新排序
    staqc_classify_mutiple_iid_title2codes=[]
    # 循环遍历
    for qid in staqc_classify_mutiple_qids:
        # 拿出指定qid 列表数据
        staqc_iid_title2codes = [i for i in staqc_mutiple_iid_title2codes_labled if i[0][0] == qid]
        # 根据qid-index排序 [(qid,index),query,code]
        staqc_iid_title2codes = sorted(staqc_iid_title2codes, key=lambda x: (x[0][0], x[0][1]))
        # 拿出qid-coun计数
        count = [i[1] for i in staqc_labled_iid_count if i[0]==qid][0]
        # 重新index排序
        for i,j in zip(range(count),staqc_iid_title2codes):
            staqc_classify_mutiple_iid_title2codes.append([(qid,i),j[1],j[2]])

    print('%s语言分类后的mutiple-qid-index总长度为%d' % (lang_type, len(staqc_classify_mutiple_iid_title2codes)))

    staqc_classify_single_qid_title2codes = [[i[0][0],i[1],i[2]] for i in staqc_classify_single_iid_title2codes]

    # [qid, q,c]
    with open('./data/code_solution_labeled_data/filter/%s_how_to_do_it_by_unlabeled_single_qid_title&codes.txt'% lang_type,"r") as f:
        # 字符串以\n分割
        staqc_unlabeled_single_qid_title2codes = [eval(i) for i in f.read().split('\n')[:-1]]

    staqc_mergeall_single_qid_title2codes = staqc_classify_single_qid_title2codes+staqc_unlabeled_single_qid_title2codes

    # 新qid-index的排序
    staqc_mergeall_single_qid_title2codes = sorted(staqc_mergeall_single_qid_title2codes, key=lambda x: x[0])
    # 拿出对应qid
    staqc_mergeall_single_qids = [i[0] for i in staqc_mergeall_single_qid_title2codes]

    print('%s语言合并后的single-qid总长度为%d' % (lang_type, len(staqc_mergeall_single_qids)))

    staqc_single_qid_titles = dict([(i[0], i[1]) for i in staqc_mergeall_single_qid_title2codes])

    pickle.dump(staqc_single_qid_titles,open('../compare_data/staqc_corpus/%s_single_qid_to_title.pickle' % lang_type, 'wb'))

    staqc_single_qid_codes = dict([(i[0], i[2]) for i in staqc_mergeall_single_qid_title2codes])

    pickle.dump(staqc_single_qid_codes, open('../compare_data/staqc_corpus/%s_single_qid_to_code.pickle' % lang_type, 'wb'))

    #############################################多候选#############################################
    # 新qid-index的排序
    staqc_classify_mutiple_iid_title2codes = sorted(staqc_classify_mutiple_iid_title2codes, key=lambda x: (x[0], x[1]))

    staqc_mutiple_qid_titles = dict([(i[0][0], i[1]) for i in staqc_classify_mutiple_iid_title2codes])

    pickle.dump(staqc_mutiple_qid_titles,open('../compare_data/staqc_corpus/%s_multiple_qid_to_title.pickle' % lang_type, 'wb'))

    staqc_mutiple_iid_codes = dict([(i[0], i[2]) for i in staqc_classify_mutiple_iid_title2codes])

    pickle.dump(staqc_mutiple_iid_codes,open('../compare_data/staqc_corpus/%s_multiple_iid_to_code.pickle' % lang_type, 'wb'))



#sql
'''
sql语言分类前的mutiple-qid总长度为43758
sql语言分类后的single-qid总长度为11251
sql语言分类后的single-qid-index总长度为11251
sql语言分类后的mutiple-qid总长度为14733
sql语言分类后的mutiple-qid-index总长度为32507
sql语言合并后的single-qid总长度为86888
'''

#python
'''
python语言分类前mutiple-qid总长度为61862
python语言分类后的single-qid总长度为23275
python语言分类后的single-qid-index总长度为23275
python语言分类后的mutiple-qid总长度为16875
python语言分类后的mutiple-qid-index总长度为38587
python语言合并后的single-qid总长度为108569
'''


sql_type = 'sql'
python_type = 'python'

if __name__ == '__main__':
    main(sql_type)
    main(python_type)