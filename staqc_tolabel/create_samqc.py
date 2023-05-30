import pickle

def main(lang_type):
    # [(id, index),[[si，si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度, 标签]
    with open('./data/code_solution_labeled_data/filter/%s_how_to_do_it_by_classify_multiple_iid_title&blocks.txt'%lang_type, "r") as f:
        samqc_mutiple_iid_title2blocks_unlabled = eval(f.read())
        # [(id, index),q，c]
        samqc_mutiple_iid_title2codes_unlabled = [[i[0],i[3][0],i[2][0][0]] for i in samqc_mutiple_iid_title2blocks_unlabled]
        # 新qid-index的排序
        samqc_mutiple_iid_title2codes_unlabled = sorted(samqc_mutiple_iid_title2codes_unlabled, key=lambda x: (x[0][0], x[0][1]))

        # 读取最终的标签
    with open("../final_collection/%s_samqc_labled_by_filter_multiple_iids.txt" % lang_type, "r") as f:
        # 读取list
        qid2index_labels = [eval(i) for i in f.read().splitlines()]
        # 新qid-index的排序
        qid2index_labels = sorted(qid2index_labels, key=lambda x: (x[0], x[1]))

    samqc_mutiple_iid_title2codes_labled = [i for i in samqc_mutiple_iid_title2codes_unlabled if i[0] in qid2index_labels]

    samqc_labled_mutiple_qids = set([i[0] for i in samqc_mutiple_iid_title2codes_labled])

    print('%s语言分类前的mutiple-qid总长度为%d' % (lang_type, len(samqc_labled_mutiple_qids)))

    # 统计qid个数
    samqc_labled_count_qids = set([i[0][0] for i in samqc_mutiple_iid_title2codes_labled])
    # 统计qid个数
    samqc_labled_iid_count = [(j, len([i for i in samqc_mutiple_iid_title2codes_labled if i[0][0] == j])) for j in samqc_labled_count_qids]

    # 单候选的拿出来
    samqc_classify_single_qids = set([i[0] for i in samqc_labled_iid_count if i[1] == 1])

    print('%s语言分类后的single-qid总长度为%d' % (lang_type, len(samqc_classify_single_qids)))

    samqc_classify_single_iid_title2codes = [i for i in samqc_mutiple_iid_title2codes_labled if i[0][0] in samqc_classify_single_qids]

    print('%s语言分类后的single-qid-index总长度为%d' % (lang_type, len(samqc_classify_single_iid_title2codes)))

    # 多候选的拿出来
    samqc_classify_mutiple_qids = set([i[0] for i in samqc_labled_iid_count if i[1] != 1])

    print('%s语言分类后的mutiple-qid总长度为%d' % (lang_type, len(samqc_classify_mutiple_qids)))

    # qid-index 重新排序
    samqc_classify_mutiple_iid_title2codes = []
    # 循环遍历
    for qid in samqc_classify_mutiple_qids:
        # 拿出指定qid 列表数据
        samqc_iid_title2codes = [i for i in samqc_mutiple_iid_title2codes_labled if i[0][0] == qid]
        # 根据qid-index排序 [(qid,index),query,code]
        samqc_iid_title2codes = sorted(samqc_iid_title2codes, key=lambda x: (x[0][0], x[0][1]))
        # 拿出qid-coun计数
        count = [i[1] for i in samqc_labled_iid_count if i[0] == qid][0]
        # 重新index排序
        for i, j in zip(range(count), samqc_iid_title2codes):
            samqc_classify_mutiple_iid_title2codes.append([(qid, i), j[1], j[2]])

    print('%s语言分类后的mutiple-qid-index总长度为%d' % (lang_type, len(samqc_classify_mutiple_iid_title2codes)))

    samqc_classify_single_qid_title2codes = [[i[0][0], i[1], i[2]] for i in samqc_classify_single_iid_title2codes]

    # [qid, q,c]
    with open('./data/code_solution_labeled_data/filter/%s_how_to_do_it_by_unlabeled_single_qid_title&codes.txt' % lang_type, "r") as f:
        # 字符串以\n分割
        samqc_unlabeled_single_qid_title2codes = [eval(i) for i in f.read().split('\n')[:-1]]

    samqc_mergeall_single_qid_title2codes = samqc_classify_single_qid_title2codes + samqc_unlabeled_single_qid_title2codes

    # 新qid-index的排序
    samqc_mergeall_single_qid_title2codes = sorted(samqc_mergeall_single_qid_title2codes, key=lambda x: x[0])
    # 拿出对应qid
    samqc_mergeall_single_qids = [i[0] for i in samqc_mergeall_single_qid_title2codes]

    print('%s语言合并后的single-qid总长度为%d' % (lang_type, len(samqc_mergeall_single_qids)))

    samqc_single_qid_titles = dict([(i[0], i[1]) for i in samqc_mergeall_single_qid_title2codes])

    pickle.dump(samqc_single_qid_titles, open('../compare_data/samqc_corpus/%s_single_qid_to_title.pickle' % lang_type, 'wb'))

    samqc_single_qid_codes = dict([(i[0], i[2]) for i in samqc_mergeall_single_qid_title2codes])

    pickle.dump(samqc_single_qid_codes, open('../compare_data/samqc_corpus/%s_single_qid_to_code.pickle' % lang_type, 'wb'))

    #############################################多候选#############################################
    # 新qid-index的排序
    samqc_classify_mutiple_iid_title2codes = sorted(samqc_classify_mutiple_iid_title2codes, key=lambda x: (x[0],x[1]))

    samqc_mutiple_qid_titles = dict([(i[0][0], i[1]) for i in samqc_classify_mutiple_iid_title2codes])

    pickle.dump(samqc_mutiple_qid_titles, open('../compare_data/samqc_corpus/%s_multiple_qid_to_title.pickle' % lang_type, 'wb'))

    samqc_mutiple_iid_codes = dict([(i[0], i[2]) for i in samqc_classify_mutiple_iid_title2codes])

    pickle.dump(samqc_mutiple_iid_codes, open('../compare_data/samqc_corpus/%s_multiple_iid_to_code.pickle' % lang_type, 'wb'))



#sql
'''
sql语言分类前的mutiple-qid总长度为44487
sql语言分类后的single-qid总长度为11189
sql语言分类后的single-qid-index总长度为11189
sql语言分类后的mutiple-qid总长度为15092
sql语言分类后的mutiple-qid-index总长度为33298
sql语言合并后的single-qid总长度为86826
'''

#python
'''
python语言分类前的mutiple-qid总长度为61981
python语言分类后的single-qid总长度为22943
python语言分类后的single-qid-index总长度为22943
python语言分类后的mutiple-qid总长度为17170
python语言分类后的mutiple-qid-index总长度为39038
python语言合并后的single-qid总长度为108237
'''



sql_type = 'sql'
python_type = 'python'

if __name__ == '__main__':
    #main(sql_type)
    main(python_type)
