
import pickle
import pandas as pd


def main(lang_type):

    # [(id, index),[[si，si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度, 标签]

    with open('../label_tool/data/code_solution_labeled_data/filter/%s_how_to_do_it_by_classifier_multiple_iid_title&blocks.txt'%lang_type,
            "r") as f:
        # 字符串列表放置
        codedb_mutiple_iid_title2blocks_unlabled = eval(f.read())
        # 取qids
        codedb_mutiple_qids = set([i[0][0] for i in codedb_mutiple_iid_title2blocks_unlabled])
        print('codedb的qid长度%s'%len(codedb_mutiple_qids))
        # 取qid-index
        codedb_mutiple_iids = [i[0] for i in codedb_mutiple_iid_title2blocks_unlabled]
        print('codedb的qid-index长度%s' % len(codedb_mutiple_iids))
        # 计算qid-index计数
        codedb_mutiple_iids_count = [(j, len([i for i in codedb_mutiple_iids  if i[0] == j])) for j in codedb_mutiple_qids]
        # 对qid-index排序
        codedb_mutiple_iids_dict = dict(codedb_mutiple_iids_count)

    # [(id,index), q 查询,c 代码]

    with open('../label_tool/data/code_solution_labeled_data/filter/%s_how_to_do_it_by_classifier_multiple_iid_title&codes.txt'%lang_type,
            "r") as f:
        # 字符串以\n分割
        staqc_mutiple_iid_title2codes_unlabled = [eval(i) for i in f.read().split('\n')[:-1]]
        # 取qids
        staqc_mutiple_qids = set([i[0][0] for i in staqc_mutiple_iid_title2codes_unlabled])
        print('staqc的qid长度%s' % len(staqc_mutiple_qids))
        # 取qid-index
        staqc_mutiple_iids = [i[0] for i in staqc_mutiple_iid_title2codes_unlabled]
        print('staqc的qid-index长度%s' % len(staqc_mutiple_iids))
        # 计算qid-index计数
        staqc_mutiple_iids_count = [(j, len([i for i in staqc_mutiple_iids if i[0] == j])) for j in  staqc_mutiple_qids]
        # 对qid-index排序
        staqc_mutiple_iids_dict = dict(staqc_mutiple_iids_count)

    # 求qid交集 (两者qid是否相同）
    common_qids = list(set(staqc_mutiple_qids) & set(codedb_mutiple_qids))

    # 求qid-index交集 (两者qid-index个数是否相同）
    common_iid_qids = [qid for qid in common_qids if staqc_mutiple_iids_dict[qid] == codedb_mutiple_iids_dict[qid]]

    print('staqc和codedb共有（qid-index)的qid长度%s'%len(common_iid_qids))

    with  open("%s_staqc_labled_by_classifier_multiple_iids.txt" % lang_type, "r") as f1:
        # 读取(qid,index) list
        staqc_qid2index = [eval(i) for i in f1.read().splitlines()]
        # 过滤
        filter_staqc_qid2index = [i for i in staqc_qid2index if i[0] in common_iid_qids]
        # 排序
        sorted_filter_staqc_qid2index = sorted(filter_staqc_qid2index,key=lambda x: x[0])

        print('staqc过滤后的qid-index长度%s' % len(sorted_filter_staqc_qid2index))

        with open("%s_staqc_labled_by_filter_multiple_iids.txt" % lang_type, "a") as f11:
            for i in sorted_filter_staqc_qid2index:
                f11.write(str(i)+'\n')
            f11.close()

    with  open("%s_codedb_labled_by_classifier_multiple_iids.txt" % lang_type, "r") as f2:
        # 读取(qid,index) list
        codedb_qid2index = [eval(i) for i in f2.read().splitlines()]
        # 过滤
        filter_codedb_qid2index = [i for i in codedb_qid2index if i[0] in common_iid_qids]
        # 排序
        sorted_filter_codedb_qid2index = sorted(filter_codedb_qid2index, key=lambda x: x[0])

        print('codedb过滤后的qid-index长度%s' % len(sorted_filter_codedb_qid2index))

        with open("%s_codedb_labled_by_filter_multiple_iids.txt" % lang_type, "a") as f21:
            for i in sorted_filter_codedb_qid2index:
                f21.write(str(i)+'\n')
            f21.close()

    # 读取codeqa数据
    # [(id, index),[[c]] 代码，[q] 查询, 块长度]
    codeqa_blocks = pickle.load(open('../codemf_crawl/corpus/%s_codeqa_qid2index_blocks_labeled.pickle' %lang_type, 'rb'))

    codeqa_qids = set([i[0][0] for i in codeqa_blocks])
    print('%s语言的codeqa的qid长度%s' % (lang_type,len(codeqa_qids)))

    codeqa_iids = [i[0] for i in codeqa_blocks]
    # 新qid-index的排序
    codeqa_iids = sorted(codeqa_iids, key=lambda x: (x[0], x[1]))
    print('%s语言的codeqa的qid-index长度%s' % (lang_type,len(codeqa_iids)))

    # 读取codedb数据
    # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度，标签]
    codedb_blocks = pickle.load(open('../sattmf_crawl/corpus/%s_codedb_qid2index_blocks_unlabeled.pickle'%lang_type, 'rb'))

    codedb_qids = set([i[0][0] for i in codedb_blocks])
    print('%s语言的codedb的qid长度%s' % (lang_type,len(codedb_qids)))

    codedb_iids = [i[0] for i in codedb_blocks]
    # 新qid-index的排序
    codedb_iids = sorted(codedb_iids, key=lambda x: (x[0], x[1]))
    print('%s语言的codedb的qid-index长度%s' % (lang_type,len(codedb_iids)))

    # 保证有小波变换值
    codeqa_sorted = pd.read_csv('../codemf_crawl/corpus/%s_codeqa_qids_scored.csv' % lang_type, index_col=0, sep='|', encoding='utf-8')
    codeqa_sqids = codeqa_sorted['qid'].tolist()

    codeqa_ranked = pd.read_csv('../codemf_crawl/corpus/%s_codeqa_qids_ranked.csv' % lang_type, index_col=0, sep='|', encoding='utf-8')
    codeqa_rqids = codeqa_ranked['qid'].tolist()

    # 保证codeqa和codedb有交集
    common_qdqids = list(set(codeqa_qids) & set(codedb_qids))
    # 保证rank和score有交集
    common_srqids = list(set(codeqa_sqids) & set(codeqa_rqids))

    common_qids = list(set(common_qdqids) & set(common_srqids))

    # sql 139774  python
    print('%s语言的codedb和codeqa共同的qid长度为%s'%(lang_type,len(common_qids)))

    # 打分值从大到小排序
    # sql 217712  python 
    filter_rqids = [i for i in codeqa_rqids if i in common_qids]
    print('%s语言的codeqa中rank的qid长度为%s' % (lang_type, len(filter_rqids)))

    # 打分值从大到小排序
    # sql 217712
    filter_sqids = [i for i in codeqa_sqids if i in common_qids]
    print('%s语言的codeqa中score的qid长度为%s'%(lang_type,len(filter_sqids)))

    # 读取codedb标签
    with open("../sattmf_crawl/corpus/%s_codedb_blocks_labled_by_multiple_iids.txt" % lang_type, "r") as f:
        # 读取list
        qid2index_labels = [eval(i) for i in f.read().splitlines()]
        # 新qid-index的排序
        qid2index_labels = sorted(qid2index_labels, key=lambda x: (x[0], x[1]))

    labeled_codedb_iids = [i for i in codedb_iids if i in qid2index_labels]
    labeled_codedb_qids = set([i[0] for i in labeled_codedb_iids])

    # sql 93993   python
    filter_codedb_qids =  [i for i in labeled_codedb_qids if i in common_srqids]
    print('%s语言的codedb中保留的qid长度为%s'%(lang_type,len(filter_codedb_qids)))
    # sql 93993  python
    filter_codeqa_rqids = filter_rqids[:len(filter_codedb_qids)]
    print('%s语言的codeqa中rank的qid长度为%s'% (lang_type,len(filter_codeqa_rqids)))
    # sql 93993  python
    filter_codeqa_sqids = filter_sqids[:len(filter_codedb_qids)]
    print('%s语言的codeqa中score的qid长度为%s'%(lang_type,len(filter_codeqa_sqids)))

    #######################################################codeqa#################################################################
    #######################################################upvote###############################################################
    # [(id, index),[[c]] 代码，[q] 查询, 块长度]
    # sql 193887
    filter_codeqa_blocks = [i for i in codeqa_blocks if i[0][0] in filter_codeqa_rqids]
    print('%s语言的codeqa中rank保留的qid-index长度为%s' % (lang_type, len(filter_codeqa_blocks)))

    # 根据qid-index排序 [(qid,index),query,code]
    codeqa_filter_mutiple_iid_title2codes = [[i[0], i[2][0], i[1][0][0]] for i in filter_codeqa_blocks]

    #############################################多候选#############################################
    codeqa_mutiple_qid_titles = dict([(i[0][0], i[1]) for i in codeqa_filter_mutiple_iid_title2codes])

    pickle.dump(codeqa_mutiple_qid_titles,open('../compare_data/upvqc_corpus/%s_multiple_qid_to_title.pickle' % lang_type, 'wb'))

    codeqa_mutiple_iid_codes = dict([(i[0], i[2]) for i in codeqa_filter_mutiple_iid_title2codes])

    pickle.dump(codeqa_mutiple_iid_codes,open('../compare_data/upvqc_corpus/%s_multiple_iid_to_code.pickle' % lang_type, 'wb'))

    #######################################################codemf###############################################################
    # [(id, index),[[c]] 代码，[q] 查询, 块长度]
    # sql
    filter_codeqa_blocks = [i for i in codeqa_blocks if i[0][0] in filter_codeqa_sqids]
    print('%s语言的codeqa中score保留的qid-index长度为%s' % (lang_type, len(filter_codeqa_blocks)))

    # 根据qid-index排序 [(qid,index),query,code]
    codeqa_filter_mutiple_iid_title2codes = [[i[0], i[2][0], i[1][0][0]] for i in filter_codeqa_blocks]

    #############################################多候选#############################################
    codeqa_mutiple_qid_titles = dict([(i[0][0], i[1]) for i in codeqa_filter_mutiple_iid_title2codes])

    pickle.dump(codeqa_mutiple_qid_titles, open('../compare_data/mucqc_corpus/%s_multiple_qid_to_title.pickle' % lang_type, 'wb'))

    codeqa_mutiple_iid_codes = dict([(i[0], i[2]) for i in codeqa_filter_mutiple_iid_title2codes])

    pickle.dump(codeqa_mutiple_iid_codes, open('../compare_data/mucqc_corpus/%s_multiple_iid_to_code.pickle' % lang_type, 'wb'))

    # 根据qid-index排序 [(qid,index),query,code]
    #  62588
    codeqa_filter_single_qid_title2codes = [i for i in codeqa_filter_mutiple_iid_title2codes if i[0][1] == 0]
    print('%s语言的codeqa的single-qid-index总长度为%d' % (lang_type, len(codeqa_filter_single_qid_title2codes)))

    #############################################单候选#############################################
    codeqa_single_qid_titles = dict([(i[0][0], i[1]) for i in codeqa_filter_single_qid_title2codes])

    pickle.dump(codeqa_single_qid_titles, open('../compare_data/sicqc_corpus/%s_single_qid_to_title.pickle' % lang_type, 'wb'))

    codeqa_single_qid_codes = dict([(i[0][0], i[2]) for i in codeqa_filter_single_qid_title2codes])

    pickle.dump(codeqa_single_qid_codes, open('../compare_data/sicqc_corpus/%s_single_qid_to_code.pickle' % lang_type, 'wb'))

    #######################################################codedb#################################################################
    # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度，标签]

    labeled_codedb_iids  =  [i for i in qid2index_labels if i[0] in filter_codedb_qids]
    filter_codedb_blocks =  [i for i in codedb_blocks if i[0] in labeled_codedb_iids]
    codedb_filter_mutiple_iid_title2codes = [[i[0], i[3][0], i[2][0][0]] for i in filter_codedb_blocks]
    # 133915
    print('%s语言的codedb最后保留的qid-index长度为%s' % (lang_type, len(codedb_filter_mutiple_iid_title2codes )))

    # 统计qid个数
    codedb_count_qids = set([i[0][0] for i in filter_codedb_blocks])
    # 统计qid个数
    codedb_iid_count = [(j, len([i for i in filter_codedb_blocks if i[0][0] == j])) for j in codedb_count_qids]

    # 单候选的拿出来
    codedb_single_qids = set([i[0] for i in codedb_iid_count if i[1] == 1])
    # sql 60983
    print('%s语言的codedb的single-qid总长度为%d' % (lang_type, len(codedb_single_qids)))

    codedb_classify_single_iid_title2codes = [i for i in codedb_filter_mutiple_iid_title2codes if i[0][0] in codedb_single_qids]
    # sql 60983
    print('%s语言的codedb的single-qid-index总长度为%d' % (lang_type, len(codedb_classify_single_iid_title2codes)))

    # 多候选的拿出来
    codedb_mutiple_qids = set([i[0] for i in codedb_iid_count if i[1] != 1])
    # sql 33010
    print('%s语言的codedb的的mutiple-qid总长度为%d' % (lang_type, len(codedb_mutiple_qids)))

    # qid-index 重新排序
    codedb_classify_mutiple_iid_title2codes = []
    # 循环遍历
    for qid in codedb_mutiple_qids:
        # 拿出指定qid 列表数据
        codedb_iid_title2codes = [i for i in codedb_filter_mutiple_iid_title2codes if i[0][0] == qid]
        # 根据qid-index排序 [(qid,index),query,code]
        codedb_iid_title2codes = sorted(codedb_iid_title2codes, key=lambda x: (x[0][0], x[0][1]))
        # 拿出qid-coun计数
        count = [i[1] for i in codedb_iid_count if i[0] == qid][0]
        # 重新index排序
        for i, j in zip(range(count), codedb_iid_title2codes):
            codedb_classify_mutiple_iid_title2codes.append([(qid, i), j[1], j[2]])
    # sql 72932
    print('%s语言的codedb的mutiple-qid-index总长度为%d' % (lang_type, len(codedb_classify_mutiple_iid_title2codes)))

    codedb_classify_single_qid_title2codes = [[i[0][0], i[1], i[2]] for i in codedb_classify_single_iid_title2codes]

    # 新qid-index的排序
    codedb_classify_single_qid_title2codes = sorted(codedb_classify_single_qid_title2codes, key=lambda x: x[0])

    codedb_single_qid_titles = dict([(i[0], i[1]) for i in codedb_classify_single_qid_title2codes])

    pickle.dump(codedb_single_qid_titles, open('../compare_data/saiqc_corpus/%s_single_qid_to_title.pickle' % lang_type, 'wb'))

    codedb_single_qid_codes = dict([(i[0], i[2]) for i in codedb_classify_single_qid_title2codes])

    pickle.dump(codedb_single_qid_codes, open('../compare_data/saiqc_corpus/%s_single_qid_to_code.pickle' % lang_type, 'wb'))

    #############################################多候选#############################################
    # 新qid-index的排序
    codedb_classify_mutiple_iid_title2codes = sorted(codedb_classify_mutiple_iid_title2codes, key=lambda x: (x[0],x[1]))

    codedb_mutiple_qid_titles = dict([(i[0][0], i[1]) for i in codedb_classify_mutiple_iid_title2codes])

    pickle.dump(codedb_mutiple_qid_titles, open('../compare_data/saiqc_corpus/%s_multiple_qid_to_title.pickle' % lang_type, 'wb'))

    codedb_mutiple_iid_codes = dict([(i[0], i[2]) for i in codedb_classify_mutiple_iid_title2codes])

    pickle.dump(codedb_mutiple_iid_codes, open('../compare_data/saiqc_corpus/%s_multiple_iid_to_code.pickle' % lang_type, 'wb'))

    


#sql
'''
sql语言的codeqa的qid长度148888
sql语言的codeqa的qid-index长度436325
sql语言的codedb的qid长度492295
sql语言的codedb的qid-index长度805249
sql语言的codedb和codeqa共同的qid长度为139774
sql语言的codeqa中rank的qid长度为139774
sql语言的codeqa中score的qid长度为139774
sql语言的codedb中保留的qid长度为93993
sql语言的codeqa中rank的qid长度为93993
sql语言的codeqa中score的qid长度为93993
sql语言的codeqa中rank保留的qid-index长度为273784
sql语言的codeqa中score保留的qid-index长度为282672
sql语言的codeqa的single-qid-index总长度为93993
sql语言的codedb最后保留的qid-index长度为133915
sql语言的codedb的single-qid总长度为60983
sql语言的codedb的single-qid-index总长度为60983
sql语言的codedb的的mutiple-qid总长度为33010
sql语言的codedb的mutiple-qid-index总长度为72932

saiqc的 负采样 4
'''


#python
'''
python语言的codeqa的qid长度245381
python语言的codeqa的qid-index长度816000
python语言的codedb的qid长度628920
python语言的codedb的qid-index长度1192482
python语言的codedb和codeqa共同的qid长度为231477
python语言的codeqa中rank的qid长度为231477
python语言的codeqa中score的qid长度为231477
python语言的codedb中保留的qid长度为161944
python语言的codeqa中rank的qid长度为161944
python语言的codeqa中score的qid长度为161944
python语言的codeqa中rank保留的qid-index长度为535151
python语言的codeqa中score保留的qid-index长度为553379
python语言的codeqa的single-qid-index总长度为161944
python语言的codedb最后保留的qid-index长度为218602
python语言的codedb的single-qid总长度为117703
python语言的codedb的single-qid-index总长度为117703
python语言的codedb的的mutiple-qid总长度为44241
python语言的codedb的mutiple-qid-index总长度为100899

saiqc的 负采样 4

'''




sql_type = 'sql'
python_type = 'python'

if __name__ == '__main__':
    main(sql_type)
    main(python_type)
