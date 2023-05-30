

def main(lang_type,data_type):
    ### 经测试 hnn的语料又放进联合模型预测 ###

    f = open("%s_%s_labled_by_classifier_multiple_iids.txt"%(lang_type,data_type),"r")
    # 读取list
    qid2index_labels = [eval(i) for i in f.read().splitlines()]
    # 新qid-index的排序
    qid2index_labels = sorted(qid2index_labels, key=lambda x: (x[0], x[1]))
    print(qid2index_labels)
    #  python: 622252   sql:43882
    print('%s语言的qid-index总长度为%d' % (lang_type,len(qid2index_labels)))
    # 统计qid
    qid2index_qids = set([i[0] for i in qid2index_labels])
    # python: 40391   sql:26052
    print('%s语言的qid总长度为%d' % (lang_type,len(qid2index_qids)))

    # 计算qid-index计数
    qid2index_count = [(j, len([i for i in qid2index_labels if i[0]==j]))  for j in qid2index_qids]
    # 对qid-index排序
    qid2index_count = sorted(qid2index_count, key=lambda x: x[0])

    qid2index_single_qids =  [i[0] for i in qid2index_count if i[1] ==1]
    #  python: 23407   sql:11276
    print('%s语言的single-qid总长度为%d' % (lang_type,len(qid2index_single_qids)))
    single_qid2index = [i for i in qid2index_labels if i[0] in qid2index_single_qids]
    #  python: 23407   sql:1276
    print('%s语言的single-qid-index总长度为%d' % (lang_type,len(single_qid2index)))

    qid2index_mutiple_qids =  [i[0] for i in qid2index_count if i[1] !=1]
    #  python: 16894   sql:14776
    print('%s语言的mutiple-qid总长度为%d' %(lang_type, len(qid2index_mutiple_qids)))
    mutiple_qid2index = [i for i in qid2index_labels if i[0] in qid2index_mutiple_qids]
    #  python: 38845   sql:32606
    print('%s语言的single-qid-index总长度为%d' %(lang_type, len(mutiple_qid2index)))


# staqc

# python  一对一  85294 Posts        一对多  68555  Posts              一共  153849 Posts
#  原始    一对一  85294 条           一对多  187035  条                 一共  272329 条
#  预测                            一对一 23407,一对多 38845 （打1=62252）
#  人工                                         共计 2169  2169+60083=62252

# 我们                              一对多  187044 条
# 预测                             一对一      ，一对多       (打1=


#  sql   一对一   75637 Posts        一对多 39752  Posts               一共  115389 Posts
#  原始   一对一   75637 条           一对多 103684 条                   一共  179321 条
#  预测                           一对一 11276,一对多 32606 （打1=43882）
#  人工                                        共计 2056   2056+41826=43882

# 我们                              一对多  103658 条
# 预测                             一对一      ，一对多      (打1=




if __name__ == '__main__':
    main('python','staqc')
    main('sql','staqc')
