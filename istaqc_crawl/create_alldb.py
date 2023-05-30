import pickle

def main(lang_type):
    # 读取codedb数据
    # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度，标签]
    codedb_blocks = pickle.load(open('./corpus/%s_codedb_qid2index_blocks_unlabeled.pickle' % lang_type, 'rb'))

    # 根据qid-index排序 [(qid,index),query,code]
    codedb_iid_title2codes = [[i[0], i[3][0], i[2][0][0]] for i in codedb_blocks]
    print('%s语言的qid-index总长度为%d' % (lang_type, len(codedb_iid_title2codes)))

    #############################################多候选#############################################
    codedb_mutiple_qid_titles = dict([(i[0][0], i[1]) for i in codedb_iid_title2codes])

    pickle.dump(codedb_mutiple_qid_titles,open('../origin_data/alldb_corpus/%s_multiple_qid_to_title.pickle' % lang_type, 'wb'))

    codedb_mutiple_iid_codes = dict([(i[0], i[2]) for i in codedb_iid_title2codes])

    pickle.dump(codedb_mutiple_iid_codes,open('../origin_data/alldb_corpus/%s_multiple_iid_to_code.pickle' % lang_type, 'wb'))


sql_type = 'sql'
python_type = 'python'


if __name__ == '__main__':
    main(sql_type)
    main(python_type)