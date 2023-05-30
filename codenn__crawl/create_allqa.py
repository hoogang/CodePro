import pickle


def main(lang_type):
    # 读取codeqa数据
    # [(id, index),[[c]] 代码，[q] 查询, 块长度]
    codeqa_blocks = pickle.load(open('./corpus/%s_codeqa_qid2index_blocks_labeled.pickle' % lang_type, 'rb'))

    # 根据qid-index排序 [(qid,index),query,code]
    codeqa_iid_title2codes = [[i[0], i[2][0], i[1][0][0]] for i in codeqa_blocks]
    print('%s语言分类后的qid-index总长度为%d' % (lang_type, len(codeqa_iid_title2codes)))

    #############################################多候选#############################################
    codeqa_mutiple_qid_titles = dict([(i[0][0], i[1]) for i in codeqa_iid_title2codes])

    pickle.dump(codeqa_mutiple_qid_titles,open('../origin_data/allqa_corpus/%s_multiple_qid_to_title.pickle' % lang_type, 'wb'))

    codeqa_mutiple_iid_codes = dict([(i[0], i[2]) for i in codeqa_iid_title2codes])

    pickle.dump(codeqa_mutiple_iid_codes,open('../origin_data/allqa_corpus/%s_multiple_iid_to_code.pickle' % lang_type, 'wb'))


sql_type = 'sql'
python_type = 'python'


if __name__ == '__main__':
    main(sql_type)
    main(python_type)