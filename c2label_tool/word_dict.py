
import pickle


def get_vocab(corpus1,corpus2):
    word_vacab = set()
    for i in range(0,len(corpus1)):
        for j in range(0,len(corpus1[i][1][0])):
            word_vacab.add(corpus1[i][1][0][j])
        for j in range(0,len(corpus1[i][1][1])):
            word_vacab.add(corpus1[i][1][1][j])
        for j in range(0,len(corpus1[i][2][0])): #len(corpus2[i][2])
            word_vacab.add(corpus1[i][2][0][j])  #注意之前是 word_vacab.add(corpus2[i][2][j])
        for j in range(0,len(corpus1[i][3])):
            word_vacab.add(corpus1[i][3][j])

    for i in range(0,len(corpus2)):
        for j in range(0,len(corpus2[i][1][0])):
            word_vacab.add(corpus2[i][1][0][j])
        for j in range(0,len(corpus2[i][1][1])):
            word_vacab.add(corpus2[i][1][1][j])
        for j in range(0,len(corpus2[i][2][0])):#len(corpus2[i][2])
            word_vacab.add(corpus2[i][2][0][j]) #注意之前是 word_vacab.add(corpus2[i][2][j])
        for j in range(0,len(corpus2[i][3])):
            word_vacab.add(corpus2[i][3][j])
    print(len(word_vacab))
    return word_vacab


def load_pickle(filename):
    return pickle.load(open(filename, 'rb'))

#构建初步词典
def vocab_propcess(filepath1,filepath2,save_path):
    with open(filepath1, 'r')as f:
        total_data1 = eval(f.read())
        f.close()

    with open(filepath2, 'r')as f:
        total_data2 = eval(f.read())
        f.close()

    x1= get_vocab(total_data1,total_data2)
    #total_data_sort = sorted(x1, key=lambda x: (x[0], x[1]))
    f = open(save_path, "w")
    f.write(str(x1))
    f.close()


def final_vocab_process(filepath1,filepath2,save_path):
    word_set = set()

    with open(filepath1, 'r')as f:
        total_data1 = set(eval(f.read()))
        f.close()

    with open(filepath2, 'r')as f:
        total_data2 = eval(f.read())
        f.close()
    total_data1 = list(total_data1)
    x1= get_vocab(total_data2,total_data2)
    #total_data_sort = sorted(x1, key=lambda x: (x[0], x[1]))

    for i in x1:
        if i in total_data1:
            continue
        else:
            word_set.add(i)
    print(len(total_data1))
    print(len(word_set))
    f = open(save_path, "w")
    f.write(str(word_set))
    f.close()




if __name__ == "__main__":

    # #====================获取staqc的词语集合===============
    python_hnn_tokens =   '../data_hnn/python_hnn_qid2index_blocks_labeled_tokennized.txt'
    python_staqc_tokens = ' ../staqc_tolabel/data/code_solution_labeled_data/filter/tokennized_data/python_staqc_multiple_iid_title&blocks_to_mutiple_qids_tokennized.txt'
    python_staqc_dict =  '../staqc_tolabel/data/code_solution_labeled_data/filter/word_dict/python_word_vocab_dict.txt'

    sql_hnn_tokens =    '../data_hnn/sql_hnn_qid2index_blocks_labeled_tokennized.txt'
    sql_staqc_tokens =  ' ../staqc_tolabel/data/code_solution_labeled_data/filter/sql_staqc_multiple_iid_title&blocks_to_mutiple_qids_tokennized.txt'
    sql_staqc_dict =    '../staqc_tolabel/data/code_solution_labeled_data/filter/word_dict/sql_word_vocab_dict.txt'
    #
    # #vocab_process(python_hnn_tokens,python_staqc_tokens,python_staqc_dict)
    # #vocab_process(sql_hnn_tokens,sql_staqc_tokens,sql_staqc_dict)
    #
    # #====================获取最后大语料的词语集合的词语集合===============
    # sql_large_tokens =  '../istaqc_crawl/corpus/tokennized_data/sql_codedb_qid2index_blocks_to_multiple_qids_tokennized.txt'
    # sql_large_dict  =   '../istaqc_crawl/corpus/word_dict/sql_word_vocab_dict.txt'
    # final_vocab_process(sql_staqc_dict, sql_large_tokens, sql_large_dict)
    #
    #
    # python_large_tokens = '../istaqc_crawl/corpus/tokennized_data/python_codedb_qid2index_blocks_to_multiple_qids_tokennized.txt'
    # python_large_dict = '../istaqc_crawl/corpus/word_dict/python_word_vocab_dict.txt'
    # final_vocab_process(python_staqc_dict,python_large_tokens,python_large_dict)



    sql_corpus_tokens =  '../hqfso_tool/corpus/tokennized_data/sql_corpus_qid2index_blocks_to_multiple_qids_tokennized.txt'
    sql_corpus_dict  =   '../hqfso_tool/corpus/word_dict/sql_word_vocab_dict.txt'
    final_vocab_process(sql_staqc_dict, sql_corpus_tokens, sql_corpus_dict)


    python_corpus_tokens = '../hqfso_tool/corpus/tokennized_data/python_corpus_qid2index_blocks_to_multiple_qids_tokennized.txt'
    python_corpus_dict = '../hqfso_tool/corpus/word_dict/python_word_vocab_dict.txt'
    final_vocab_process(python_staqc_dict,python_corpus_tokens,python_corpus_dict)




