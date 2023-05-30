import pickle
from collections import Counter

def load_pickle(filename):
    return pickle.load(open(filename, 'rb'))


def single_list(arr, target):
    return arr.count(target)

#staqc：把语料中的单候选和多候选分隔开
def staqc_process(filepath,save_single_path,save_mutiple_path):
    with open(filepath,'r')as f:
        total_data= eval(f.read())
        f.close()
    qids = []
    for i in range(0, len(total_data)):
        qids.append(total_data[i][0][0])

    result = Counter(qids)

    total_data_single = []
    total_data_multiple = []
    for i in range(0, len(total_data)):
        if(result[total_data[i][0][0]]==1):
            total_data_single.append(total_data[i])

        else:
            total_data_multiple.append(total_data[i])

    f = open(save_single_path, "w")
    f.write(str(total_data_single))
    f.close()

    f = open(save_mutiple_path, "w")
    f.write(str(total_data_multiple))
    f.close()


#large:把语料中的单候选和多候选分隔开
def large_process(filepath,save_single_path,save_mutiple_path):

    total_data = load_pickle(filepath)
    qids = []
    print(len(total_data))
    for i in range(0, len(total_data)):
        qids.append(total_data[i][0][0])
    print(len(qids))
    result = Counter(qids)
    total_data_single = []
    total_data_multiple = []
    for i in range(0, len(total_data)):
        if (result[total_data[i][0][0]] == 1 ):
            total_data_single.append(total_data[i])
        else:
            total_data_multiple.append(total_data[i])
    print(len(total_data_single))


    with open(save_single_path, 'wb') as f:
        pickle.dump(total_data_single, f)
    with open(save_mutiple_path, 'wb') as f:
        pickle.dump(total_data_multiple, f)


#把单候选只保留其qid
def single_u2label(path1,path2):
    total_data = load_pickle(path1)
    labels=[]

    for i in range(0,len(total_data)):
        labels.append([total_data[i][0],1])

    total_data_sort = sorted(labels, key=lambda x: (x[0], x[1]))
    f = open(path2, "w")
    f.write(str(total_data_sort))
    f.close()


if __name__ == "__main__":
    # #将staqc_python中的单候选和多候选分开
    # python_staqc_path = '../staqc_tolabel/data/code_solution_labeled_data/filter/python_how_to_do_it_by_classifier_multiple_iid_title&blocks.txt'
    # python_staqc_sigle =  '../staqc_tolabel/data/code_solution_labeled_data/filter/iidto_single/python_staqc_multiple_iid_title&blocks_to_single_qids.txt'
    # python_staqc_multiple = '../staqc_tolabel/data/code_solution_labeled_data/filter/iidto_mutiple/python_staqc_multiple_iid_title&blocks_to_mutiple_qids.txt'
    # staqc_process(python_staqc_path,python_staqc_sigle,python_staqc_multiple)
    #
    # #将staqc_sql中的单候选和多候选分开
    # staqc_sql_path = '../staqc_tolabel/data/code_solution_labeled_data/filter/sql_how_to_do_it_by_classifier_multiple_iid_title&blocks.txt'
    # staqc_sql_sigle = '../staqc_tolabel/data/code_solution_labeled_data/filter/iidto_single/sql_staqc_multiple_iid_title&blocks_to_single_qids.txt'
    # staqc_sql_multiple = '../staqc_tolabel/data/code_solution_labeled_data/filter/iidto_mutiple/sql_staqc_multiple_iid_title&blocks_to_mutiple_qids.txt'
    # staqc_process(staqc_sql_path, staqc_sql_sigle, staqc_sql_multiple)
    #
    # #将large_python中的单候选和多候选分开
    # python_codedb_path =    '../istaqc_crawl/corpus/python_codedb_qid2index_blocks_unlabeled.pickle'
    # python_codedb_single =  '../istaqc_crawl/corpus/iidto_single/python_codedb_qid2index_blocks_to_single_qids.pickle'
    # python_codedb_multiple = '../istaqc_crawl/corpus/iidto_mutiple/python_codedb_qid2index_blocks_to_multiple_qids.pickle'
    # large_process(python_codedb_path, python_codedb_single, python_codedb_multiple)
    #
    # # 将large_sql中的单候选和多候选分开
    # sql_codedb_path = '../istaqc_crawl/corpus/sql_codedb_qid2index_blocks_unlabeled.pickle'
    # sql_codedb_single = '../istaqc_crawl/corpus/iidto_single/sql_codedb_qid2index_blocks_to_single_qids.pickle'
    # sql_codedb_multiple = '../istaqc_crawl/corpus/iidto_mutiple/sql_codedb_qid2index_blocks_to_multiple_qids.pickle'
    # large_process(sql_codedb_path, sql_codedb_single, sql_codedb_multiple)
    #
    # sql_codedb_single_label   =  '../istaqc_crawl/corpus/sql_codedb_qid2index_blocks_to_single_qids_labeled.txt'
    # python_codedb_single_label = '../istaqc_crawl/corpus/python_codedb_qid2index_blocks_to_single_qids_labeled.txt'
    #
    # single_u2label(sql_codedb_single,sql_codedb_single_label)
    # single_u2label(python_codedb_single,python_codedb_single_label)


    #将large_python中的单候选和多候选分开
    python_corpus_path =    '../hqfso_tool/corpus/python_corpus_qid2index_blocks_unlabeled.pickle'
    python_corpus_single =  '../hqfso_tool/corpus/iidto_single/python_corpus_qid2index_blocks_to_single_qids.pickle'
    python_corpus_multiple = '../hqfso_tool/corpus/iidto_mutiple/python_corpus_qid2index_blocks_to_multiple_qids.pickle'
    large_process(python_corpus_path, python_corpus_single, python_corpus_multiple)

    # 将large_sql中的单候选和多候选分开
    sql_corpus_path = '../hqfso_tool/corpus/sql_corpus_qid2index_blocks_unlabeled.pickle'
    sql_corpus_single = '../hqfso_tool/corpus/iidto_single/sql_corpus_qid2index_blocks_to_single_qids.pickle'
    sql_corpus_multiple = '../hqfso_tool/corpus/iidto_mutiple/sql_corpus_qid2index_blocks_to_multiple_qids.pickle'
    large_process(sql_corpus_path, sql_corpus_single, sql_corpus_multiple)


    sql_corpus_single_label   =  '../hqfso_tool/corpus/sql_corpus_qid2index_blocks_to_single_qids_labeled.txt'
    python_corpus_single_label = '../hqfso_tool/corpus/python_corpus_qid2index_blocks_to_single_qids_labeled.txt'

    single_u2label(sql_corpus_single,sql_corpus_single_label)
    single_u2label(python_corpus_single,python_corpus_single_label)
