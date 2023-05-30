
'''
并行分词
'''

import pickle

import sys
sys.path.append("../crawl_process/")

# 解析结构
from parsers.python_structured import *
from parsers.sqlang_structured import *

# 多进程
from multiprocessing import Pool as ThreadPool

#python解析
def multipro_python_query(data_list):
    result=[python_query_parse(line) for line in data_list]
    return result

def multipro_python_code(data_list):
    result = [python_code_parse(line) for line in data_list]
    return result

def multipro_python_context(data_list):
    result = []
    for line in data_list:
        if (line == '-10000'):
            result.append(['-10000'])
        else:
            result.append(python_context_parse(line))
    return result


#sql解析
def multipro_sqlang_query(data_list):
    result=[sqlang_query_parse(line) for line in data_list]
    return result

def multipro_sqlang_code(data_list):
    result = [sqlang_code_parse(line) for line in data_list]
    return result

def multipro_sqlang_context(data_list):
    result = []
    for line in data_list:
        if (line == '-10000'):
            result.append(['-10000'])
        else:
            result.append(sqlang_context_parse(line))
    return result


def parse_python(python_list,split_num):

    acont1_data =  [i[1][0][0] for i in python_list]

    acont1_split_list = [acont1_data[i:i + split_num] for i in range(0, len(acont1_data), split_num)]
    pool = ThreadPool(10)
    acont1_list = pool.map(multipro_python_context, acont1_split_list)
    pool.close()
    pool.join()
    acont1_cut = []
    for p in acont1_list:
        acont1_cut += p
    print('acont1条数：%d' % len(acont1_cut))

    acont2_data = [i[1][1][0] for i in python_list]

    acont2_split_list = [acont2_data[i:i + split_num] for i in range(0, len(acont2_data), split_num)]
    pool = ThreadPool(10)
    acont2_list = pool.map(multipro_python_context, acont2_split_list)
    pool.close()
    pool.join()
    acont2_cut = []
    for p in acont2_list:
        acont2_cut += p
    print('acont2条数：%d' % len(acont2_cut))


    query_data = [i[3][0] for i in python_list]

    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    pool = ThreadPool(10)
    query_list = pool.map(multipro_python_query, query_split_list)
    pool.close()
    pool.join()
    query_cut = []
    for p in query_list:
        query_cut += p
    print('query条数：%d' % len(query_cut))

    code_data = [i[2][0][0] for i in python_list]

    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
    pool = ThreadPool(10)
    code_list = pool.map(multipro_python_code, code_split_list)
    pool.close()
    pool.join()
    code_cut = []
    for p in code_list:
        code_cut += p
    print('code条数：%d' % len(code_cut))

    qids = [i[0] for i in python_list]
    print(qids[0])
    print(len(qids))

    return acont1_cut,acont2_cut,query_cut,code_cut,qids


def parse_sqlang(sqlang_list,split_num):

    acont1_data =  [i[1][0][0] for i in sqlang_list]

    acont1_split_list = [acont1_data[i:i + split_num] for i in range(0, len(acont1_data), split_num)]
    pool = ThreadPool(10)
    acont1_list = pool.map(multipro_sqlang_context, acont1_split_list)
    pool.close()
    pool.join()
    acont1_cut = []
    for p in acont1_list:
        acont1_cut += p
    print('acont1条数：%d' % len(acont1_cut))

    acont2_data = [i[1][1][0] for i in sqlang_list]

    acont2_split_list = [acont2_data[i:i + split_num] for i in range(0, len(acont2_data), split_num)]
    pool = ThreadPool(10)
    acont2_list = pool.map(multipro_sqlang_context, acont2_split_list)
    pool.close()
    pool.join()
    acont2_cut = []
    for p in acont2_list:
        acont2_cut += p
    print('acont2条数：%d' % len(acont2_cut))

    query_data = [i[3][0] for i in sqlang_list]

    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    pool = ThreadPool(10)
    query_list = pool.map(multipro_sqlang_query, query_split_list)
    pool.close()
    pool.join()
    query_cut = []
    for p in query_list:
        query_cut += p
    print('query条数：%d' % len(query_cut))

    code_data = [i[2][0][0] for i in sqlang_list]

    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
    pool = ThreadPool(10)
    code_list = pool.map(multipro_sqlang_code, code_split_list)
    pool.close()
    pool.join()
    code_cut = []
    for p in code_list:
        code_cut += p
    print('code条数：%d' % len(code_cut))
    qids = [i[0] for i in sqlang_list]

    return acont1_cut ,acont2_cut,query_cut,code_cut,qids


def main_totokens(lang_type,split_num,source_path,save_path):
    total_data = []
    with open(source_path, "rb") as f:
        #  存储为字典 有序
        corpus_lis = eval(f.read())  #txt

        # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度]

        if lang_type=='python':

            parse_acont1, parse_acont2,parse_query, parse_code,qids  = parse_python(corpus_lis,split_num)
            for i in range(0,len(qids)):
                total_data.append([qids[i],[parse_acont1[i],parse_acont2[i]],[parse_code[i]],parse_query[i]])

        if lang_type == 'sql':

            parse_acont1,parse_acont2,parse_query, parse_code,qids = parse_sqlang(corpus_lis, split_num)
            for i in range(0,len(qids)):
                total_data.append([qids[i],[parse_acont1[i],parse_acont2[i]],[parse_code[i]],parse_query[i]])


    f = open(save_path, "w")
    f.write(str(total_data))
    f.close()


def large_totokens(lang_type,split_num,source_path,save_path):
    total_data = []
    with open(source_path, "rb") as f:
        #  存储为字典 有序
        corpus_lis = pickle.load(f) #pickle

        # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, 块长度]

        print(corpus_lis[0])

        if lang_type=='python':

            parse_acont1, parse_acont2,parse_query, parse_code,qids  = parse_python(corpus_lis,split_num)
            for i in range(0,len(qids)):
                total_data.append([qids[i],[parse_acont1[i],parse_acont2[i]],[parse_code[i]],parse_query[i]])

        if lang_type == 'sql':

            parse_acont1,parse_acont2,parse_query, parse_code,qids = parse_sqlang(corpus_lis, split_num)
            for i in range(0,len(qids)):
                total_data.append([qids[i],[parse_acont1[i],parse_acont2[i]],[parse_code[i]],parse_query[i]])


    f = open(save_path, "w")
    f.write(str(total_data))
    f.close()


python_type= 'python'
sqlang_type ='sql'

words_top = 100
split_num = 1000

def test(path1,path2):
    with open(path1, "rb") as f:
        #  存储为字典 有序
        corpus_lis1  = pickle.load(f) #pickle
    with open(path2, "rb") as f:
        corpus_lis2 = eval(f.read()) #txt

    print(corpus_lis1[10])
    print(corpus_lis2[10])


if __name__ == '__main__':

    # python_hnn_path = '../data_hnn/python_hnn_qid2index_blocks_labeled.txt'
    # python_hnn_tokens = '../data_hnn/python_hnn_qid2index_blocks_labeled_tokennized.txt'
    # #main_tokens(python_type, split_num, python_hnn_path, python_hnn_tokens)
    #
    # sql_hnn_path =   '../data_hnn/sql_hnn_qid2index_blocks_labeled.txt'
    # sql_hnn_tokens = '../data_hnn/sql_hnn_qid2index_blocks_labeled_tokennized.txt'
    # #main_tokens(sqlang_type,split_num,sql_hnn_path,sql_hnn_tokens)
    #
    # staqc_python_path = '../staqc_tolabel/data/code_solution_labeled_data/filter/python_how_to_do_it_by_classifier_multiple_iid_title&blocks.txt'
    # staqc_python_tokens = '../staqc_tolabel/data/code_solution_labeled_data/filter/tokennized_data/python_staqc_multiple_iid_title&blocks_to_mutiple_qids_tokennized.txt'
    # #main_totokens(python_type, split_num, staqc_python_path, staqc_python_tokens)
    #
    # sql_staqc_path =  '../staqc_tolabel/data/code_solution_labeled_data/filter/sql_how_to_do_it_by_classifier_multiple_iid_title&blocks.txt'
    # sql_staqc_tokens =  '../staqc_tolabel/data/code_solution_labeled_data/filter/tokennized_data/sql_staqc_multiple_iid_title&blocks_to_mutiple_qids_tokennized.txt'
    # #main_totokens(sqlang_type,split_num,sql_staqc_path,sql_staqc_tokens)
    #
    # python_codedb_path= '../istaqc_crawl/corpus/iidto_mutiple/python_codedb_qid2index_blocks_to_multiple_qids.pickle'
    # python_codedb_tokens='../istaqc_crawl/corpus/tokennized_data/python_codedb_qid2index_blocks_to_multiple_qids_tokennized.txt'
    # large_totokens(python_type, split_num, python_codedb_path, python_codedb_tokens)
    #
    # sql_codedb_path= '../istaqc_crawl/corpus/iidto_mutiple/sql_codedb_qid2index_blocks_to_multiple_qids.pickle'
    # sql_codedb_tokens='../istaqc_crawl/corpus/tokennized_data/sql_codedb_qid2index_blocks_to_multiple_qids_tokennized.txt'
    # large_totokens(sqlang_type, split_num, sql_codedb_path, sql_codedb_tokens)


    python_corpus_path= '../hqfso_tool/corpus/iidto_mutiple/python_corpus_qid2index_blocks_to_multiple_qids.pickle'
    python_corpus_tokens='../hqfso_tool/corpus/tokennized_data/python_corpus_qid2index_blocks_to_multiple_qids_tokennized.txt'
    large_totokens(python_type, split_num, python_corpus_path, python_corpus_tokens)

    sql_corpus_path= '../hqfso_tool/corpus/iidto_mutiple/sql_corpus_qid2index_blocks_to_multiple_qids.pickle'
    sql_corpus_tokens='../hqfso_tool/corpus/tokennized_data/sql_corpus_qid2index_blocks_to_multiple_qids_tokennized.txt'
    large_totokens(sqlang_type, split_num, sql_corpus_path, sql_corpus_tokens)
