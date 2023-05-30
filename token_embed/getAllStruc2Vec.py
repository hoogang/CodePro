
import pickle
import logging
import joblib


import sys
sys.path.append("../")

# 解析结构
from crawl_process.parsers import python_structured
from crawl_process.parsers import javang_structured
from crawl_process.parsers import sqlang_structured
# 解析分词
from crawl_process.parsers import langtrans_withtag

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
#多进程
from multiprocessing import Pool as ThreadPool

import numpy as np
# FastText库  gensim 3.4.0
from gensim.models import FastText
from collections import Counter
from collections import defaultdict
# 数据归一化/标准化
from sklearn  import preprocessing


# 保证以CSV保存字符串
def filter_inva_char(line):
    # 去除横杠
    line = line.replace('|', ' ').replace('¦', ' ')
    # 去除\r
    line = line.replace('\r', ' ')
    return line

# 解析python数据
def multil_process_python_codeTag(data_list):
    data_list = [filter_inva_char(i) for i in data_list]
    result = [(langtrans_withtag.codetag_parse('python',line) if langtrans_withtag.codetag_parse('python',line)!=[] else ['-100']) if line!='-10000' else ['-100']  for line in data_list]
    return result

def multil_process_python_query(data_list):
    data_list = [filter_inva_char(i) for i in data_list]
    result = [python_structured.python_query_parse(line) if python_structured.python_query_parse(line)!=[] else ['-100'] for line in data_list]
    return result

def multil_process_python_code(data_list):
    data_list = [filter_inva_char(i) for i in data_list]
    result = [(python_structured.python_code_parse(line) if python_structured.python_code_parse(line)!=[] else ['-100']) if line!='-10000' else ['-100']  for line in data_list]
    return result

def multil_process_python_context(data_list):
    data_list = [filter_inva_char(i) for i in data_list]
    result=[(python_structured.python_context_parse(line) if python_structured.python_context_parse(line)!=[] else ['-100']) if line!='-10000' else ['-100']  for line in data_list]
    return result

def multil_process_python_queryTag(data_list):
    data_list = [filter_inva_char(i) for i in data_list]
    result = [(langtrans_withtag.querytag_parse('python',line) if langtrans_withtag.querytag_parse('python',line)!=[] else ['-100']) if line!='-10000' else ['-100']  for line in data_list]
    return result

# 解析sqlang数据
def multil_process_javang_codeTag(data_list):
    data_list = [filter_inva_char(i) for i in data_list]
    result = [(langtrans_withtag.codetag_parse('java',line) if langtrans_withtag.codetag_parse('java',line)!=[] else ['-100']) if line!='-10000' else ['-100']  for line in data_list]
    return result

def multil_process_javang_query(data_list):
    data_list = [filter_inva_char(i) for i in data_list]
    result = [javang_structured.javang_query_parse(line) if javang_structured.javang_query_parse(line)!=[] else ['-100'] for line in data_list]
    return result

def multil_process_javang_code(data_list):
    data_list = [filter_inva_char(i) for i in data_list]
    result = [(javang_structured.javang_code_parse(line)  if javang_structured.javang_code_parse(line)!=[] else ['-100']) if line!='-10000' else ['-100']  for line in data_list]
    return result

def multil_process_javang_context(data_list):
    data_list = [filter_inva_char(i) for i in data_list]
    result=[(javang_structured.javang_context_parse(line) if javang_structured.javang_context_parse(line)!=[] else ['-100']) if line!='-10000' else ['-100']  for line in data_list]
    return result

def multil_process_javang_queryTag(data_list):
    data_list = [filter_inva_char(i) for i in data_list]
    result = [(langtrans_withtag.querytag_parse('java',line) if langtrans_withtag.querytag_parse('java',line)!=[] else ['-100']) if line!='-10000' else ['-100']  for line in data_list]
    return result



# 解析sqlang数据
def multil_process_sqlang_codeTag(data_list):
    data_list = [filter_inva_char(i) for i in data_list]
    result = [(langtrans_withtag.codetag_parse('sql',line) if langtrans_withtag.codetag_parse('sql',line)!=[] else ['-100']) if line!='-10000' else ['-100']  for line in data_list]
    return result

def multil_process_sqlang_query(data_list):
    data_list = [filter_inva_char(i) for i in data_list]
    result = [sqlang_structured.sqlang_query_parse(line) if sqlang_structured.sqlang_query_parse(line)!=[] else ['-100'] for line in data_list]
    return result

def multil_process_sqlang_code(data_list):
    data_list = [filter_inva_char(i) for i in data_list]
    result = [(sqlang_structured.sqlang_code_parse(line)  if sqlang_structured.sqlang_code_parse(line)!=[] else ['-100']) if line!='-10000' else ['-100']  for line in data_list]
    return result

def multil_process_sqlang_context(data_list):
    data_list = [filter_inva_char(i) for i in data_list]
    result=[(sqlang_structured.sqlang_context_parse(line) if sqlang_structured.sqlang_context_parse(line)!=[] else ['-100']) if line!='-10000' else ['-100']  for line in data_list]
    return result


def parse_like_staqc_python_corpus(corpus_list,split_num):
    # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, [qcont] 查询上下文, [acont] 回复上下文, 块长度，标签]

    scont1_data = [i[1][0][0] for i in corpus_list]
    scont1_split_list = [scont1_data[i:i + split_num] for i in range(0, len(scont1_data), split_num)]
    pool = ThreadPool(10)
    scont1_list = pool.map(multil_process_python_context, scont1_split_list)
    pool.close()
    pool.join()
    scont1_cut = []
    for p in scont1_list:
        scont1_cut += p
    print('scont1条数：%d' % len(scont1_cut))

    scont2_data = [i[1][1][0] for i in corpus_list]
    scont2_split_list = [scont2_data[i:i + split_num] for i in range(0, len(scont2_data), split_num)]
    pool = ThreadPool(10)
    scont2_list = pool.map(multil_process_python_context, scont2_split_list)
    pool.close()
    pool.join()
    scont2_cut = []
    for p in scont2_list:
        scont2_cut += p
    print('scont2条数：%d' % len(scont2_cut))

    qcont_data = [i[4][0] for i in corpus_list]
    qcont_split_list = [qcont_data[i:i + split_num] for i in range(0, len(qcont_data), split_num)]
    pool = ThreadPool(10)
    qcont_list = pool.map(multil_process_python_codeTag, qcont_split_list)
    pool.close()
    pool.join()
    qcont_cut = []
    for p in qcont_list:
        qcont_cut += p
    print('qcont条数：%d' % len(qcont_cut))

    acont_data = [i[5][0] for i in corpus_list]
    acont_split_list = [qcont_data[i:i + split_num] for i in range(0, len(acont_data), split_num)]
    pool = ThreadPool(10)
    acont_list = pool.map(multil_process_python_codeTag, acont_split_list)
    pool.close()
    pool.join()
    acont_cut = []
    for p in acont_list:
        acont_cut += p
    print('acont条数：%d' % len(acont_cut))

    query_data = [i[3][0] for i in corpus_list]
    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    pool = ThreadPool(10)
    query_list = pool.map(multil_process_python_query, query_split_list)
    pool.close()
    pool.join()
    query_cut = []
    for p in query_list:
        query_cut += p
    print('query条数：%d' % len(query_cut))

    code_data = [i[2][0][0] for i in corpus_list]
    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
    pool = ThreadPool(10)
    code_list = pool.map(multil_process_python_code, code_split_list)
    pool.close()
    pool.join()
    code_cut = []
    for p in code_list:
        code_cut += p
    print('code条数：%d' % len(code_cut))

    return scont1_cut, scont2_cut, qcont_cut, acont_cut, query_cut, code_cut


def parse_like_staqc_javang_corpus(corpus_list,split_num):
    # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, [qcont] 查询上下文, [acont] 回复上下文, 块长度，标签]

    scont1_data =  [i[1][0][0] for i in corpus_list]
    scont1_split_list = [scont1_data[i:i + split_num] for i in range(0, len(scont1_data), split_num)]
    pool = ThreadPool(10)
    scont1_list = pool.map(multil_process_javang_context, scont1_split_list)
    pool.close()
    pool.join()
    scont1_cut = []
    for p in scont1_list:
        scont1_cut += p
    print('scont1条数：%d' % len(scont1_cut))

    scont2_data = [i[1][1][0] for i in corpus_list]
    scont2_split_list = [scont2_data[i:i + split_num] for i in range(0, len(scont2_data), split_num)]
    pool = ThreadPool(10)
    scont2_list = pool.map(multil_process_javang_context, scont2_split_list)
    pool.close()
    pool.join()
    scont2_cut = []
    for p in scont2_list:
        scont2_cut += p
    print('scont2条数：%d' % len(scont2_cut))

    qcont_data =  [i[4][0] for i in corpus_list]
    qcont_split_list = [qcont_data[i:i + split_num] for i in range(0, len(qcont_data), split_num)]
    pool = ThreadPool(10)
    qcont_list = pool.map(multil_process_javang_codeTag, qcont_split_list)
    pool.close()
    pool.join()
    qcont_cut = []
    for p in qcont_list:
        qcont_cut += p
    print('qcont条数：%d' % len(qcont_cut))

    acont_data =  [i[5][0] for i in corpus_list]
    acont_split_list = [qcont_data[i:i + split_num] for i in range(0, len(acont_data), split_num)]
    pool = ThreadPool(10)
    acont_list = pool.map(multil_process_javang_codeTag, acont_split_list)
    pool.close()
    pool.join()
    acont_cut = []
    for p in acont_list:
        acont_cut += p
    print('acont条数：%d' % len(acont_cut))

    query_data = [i[3][0] for i in corpus_list]
    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    pool = ThreadPool(10)
    query_list = pool.map(multil_process_javang_query, query_split_list)
    pool.close()
    pool.join()
    query_cut = []
    for p in query_list:
        query_cut += p
    print('query条数：%d' % len(query_cut))

    code_data = [i[2][0][0] for i in corpus_list]
    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
    pool = ThreadPool(10)
    code_list = pool.map(multil_process_javang_code, code_split_list)
    pool.close()
    pool.join()
    code_cut = []
    for p in code_list:
        code_cut += p
    print('code条数：%d' % len(code_cut))

    return scont1_cut,scont2_cut,qcont_cut,acont_cut,query_cut,code_cut


def parse_like_staqc_sqlang_corpus(corpus_list,split_num):
    # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, [qcont] 查询上下文, [acont] 回复上下文, 块长度，标签]

    scont1_data =  [i[1][0][0] for i in corpus_list]
    scont1_split_list = [scont1_data[i:i + split_num] for i in range(0, len(scont1_data), split_num)]
    pool = ThreadPool(10)
    scont1_list = pool.map(multil_process_sqlang_context, scont1_split_list)
    pool.close()
    pool.join()
    scont1_cut = []
    for p in scont1_list:
        scont1_cut += p
    print('scont1条数：%d' % len(scont1_cut))

    scont2_data = [i[1][1][0] for i in corpus_list]
    scont2_split_list = [scont2_data[i:i + split_num] for i in range(0, len(scont2_data), split_num)]
    pool = ThreadPool(10)
    scont2_list = pool.map(multil_process_sqlang_context, scont2_split_list)
    pool.close()
    pool.join()
    scont2_cut = []
    for p in scont2_list:
        scont2_cut += p
    print('scont2条数：%d' % len(scont2_cut))

    qcont_data =  [i[4][0] for i in corpus_list]
    qcont_split_list = [qcont_data[i:i + split_num] for i in range(0, len(qcont_data), split_num)]
    pool = ThreadPool(10)
    qcont_list = pool.map(multil_process_sqlang_codeTag, qcont_split_list)
    pool.close()
    pool.join()
    qcont_cut = []
    for p in qcont_list:
        qcont_cut += p
    print('qcont条数：%d' % len(qcont_cut))

    acont_data =  [i[5][0] for i in corpus_list]
    acont_split_list = [qcont_data[i:i + split_num] for i in range(0, len(acont_data), split_num)]
    pool = ThreadPool(10)
    acont_list = pool.map(multil_process_sqlang_codeTag, acont_split_list)
    pool.close()
    pool.join()
    acont_cut = []
    for p in acont_list:
        acont_cut += p
    print('acont条数：%d' % len(acont_cut))

    query_data = [i[3][0] for i in corpus_list]
    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    pool = ThreadPool(10)
    query_list = pool.map(multil_process_sqlang_query, query_split_list)
    pool.close()
    pool.join()
    query_cut = []
    for p in query_list:
        query_cut += p
    print('query条数：%d' % len(query_cut))

    code_data = [i[2][0][0] for i in corpus_list]
    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
    pool = ThreadPool(10)
    code_list = pool.map(multil_process_sqlang_code, code_split_list)
    pool.close()
    pool.join()
    code_cut = []
    for p in code_list:
        code_cut += p
    print('code条数：%d' % len(code_cut))

    return scont1_cut,scont2_cut,qcont_cut,acont_cut,query_cut,code_cut


def parse_like_codenn_python_corpus(corpus_list,split_num):
    # [(id, index), [[c]] 代码，[q] 查询, [qcont] 查询上下文, [acont] 回复上下文, 块长度，标签]

    qcont_data =  [i[3][0] for i in corpus_list]
    qcont_split_list = [qcont_data[i:i + split_num] for i in range(0, len(qcont_data), split_num)]
    pool = ThreadPool(10)
    qcont_list = pool.map(multil_process_python_codeTag, qcont_split_list)
    pool.close()
    pool.join()
    qcont_cut = []
    for p in qcont_list:
        qcont_cut += p
    print('qcont条数：%d' % len(qcont_cut))

    acont_data =  [i[4][0] for i in corpus_list]
    acont_split_list = [qcont_data[i:i + split_num] for i in range(0, len(acont_data), split_num)]
    pool = ThreadPool(10)
    acont_list = pool.map(multil_process_python_codeTag, acont_split_list)
    pool.close()
    pool.join()
    acont_cut = []
    for p in acont_list:
        acont_cut += p
    print('acont条数：%d' % len(acont_cut))

    query_data = [i[2][0] for i in corpus_list]
    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    pool = ThreadPool(10)
    query_list = pool.map(multil_process_python_query, query_split_list)
    pool.close()
    pool.join()
    query_cut = []
    for p in query_list:
        query_cut += p
    print('query条数：%d' % len(query_cut))

    code_data = [i[1][0][0] for i in corpus_list]
    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
    pool = ThreadPool(10)
    code_list = pool.map(multil_process_python_code, code_split_list)
    pool.close()
    pool.join()
    code_cut = []
    for p in code_list:
        code_cut += p
    print('code条数：%d' % len(code_cut))

    return qcont_cut,acont_cut,query_cut,code_cut


def parse_like_codenn_javang_corpus(corpus_list,split_num):
    # [(id, index), [[c]] 代码，[q] 查询, [qcont] 查询上下文, [acont] 回复上下文, 块长度，标签]

    qcont_data =  [i[3][0] for i in corpus_list]
    qcont_split_list = [qcont_data[i:i + split_num] for i in range(0, len(qcont_data), split_num)]
    pool = ThreadPool(10)
    qcont_list = pool.map(multil_process_javang_codeTag, qcont_split_list)
    pool.close()
    pool.join()
    qcont_cut = []
    for p in qcont_list:
        qcont_cut += p
    print('qcont条数：%d' % len(qcont_cut))

    acont_data =  [i[4][0] for i in corpus_list]
    acont_split_list = [qcont_data[i:i + split_num] for i in range(0, len(acont_data), split_num)]
    pool = ThreadPool(10)
    acont_list = pool.map(multil_process_javang_codeTag, acont_split_list)
    pool.close()
    pool.join()
    acont_cut = []
    for p in acont_list:
        acont_cut += p
    print('acont条数：%d' % len(acont_cut))

    query_data = [i[2][0] for i in corpus_list]
    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    pool = ThreadPool(10)
    query_list = pool.map(multil_process_javang_query, query_split_list)
    pool.close()
    pool.join()
    query_cut = []
    for p in query_list:
        query_cut += p
    print('query条数：%d' % len(query_cut))

    code_data = [i[1][0][0] for i in corpus_list]
    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
    pool = ThreadPool(10)
    code_list = pool.map(multil_process_javang_code, code_split_list)
    pool.close()
    pool.join()
    code_cut = []
    for p in code_list:
        code_cut += p
    print('code条数：%d' % len(code_cut))

    return qcont_cut,acont_cut,query_cut,code_cut


def parse_like_codenn_sqlang_corpus(corpus_list,split_num):
    # [(id, index), [[c]] 代码，[q] 查询, [qcont] 查询上下文, [acont] 回复上下文, 块长度，标签]

    qcont_data =  [i[3][0] for i in corpus_list]
    qcont_split_list = [qcont_data[i:i + split_num] for i in range(0, len(qcont_data), split_num)]
    pool = ThreadPool(10)
    qcont_list = pool.map(multil_process_sqlang_codeTag, qcont_split_list)
    pool.close()
    pool.join()
    qcont_cut = []
    for p in qcont_list:
        qcont_cut += p
    print('qcont条数：%d' % len(qcont_cut))

    acont_data =  [i[4][0] for i in corpus_list]
    acont_split_list = [qcont_data[i:i + split_num] for i in range(0, len(acont_data), split_num)]
    pool = ThreadPool(10)
    acont_list = pool.map(multil_process_sqlang_codeTag, acont_split_list)
    pool.close()
    pool.join()
    acont_cut = []
    for p in acont_list:
        acont_cut += p
    print('acont条数：%d' % len(acont_cut))

    query_data = [i[2][0] for i in corpus_list]
    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    pool = ThreadPool(10)
    query_list = pool.map(multil_process_sqlang_query, query_split_list)
    pool.close()
    pool.join()
    query_cut = []
    for p in query_list:
        query_cut += p
    print('query条数：%d' % len(query_cut))

    code_data = [i[1][0][0] for i in corpus_list]
    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
    pool = ThreadPool(10)
    code_list = pool.map(multil_process_sqlang_code, code_split_list)
    pool.close()
    pool.join()
    code_cut = []
    for p in code_list:
        code_cut += p
    print('code条数：%d' % len(code_cut))

    return qcont_cut,acont_cut,query_cut,code_cut

def parse_same_codesn_python_corpus(corpus_list,split_num):
    # [(id, index),[[c]] 代码，[q] 查询, [ccont] 回复上下文， 块标签]

    ccont_data = [i[3][0] for i in corpus_list]
    ccont_split_list = [ccont_data[i:i + split_num] for i in range(0, len(ccont_data), split_num)]
    pool = ThreadPool(10)
    ccont_list = pool.map(multil_process_python_queryTag, ccont_split_list)
    pool.close()
    pool.join()
    ccont_cut = []
    for p in ccont_list:
        ccont_cut += p
    print('ccont条数：%d' % len(ccont_cut))

    query_data = [i[2][0] for i in corpus_list]
    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    pool = ThreadPool(10)
    query_list = pool.map(multil_process_python_query, query_split_list)
    pool.close()
    pool.join()
    query_cut = []
    for p in query_list:
        query_cut += p
    print('query条数：%d' % len(query_cut))

    code_data = [i[1][0][0] for i in corpus_list]
    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
    pool = ThreadPool(10)
    code_list = pool.map(multil_process_python_code, code_split_list)
    pool.close()
    pool.join()
    code_cut = []
    for p in code_list:
        code_cut += p
    print('code条数：%d' % len(code_cut))

    return ccont_cut, query_cut, code_cut


def parse_same_codesn_javang_corpus(corpus_list,split_num):
    # [(id, index),[[c]] 代码，[q] 查询, [ccont] 回复上下文， 块标签]

    ccont_data = [i[3][0] for i in corpus_list]
    ccont_split_list = [ccont_data[i:i + split_num] for i in range(0, len(ccont_data), split_num)]
    pool = ThreadPool(10)
    ccont_list = pool.map(multil_process_javang_queryTag, ccont_split_list)
    pool.close()
    pool.join()
    ccont_cut = []
    for p in ccont_list:
        ccont_cut += p
    print('ccont条数：%d' % len(ccont_cut))

    query_data = [i[2][0] for i in corpus_list]
    query_split_list = [query_data[i:i + split_num] for i in range(0, len(query_data), split_num)]
    pool = ThreadPool(10)
    query_list = pool.map(multil_process_javang_query, query_split_list)
    pool.close()
    pool.join()
    query_cut = []
    for p in query_list:
        query_cut += p
    print('query条数：%d' % len(query_cut))

    code_data = [i[1][0][0] for i in corpus_list]
    code_split_list = [code_data[i:i + split_num] for i in range(0, len(code_data), split_num)]
    pool = ThreadPool(10)
    code_list = pool.map(multil_process_javang_code, code_split_list)
    pool.close()
    pool.join()
    code_cut = []
    for p in code_list:
        code_cut += p
    print('code条数：%d' % len(code_cut))

    return ccont_cut, query_cut, code_cut


def python_parse(split_num):
    # 读取语料数据集
    with open('./corpus/python_like_staqc_corpus_qid2index_blocks_unlabeled.pickle', "rb") as f:
        #  存储为列表 有序  1586155条
        like_staqc_corpus_lis = pickle.load(f)
        print('like staqc语料的长度：', len(like_staqc_corpus_lis))

    with open('./corpus/python_like_codenn_corpus_qid2index_blocks_labeled.pickle', "rb") as f:
        #  存储为列表 有序  #
        like_codenn_corpus_lis = pickle.load(f)
        print('like codenn语料的长度：', len(like_codenn_corpus_lis))

    with open('./corpus/python_same_codesn_corpus_qid2index_blocks_labeled.pickle', "rb") as f:
        #  存储为列表 有序
        same_codesn_corpus_lis = pickle.load(f)
        print('same codesn语料的长度：', len(like_codenn_corpus_lis))

    #3200885条
    print('python语言总计%s条语料数据'%(len(like_staqc_corpus_lis) + len(like_codenn_corpus_lis) + len(same_codesn_corpus_lis)))

    parse_scont1, parse_scont2, parse_qcont, parse_acont, parse_query, parse_code = \
        parse_like_staqc_python_corpus(like_staqc_corpus_lis, split_num)

    # 语料重组
    like_staqc_corpora1 = [' '.join([' '.join(i), ' '.join(j), ' '.join(a), ' '.join(b), ' '.join(x), ' '.join(y)]) for
                           i, j, a, b, x, y in
                           zip(parse_scont1, parse_scont2, parse_qcont, parse_acont, parse_query, parse_code)]

    # 语料重组
    like_staqc_corpora2 = parse_scont1 + parse_scont2 + parse_qcont + parse_acont + parse_query + parse_code
    # 去除-100
    like_staqc_corpora2 = list(filter(lambda x: x != ['-100'], like_staqc_corpora2))
    print('like_staqc的python语料解析完毕！')

    parse_qcont, parse_acont, parse_query, parse_code = parse_like_codenn_python_corpus(like_codenn_corpus_lis,
                                                                                        split_num)
    # 语料重组
    like_codenn_corpora1 = [' '.join([' '.join(i), ' '.join(j), ' '.join(a), ' '.join(b)]) for i, j, a, b in
                            zip(parse_qcont, parse_acont, parse_query, parse_code)]

    # 语料重组
    like_codenn_corpora2 = parse_qcont + parse_acont + parse_query + parse_code
    # 去除-100
    like_codenn_corpora2 = list(filter(lambda x: x != ['-100'], like_codenn_corpora2))
    print('like_codenn的python语料解析完毕！')

    parse_ccont, parse_query, parse_code = parse_same_codesn_python_corpus(same_codesn_corpus_lis, split_num)

    # 语料重组
    same_codesn_corpora1 = [' '.join([' '.join(i), ' '.join(j), ' '.join(k)]) for i, j, k in
                            zip(parse_ccont, parse_query, parse_code)]

    # 语料重组
    same_codesn_corpora2 = parse_ccont + parse_query + parse_code
    # 去除-100
    same_codesn_corpora2 = list(filter(lambda x: x != ['-100'], same_codesn_corpora2))
    print('same_codesn的python语料解析完毕！')

    corpora_pairs = like_staqc_corpora1 + like_codenn_corpora1 + same_codesn_corpora1
    # 提取context的software-words (去除-100)
    corpora_pairs = [[i for i in context.split(' ') if i != '-100'] for context in corpora_pairs]

    corpora_toklis = like_staqc_corpora2 + like_codenn_corpora2 + same_codesn_corpora2

    with open('./parsed/python_corpora_pairs.pickle', "wb") as f:
        pickle.dump(corpora_pairs, f)

    with open('./parsed/python_corpora_toklis.pickle', "wb") as f:
        pickle.dump(corpora_toklis, f)


def javang_parse(split_num):
    # 读取语料数据集
    with open('./corpus/java_like_staqc_corpus_qid2index_blocks_unlabeled.pickle', "rb") as f:
        #  存储为列表 有序  #
        like_staqc_corpus_lis = pickle.load(f)
        print('like staqc语料的长度：',len(like_staqc_corpus_lis))

    with open('./corpus/java_like_codenn_corpus_qid2index_blocks_labeled.pickle', "rb") as f:
        #  存储为列表 有序
        like_codenn_corpus_lis = pickle.load(f)
        print('like codenn语料的长度：', len(like_codenn_corpus_lis))

    with open('./corpus/java_same_codesn_corpus_qid2index_blocks_labeled.pickle', "rb") as f:
        #  存储为列表 有序
        same_codesn_corpus_lis = pickle.load(f)
        print('same codesn语料的长度：', len(like_codenn_corpus_lis))

    print('java语言总计%s条语料数据'%(len(like_staqc_corpus_lis)+len(like_codenn_corpus_lis)+len(same_codesn_corpus_lis)))

    parse_scont1, parse_scont2, parse_qcont, parse_acont, parse_query, parse_code = \
        parse_like_staqc_javang_corpus(like_staqc_corpus_lis, split_num)

    # 语料重组
    like_staqc_corpora1 = [' '.join([' '.join(i), ' '.join(j), ' '.join(a), ' '.join(b), ' '.join(x), ' '.join(y)]) for
                          i, j, a, b, x, y in
                          zip(parse_scont1, parse_scont2, parse_qcont, parse_acont, parse_query, parse_code)]

    # 语料重组
    like_staqc_corpora2 = parse_scont1 + parse_scont2 + parse_qcont + parse_acont + parse_query + parse_code
    # 去除-100
    like_staqc_corpora2 = list(filter(lambda x: x != ['-100'], like_staqc_corpora2))
    print('like_staqc的java语料解析完毕！')

    parse_qcont, parse_acont, parse_query, parse_code = parse_like_codenn_javang_corpus(like_codenn_corpus_lis, split_num)

    # 语料重组
    like_codenn_corpora1 = [' '.join([' '.join(i), ' '.join(j), ' '.join(a), ' '.join(b)]) for i, j, a, b in
                       zip(parse_qcont, parse_acont, parse_query, parse_code)]

    # 语料重组
    like_codenn_corpora2 = parse_qcont + parse_acont + parse_query + parse_code
    # 去除-100
    like_codenn_corpora2 = list(filter(lambda x: x != ['-100'], like_codenn_corpora2))
    print('like_codenn的java语料解析完毕！')

    parse_ccont, parse_query, parse_code = parse_same_codesn_javang_corpus(same_codesn_corpus_lis, split_num)

    # 语料重组
    same_codesn_corpora1  = [' '.join([' '.join(i), ' '.join(j), ' '.join(k)]) for i, j, k in zip(parse_ccont, parse_query, parse_code)]

    # 语料重组
    same_codesn_corpora2 = parse_ccont + parse_query + parse_code
    # 去除-100
    same_codesn_corpora2 = list(filter(lambda x: x != ['-100'], same_codesn_corpora2))
    print('same_codesn的java语料解析完毕！')

    corpora_pairs = like_staqc_corpora1 + like_codenn_corpora1+ same_codesn_corpora1
    # 提取context的software-words (去除-100)
    corpora_pairs = [[i for i in context.split(' ') if i != '-100'] for context in corpora_pairs]

    corpora_toklis = like_staqc_corpora2 + like_codenn_corpora2 + same_codesn_corpora2

    with open('./parsed/java_corpora_pairs.pickle', "wb") as f:
        pickle.dump(corpora_pairs, f)

    with open('./parsed/java_corpora_toklis.pickle', "wb") as f:
        pickle.dump(corpora_toklis, f)


def sqlang_parse(split_num):
    # 读取语料数据集
    with open('./corpus/sql_like_staqc_corpus_qid2index_blocks_unlabeled.pickle', "rb") as f:
        #  存储为列表 有序  1062007条
        like_staqc_corpus_lis = pickle.load(f)
        print('like staqc语料的长度：', len(like_staqc_corpus_lis))

    with open('./corpus/sql_like_codenn_corpus_qid2index_blocks_labeled.pickle', "rb") as f:
        #  存储为列表 有序
        like_codenn_corpus_lis = pickle.load(f)
        print('like codenn语料的长度：', len(like_codenn_corpus_lis))

    # 1678404
    print('sql语言总计%s条语料数据'%(len(like_staqc_corpus_lis) + len(like_codenn_corpus_lis)))

    parse_scont1, parse_scont2, parse_qcont, parse_acont, parse_query, parse_code = \
        parse_like_staqc_sqlang_corpus(like_staqc_corpus_lis, split_num)

    # 语料重组
    like_staqc_corpora1 = [' '.join([' '.join(i), ' '.join(j), ' '.join(a), ' '.join(b), ' '.join(x), ' '.join(y)]) for
                           i, j, a, b, x, y in
                           zip(parse_scont1, parse_scont2, parse_qcont, parse_acont, parse_query, parse_code)]

    # 语料重组
    like_staqc_corpora2 = parse_scont1 + parse_scont2 + parse_qcont + parse_acont + parse_query + parse_code
    # 去除-100
    like_staqc_corpora2 = list(filter(lambda x: x != ['-100'], like_staqc_corpora2))
    print('like_staqc的sql语料解析完毕！')

    parse_qcont, parse_acont, parse_query, parse_code = parse_like_codenn_sqlang_corpus(like_codenn_corpus_lis,
                                                                                        split_num)
    # 语料重组
    like_codenn_corpora1 = [' '.join([' '.join(i), ' '.join(j), ' '.join(a), ' '.join(b)]) for i, j, a, b in
                            zip(parse_qcont, parse_acont, parse_query, parse_code)]

    # 语料重组
    like_codenn_corpora2 = parse_qcont + parse_acont + parse_query + parse_code
    # 去除-100
    like_codenn_corpora2 = list(filter(lambda x: x != ['-100'], like_codenn_corpora2))
    print('like_codenn的sql语料解析完毕！')

    corpora_pairs = like_staqc_corpora1 + like_codenn_corpora1
    # 提取context的software-words (去除-100)
    corpora_pairs = [[i for i in context.split(' ') if i != '-100'] for context in corpora_pairs]

    corpora_toklis = like_staqc_corpora2 + like_codenn_corpora2

    with open('./parsed/sql_corpora_pairs.pickle', "wb") as f:
        pickle.dump(corpora_pairs, f)

    with open('./parsed/sql_corpora_toklis.pickle', "wb") as f:
        pickle.dump(corpora_toklis, f)


def main(lang_type):

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    corpora_pairs = joblib.load(open('./parsed/%s_corpora_pairs.pickle'%lang_type, "rb"))

    # 训练tokdif2vec词向量
    # 计数[Counter({'我': 1, '正在': 1, '学习': 1, '计算机': 1}), Counter({'它': 1, '正在': 1, '吃饭': 1})]
    qadoc_tf = [Counter(doc) for doc in corpora_pairs]
    # 对于不存在的词idf[null]=0 构建字典，不存在为0,dict1[‘a’] 会有默认类型int 初始值为0，
    word_idf = defaultdict(int)
    for doc in qadoc_tf:
        for word in doc:
            word_idf[word] += 1
    # word_idf defaultdict(<class 'int'>, {'我': 2, '正在': 2, '学习':1})
    for word in word_idf:  # 文档总数/包含词的文档总数+1
        word_idf[word] = np.log(len(qadoc_tf) / (word_idf[word] + 1))
    #############################对词的TF值进行归一化处理################################################
    #  取出词的IDF值  变成 x*1 格式 [[],[],[]]^T  其属性（按列进行)
    idf_values = np.array([*word_idf.values()]).reshape(-1, 1)
    #  将IDF值标准化缩放)  StandardScaler 标准化  MinMaxScaler 归一化
    data_scaler = preprocessing.MinMaxScaler()
    #  [[1,2,3]]
    norm_values = data_scaler.fit_transform(idf_values).reshape(1, -1)[0]
    #############################对词的TF值进行归一化处理################################################
    #  (word，tf值）词和TF值写入txt
    tokidf2vec = ''.join([' '.join([i, str(j)]) + '\n' for i, j in zip(word_idf.keys(), norm_values)])

    # 语料保存txt
    with open('../use_embeddings/%s_idfvec.txt' % lang_type, 'w') as f:
        f.write(tokidf2vec)
    print('%s语料的词tokidf2vec词向量保存完毕！' % lang_type)

    corpora_toklis = joblib.load(open('./parsed/%s_corpora_toklis.pickle' % lang_type, "rb"))

    # 训练stru2vec词向量
    # --------------------------训练stru2vec词向量-------------------------------#
    model = FastText(corpora_toklis, size=300, min_count=0, window=10, iter=20, workers=5)

    # --------------------------训练code2vec词向量-------------------------------#
    model.save("../use_embeddings/%s_struc2vec.model" % lang_type)
    print('%s语料的词向量模型训练完毕并保存！'% lang_type)

    # 保存code2vec词典
    word_list = model.wv.index2word
    print('%s的语料中词典大小为%d' % (lang_type, len(word_list)))

    # 保存在vocab文件里
    with open( '../use_embeddings/%s_vocab.txt' % lang_type, 'w') as f:
        # 写入每一个词
        for word in word_list:
            f.write(word + '\n')

    # 保存词向量 code2vec
    model.wv.save_word2vec_format('../use_embeddings/%s_struc2vec.txt' % lang_type, binary=False)
    # linux命令
    eping_cmd = 'sed \'1d\' ../use_embeddings/%s_struc2vec.txt >  ../use_embeddings/%s_glove.txt' % (lang_type, lang_type)
    # 转化为glove词向量
    os.popen(eping_cmd).read()

    # linux命令
    eping_cmd ='7za a ../use_embeddings/%s_struc2vec.7z  ../use_embeddings/%s_struc2vec.txt' % (lang_type, lang_type)
    # 压缩词向量
    os.popen(eping_cmd).read()

    print('%s语料的struc2vec词向量保存完毕！'%lang_type)




#python
'''
like staqc语料的长度： 1595943
like codenn语料的长度： 1163094
same codesn语料的长度： 1163094
python语言总计3210673条语料数据
scont1条数：1595943
scont2条数：1595943
qcont条数：1595943
acont条数：1595943
query条数：1595943
code条数：1595943
like_staqc的python语料解析完毕！
qcont条数：1163094
acont条数：1163094
query条数：1163094
code条数：1163094
like_codenn的python语料解析完毕！
ccont条数：451636
query条数：451636
code条数：451636
same_codesn的python语料解析完毕！

python语料的词向量模型训练完毕并保存！
python的语料中词典大小为2450144
python语料的struc2vec词向量保存完毕！

'''


#java
'''
java语言总计4706911条语料数据
scont1条数：2695538
scont2条数：2695538
qcont条数：2695538
acont条数：2695538
query条数：2695538
code条数：2695538
like_staqc的java语料解析完毕！
qcont条数：1530644
acont条数：1530644
query条数：1530644
code条数：1530644
like_codenn的java语料解析完毕！
ccont条数：480729
query条数：480729
code条数：480729
same_codesn的java语料解析完毕！

java语料的词向量模型训练完毕并保存！
java的语料中词典大小为3745838
java语料的struc2vec词向量保存完毕！

'''


#sql
'''
like staqc语料的长度： 1074407
like codenn语料的长度： 616397
sql语言总计1690804条语料数据
scont1条数：1074407
scont2条数：1074407
qcont条数：1074407
acont条数：1074407
query条数：1074407
code条数：1074407
like_staqc的sql语料解析完毕！
qcont条数：616397
acont条数：616397
query条数：616397
code条数：616397
like_codenn的sql语料解析完毕！

sql语料的词向量模型训练完毕并保存！
sql的语料中词典大小为688383
sql语料的struc2vec词向量保存完毕！

'''



#参数配置
python_type= 'python'
java_type ='java'
sql_type ='sql'

#分割参数
split_num = 10000


if __name__ == '__main__':
    python_parse(split_num)
    javang_parse(split_num)
    sqlang_parse(split_num)
    main(python_type)
    main(java_type)
    main(sql_type)
