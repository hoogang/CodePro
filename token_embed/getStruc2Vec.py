
import os
import pickle
import logging

import sys
sys.path.append("../")

# 解析结构
from crawl_process.parsers import python_structured
from crawl_process.parsers import sqlang_structured

#FastText库  gensim 3.4.0
from gensim.models import FastText

# 多进程
from multiprocessing import Pool as ThreadPool


#python解析
def multipro_python_query(data_list):
    result=[python_structured.python_query_parse(line) for line in data_list]
    return result

def multipro_python_code(data_list):
    result=[python_structured.python_code_parse(line) if python_structured.python_code_parse(line)!=[] else ['-100'] for line in data_list]
    return result

def multipro_python_context(data_list):
    result=[(python_structured.python_context_parse(line)  if python_structured.python_context_parse(line)!=[] else ['-100']) if line!='-10000' else ['-100']  for line in data_list]
    return result

#sql解析
def multipro_sqlang_query(data_list):
    result=[sqlang_structured.sqlang_query_parse(line) for line in data_list]
    return result

def multipro_sqlang_code(data_list):
    result=[sqlang_structured.sqlang_code_parse(line) if sqlang_structured.sqlang_code_parse(line)!=[] else ['-100'] for line in data_list]
    return result

def multipro_sqlang_context(data_list):
    result=[(sqlang_structured.sqlang_context_parse(line) if sqlang_structured.sqlang_context_parse(line)!=[] else ['-100']) if line!='-10000' else ['-100']  for line in data_list]
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

    qcont_data = [i[4][0] for i in python_list]

    qcont_split_list = [qcont_data[i:i + split_num] for i in range(0, len(qcont_data), split_num)]
    pool = ThreadPool(10)
    qcont_list = pool.map(multipro_python_context, qcont_split_list)
    pool.close()
    pool.join()
    qcont_cut = []
    for p in qcont_list:
        qcont_cut += p
    print('qcont条数：%d' % len(qcont_cut))

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

    return acont1_cut,acont2_cut,qcont_cut,query_cut,code_cut


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

    qcont_data = [i[4][0] for i in sqlang_list]

    qcont_split_list = [qcont_data[i:i + split_num] for i in range(0, len(qcont_data), split_num)]
    pool = ThreadPool(10)
    qcont_list = pool.map(multipro_sqlang_context, qcont_split_list)
    pool.close()
    pool.join()
    qcont_cut = []
    for p in qcont_list:
        qcont_cut += p
    print('qcont条数：%d' % len(qcont_cut))

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

    return acont1_cut,acont2_cut,qcont_cut,query_cut,code_cut



def main(lang_type,split_num):
    # 再次判定staqc和hnn中语料来自相同xml
    staqc_qidsfind_inxml = pickle.load(open('../label_tool/data/code_solution_labeled_data/filter/%s_staqc_qidsfind_inxml.pickle' % lang_type, 'rb'))

    hnn_qidsfind_inxml = pickle.load(open('../data_hnn/%s_hnn_qidsfind_inxml.pickle' % lang_type, 'rb'))

    # 不重复一个集合是否包含另一个集合
    judge_cover = [set(hnn_qidsfind_inxml[i]).issubset(set(staqc_qidsfind_inxml[i])) for i in [2017, 2018]]
    # 打印True 覆盖
    print(judge_cover)

    all_staqc_qids= set(staqc_qidsfind_inxml[2017]+staqc_qidsfind_inxml[2018])

    with open('./corpus/%s_like_staqc_corpus_qid2index_blocks_unlabeled.pickle' % lang_type, "rb") as f:
        #  存储为字典 有序
        corpus_lis  = pickle.load(f)
        corpus_qids = set([i[0][0] for i in corpus_lis])
        # 判定corpus是否包含staqc待测语料
        judge_cover = [all_staqc_qids.issubset(corpus_qids)]
        # #判断集合中所有元素是否包含在指定集合；是True，否False
        print(judge_cover)

        # [(id, index),[[si]，[si+1]] 文本块，[[c]] 代码，[q] 查询, [qcont] 查询上下文, 块长度，标签]

        if lang_type=='python':

            parse_acont1, parse_acont2, parse_qcont,parse_query, parse_code  = parse_python(corpus_lis,split_num)
            print('%s语料解析完毕！' % lang_type)

            # 语料重组
            corpora = parse_acont1 + parse_acont2+ parse_qcont + parse_query + parse_code
            # 去除-100
            corpora = list(filter(lambda x: x!=['-100'], corpora))
            print('%s语料加载完毕开始训练过程！......' % lang_type)

            # 训练stru2vec词向量
            # --------------------------训练stru2vec词向量-------------------------------#
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

            model = FastText(corpora, size=300, min_count=0, window=10, iter=20, workers=5)

            # --------------------------训练code2vec词向量-------------------------------#
            model.save("../crawl_embeddings/%s_struc2vec.model" % lang_type)
            print('%s语料的词向量模型训练完毕并保存！'% lang_type)

            # 保存code2vec词典
            word_list = model.wv.index2word
            print('%s的语料中词典大小为%d' % (lang_type, len(word_list)))

            # 保存在vocab文件里
            with open( '../crawl_embeddings/%s_vocab.txt' % lang_type, 'w') as f:
                # 写入每一个词
                for word in word_list:
                    f.write(word + '\n')

            # 保存词向量 code2vec
            model.wv.save_word2vec_format('../crawl_embeddings/%s_struc2vec.txt' % lang_type, binary=False)
            # linux命令
            eping_cmd = 'sed \'1d\' ../crawl_embeddings/%s_struc2vec.txt >  ../crawl_embeddings/%s_glove.txt' % (lang_type, lang_type)
            # 转化为glove词向量
            os.popen(eping_cmd).read()

            # linux命令
            eping_cmd ='7z a ../crawl_embeddings/%s_struc2vec.7z   ../crawl_embeddings/%s_struc2vec.txt' % (lang_type, lang_type)
            # 压缩词向量
            os.popen(eping_cmd).read()

            print('%s语料的词分布向量文件保存完毕！'%lang_type)

        if lang_type == 'sql':

            parse_acont1, parse_acont2, parse_qcont, parse_query, parse_code = parse_sqlang(corpus_lis, split_num)
            print('%s语料解析完毕！' % lang_type)

            # 语料重组
            corpora = parse_acont1 + parse_acont2+ parse_qcont + parse_query + parse_code
            # 去除-100
            corpora = list(filter(lambda x: x!=['-100'], corpora))
            print('%s语料加载完毕开始训练过程！......' % lang_type)

            # 训练stru2vec词向量
            # --------------------------训练stru2vec词向量-------------------------------#
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

            model = FastText(corpora, size=300, min_count=0, window=10, iter=20, workers=5)

            # --------------------------训练code2vec词向量-------------------------------#
            model.save("../crawl_embeddings/%s_struc2vec.model" % lang_type)
            print('%s语料的词向量模型训练完毕并保存！'% lang_type)

            # 保存code2vec词典
            word_list = model.wv.index2word
            print('%s的语料中词典大小为%d' % (lang_type, len(word_list)))

            # 保存在vocab文件里
            with open( '../crawl_embeddings/%s_vocab.txt' % lang_type, 'w') as f:
                # 写入每一个词
                for word in word_list:
                    f.write(word + '\n')

            # 保存词向量 code2vec
            model.wv.save_word2vec_format('../crawl_embeddings/%s_struc2vec.txt' % lang_type, binary=False)
            # linux命令
            eping_cmd = 'sed \'1d\' ../crawl_embeddings/%s_struc2vec.txt >  ../crawl_embeddings/%s_glove.txt' % (lang_type, lang_type)
            # 转化为glove词向量
            os.popen(eping_cmd).read()

            # linux命令
            eping_cmd = '7z a ../crawl_embeddings/%s_struc2vec.7z   ../crawl_embeddings/%s_struc2vec.txt' % (lang_type, lang_type)
            # 压缩词向量
            os.popen(eping_cmd).read()



sql_type ='sql'
python_type= 'python'

split_num = 10000

if __name__ == '__main__':
    main(python_type,split_num,)
    main(sql_type,split_num)
