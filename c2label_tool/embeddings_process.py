# coding=utf-8

'''
从大词典中获取特定于于语料的词典
将数据处理成待打标签的形式
'''

import numpy as np
import pickle
from gensim.models import KeyedVectors

#词向量文件保存成bin文件
def trans_bin(word2vec_txt,word2vec_bin):
    wv_from_text = KeyedVectors.load_word2vec_format(word2vec_txt, binary=False)
    # 如果每次都用上面的方法加载，速度非常慢
    # 保存成bin文件加载bin文件，速度会变快
    wv_from_text.init_sims(replace=True)
    wv_from_text.save(word2vec_bin)
    '''
    读取用一下代码
    model = KeyedVectors.load(embed_path, mmap='r')
    '''

#构建新的词典 和词向量矩阵
def get_dict(type_vpath,type_wpath,final_vpath,final_wpath):  #词标签，词向量
    # 原词159018 找到的词133959 找不到的词25059
    # 添加unk过后 159019 找到的词133960 找不到的词25059
    # 添加pad过后 词向量 133961
    # 加载转换文件
    model = KeyedVectors.load(type_vpath, mmap='r')

    with open(type_wpath,'r')as f:
        total_word= eval(f.read())
        f.close()

    # 输出词向量
    #其中0 PAD_ID,1 SOS_ID,2 E0S_ID,3 UNK_ID
    word_dict = ['PAD','SOS','EOS','UNK']

    fail_word = []
    rng = np.random.RandomState(None)
    pad_embedding = np.zeros(shape=(1, 300)).squeeze()
    unk_embediing = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    sos_embediing = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    eos_embediing = rng.uniform(-0.25, 0.25, size=(1, 300)).squeeze()
    word_vectors = [pad_embedding,sos_embediing,eos_embediing,unk_embediing]
    print(len(total_word))
    for word in total_word:
        try:
            word_vectors.append(model[word]) #加载词向量
            word_dict.append(word)
        except:
            print(word)
            fail_word.append(word)

    #关于有多少个词，以及多少个词没有找到
    print(len(word_dict))
    print(len(word_vectors))
    print(len(fail_word))


    #判断词向量是否正确
    '''
    couunt = 0
    for i in range(4,len(word_dict)):
        if word_vectors[i].all() == model.wv[word_dict[i]].all():
            continue
        else:
            couunt +=1

    print(couunt)
    '''

    word_vectors = np.array(word_vectors)
    #print(word_vectors.shape)
    word_dict = dict(map(reversed, enumerate(word_dict)))
    #np.savetxt(final_vec_path,word_vectors)
    with open(final_vpath, 'wb') as file:
        pickle.dump(word_vectors, file)

    with open(final_wpath, 'wb') as file:
        pickle.dump(word_dict, file)



def get_dict_append(type_vec_path,previous_dict,previous_vec,append_word_path,final_vec_path,final_word_path):  #词标签，词向量
    #原词159018 找到的词133959 找不到的词25059
    #添加unk过后 159019 找到的词133960 找不到的词25059
    #添加pad过后 词典：133961 词向量 133961

    # 加载转换文件

    model = KeyedVectors.load(type_vec_path, mmap='r')

    with open(previous_dict, 'rb') as f:
        pre_word_dict = pickle.load(f)

    with open(previous_vec, 'rb') as f:
        pre_word_vec = pickle.load(f)

    with open(append_word_path,'r')as f:
        append_word= eval(f.read())
        f.close()

    # 输出词向量

    print(type(pre_word_vec))
    word_dict =  list(pre_word_dict.keys()) #'#其中0 PAD_ID,1 SOS_ID,2 E0S_ID,3 UNK_ID
    print(len(word_dict))
    word_vectors = pre_word_vec.tolist()
    print(word_dict[:100])
    fail_word =[]
    print(len(append_word))
  
    for word in append_word:
        try:
            # 加载词向量
            word_vectors.append(model[word])
            word_dict.append(word)
        except:
            fail_word.append(word)
    #关于有多少个词，以及多少个词没有找到
    print(len(word_dict))
    print(len(word_vectors))
    print(len(fail_word))
    print(word_dict[:100])


    '''
    #判断词向量是否正确
    print("----------------------------")
    couunt = 0

    import operator
    for i in range(159035,len(word_dict)):
        if operator.eq(word_vectors[i].tolist(), model.wv[word_dict[i]].tolist()) == True:
            continue
        else:
            couunt +=1

    print(couunt)
    '''


    word_vectors = np.array(word_vectors)

    word_dict = dict(map(reversed, enumerate(word_dict)))
    #np.savetxt(final_vec_path,word_vectors)
    with open(final_vec_path, 'wb') as file:
        pickle.dump(word_vectors, file)

    with open(final_word_path, 'wb') as file:
        pickle.dump(word_dict, file)



#得到词在词典中的位置
def get_index(type,text,word_dict):
    location = []
    if type == 'code':
        location.append(1)
        len_c = len(text)
        if len_c+1 <350:
            if len_c == 1 and text[0] == '-1000':
                location.append(2)
            else:
                for i in range(0, len_c):
                    if word_dict.get(text[i]) != None:
                        index = word_dict.get(text[i])
                        location.append(index)
                    else:
                        index = word_dict.get('UNK')
                        location.append(index)

                location.append(2)
        else:
            for i in range(0, 348):
                if word_dict.get(text[i]) != None:
                    index = word_dict.get(text[i])
                    location.append(index)
                else:
                    index = word_dict.get('UNK')
                    location.append(index)
            location.append(2)
    else:
        if len(text) == 0:
            location.append(0)
        elif text[0] == '-10000':
            location.append(0)
        else:
            for i in range(0, len(text)):
                if word_dict.get(text[i]) != None:
                    index = word_dict.get(text[i])
                    location.append(index)
                else:
                    index = word_dict.get('UNK')
                    location.append(index)

    return location


#将训练、测试、验证语料序列化
#查询：25 上下文：100 代码：350
def serialization(word_dict_path,type_path,final_type_path):

    with open(word_dict_path, 'rb') as f:
        word_dict = pickle.load(f)

    with open(type_path,'r')as f:
        corpus= eval(f.read())
        f.close()

    total_data = []

    for i in range(0, len(corpus)):
        qid = corpus[i][0]

        # Si
        si_word_list =   get_index('text',corpus[i][1][0],word_dict)
        # Si+1
        si1_word_list =  get_index('text',corpus[i][1][1],word_dict)
        # code
        tokenized_code =  get_index('code', corpus[i][2][0], word_dict)
        # query
        query_word_list = get_index('text',corpus[i][3],word_dict)

        block_length = 4
        label = 0
        if(len(si_word_list)>100):
            si_word_list = si_word_list[:100]
        else:
            for k in range(0, 100 - len(si_word_list)):
                si_word_list.append(0)

        if (len(si1_word_list) > 100):
            si1_word_list = si1_word_list[:100]
        else:
            for k in range(0, 100 - len(si1_word_list)):
                si1_word_list.append(0)

        if (len(tokenized_code) < 350):
            for k in range(0, 350 - len(tokenized_code)):
                tokenized_code.append(0)
        else:
            tokenized_code = tokenized_code[:350]

        if (len(query_word_list) > 25):
            query_word_list = query_word_list[:25]
        else:
            for k in range(0, 25 - len(query_word_list)):
                query_word_list.append(0)

        one_data = [qid, [si_word_list, si1_word_list], [tokenized_code], query_word_list, block_length, label]
        total_data.append(one_data)

    with open(final_type_path, 'wb') as file:
        pickle.dump(total_data, file)


if __name__ == '__main__':

    python_embed_txt = '../crawl_embeddings/python_struc2vec.txt'
    python_embed_bin = '../crawl_embeddings/python_struc2vec.bin'

    sql_embed_txt = '../crawl_embeddings/sql_struc2vec.txt'
    sql_embed_bin = '../crawl_embeddings/sql_struc2vec.bin'

    #trans_bin(python_embed_txt, python_embed_bin)
    #trans_bin(sql_embed_txt,sql_embed_bin)

    #====================================基于Staqc的词典和词向量==========================

    python_staqc_dict = '../staqc_tolabel/data/code_solution_labeled_data/filter/word_dict/python_word_vocab_dict.txt'
    python_staqc_fvocab = '../staqc_tolabel/data/code_solution_labeled_data/filter/embeddings/python_word_vocab_final.pkl'
    python_staqc_fdict = '../staqc_tolabel/data/code_solution_labeled_data/filter/embeddings/python_word_dict_final.pkl'

    sql_staqc_dict =    '../staqc_tolabel/data/code_solution_labeled_data/filter/word_dict/sql_word_vocab_dict.txt'
    sql_staqc_fvocab = '../staqc_tolabel/data/code_solution_labeled_data/filter/embeddings/sql_word_vocab_final.pkl'
    sql_staqc_fdict = '../staqc_tolabel/data/code_solution_labeled_data/filter/embeddings/sql_word_dict_final.pkl'

    #txt存储数组向量，读取时间：30s,以pickle文件存储0.23s,所以最后采用pkl文件

    #get_dict(python_embed_bin,python_staqc_dict,python_staqc_fvocab,python_staqc_fdict)
    #get_dict(sql_embed_bin,sql_staqc_dict, sql_staqc_fvocab, sql_staqc_fdict)

  
    #sql 待处理语料地址
    sql_large_dict = '../istaqc_crawl/corpus/word_dict/sql_word_vocab_dict.txt'
    # sql最后的词典和对应的词向量
    sql_large_fvocab  = '../istaqc_crawl/corpus/embeddings/sql_word_vocab_final.pkl'
    sql_large_fdict  = '../istaqc_crawl/corpus/embeddings/sql_word_dict_final.pkl'
    #get_dict_append(sql_embed_bin, sql_staqc_fdict, sql_staqc_fvocab, sql_large_dict, sql_large_fvocab,sql_large_fdict)


    #处理成打标签的形式
    sql_staqc_data = '../staqc_tolabeldata/code_solution_labeled_data/filter/tokennized_data/sql_staqc_multiple_iid_title&blocks_to_mutiple_qids_tokennized.txt'
    sql_staqc_format = '../staqc_tolabel/data/code_solution_labeled_data/filter/format_tolabel/sql_how_to_do_it_by_classifier_multiple_iid_title&blocks_to_mutiple_qids_format_tolabel.pkl'
    #serialization(sql_staqc_fdict, sql_staqc_data,  sql_staqc_format)

    sql_large_data = '../istaqc_crawl/corpus/tokennized_data/sql_codedb_qid2index_blocks_to_multiple_qids_tokennized.txt'
    sql_large_format ='../istaqc_crawl/corpus/format_tolabel/sql_codedb_qid2index_blocks_to_multiple_qids_format_tolabel.pkl'
    #serialization(sql_large_fdict, sql_large_data,  sql_large_format)

    #python 待处理语料地址
    python_large_dict = '../istaqc_crawl/corpus/word_dict/python_word_vocab_dict.txt'
    #python最后的词典和对应的词向量
    python_large_fvocab =  '../istaqc_crawl/corpus/embeddings/python_word_vocab_final.pkl'
    python_large_fdict = '../istaqc_crawl/corpus/embeddings/python_word_dict_final.pkl'
    #get_dict_append(python_embed_bin, python_staqc_fdict, python_staqc_fvocab, python_large_dict, python_large_fvocab, python_large_fdict)


    #处理成打标签的形式
    python_staqc_data = '../staqc_tolabe/ldata/code_solution_labeled_data/filter/tokennized_data/python_staqc_multiple_iid_title&blocks_to_mutiple_qids_tokennized.txt'
    python_staqc_format = '../staqc_tolabel/data/code_solution_labeled_data/filter/format_tolabel/python_how_to_do_it_by_classifier_multiple_iid_title&blocks_to_mutiple_qids_format_tolabel.pkl'
    #serialization(python_staqc_fdict, python_staqc_data, python_staqc_format)

    python_large_data = '../istaqc_crawl/corpus/tokennized_data/python_codedb_qid2index_blocks_to_multiple_qids_tokennized.txt'
    python_large_format ='../istaqc_crawl/corpus/format_tolabel/python_codedb_qid2index_blocks_to_multiple_qids_format_tolabel.pkl'
    #serialization(python_large_fdict, python_large_data, python_large_format)


    #python 待处理语料地址
    sql_corpus_dict = '../hqfso_tool/corpus/word_dict/sql_word_vocab_dict.txt'
    #python最后的词典和对应的词向量
    sql_corpus_fvocab =  '../hqfso_tool/corpus/embeddings/sql_word_vocab_final.pkl'
    sql_corpus_fdict = '../hqfso_tool/corpus/embeddings/sql_word_dict_final.pkl'
    get_dict_append(sql_embed_bin, sql_staqc_fdict, sql_staqc_fvocab, sql_corpus_dict, sql_corpus_fvocab, sql_corpus_fdict)

    sql_corpus_data = '../hqfso_tool/corpus/tokennized_data/sql_corpus_qid2index_blocks_to_multiple_qids_tokennized.txt'
    sql_corpus_format ='../hqfso_tool/corpus/format_tolabel/sql_corpus_qid2index_blocks_to_multiple_qids_format_tolabel.pkl'
    serialization(sql_corpus_fdict, sql_corpus_data, sql_corpus_format)

    #python 待处理语料地址
    python_corpus_dict = '../hqfso_tool/corpus/word_dict/python_word_vocab_dict.txt'
    #python最后的词典和对应的词向量
    python_corpus_fvocab =  '../hqfso_tool/corpus/embeddings/python_word_vocab_final.pkl'
    python_corpus_fdict = '../hqfso_tool/corpus/embeddings/python_word_dict_final.pkl'
    get_dict_append(python_embed_bin, python_staqc_fdict, python_staqc_fvocab, python_corpus_dict, python_corpus_fvocab, python_corpus_fdict)

    python_corpus_data = '../hqfso_tool/corpus/tokennized_data/python_corpus_qid2index_blocks_to_multiple_qids_tokennized.txt'
    python_corpus_format ='../hqfso_tool/corpus/format_tolabel/python_corpus_qid2index_blocks_to_multiple_qids_format_tolabel.pkl'
    serialization(python_corpus_fdict, python_corpus_data, python_corpus_format)










