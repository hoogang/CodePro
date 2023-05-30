    # -*- coding: utf-8 -*-
import pickle
import numpy as np


#-----------------------------------训练集处理成三元组模式---------------------------------

def gen_train_samples(candi_type,data_type,lang_type):
    #处理训练集
    train_ipath='./corpus/data/%s_%s_%s_parse_train.txt'%(candi_type,data_type,lang_type)
    train_opath='./corpus/pair/%s_%s_%s_train_triplets.txt'%(candi_type,data_type,lang_type)
    with open (train_ipath, 'r',encoding='utf-8') as fin:
        with open(train_opath,'w',encoding='utf-8') as fout:
            #正样本
            pos_list = []
            count=0  #计数
            # 循环读取每一行
            while True:
                line=fin.readline()
                # 如果不存在跳出循环
                if not line:
                    break
                #元素拆分
                line_info=line.split('\t')
                c_id=line_info[2]
                id=int(c_id.split('_')[2])
                if id==0:
                    count += 1
                    pos_code=line_info[3]
                    pos_list.append(pos_code)
                else:
                    query=line_info[1]
                    neg_code=line_info[3]
                    #写入（正样本，查询，负样本）
                    fout.write('\t'.join([pos_list[count-1],query,neg_code]) + '\n')
    print('triplets数据转换完毕！')

#-----------------------------------收集所有数据的词汇--------------------------------

def gen_vocab(candi_type,data_type,lang_type):

    # 词典包含了查询和代码的词
    words = []
    # 训练、验证、测试
    data_sets = ['%s_%s_%s_parse_train.txt'%(candi_type,data_type,lang_type),'%s_%s_%s_parse_valid.txt'%(candi_type,data_type,lang_type),'%s_%s_%s_parse_test.txt'%(candi_type,data_type,lang_type)]

    for set_name in data_sets:
        fin_path ='./corpus/data/%s'%set_name
        with open(fin_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                line_in =line.strip('\n').split('\t')
                query=line_in[1].split(' ')
                code =line_in[3].split(' ')
                for r1 in query:
                    if r1 not in words:
                        words.append(r1)
                for r2 in code:
                    if r2 not in words:
                        words.append(r2)

    # 预测
    cfpred_path = './pred/codemf_%s_pred.txt' % lang_type
    with open(cfpred_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line_in = line.strip('\n').split('\t')
            query = line_in[1].split(' ')
            code = line_in[3].split(' ')
            for r1 in query:
                if r1 not in words:
                    words.append(r1)
            for r2 in code:
                if r2 not in words:
                    words.append(r2)

    # 预测
    snpred_path = './pred/sattmf_%s_pred.txt' % lang_type
    with open(snpred_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            line_in = line.strip('\n').split('\t')
            query = line_in[1].split(' ')
            code = line_in[3].split(' ')
            for r1 in query:
                if r1 not in words:
                    words.append(r1)
            for r2 in code:
                if r2 not in words:
                    words.append(r2)

    if lang_type=='python':
        # 预测
        ccpred_path = './pred/codesn_%s_pred.txt' % lang_type
        with open(ccpred_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                line_in = line.strip('\n').split('\t')
                query = line_in[1].split(' ')
                code = line_in[3].split(' ')
                for r1 in query:
                    if r1 not in words:
                        words.append(r1)
                for r2 in code:
                    if r2 not in words:
                        words.append(r2)

            # 预测
        hmpred_path = './pred/human_%s_pred.txt' % lang_type
        with open(hmpred_path, 'r', encoding='utf-8') as fin:
            for line in fin:
                line_in = line.strip('\n').split('\t')
                query = line_in[1].split(' ')
                code = line_in[3].split(' ')
                for r1 in query:
                    if r1 not in words:
                        words.append(r1)
                for r2 in code:
                    if r2 not in words:
                        words.append(r2)

    fout_path = './vocab/%s_%s_%s_vocab.txt'%(candi_type,data_type,lang_type)
    with open(fout_path,'w',encoding='utf-8') as fout:
        for i, j in enumerate(words):
            fout.write('{}\t{}\n'.format(i, j))

    print('vocab词典数据转换完毕！')


#-----------------------------------根据词表生成对应的embedding--------------------------------

def data_transform(struc2vec_folder,adapt2vec_folder,candi_type,data_type,lang_type,embedding_size):

    vocab_in = './vocab/%s_%s_%s_vocab.txt'%(candi_type,data_type,lang_type)
    # add 2 words: <PAD> and <UNK>
    clean_vocab_out = './vocab/%s_%s_%s_clean_vocab.txt'%(candi_type,data_type,lang_type)

    struc2vec_in  = '../../use_embeddings/%s_glove.txt'%lang_type
    struc2vec_out = struc2vec_folder+'%s_%s_%s_struc2vec.pkl'%(candi_type, data_type, lang_type)

    tokidfvec_in = '../../use_embeddings/%s_idfvec.txt' % lang_type
    tokidfvec_out = adapt2vec_folder+'%s_%s_%s_tokidf2vec.pkl'% (candi_type, data_type,lang_type)

    words = []
    with open(vocab_in, 'r', encoding='utf-8') as f:
        for line in f:
            word = line.strip('\n').split('\t')[1]
            words.append(word)
    print('%s候选的%s类型的%s语言中的vocab.txt总共有%d个词'%(candi_type,data_type,lang_type,len(words)))

   ##############################################idfvec#####################################
    # 打开词向量
    with open(tokidfvec_in, 'r', encoding='utf-8') as f:
        #  dict 将['a' 'b']转换为 {'a':'b'} 前面把 pad和unk的idf填上
        tokidf_lis = []
        for line in f:
            try:  # 去掉换行符\n 将每一行以空格为分隔符转换成列表
                word_idf = line.strip('\n').split(' ')
                tokidf_lis.append(word_idf)
            except:
                print('这行加载失败: %s' % line.strip())
    # 变成word:idf词典
    tokidf_dict=dict(tokidf_lis)

    ##############################################embedding####################################
    rng = np.random.RandomState(None)
    pad_embedding = rng.uniform(-0.25, 0.25, size=(1, embedding_size))
    unk_embedding = rng.uniform(-0.25, 0.25, size=(1, embedding_size))

    struc2vecs = []
    # 加载pad和unk词
    clean_words = ['<pad>', '<unk>']
    struc2vecs.append(pad_embedding.reshape(-1).tolist())
    struc2vecs.append(unk_embedding.reshape(-1).tolist())
    print('pad和unk服从均匀分布的随机变量初始化......')

    tokidfvecs= [[0],[0]]
    # 打开词向量
    with open(struc2vec_in, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line_info = line.strip('\n').split(' ')
                word = line_info[0]
                embedding = [float(val) for val in line_info[1:]]
                if word in words:
                    #词在词典中
                    clean_words.append(word)
                    struc2vecs.append(embedding)
                    tokidfvecs.append([tokidf_dict[word]])
            except:
                print('这行加载失败: %s'%line.strip())

    print('%s候选的%s类型的%s语言中的clean_vocab.txt总共有%d个词'%(candi_type, data_type, lang_type, len(clean_words)))

    ##############################################embedding#####################################
    print('%s候选的%s类型的%s语言在glove词表加上pad和unk总共有%d个词'%(candi_type,data_type,lang_type,len(clean_words)))

    print('%s候选的%s类型的%s语言的embedding总共有%d个词'%(candi_type,data_type,lang_type,len(struc2vecs)))

    print('{}候选的{}类型的{}语言的embedding的维度为:{}'.format(candi_type,data_type,lang_type,np.shape(struc2vecs)))

    # 保存在词库的的词
    with open(clean_vocab_out, 'w', encoding='utf-8') as f:
        for i, j in enumerate(clean_words):
            f.write('{}\t{}\n'.format(i, j))

    # 保存embedding为pickle文件
    with open(struc2vec_out, 'wb') as f:
        pickle.dump(struc2vecs, f)

    # 保存wordidfvec为pickle文件
    with open(tokidfvec_out, 'wb') as f:
        pickle.dump(tokidfvecs, f)

    print('%s候选的%s类型的%s语言的wordidfvec总共有%d个词'%(candi_type,data_type,lang_type, len(tokidfvecs)))

    print('{}候选的{}类型的{}语言的wordidfvec的维度为:{}'.format(candi_type,data_type,lang_type,np.shape(tokidfvecs)))



'''
triplets数据转换完毕！
vocab词典数据转换完毕！
mutiple候选的mucqc类型的sql语言中的vocab.txt总共有28191个词
pad和unk服从均匀分布的随机变量初始化......
mutiple候选的mucqc类型的sql语言中的clean_vocab.txt总共有23644个词
mutiple候选的mucqc类型的sql语言在glove词表加上pad和unk总共有23644个词
mutiple候选的mucqc类型的sql语言的embedding总共有23644个词
mutiple候选的mucqc类型的sql语言的embedding的维度为:(23644, 300)
mutiple候选的mucqc类型的sql语言的wordidfvec总共有23644个词
mutiple候选的mucqc类型的sql语言的wordidfvec的维度为:(23644, 1)

triplets数据转换完毕！
vocab词典数据转换完毕！
mutiple候选的mucqc类型的python语言中的vocab.txt总共有118321个词
pad和unk服从均匀分布的随机变量初始化......
mutiple候选的mucqc类型的python语言中的clean_vocab.txt总共有116366个词
mutiple候选的mucqc类型的python语言在glove词表加上pad和unk总共有116366个词
mutiple候选的mucqc类型的python语言的embedding总共有116366个词
mutiple候选的mucqc类型的python语言的embedding的维度为:(116366, 300)
mutiple候选的mucqc类型的python语言的wordidfvec总共有116366个词
mutiple候选的mucqc类型的python语言的wordidfvec的维度为:(116366, 1)

triplets数据转换完毕！
vocab词典数据转换完毕！
mutiple候选的upvqc类型的sql语言中的vocab.txt总共有25656个词
pad和unk服从均匀分布的随机变量初始化......
mutiple候选的upvqc类型的sql语言中的clean_vocab.txt总共有22002个词
mutiple候选的upvqc类型的sql语言在glove词表加上pad和unk总共有22002个词
mutiple候选的upvqc类型的sql语言的embedding总共有22002个词
mutiple候选的upvqc类型的sql语言的embedding的维度为:(22002, 300)
mutiple候选的upvqc类型的sql语言的wordidfvec总共有22002个词
mutiple候选的upvqc类型的sql语言的wordidfvec的维度为:(22002, 1)

triplets数据转换完毕！
vocab词典数据转换完毕！
mutiple候选的upvqc类型的python语言中的vocab.txt总共有119513个词
pad和unk服从均匀分布的随机变量初始化......
mutiple候选的upvqc类型的python语言中的clean_vocab.txt总共有117347个词
mutiple候选的upvqc类型的python语言在glove词表加上pad和unk总共有117347个词
mutiple候选的upvqc类型的python语言的embedding总共有117347个词
mutiple候选的upvqc类型的python语言的embedding的维度为:(117347, 300)
mutiple候选的upvqc类型的python语言的wordidfvec总共有117347个词
mutiple候选的upvqc类型的python语言的wordidfvec的维度为:(117347, 1)

triplets数据转换完毕！
vocab词典数据转换完毕！
mutiple候选的saiqc类型的sql语言中的vocab.txt总共有29479个词
pad和unk服从均匀分布的随机变量初始化......
mutiple候选的saiqc类型的sql语言中的clean_vocab.txt总共有25983个词
mutiple候选的saiqc类型的sql语言在glove词表加上pad和unk总共有25983个词
mutiple候选的saiqc类型的sql语言的embedding总共有25983个词
mutiple候选的saiqc类型的sql语言的embedding的维度为:(25983, 300)
mutiple候选的saiqc类型的sql语言的wordidfvec总共有25983个词
mutiple候选的saiqc类型的sql语言的wordidfvec的维度为:(25983, 1)

triplets数据转换完毕！
vocab词典数据转换完毕！
mutiple候选的saiqc类型的python语言中的vocab.txt总共有170862个词
pad和unk服从均匀分布的随机变量初始化......
mutiple候选的saiqc类型的python语言中的clean_vocab.txt总共有165712个词
mutiple候选的saiqc类型的python语言在glove词表加上pad和unk总共有165712个词
mutiple候选的saiqc类型的python语言的embedding总共有165712个词
mutiple候选的saiqc类型的python语言的embedding的维度为:(165712, 300)
mutiple候选的saiqc类型的python语言的wordidfvec总共有165712个词
mutiple候选的saiqc类型的python语言的wordidfvec的维度为:(165712, 1)

triplets数据转换完毕！
vocab词典数据转换完毕！
mutiple候选的samdb类型的sql语言中的vocab.txt总共有89791个词
pad和unk服从均匀分布的随机变量初始化......
mutiple候选的samdb类型的sql语言中的clean_vocab.txt总共有66569个词
mutiple候选的samdb类型的sql语言在glove词表加上pad和unk总共有66569个词
mutiple候选的samdb类型的sql语言的embedding总共有66569个词
mutiple候选的samdb类型的sql语言的embedding的维度为:(66569, 300)
mutiple候选的samdb类型的sql语言的wordidfvec总共有66569个词
mutiple候选的samdb类型的sql语言的wordidfvec的维度为:(66569, 1)

triplets数据转换完毕！
vocab词典数据转换完毕！
mutiple候选的samdb类型的python语言中的vocab.txt总共有405419个词
pad和unk服从均匀分布的随机变量初始化......
mutiple候选的samdb类型的python语言中的clean_vocab.txt总共有356368个词
mutiple候选的samdb类型的python语言在glove词表加上pad和unk总共有356368个词
mutiple候选的samdb类型的python语言的embedding总共有356368个词
mutiple候选的samdb类型的python语言的embedding的维度为:(356368, 300)
mutiple候选的samdb类型的python语言中的vocab.txt总共有405419个词
mutiple候选的samdb类型的python语言的embedding的维度为:(139162, 300)
mutiple候选的samdb类型的python语言的wordidfvec总共有139162个词
mutiple候选的samdb类型的python语言的wordidfvec的维度为:(139162, 1)

triplets数据转换完毕！
vocab词典数据转换完毕！
mutiple候选的staqc类型的sql语言中的vocab.txt总共有26822个词
pad和unk服从均匀分布的随机变量初始化......
mutiple候选的staqc类型的sql语言中的clean_vocab.txt总共有21847个词
mutiple候选的staqc类型的sql语言在glove词表加上pad和unk总共有21847个词
mutiple候选的staqc类型的sql语言的embedding总共有21847个词
mutiple候选的staqc类型的sql语言的embedding的维度为:(21847, 300)
mutiple候选的staqc类型的sql语言的wordidfvec总共有21847个词
mutiple候选的staqc类型的sql语言的wordidfvec的维度为:(21847, 1)

triplets数据转换完毕！
vocab词典数据转换完毕！
mutiple候选的staqc类型的python语言中的vocab.txt总共有150682个词
pad和unk服从均匀分布的随机变量初始化......
mutiple候选的staqc类型的python语言中的clean_vocab.txt总共有136122个词
mutiple候选的staqc类型的python语言在glove词表加上pad和unk总共有136122个词
mutiple候选的staqc类型的python语言的embedding总共有136122个词
mutiple候选的staqc类型的python语言的embedding的维度为:(136122, 300)
mutiple候选的staqc类型的python语言的wordidfvec总共有136122个词
mutiple候选的staqc类型的python语言的wordidfvec的维度为:(136122, 1)

triplets数据转换完毕！
vocab词典数据转换完毕！
mutiple候选的samqc类型的sql语言中的vocab.txt总共有27337个词
pad和unk服从均匀分布的随机变量初始化......
mutiple候选的samqc类型的sql语言中的clean_vocab.txt总共有22253个词
mutiple候选的samqc类型的sql语言在glove词表加上pad和unk总共有22253个词
mutiple候选的samqc类型的sql语言的embedding总共有22253个词
mutiple候选的samqc类型的sql语言的embedding的维度为:(22253, 300)
mutiple候选的samqc类型的sql语言的wordidfvec总共有22253个词
mutiple候选的samqc类型的sql语言的wordidfvec的维度为:(22253, 1)

triplets数据转换完毕！
vocab词典数据转换完毕！
mutiple候选的samqc类型的python语言中的vocab.txt总共有154393个词
pad和unk服从均匀分布的随机变量初始化......
mutiple候选的samqc类型的python语言中的clean_vocab.txt总共有139454个词
mutiple候选的samqc类型的python语言在glove词表加上pad和unk总共有139454个词
mutiple候选的samqc类型的python语言的embedding总共有139454个词
mutiple候选的samqc类型的python语言的embedding的维度为:(139454, 300)
mutiple候选的samqc类型的python语言的wordidfvec总共有139454个词
mutiple候选的samqc类型的python语言的wordidfvec的维度为:(139454, 1)

triplets数据转换完毕！
vocab词典数据转换完毕！
single候选的sicqc类型的sql语言中的vocab.txt总共有23867个词
pad和unk服从均匀分布的随机变量初始化......
single候选的sicqc类型的sql语言中的clean_vocab.txt总共有20810个词
single候选的sicqc类型的sql语言在glove词表加上pad和unk总共有20810个词
single候选的sicqc类型的sql语言的embedding总共有20810个词
single候选的sicqc类型的sql语言的embedding的维度为:(20810, 300)
single候选的sicqc类型的sql语言的wordidfvec总共有20810个词
single候选的sicqc类型的sql语言的wordidfvec的维度为:(20810, 1)

triplets数据转换完毕！
vocab词典数据转换完毕！
single候选的sicqc类型的python语言中的vocab.txt总共有96995个词
pad和unk服从均匀分布的随机变量初始化......
single候选的sicqc类型的python语言中的clean_vocab.txt总共有95393个词
single候选的sicqc类型的python语言在glove词表加上pad和unk总共有95393个词
single候选的sicqc类型的python语言的embedding总共有95393个词
single候选的sicqc类型的python语言的embedding的维度为:(95393, 300)
single候选的sicqc类型的python语言的wordidfvec总共有95393个词
single候选的sicqc类型的python语言的wordidfvec的维度为:(95393, 1)

triplets数据转换完毕！
vocab词典数据转换完毕！
single候选的csnqc类型的python语言中的vocab.txt总共有983824个词
pad和unk服从均匀分布的随机变量初始化......
single候选的csnqc类型的python语言中的clean_vocab.txt总共有978811个词
single候选的csnqc类型的python语言在glove词表加上pad和unk总共有978811个词
single候选的csnqc类型的python语言的embedding总共有978811个词
single候选的csnqc类型的python语言的embedding的维度为:(978811, 300)
cleaned候选的csnqc类型的python语言的wordidfvec总共有978811个词
cleaned候选的csnqc类型的python语言的wordidfvec的维度为:(978811, 1)

triplets数据转换完毕！
vocab词典数据转换完毕！
cleaned候选的csnqc类型的python语言中的vocab.txt总共有686682个词
pad和unk服从均匀分布的随机变量初始化......
cleaned候选的csnqc类型的python语言中的clean_vocab.txt总共有682328个词
cleaned候选的csnqc类型的python语言在glove词表加上pad和unk总共有682328个词
cleaned候选的csnqc类型的python语言的embedding总共有682328个词
cleaned候选的csnqc类型的python语言的embedding的维度为:(682328, 300)
cleaned候选的csnqc类型的python语言的wordidfvec总共有682328个词
cleaned候选的csnqc类型的python语言的wordidfvec的维度为:(682328, 1)

triplets数据转换完毕！
vocab词典数据转换完毕！
nlqfpro候选的csnqc类型的python语言中的vocab.txt总共有700098个词
pad和unk服从均匀分布的随机变量初始化......
nlqfpro候选的csnqc类型的python语言中的clean_vocab.txt总共有694850个词
nlqfpro候选的csnqc类型的python语言在glove词表加上pad和unk总共有694850个词
nlqfpro候选的csnqc类型的python语言的embedding总共有694850个词
nlqfpro候选的csnqc类型的python语言的embedding的维度为:(694850, 300)
nlqfpro候选的csnqc类型的python语言的wordidfvec总共有694850个词
nlqfpro候选的csnqc类型的python语言的wordidfvec的维度为:(694850, 1)

triplets数据转换完毕！
vocab词典数据转换完毕！
codepro候选的mixqc类型的python语言中的vocab.txt总共有999056个词
pad和unk服从均匀分布的随机变量初始化......
codepro候选的mixqc类型的python语言中的clean_vocab.txt总共有976311个词
codepro候选的mixqc类型的python语言在glove词表加上pad和unk总共有976311个词
codepro候选的mixqc类型的python语言的embedding总共有976311个词
codepro候选的mixqc类型的python语言的embedding的维度为:(976311, 300)
codepro候选的mixqc类型的python语言的wordidfvec总共有976311个词
codepro候选的mixqc类型的python语言的wordidfvec的维度为:(976311, 1)
'''

#--------参数配置----------
struc2vec_folder='./struc2vec/'
adpa2vec_folder='./adapt2vec/'

#候选类型
single_type ='single'
mutiple_type='mutiple'
cleaned_type='cleaned'
nlqfpro_type='nlqfpro'


#数据类型
mucqc_type='mucqc'
upvqc_type='upvqc'
staqc_type='staqc'
samqc_type='samqc'
saiqc_type='saiqc'
samdb_type='samdb'

sicqc_type='sicqc'
csnqc_type='csnqc'


#语言类型
sql_type='sql'
python_type='python'

# 向量维度
embedding_size=300


if __name__ == '__main__':

    #mutiple
    # gen_train_samples(mutiple_type,mucqc_type,sql_type)
    # gen_vocab(mutiple_type,mucqc_type,sql_type)
    # data_transform(struc2vec_folder, adpa2vec_folder,mutiple_type, mucqc_type, sql_type, embedding_size)
    #
    # gen_train_samples(mutiple_type, mucqc_type,python_type)
    # gen_vocab(mutiple_type, mucqc_type, python_type)
    # data_transform(struc2vec_folder, adpa2vec_folder,mutiple_type, mucqc_type, python_type, embedding_size)
    # #
    # gen_train_samples(mutiple_type, upvqc_type, sql_type)
    # gen_vocab(mutiple_type, upvqc_type, sql_type)
    # data_transform(struc2vec_folder, adpa2vec_folder, mutiple_type, upvqc_type, sql_type, embedding_size)
    #
    #
    # gen_train_samples(mutiple_type, upvqc_type, python_type)
    # gen_vocab(mutiple_type, upvqc_type, python_type)
    # data_transform(struc2vec_folder, adpa2vec_folder, mutiple_type, upvqc_type, python_type, embedding_size)
    #
    #
    # gen_train_samples(mutiple_type, saiqc_type, sql_type)
    # gen_vocab(mutiple_type, saiqc_type, sql_type)
    # data_transform(struc2vec_folder, adpa2vec_folder, mutiple_type, saiqc_type, sql_type, embedding_size)
    #
    #
    # gen_train_samples(mutiple_type, saiqc_type, python_type)
    # gen_vocab(mutiple_type, saiqc_type, python_type)
    # data_transform(struc2vec_folder, adpa2vec_folder, mutiple_type, saiqc_type, python_type, embedding_size)
    #
    #
    # gen_train_samples(mutiple_type, samdb_type, sql_type)
    # gen_vocab(mutiple_type, samdb_type, sql_type)
    # data_transform(struc2vec_folder, adpa2vec_folder, mutiple_type, samdb_type, sql_type, embedding_size)
    #
    #
    # gen_train_samples(mutiple_type, samdb_type, python_type)
    # gen_vocab(mutiple_type, samdb_type, python_type)
    # data_transform(struc2vec_folder, adpa2vec_folder, mutiple_type, samdb_type, python_type, embedding_size)
    #
    #
    # gen_train_samples(mutiple_type, staqc_type,sql_type)
    # gen_vocab(mutiple_type, staqc_type, sql_type)
    # data_transform(struc2vec_folder, adpa2vec_folder, mutiple_type, staqc_type, sql_type, embedding_size)
    #
    #
    # gen_train_samples(mutiple_type, staqc_type,python_type)
    # gen_vocab(mutiple_type, staqc_type, python_type)
    # data_transform(struc2vec_folder, adpa2vec_folder, mutiple_type, staqc_type, python_type, embedding_size)
    #
    #
    # gen_train_samples(mutiple_type, samqc_type,sql_type)
    # gen_vocab(mutiple_type, samqc_type,sql_type)
    # data_transform(struc2vec_folder, adpa2vec_folder, mutiple_type, samqc_type,sql_type, embedding_size)
    #
    #
    # gen_train_samples(mutiple_type, samqc_type,python_type)
    # gen_vocab(mutiple_type, samqc_type,python_type)
    # data_transform(struc2vec_folder, adpa2vec_folder, mutiple_type, samqc_type,python_type, embedding_size)
    #
    #
    # #single
    # gen_train_samples(single_type, sicqc_type,sql_type)
    # gen_vocab(single_type, sicqc_type,sql_type)
    # data_transform(struc2vec_folder, adpa2vec_folder, single_type, sicqc_type,sql_type, embedding_size)
    #
    # gen_train_samples(single_type, sicqc_type, python_type)
    # gen_vocab(single_type, sicqc_type, python_type)
    # data_transform(struc2vec_folder, adpa2vec_folder, single_type, sicqc_type, python_type, embedding_size)

    #gen_train_samples(single_type, csnqc_type, python_type)
    #gen_vocab(single_type, csnqc_type, python_type)
    #data_transform(struc2vec_folder, adpa2vec_folder, single_type, csnqc_type, python_type, embedding_size)

    # gen_train_samples(cleaned_type, csnqc_type, python_type)
    # gen_vocab(cleaned_type, csnqc_type, python_type)
    # data_transform(struc2vec_folder, adpa2vec_folder, cleaned_type, csnqc_type, python_type, embedding_size)

    gen_train_samples(nlqfpro_type, csnqc_type, python_type)
    gen_vocab(nlqfpro_type, csnqc_type, python_type)
    data_transform(struc2vec_folder, adpa2vec_folder, nlqfpro_type, csnqc_type, python_type, embedding_size)











