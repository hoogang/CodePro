#python3

# tensorflow-gpu=1.0.0

#在命令行输入 tensorboard –-logdir=「log目录」
#本地访问地址 https://locallhost/6006/

import os
import codecs
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from gensim.models import KeyedVectors

def vector_show(words_list,lang_type):

    words, embeddings = [], []

    # 打印全部词向量
    with codecs.open("%s_struc2vec.txt"%lang_type,'r') as f:
        header = f.readline()
        vocab_size, vector_size = map(int, header.split())
        print('字典的大小：%d，词向量的维度：%d'%(vocab_size,vector_size))
        for line in tqdm(range(vocab_size)):
            word_list = f.readline().strip('\n').split(' ')
            word = word_list[0]
            vector = word_list[1:]
            if word == "":
                continue
            words.append(word)
            embeddings.append(np.array(vector))
    assert len(words) == len(embeddings)

    with tf.Session() as sess:
        X = tf.Variable([0.0], name='embedding')
        place = tf.placeholder(tf.float32, shape=[len(words), vector_size])
        set_x = tf.assign(X, place, validate_shape=False)
        sess.run(tf.global_variables_initializer())
        sess.run(set_x, feed_dict={place: embeddings})

        if not os.path.exists('./dict_words/%s_logs/' % lang_type):
            os.makedirs('./dict_words/%s_logs/' % lang_type)

        with codecs.open('./dict_words/%s_logs/metadata.tsv' % lang_type, 'w') as f:
            for word in tqdm(words):
                f.write(word + '\n')

        # with summary
        summary_writer = tf.summary.FileWriter("./dict_words/%s_logs/" % lang_type, sess.graph)

        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = 'embedding'
        embedding_conf.metadata_path = os.path.join("./dict_words/%s_logs/" % lang_type, 'metadata.tsv')
        projector.visualize_embeddings(summary_writer, config)

        # save
        saver = tf.train.Saver()
        saver.save(sess, os.path.join("./dict_words/%s_logs/" % lang_type, "model.ckpt"))


    # 加载转换文件
    model = KeyedVectors.load_word2vec_format("%s_struc2vec.txt" % lang_type, binary=False)
    # 输出词向量
    embeddings = []
    for word in words_list:
        embeddings.append(model.wv[word])  # 加载词向量


    with tf.Session() as sess:
        X = tf.Variable([0.0], name='embedding')
        place = tf.placeholder(tf.float32, shape=[len(words_list), vector_size])
        set_x = tf.assign(X, place, validate_shape=False)
        sess.run(tf.global_variables_initializer())
        sess.run(set_x, feed_dict={place: embeddings})

        if not os.path.exists('./part_words/%s_logs/' % lang_type):
            os.makedirs('./part_words/%s_logs/' % lang_type)

        with codecs.open('./part_words/%s_logs/metadata.tsv'%lang_type, 'w') as f:
            for word in tqdm(words_list):
                f.write(word + '\n')

        # with summary
        summary_writer = tf.summary.FileWriter("./part_words/%s_logs/"%lang_type, sess.graph)

        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = 'embedding'
        embedding_conf.metadata_path = os.path.join("./part_words/%s_logs/"%lang_type,'metadata.tsv')
        projector.visualize_embeddings(summary_writer, config)

        # save
        saver = tf.train.Saver()
        saver.save(sess, os.path.join("./part_words/%s_logs/"%lang_type, "model.ckpt"))




#参数配置
#字典的大小：1557800，词向量的维度：300
sqlang_type = 'sql'

# 选的词语列表
sql_words=[
    'row','column','table','group','tagstr','tagint','codint','value','data','server','sql','database','if','with','from','for','to','on','in','into','between','or','and','but','where','what','when','which','order','by','of','as','like','set','select','insert','update','delete','join','inner','create','check','view','use','get','*','(',')','/','=']



#字典的大小：2025408，词向量的维度：300
python_type = 'python'

# 选的词语列表
python_words=[
    'or','and','not','as','if','else','elif','str','int','float','list','tuple','dict','bool','set','false','none','true','assert', 'break', 'class', 'continue','del', 'for', 'range', 'in','global', 'from','import', 'is', 'lambda', 'nonlocal', 'pass', 'raise','def','return', 'try','except','finally', 'while', 'with', 'yield','tagstr','tagint','{','}','<','>','(',')']


if __name__ == '__main__':
    vector_show(sql_words,sqlang_type)
    vector_show(python_words,python_type)