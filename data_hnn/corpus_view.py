# -*- coding: utf-8 -*-

# python2 运行
import pickle

def main(lang_type):

    train_data_partialcontext= pickle.load(open('./%s/train/data_partialcontext_shared_text_vocab_in_buckets.pickle'%lang_type,'rb'))

    # 长度是4，分为4个桶
    print'train_data_partialcontext—seqlen:\n',len(train_data_partialcontext)

    # 打印前4个桶里的前4个数据
    post_train_iids_labels= []
    for i in range(4):
        # 按帖子ID从小到大排序
        bucket_data_partialcontext = train_data_partialcontext[i]
        each_train_data_partialcontext = sorted(bucket_data_partialcontext, key=lambda x: (x[0][0], x[0][1]))
        for j in range(2):
            # [(id, index),[[si，si+1]] 文本块，[true_len(si),true_len(si+1)] 文本长度，[[c]] 代码，[true_len(c)] 代码长度，[q] 查询, 查询长度， 标签，指标， 块长度
            print 'train_data_partialcontext%d-%d:\n'%(i,j), each_train_data_partialcontext[j]
            bucket_iids_labels = [(k[0][0], k[0][1], k[7]) for k in each_train_data_partialcontext]
            post_train_iids_labels.extend(bucket_iids_labels)

    post_train_iids_labels = sorted(set(post_train_iids_labels), key=lambda x : (x[0],x[1]))
    print 'post_train_iids_labels:\n', post_train_iids_labels[:4]

    post_train_iids = [i[:2] for i in post_train_iids_labels]
    print 'post_train_iids:\n', post_train_iids[:4]

    # 打印前两项iids
    train_iids = pickle.load(open('%s/train/iids.pickle' % lang_type, 'rb'))
    train_iids = sorted(train_iids, key=lambda x: (x[0],x[1]))
    print 'train_iids:\n', train_iids[:4]

    assert post_train_iids == train_iids


    # 处理自然语言词向量
    train_data_partialcontext_word2vec = pickle.load(open('./%s/train/rnn_partialcontext_word_embedding_text_150.pickle' % lang_type, 'rb'))
    #  Si, Si+1, q的词向量  24379个词，150维度 （24379,150）
    print 'train_data_rnncontext_word2vec-seqlen:\n', len(train_data_partialcontext_word2vec)

    train_data_partialcontext_code2vec = pickle.load(open('./%s/train/rnn_partialcontext_word_embedding_code_150.pickle' % lang_type, 'rb'))
    #  c的词向量  218900个词，150维度  （218900,150）
    print 'train_data_rnncontext_code2vec-seqlen:\n', len(train_data_partialcontext_code2vec)


    valid_data_partialcontext = pickle.load(open('./%s/valid/data_partialcontext_shared_text_vocab_in_buckets.pickle' % lang_type, 'rb'))

    # 长度是4，分为4个桶
    print 'valid_data_partialcontext—seqlen:\n', len(valid_data_partialcontext)

    # 打印前4个桶里的前4个数据
    post_valid_iids_labels = []
    for i in range(4):
        # 按帖子ID从小到大排序
        bucket_data_partialcontext = valid_data_partialcontext[i]
        each_valid_data_partialcontext = sorted(bucket_data_partialcontext, key=lambda x: (x[0][0], x[0][1]))
        for j in range(2):
            # [(id, index),[[si，si+1]] 文本块，[true_len(si),true_len(si+1)] 文本长度，[[c]] 代码，[true_len(c)] 代码长度，[q] 查询, 查询长度， 标签，指标， 块长度
            print 'valid_data_partialcontext%d-%d:\n' % (i, j), each_valid_data_partialcontext[j]
            bucket_iids_labels = [(k[0][0], k[0][1], k[7]) for k in each_valid_data_partialcontext]
            post_valid_iids_labels.extend(bucket_iids_labels)

    post_valid_iids_labels = sorted(set(post_valid_iids_labels), key=lambda x: (x[0], x[1]))
    print 'post_valid_iids_labels:\n', post_valid_iids_labels[:4]

    post_valid_iids = [i[:2] for i in post_valid_iids_labels]
    print 'post_train_iids:\n', post_valid_iids[:4]

    # 打印前两项iids
    valid_iids = pickle.load(open('%s/valid/iids.pickle' % lang_type, 'rb'))
    valid_iids = sorted(valid_iids, key=lambda x: (x[0], x[1]))
    print 'valid_iids:\n', valid_iids[:4]

    assert post_valid_iids == valid_iids


    test_data_partialcontext = pickle.load(open('./%s/test/data_partialcontext_shared_text_vocab_in_buckets.pickle' % lang_type, 'rb'))
    # 长度是4，分为4个桶
    print 'test_data_partialcontext—seqlen:\n', len(test_data_partialcontext)

    # 打印前4个桶里的前4个数据
    post_test_iids_labels = []
    for i in range(4):
        # 按帖子ID从小到大排序
        bucket_data_partialcontext = test_data_partialcontext[i]
        each_test_data_partialcontext = sorted(bucket_data_partialcontext, key=lambda x: (x[0][0], x[0][1]))
        for j in range(2):
            # [(id, index),[[si，si+1]] 文本块，[true_len(si),true_len(si+1)] 文本长度，[[c]] 代码，[true_len(c)] 代码长度，[q] 查询, 查询长度， 标签，指标， 块长度
            print 'test_data_partialcontext%d-%d:\n' % (i, j), each_test_data_partialcontext[j]
            bucket_iids_labels = [(k[0][0], k[0][1], k[7]) for k in each_test_data_partialcontext]
            post_test_iids_labels.extend(bucket_iids_labels)

    post_test_iids_labels = sorted(set(post_test_iids_labels), key=lambda x: (x[0], x[1]))
    print 'post_valid_iids_labels:\n', post_test_iids_labels[:4]

    post_test_iids = [i[:2] for i in post_test_iids_labels]
    print 'post_test_iids:\n', post_test_iids[:4]

    # 打印前两项iids
    test_iids = pickle.load(open('%s/test/iids.pickle' % lang_type, 'rb'))
    test_iids = sorted(test_iids, key=lambda x: (x[0], x[1]))
    print 'test_iids:\n', test_iids[:4]

    assert post_test_iids == test_iids

    post_iids_labels = post_train_iids_labels+post_valid_iids_labels+post_test_iids_labels

    print '%s语言数据的HNN模型训练有%d条\n'%(lang_type, len(set(post_iids_labels))), post_iids_labels[:4]

    pickle.dump(post_iids_labels,open('./%s_hnn_iids_labeles.pickle'%lang_type,'wb'))


if __name__ == "__main__":
    main('python')
    main('sql')