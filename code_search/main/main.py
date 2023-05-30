# -*- coding: utf-8 -*-
import os
import time
import logging

# 数据处理
from data_helper import *
# 模型选择
#from models.M03_PSCS import SiameseCSNN
from models.M04_AdaCS import SiameseCSNN


# 额外处理
from model_utils import *
# 参数配置
import argparse


# 创建一个logger
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)


class NNConfig(object):
    def __init__(self, struc2vecs=None,tokidf2vecs=None):

        #  序列长度
        self.query_length = 20
        self.code_length  = 200
        #  迭代次数
        self.num_epochs = 50
        #  批次大小
        self.batch_size = 256
        #  评测的批次
        self.eval_batch = 256

        #  隐层大小
        self.hidden_size = 256
        #  丢失率
        self.keep_prob = 0.5
        #  循环器层数
        self.n_layers =  2
        #  中间维数
        self.layer_size = 200

        #  词嵌入矩阵
        self.struc2vecs = np.array(struc2vecs).astype(np.float32)
        self.struc2vec_size = 300

        #  词IDF矩阵
        self.tokidf2vecs = np.array(tokidf2vecs).astype(np.float32)

        #  优化器的选择
        self.optimizer = 'adam'
        #  正则化
        self.l2_lambda = 0.02

        #  学习率
        self.learning_rate = 0.02
        #  间距值
        self.margin = 0.5

        self.save_path='./'

        self.best_path='./'


def evaluate(sess, model, corpus, config):
    """
    using corpus to evaluate the session and model’s MAP MRR
    """
    iterator = Iterator(corpus)

    count = 0
    total_qids =  []
    total_cids =  []
    total_preds = []
    total_labels =[]
    total_loss = 0


    for batch_x in iterator.next(config.eval_batch, shuffle=True):
        # 查询id, 样例, 查询长度, 样例长度
        batch_qids, batch_q, batch_cids, batch_c, batch_qmask, batch_cmask, labels = zip(*batch_x)

        batch_q = np.asarray(batch_q)
        batch_c = np.asarray(batch_c)

        #距离
        q_cp_cosine, loss = sess.run([model.q_cpos_cosine, model.total_loss],
                                     feed_dict ={model.code_pos:batch_c,
                                                model.query:batch_q,
                                                model.code_neg:batch_c,
                                                model.dropout_keep_prob: 1.0})

        total_loss += loss

        count += 1

        total_qids.append(batch_qids)
        total_cids.append(batch_cids)
        total_preds.append(q_cp_cosine)
        total_labels.append(labels)


    total_qids   = np.concatenate(total_qids, axis=0)
    total_cids   = np.concatenate(total_cids, axis=0)
    total_preds  = np.concatenate(total_preds, axis=0)
    total_labels = np.concatenate(total_labels,axis=0)

    # 评价指标
    recall_1, recall_2, mrr, l_mrr, h_mrr,  frank, l_frank, h_frank, prec_1, prec_3, prec_5= eval_metric(total_qids, total_cids, total_preds, total_labels)
    # 平均损失
    ave_loss = total_loss/count

    return ave_loss, recall_1, recall_2, mrr, l_mrr, h_mrr, frank, l_frank, h_frank, prec_1, prec_3, prec_5



def pred(pred_corpus,config):

    with tf.Session() as sess:
        # 搜索模型导入
        model = SiameseCSNN(config)

        # 恢复模型参数
        print('.........................加载模型的参数....................')
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint(config.best_path))

        print('#######################测试中开始评测######################')

        pred_loss, pred_recall_1, pred_recall_2, pred_mrr, pred_l_mrr, pred_h_mrr, pred_frank, pred_l_frank, pred_h_frank, pred_prec_1, pred_prec_3, pred_prec_5 = evaluate(sess, model, pred_corpus, config)

        # 评测指标打印
        print("-- pred recall@1: %.4f -- pred_recall_2 %.4f \n"
              "-- pred mrr: %.4f -- pred mrr [l_mrr: %.4f,h_mrr: %.4f] \n"
              "-- pred frank: %.4f -- pred [l_frank: %.4f,h_frank: %.4f] \n"
              "-- pred prec_1: %.4f -- pred prec_3: %.4f -- pred prec_5: %.4f" % (
                pred_recall_1, pred_recall_2, pred_mrr,
                pred_l_mrr, pred_h_mrr,
                pred_frank,pred_l_frank,pred_h_frank,
                pred_prec_1,pred_prec_3,pred_prec_5))


def train(train_corpus, valid_corpus, test_corpus, config):

    iterator = Iterator(train_corpus)

    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    if not os.path.exists(config.best_path):
        os.makedirs(config.best_path)

    with tf.Session() as sess:
        # 训练的主程序模块
        print('#######################开始训练和评价#######################')

        # 训练开始的时间
        start_time = time.time()

        model = SiameseCSNN(config)

        saver = tf.train.Saver(tf.global_variables(), max_to_keep=10)
        best_saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        ckpt = tf.train.get_checkpoint_state(config.save_path)

        print('#######################配置TensorBoard#######################')

        summary_writer = tf.summary.FileWriter(config.save_path, graph=sess.graph)

        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('..........重新加载模型的参数..........')
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('..........创建新建模型的参数..........')
            sess.run(tf.global_variables_initializer())

        # 计算模型的参数个数
        total_parameters = count_parameters()
        print('统计全部的参数个数:%d'%total_parameters)

        current_step = 0

        best_mrr_val = 0.0
        best_frank_val = 10.0

        best_re1_val = 0.0
        best_re2_val = 0.0


        for epoch in range(config.num_epochs):
            # 开始迭代
            print("----- Epoch {}/{} -----".format(epoch+1, config.num_epochs))

            count = 0

            for batch_x in iterator.next(config.batch_size, shuffle=True):
                # 正列, 查询,  负例, 正例长度, 查询长度，负例长度
                batch_c_pos, batch_q, batch_c_neg, batch_c_pos_mask,batch_qmask,batch_c_neg_mask = zip(*batch_x)

                batch_c_pos = np.asarray(batch_c_pos)
                batch_q  = np.asarray(batch_q)
                batch_c_neg = np.asarray(batch_c_neg)

                _, loss, summary= sess.run([model.train_op, model.total_loss, model.summary_op],
                                           feed_dict={model.code_pos:batch_c_pos,
                                                      model.query:batch_q,
                                                      model.code_neg:batch_c_neg,  # 填补参数
                                                      model.dropout_keep_prob: config.keep_prob})
                count += 1
                current_step += 1

                if count % 50 == 0:
                    print('[epoch {}, batch {}], Loss:{}'.format(epoch+1, count, loss))

                summary_writer.add_summary(summary, current_step)

            if valid_corpus is not None:

                #valid_loss和test_loss看起来一样，只是保留四位小数

                valid_loss, valid_recall_1, valid_recall_2, valid_mrr, valid_l_mrr, valid_h_mrr, valid_frank, valid_l_frank, valid_h_frank, valid_prec_1, valid_prec_3, valid_prec_5= evaluate(sess, model, valid_corpus, config)
                print("-- epoch: %d -- valid Loss: %.4f \n-- valid recall@1: %.4f -- valid mrr: %.4f"% (epoch+1, valid_loss, valid_recall_1,valid_mrr))


                logger.info("\nEval:")
                logger.info("-- epoch: %d -- valid Loss: %.4f \n-- valid recall@1: %.4f -- valid mrr: %.4f"% (epoch+1, valid_loss, valid_recall_1,valid_mrr))

                #################################################################################################################################################
                test_loss, test_recall_1, test_recall_2, test_mrr, test_l_mrr, test_h_mrr, test_frank, test_l_frank, test_h_frank, test_prec_1, test_prec_3, test_prec_5= evaluate(sess, model, test_corpus, config)
                print("-- epoch: %d -- test Loss: %.4f \n-- test recall@1: %.4f  -- test mrr: %.4f"% (epoch+1, test_loss, test_recall_1,test_mrr))

                # 打印
                logger.info("\nTest:")
                logger.info("-- epoch: %d -- test Loss: %.4f \n-- test recall@1: %.4f  -- test mrr: %.4f"% (epoch+1, test_loss, test_recall_1,test_mrr))

                #实时模型的文件
                checkpoint_path =os.path.join(config.save_path, 'mrr_{:.4f}_{}.ckpt'.format(test_mrr, current_step))
                #最好模型的文件
                bestcheck_path = os.path.join(config.best_path, 'mrr_{:.4f}_{}.ckpt'.format(test_mrr, current_step))

                #保存的地址
                saver.save(sess, checkpoint_path, global_step=epoch)

                # 设置最佳模型的阈值,越大越好
                if  test_mrr > best_mrr_val or test_frank < best_frank_val:
                    # 以MRR作为模型保存指标
                    best_mrr_val = test_mrr
                    best_frank_val = test_frank
                    best_saver.save(sess, bestcheck_path, global_step=epoch)

                if  test_recall_1 > best_re1_val:
                    best_re1_val = test_recall_1

                # 最佳排名recall@2,越大越好
                if test_recall_2 > best_re2_val:
                    best_re2_val = test_recall_2


        # 训练结束的时间
        end_time=time.time()

        print('训练稳定后程序运行的时间：%s 秒'%(end_time-start_time))

        logger.info("\nBest and Last:")
        logger.info('-- best_recall@1 %.4f -- best_recall@2 %.4f -- best_mrr %.4f --best_frank %.4f'% (best_re1_val,best_re2_val,best_mrr_val,best_frank_val))

        print('-- best_recall@1 %.4f -- best_recall@2 %.4f \n-- best_mrr %.4f --best_frank %.4f'% (best_re1_val,best_re2_val,best_mrr_val,best_frank_val))



def main(args):

    # 按照PCI_BUS_ID顺序从0开始排列GPU设备
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # 使用第1, 2块GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    # 不同候选类型 single  mutiple  nlqfpro
    candi_type = 'codepro'

    # 不同数据类型  mucqc  staqc samqc  sicqc  csnqc
    data_type  = 'mixqc'

    # 不同语言的类型   sql   python
    lang_type = 'python'

    # 测试集类型     codemf  sattmf  codesn  human
    pred_type = 'sattmf'

    # 模型种类
    model_id = 4

    # 创建一个handler，用于写入日志文件
    timestamp = str(time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time())))
    modelstamp = 'model-%s-%s-%s-%d-' % (candi_type, data_type, lang_type, model_id) + timestamp

    # 加载记录文件
    fh = logging.FileHandler('./log/' + modelstamp + '.txt')
    fh.setLevel(logging.DEBUG)

    # 定义handler的输出格式
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s] ## %(message)s')
    fh.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)

    # 序列长度
    max_q_length = 20    #查询
    max_c_length = 200   #代码

    # 读取训练路径
    pair_path = './corpus/pair'
    # 读取其他路径
    data_path =  './corpus/data'
    # 预测文件路径
    pred_path = '../pred_data'

    # 手工设置路径
    save_path = "./ckpt/%s_%s_%s_%d/checkpoint"%(candi_type,data_type,lang_type,model_id)
    best_path = "./ckpt/%s_%s_%s_%d/bestval"%(candi_type,data_type,lang_type,model_id)

    vocab_path = './vocab'

    # 词典文件
    vocab = os.path.join(vocab_path, '%s_%s_%s_clean_vocab.txt' % (candi_type, data_type, lang_type))

    # -----------------------------------------------加载词向量矩阵-------------------------
    # 词向量文件
    struc2vec_file = os.path.join('./struc2vec','%s_%s_%s_struc2vec.pkl'%(candi_type,data_type,lang_type))
    # 词向量编码 (?,300)
    struc2vecs = load_embedding(struc2vec_file)

    print('struc2vec词向量维度',np.shape(struc2vecs))
    # -----------------------------------------------加载词IDF矩阵-------------------------
    # IDF文件
    tokidf2vec_file = os.path.join('./adapt2vec','%s_%s_%s_tokidf2vec.pkl'%(candi_type,data_type,lang_type))
    # 词向量编码 (?,300)
    tokidf2vecs = load_embedding(tokidf2vec_file)

    print('tokidf2vecs词向量维度',np.shape(tokidf2vecs))

    # 网络参数
    config = NNConfig(struc2vecs=struc2vecs,tokidf2vecs=tokidf2vecs)

    config.query_length = max_q_length
    config.code_length = max_c_length

    config.save_path = save_path
    config.best_path = best_path

    # 读取训练数据
    train_file = os.path.join(pair_path,'%s_%s_%s_train_triplets.txt'%(candi_type,data_type,lang_type))
    # 读取验证数据
    valid_file = os.path.join(data_path,'%s_%s_%s_parse_valid.txt'%(candi_type,data_type,lang_type))
    # 读取测试数据
    test_file  = os.path.join(data_path,'%s_%s_%s_parse_test.txt'%(candi_type,data_type,lang_type))

    # 读取预测数据
    pred_file =  os.path.join(pred_path,'%s_%s_pred.txt'%(pred_type,lang_type))

    # [q_id, query, c_id, code, q_mask, c_mask, label]
    train_transform = transform_train(train_file, vocab)
    # 转换ID
    valid_transform = transform(valid_file, vocab)
    test_transform  = transform(test_file, vocab)
    # 转换ID
    pred_transform  = transform(pred_file, vocab)

    # padding处理
    train_corpus = load_train_data(train_transform, max_q_length, max_c_length)
    # padding处理
    valid_corpus = load_data(valid_transform, max_q_length, max_c_length, keep_ids=True)
    test_corpus  = load_data(test_transform, max_q_length, max_c_length, keep_ids=True)

    # 预测集处理
    pred_corpus  = load_data(pred_transform, max_q_length, max_c_length, keep_ids=True)

    # 加载训练参数
    if args.train:
        train(train_corpus, valid_corpus, test_corpus, config)

    # 加载预测参数
    elif args.pred:
        pred(pred_corpus, config)



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train",help="whether to train", action='store_true')
    parser.add_argument("--pred", help="whether to test", action='store_true')
    args = parser.parse_args()
    # 加载参数配置
    main(args)
