
import os
import pickle
import numpy as np

import random
random.seed(42)

from keras.preprocessing.sequence import pad_sequences


class C2labelbyCodeSa:

    def __init__(self, confs):

        self._buckets = confs.get('buckets')
        self._buckets_text_max = (max([i for i, _, _, _ in self._buckets]), max([j for _, j, _, _ in self._buckets]))
        self._buckets_code_max = (max([i for _, _, i, _ in self._buckets]), max([j for _, _, _, j in self._buckets]))

        # 模型位置
        self.params_path = confs.get('params_path')
        # 训练参数
        self.train_params = confs.get('train_params', dict())
        # 数据读取
        self.data_params = confs.get('data_params', dict())
        # 模型名字
        self.model_params = confs.get('model_params', dict())
        self._eval_sets = None

    def load_pickle(self, filename):
        with open(filename, 'rb') as f:
            word_dict = pickle.load(f)
        return word_dict

    ##### Padding #####
    def pad(self, data, len=None):
        return pad_sequences(data, maxlen=len, padding='post', truncating='post', value=0)

    def load_model(self, model, epoch, d12, d3, d4, d5, r):
        print(self.params_path)
        print( "{}/pysparams_d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(self.params_path, d12, d3, d4, d5, r, epoch))

        assert os.path.exists(
            "{}/pysparams_d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(self.params_path,d12, d3, d4, d5, r, epoch)) \
            ,"Weights at epoch {:d} not found".format(epoch)

        model.load("{}/pysparams_d12={}_d3={}_d4={}_d5={}_r={}_epo={:d}_class.h5".format(self.params_path, d12, d3, d4, d5, r, epoch))


    def process_matrix(self,inputs, trans_length):
        inputs_trans = np.split(inputs, trans_length, axis=1)
        processed_inputs = []
        for item in inputs_trans:
            item_trans = np.squeeze(item, axis=1).tolist()
            processed_inputs.append(item_trans)
        return processed_inputs

    def get_data(self,path):

        data = self.load_pickle(path)

        text_block_length, text_word_length, query_word_length, code_token_length = 2, 100, 25, 350
        text_blocks = self.process_matrix(np.array([samples_term[1] for samples_term in data]),text_block_length)

        text_S1 = text_blocks[0]
        text_S2 = text_blocks[1]

        code_blocks = self.process_matrix(np.array([samples_term[2] for samples_term in data]), text_block_length-1)
        code = code_blocks[0]

        queries = [samples_term[3] for samples_term in data]
        labels = [samples_term[5] for samples_term in data]
        ids = [samples_term[0] for samples_term in data]

        return text_S1, text_S2, code, queries, labels, ids

    #给语料打codesa标签
    def u2l_codesa(self, model, format_path, label_path):

        with open(label_path, 'r')as f:
            pre = eval(f.read())
        f.close()

        my_pre1 = pre[1]  # saihnn_label
        my_pre2 = pre[2]  # textsa_label

        total_label = []
        text_S1, text_S2, code, queries, labels, ids1 = self.get_data(format_path)
        labelpred = model.predict([np.array(text_S1), np.array(text_S2), np.array(code), np.array(queries)],batch_size=100)
        labelpred1 = np.argmax(labelpred, axis=1)

        total_label.append(ids1)
        total_label.append(my_pre1)
        total_label.append(my_pre2)
        total_label.append(labelpred1.tolist())
        f = open(label_path, "w")
        f.write(str(total_label))
        f.close()
        print("codesa标签已打完")
