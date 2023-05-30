import tensorflow as tf


print('model06_NJACS')

# todo 参数配置

# #  序列长度
# self.query_length = 20
# self.code_length = 200
# #  迭代次数
# self.num_epochs = 50
# #  批次大小
# self.batch_size = 256
# #  评测的批次
# self.eval_batch = 256
#
# #  隐层大小
# self.hidden_size = 256
# #  保存率
# self.keep_prob = 0.5
# #  循环器个数
# self.n_layers = 2
#  中间维数
# self.layer_size =300

# #  词嵌入矩阵
# self.struc2vecs = np.array(struc2vecs).astype(np.float32)
# self.struc2vec_size = 300
#
# #  优化器的选择
# self.optimizer = 'adam'
# #  正则化
# self.l2_lambda = 0.02
#
# #  学习率
# self.learning_rate = 0.002
# #  间距值
# self.margin = 0.5


class SiameseCSNN(object):
    def __init__(self, config):
        #  序列长度
        self.query_len = config.query_length
        self.code_len = config.code_length
        #  隐藏个数
        self.hidden_size = config.hidden_size
        #  学习率
        self.learning_rate = config.learning_rate
        #  优化器
        self.optimizer = config.optimizer
        #  正则化
        self.l2_lambda = config.l2_lambda
        #  词向量矩阵
        self.struc2vecs = config.struc2vecs
        #  滤波器大小
        self.filter_sizes = config.filter_sizes
        #  滤波器个数
        self.n_filters = config.n_filters
        #  循环器层数
        self.n_layers = config.n_layers
        #  中间维数
        self.layer_size = config.layer_size
        #  间隔阈值
        self.margin = config.margin

        self.struc2vec_size = config.struc2vec_size

        self.placeholder_init()

        # 正负样本距离
        self.q_cpos_cosine, self.q_cneg_cosine = self.build(self.struc2vecs)
        # 损失和精确度
        self.total_loss = self.add_loss_op(self.q_cpos_cosine, self.q_cneg_cosine,self.l2_lambda)
        # 训练节点
        self.train_op = self.add_train_op(self.total_loss)


    def placeholder_init(self):

        self.code_pos = tf.placeholder(tf.int32, [None, self.code_len], name='code_point')
        self.query = tf.placeholder(tf.int32, [None, self.query_len], name='query_point')
        self.code_neg = tf.placeholder(tf.int32, [None, self.code_len], name='code_point')

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size, self.list_size = tf.shape(self.code_pos)[0], tf.shape(self.code_neg)[1]


    def mlp_layer(self, bottom, r_weight, scope):
        """
        全连接层
        """
        assert len(bottom.get_shape().as_list()) == 2

        l_weight = bottom.get_shape().as_list()[1]

        W = tf.get_variable('W' + scope, shape=[l_weight, r_weight],initializer=tf.truncated_normal_initializer(stddev=0.1))

        b = tf.get_variable('b' + scope, shape=[r_weight],initializer=tf.constant_initializer(0.01))


        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)

        return fc


    def max_pooling(self,lstm_out): #(?,200,512)
        #  200, 512
        height, width = lstm_out.get_shape().as_list()[1],lstm_out.get_shape().as_list()[2]

        # (?,#200,#512,1)  (1,#200,#1,1)
        lstm_out = tf.expand_dims(lstm_out, -1)
        # (?,#1,512,1)
        output = tf.nn.max_pool(lstm_out,ksize=[1, height, 1, 1],strides=[1, 1, 1, 1],padding='VALID')

        # (?,512)
        max_pool= tf.reshape(output, [-1, width])

        return max_pool

    def avg_pooling(self, lstm_out):# (?,15,512)

        #  15, 512
        height, width = lstm_out.get_shape().as_list()[1], lstm_out.get_shape().as_list()[2]

        # (?,15,512,1)
        lstm_out = tf.expand_dims(lstm_out, -1)

        # (?,15,512,1) (1,15,1,1)
        output = tf.nn.avg_pool(lstm_out, ksize=[1, height, 1, 1], strides=[1, 1, 1, 1], padding='VALID')

        # (?,512)
        avg_pool= tf.reshape(output, [-1, width])

        return avg_pool


    def atten_xydir_wrapper(self, input_q, input_c,scope):

        # (?,15,512) (?,200,512)

        Wq = tf.get_variable('Wc'+scope,shape=[2*self.hidden_size,self.layer_size],
                            initializer=tf.truncated_normal_initializer(stddev=0.1))

        Wc = tf.get_variable('Wq'+scope,shape=[2*self.hidden_size, self.layer_size],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))

        V  = tf.get_variable('W'+scope,shape=[self.layer_size,1],
                             initializer=tf.truncated_normal_initializer(stddev=0.1))

        #  (?,15,512)  * (512,300)  = (?,15,300)
        fq = tf.einsum('abc,cd->abd', input_q, Wq)
        #  (?,200,512) * (512,300)  = (?,200,300)
        fc = tf.einsum('abc,cd->abd', input_c, Wc)

        re_fq = tf.tile(tf.expand_dims(fq,2),[1,1,self.code_len,1])    #  (?,15,1,300)---(?,15,200,300)
        re_fc = tf.tile(tf.expand_dims(fc,1),[1,self.query_len,1,1])   # (?,1,200,300)---(?,15,200,300)

        # 注意力矩阵（？15,200,300) *(300,1) =（?,15,200)
        M = tf.squeeze(tf.einsum('abcd,df->abcf',tf.tanh(tf.add(re_fq,re_fc)),V),[3])

        # 注意力 (?,20,200) 映射到查询维度和代码维度
        # 维度 (?, 20)
        delta_q = tf.nn.softmax(self.max_pooling(tf.transpose(M,[0,2,1])))
        # 维度 (?, 200)
        delta_c = tf.nn.softmax(self.max_pooling(M))

        # # 维度 (?, 20)
        # delta_q = tf.nn.softmax(tf.reduce_max(M, 2,keep_dims=False))
        # # 维度 (?, 200)
        # delta_c = tf.nn.softmax(tf.reduce_max(M, 1,keep_dims=False))

        # 维度 (?,512)  (?,512,15)  * (?,20,1)
        output_q = tf.squeeze(tf.matmul(input_q,tf.expand_dims(delta_q, -1), transpose_a=True))
        # 维度 (?,512)  (?,512,200) * (?,200,1)
        output_c = tf.squeeze(tf.matmul(input_c,tf.expand_dims(delta_c, -1), transpose_a=True))

        return tf.nn.tanh(output_q), tf.nn.tanh(output_c) , M


    def LSTM(self, x, dropout, scope, hidden_size):
        """
        LSTM编码层
        x: [batch, height, width]   / [batch, step, struc2vec_size]
        output: [batch, height, 2*hidden_size]  / [batch, step, 2*hidden_size]
        """
        input_x = tf.transpose(x, [1, 0, 2])
        input_x = tf.unstack(input_x)

        with tf.variable_scope("cell" + scope):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=dropout)

            multi_cell= tf.contrib.rnn.MultiRNNCell([lstm_cell]*self.n_layers, state_is_tuple=True)

        with tf.variable_scope("decode" + scope):
            output, _ = tf.contrib.rnn.static_rnn(multi_cell,  input_x, dtype=tf.float32)


        output = tf.stack(output)
        output = tf.transpose(output, [1, 0, 2])

        return output


    def BiLSTM(self, x ,dropout, scope, hidden_size):
        """
        BiLSTM编码层
        """
        input_x = tf.transpose(x, [1, 0, 2])
        input_x = tf.unstack(input_x)

        with tf.variable_scope("fcell"+scope):
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell,output_keep_prob=dropout)

            multi_fw_cell = tf.contrib.rnn.MultiRNNCell([lstm_fw_cell] * self.n_layers, state_is_tuple=True)

        with tf.variable_scope("bcell"+scope):
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell,output_keep_prob=dropout)

            multi_bw_cell = tf.contrib.rnn.MultiRNNCell([lstm_bw_cell] * self.n_layers, state_is_tuple=True)

        with tf.variable_scope("decode"+scope):
            output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(multi_fw_cell,multi_bw_cell,input_x,dtype=tf.float32)

        output = tf.stack(output)
        output = tf.transpose(output, [1, 0, 2])

        return output


    def build(self, struc2vecs):

        self.Embedding = tf.Variable(tf.to_float(struc2vecs), trainable=False, name='Embedding')

        # 维度(?, 200, 300)
        c_pos_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.Embedding, self.code_pos), self.dropout_keep_prob)

        # 维度(?, 15, 300)
        q_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.Embedding, self.query), self.dropout_keep_prob)

        # 维度(?, 200, 300)
        c_neg_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.Embedding, self.code_neg), self.dropout_keep_prob)


        # --------------------------------直接循环分开-----------------------------
        # todo 2种不同的循环方式

        # todo 1: 单向LSTM编码

        # with tf.variable_scope('lstm') as scope0:
        #
        #     # LSTM (?, 200, 512)
        #     lstm_c_pos = self.LSTM(c_pos_embed, self.dropout_keep_prob, "code", 2*self.hidden_size*2)
        #
        #     # LSTM (?, 15, 512)
        #     lstm_q = self.LSTM(q_embed, self.dropout_keep_prob, "query", 2*self.hidden_size*2)
        #
        #     scope0.reuse_variables()
        #     # LSTM (?, 200, 512)
        #     lstm_c_neg = self.LSTM(c_neg_embed, self.dropout_keep_prob, "code", 2*self.hidden_size*2)


        # todo 2: 双向LSTM编码

        with tf.variable_scope('bilstm') as scope0:

            # LSTM (?, 200, 512)
            lstm_c_pos = self.BiLSTM(c_pos_embed, self.dropout_keep_prob, "code", self.hidden_size)

            # LSTM (?, 15, 512)
            lstm_q = self.BiLSTM(q_embed, self.dropout_keep_prob, "query", self.hidden_size)

            scope0.reuse_variables()
            # LSTM (?, 200, 512)
            lstm_c_neg = self.BiLSTM(c_neg_embed, self.dropout_keep_prob, "code", self.hidden_size)

        #交互式注意力
        with tf.variable_scope('attention') as scope1:

            # 维度 (?,512,200)  (?,512,15)
            atted_q_pos, atted_c_pos, self.cpos_M = self.atten_xydir_wrapper(lstm_q, lstm_c_pos, 'xydiratt')

            scope1.reuse_variables()  # 复制参数
            # 维度 (?,512,200)  (?,512,15)
            atted_q_neg, atted_c_neg, self.cneg_M = self.atten_xydir_wrapper(lstm_q, lstm_c_neg, 'xydiratt')


        # todo 2：COS式

        q_pos_cosine = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(atted_q_pos, dim=1),
                                                           tf.nn.l2_normalize(atted_c_pos, dim=1)), axis=1)

        q_neg_cosine = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(atted_q_neg, dim=1),
                                                           tf.nn.l2_normalize(atted_c_neg, dim=1)), axis=1)

        return q_pos_cosine, q_neg_cosine


    def margin_loss(self, pos_sim, neg_sim):
        original_loss = self.margin - pos_sim + neg_sim
        l = tf.maximum(tf.zeros_like(original_loss), original_loss)
        loss = tf.reduce_sum(l)
        return loss, l

    def add_loss_op(self, p_sim, n_sim,l2_lambda=0.0001):
        """
        损失节点
        """
        loss, l = self.margin_loss(p_sim, n_sim)

        tv = tf.trainable_variables()
        l2_loss = l2_lambda * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])

        pairwise_loss = loss + l2_loss
        tf.summary.scalar('pairwise_loss', pairwise_loss)
        self.summary_op = tf.summary.merge_all()
        return pairwise_loss

    def add_train_op(self, loss):
        """
        训练节点
        """
        with tf.name_scope('train_op'):
            # 记录训练步骤
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            opt = tf.train.AdamOptimizer(self.learning_rate)
            train_op = opt.minimize(loss, self.global_step)
            return train_op
