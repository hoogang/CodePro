import tensorflow as tf

print('model04_AdaCS')

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
        #  词IDF矩阵
        self.tokidf2vecs  = config.tokidf2vecs
        #  循环器层数
        self.n_layers = config.n_layers
        #  注意力大小
        self.layer_size = config.layer_size
        #  间隔阈值
        self.margin = config.margin

        self.struc2vec_size = config.struc2vec_size

        self.placeholder_init()

        # 正负样本距离
        self.q_cpos_cosine, self.q_cneg_cosine = self.build(self.struc2vecs,self.tokidf2vecs)
        # 损失和精确度
        self.total_loss = self.add_loss_op(self.q_cpos_cosine, self.q_cneg_cosine, self.l2_lambda)
        # 网络训练节点
        self.train_op   = self.add_train_op(self.total_loss)

    def placeholder_init(self):
        self.code_pos = tf.placeholder(tf.int32, [None, self.code_len], name='code_point')
        self.query = tf.placeholder(tf.int32, [None, self.query_len], name='query_point')
        self.code_neg = tf.placeholder(tf.int32, [None, self.code_len], name='code_point')

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.batch_size, self.list_size = tf.shape(self.code_pos)[0], tf.shape(self.code_neg)[1]

    def mlp_layer(self, bottom, n_weight, scope):
        """
        全连接层
        """
        assert len(bottom.get_shape().as_list()) == 2

        n_prev_weight = bottom.get_shape().as_list()[1]

        W = tf.Variable(
            tf.truncated_normal(shape=[n_prev_weight, n_weight], dtype=tf.float32, stddev=0.01, name='W' + scope))

        b = tf.Variable(tf.constant(0.01, shape=[n_weight], dtype=tf.float32, name='b' + scope))

        fc = tf.nn.bias_add(tf.matmul(bottom, W), b)

        return fc

    def LSTM(self, x, dropout, scope, hidden_size):
        """
        LSTM编码层
        """
        input_x = tf.transpose(x, [1, 0, 2])
        input_x = tf.unstack(input_x)

        with tf.variable_scope("cell" + scope):
            lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
            lstm_cell = tf.contrib.rnn.DropoutWrapper(lstm_cell, output_keep_prob=dropout)

            multi_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell] * self.n_layers, state_is_tuple=True)

        with tf.variable_scope("decode" + scope):
            output, _ = tf.contrib.rnn.static_rnn(multi_cell, input_x, dtype=tf.float32)

        output = tf.stack(output)
        output = tf.transpose(output, [1, 0, 2])

        return output

    def BiLSTM(self, x, dropout, scope, hidden_size):
        """
        BiLSTM编码层 (?,20,300) (20,?,300)
        """
        input_x = tf.transpose(x, [1, 0, 2])
        input_x = tf.unstack(input_x)

        with tf.variable_scope("fcell" + scope):
            lstm_fw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
            lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(lstm_fw_cell, output_keep_prob=dropout)

            multi_fw_cell = tf.contrib.rnn.MultiRNNCell([lstm_fw_cell] * self.n_layers, state_is_tuple=True)

        with tf.variable_scope("bcell" + scope):
            lstm_bw_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)
            lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(lstm_bw_cell, output_keep_prob=dropout)

            multi_bw_cell = tf.contrib.rnn.MultiRNNCell([lstm_bw_cell] * self.n_layers, state_is_tuple=True)

        with tf.variable_scope("decode" + scope):
            # fw_state, bw_state  前项状态，后项状态
            output, _, _ = tf.contrib.rnn.static_bidirectional_rnn(multi_fw_cell, multi_bw_cell, input_x,
                                                                   dtype=tf.float32)

        output = tf.stack(output)
        output = tf.transpose(output, [1, 0, 2])

        return output

    def build(self, struc2vecs,tokidf2vecs):

        self.Embedding = tf.Variable(tf.to_float(struc2vecs), trainable=False, name='Embedding')
        self.Tokidfvec = tf.Variable(tf.to_float(tokidf2vecs), trainable=False, name='Embedding')

        # 维度 (?, 200, 300)
        c_pos_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.Embedding, self.code_pos), self.dropout_keep_prob)
        # 维度 （？，200,1）
        c_pos_idfv = tf.nn.dropout(tf.nn.embedding_lookup(self.Tokidfvec, self.code_pos), self.dropout_keep_prob)

        # 维度 (?, 20, 300)
        q_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.Embedding, self.query), self.dropout_keep_prob)

        # 维度 (?, 200, 300)
        c_neg_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.Embedding, self.code_neg), self.dropout_keep_prob)
        # 维度 （？，200,1）
        c_neg_idfv = tf.nn.dropout(tf.nn.embedding_lookup(self.Tokidfvec, self.code_neg), self.dropout_keep_prob)

        # ----------------------------查询代码矩阵交互-----------------------------
        with tf.variable_scope('attentive_matrix') as scope0:

            #  tf.nn.l2_normalize  dim=0 按列，dim=1 按行

            # 交互结果  (?,20,200)  (?,20,300)  * (?,300,200)
            M_qcpos_matrix = tf.matmul(tf.nn.l2_normalize(q_embed,dim=2),tf.nn.l2_normalize(c_pos_embed,dim=2),transpose_b=True)
            # 维度 (?,21,200)   (?,20,200) + (?,1,200)
            M_qcpos_embed = tf.concat([tf.transpose(c_pos_idfv,[0,2,1]),M_qcpos_matrix],1)

            # 交互结果  (?,20,200)  (?,20,300)  * (?,300,200)
            M_qcneg_matrix = tf.matmul(tf.nn.l2_normalize(q_embed,dim=2),tf.nn.l2_normalize(c_neg_embed,dim=2),transpose_b=True)
            # 维度(?,21,200)   (?, 20, 200) + (?,1,200)
            M_qcneg_embed = tf.concat([tf.transpose(c_neg_idfv,[0,2,1]),M_qcneg_matrix],1)

        # ----------------------------直接RNN循环分开-----------------------------
        # todo 2种不同的循环方式

        # todo 1: 单向LSTM编码
        # with tf.variable_scope('lstm_encode') as scope1:

        #     # 维度 (？,21,200)
        #     lstm_qcpos_embed = self.LSTM(M_qcpos_embed, self.dropout_keep_prob, "query-code", self.hidden_size)
        #     scope1.reuse_variables()
        #     # 维度 (？,21,200)
        #     lstm_qcneg_embed = self.LSTM(M_qcneg_embed, self.dropout_keep_prob, "query-code", self.hidden_size)

        # todo 2: 双向LSTM编码
        with tf.variable_scope('bilstm_encode') as scope1:

            # 维度 (？,21,512)
            lstm_qcpos_embed = self.BiLSTM(M_qcpos_embed, self.dropout_keep_prob, "qcode_embed", self.hidden_size)

            scope1.reuse_variables()
            # 维度 (？,21,512)
            lstm_qcneg_embed = self.BiLSTM(M_qcneg_embed, self.dropout_keep_prob, "qcode_embed", self.hidden_size)

        # todo 2：dense式
        with tf.variable_scope('dense_output') as scope2:

            # 维度 （？，21,1)---(?,21)
            q_pos_dense = tf.squeeze(tf.layers.dense(lstm_qcpos_embed, 1, activation=tf.nn.sigmoid, name='qcode_dense'),axis=-1)
            # 维度  (?,1)
            q_cpos_cosine = tf.reduce_mean(q_pos_dense, axis=1)

            scope2.reuse_variables()
            # 维度 （？，21,1)---(?,21)
            q_neg_dense = tf.squeeze(tf.layers.dense(lstm_qcneg_embed, 1, activation=tf.nn.sigmoid, name='qcode_dense'),axis=-1)
            # 维度  (?,1)
            q_cneg_cosine = tf.reduce_mean(q_neg_dense, axis=1)

            return q_cpos_cosine, q_cneg_cosine

    def margin_loss(self, pos_sim, neg_sim):
        original_loss = self.margin - pos_sim + neg_sim
        l = tf.maximum(tf.zeros_like(original_loss), original_loss)
        loss = tf.reduce_sum(l)
        return loss, l

    def add_loss_op(self, p_sim, n_sim, l2_lambda=0.0001):
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

            #decayed_learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, 10, 0.96, staircase=True)
            #opt = tf.train.AdamOptimizer(decayed_learning_rate)

            opt = tf.train.AdamOptimizer(self.learning_rate)
            train_op = opt.minimize(loss, self.global_step)

            return train_op
