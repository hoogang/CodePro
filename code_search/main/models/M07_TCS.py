import tensorflow as tf
import numpy as np

print('model07_TCS')

# todo 参数配置

# #  序列长度
# self.query_length = 20
# self.code_length =  200
# #  迭代次数
# self.num_epochs = 50
# #  批次大小
# self.batch_size = 256
# #  评测的批次
# self.eval_batch = 256
#
# #  隐层大小
# self.hidden_size = 256
# #  隐藏维数
# self.output_size = 0
# #  丢失率
# self.keep_prob = 0.5
# #  滤波器 [1, 2, 3, 5, 7, 9]
# self.filter_sizes = 0
# #  滤波器个数
# self.n_filters = 0
# #  循环器个数
# self.n_layers = 2
# #  注注意力机制
# self.layer_size = 0
#
# #  词嵌入矩阵
# self.struc2vecs = np.array(struc2vecs).astype(np.float32)
# self.struc2vec_size = 300


# #  优化器的选择
# self.optimizer = 'adam'
# #  正则化
# self.l2_lambda = 0.0001
#
# #  学习率
# self.learning_rate =0.0002
# #  间距值
# self.margin =0.05
#
# self.save_path = './'
#
# self.best_path = './'


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

        # self-attention
        self.num_blocks = 6
        #  注意力头数
        self.num_heads = 6

        self.placeholder_init()

        # 正负样本距离
        self.q_cpos_cosine, self.q_cneg_cosine = self.build(self.struc2vecs)
        # 损失和精确度
        self.total_loss = self.add_loss_op(self.q_cpos_cosine, self.q_cneg_cosine, self.l2_lambda)
        # 训练节点
        self.train_op = self.add_train_op(self.total_loss)


    def placeholder_init(self):
        self.code_pos = tf.placeholder(tf.int32, [None, self.code_len], name='code_point')
        self.query = tf.placeholder(tf.int32, [None, self.query_len], name='query_point')
        self.code_neg = tf.placeholder(tf.int32, [None, self.code_len], name='code_point')

        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
        self.batch_size, self.list_size = tf.shape(self.code_pos)[0], tf.shape(self.code_neg)[1]

    def max_pooling(self, lstm_out):  # (?,200,512)
        #  200, 512
        height, width = lstm_out.get_shape().as_list()[1], lstm_out.get_shape().as_list()[2]
        # (?,#200,#512,1)  (1,#200,#1,1)
        lstm_out = tf.expand_dims(lstm_out, -1)
        # (?,#1,512,1)
        output = tf.nn.max_pool(lstm_out, ksize=[1, height, 1, 1], strides=[1, 1, 1, 1], padding='VALID')
        # (?,512)
        max_pool = tf.reshape(output, [-1, width])

        return max_pool

    # transformer 原始
    def pos_encodings(self, scope, inputs, masking=True, scale=True):

        N = tf.shape(inputs)[0]  # dynamic  bathSize
        T = inputs.get_shape().as_list()[1]  # static  length
        E = inputs.get_shape().as_list()[-1]  # static  embeddingSize

        with tf.variable_scope("pos_encoding" + scope):
            # position indices
            position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)
            # First part of the PE function: sin and cos argument
            position_enc = np.array([
                [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
                for pos in range(T)])
            # Second part, apply the cosine to even columns and sin to odds.
            position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
            position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
            position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (T, E)

            # lookup
            outputs = tf.nn.embedding_lookup(position_enc, position_ind)

            # masks
            if masking:
                outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

            if scale:
                outputs *= E ** 0.5

        return tf.to_float(outputs)

    def scal_attention(self, scope, Q, K, V, dropout_rate):

        with tf.variable_scope("scalatt" + scope):
            # 点积
            outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))
            # 缩放（开平方）
            outputs /= Q.get_shape().as_list()[-1] ** 0.5

            # key masking
            # 对填充的部分进行一个mask，这些位置的attention score变为极小，embedding是padding操作的，
            # 填充的部分其embedding都是0，加起来也是0，我们就会填充一个很小的数。
            key_masks = tf.sign(tf.abs(tf.reduce_sum(K, axis=-1)))
            key_masks = tf.expand_dims(key_masks, 1)
            key_masks = tf.tile(key_masks, [1, tf.shape(Q)[1], 1])

            paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)

            # softmax操作
            outputs = tf.nn.softmax(outputs)
            # attention = tf.transpose(outputs, [0, 2, 1])
            # tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

            # query masking
            query_masks = tf.sign(tf.reduce_sum(tf.abs(Q), axis=-1))
            query_masks = tf.expand_dims(query_masks, -1)  #
            query_masks = tf.tile(query_masks, [1, 1, tf.shape(K)[1]])

            outputs *= query_masks

            # dropout操作
            outputs = tf.nn.dropout(outputs, dropout_rate)

            # weighted sum (context vectors)
            outputs = tf.matmul(outputs, V)

        return outputs

    def normalize(self, inputs, epsilon=1e-8):

        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (0.5))
        outputs = gamma * normalized + beta

        return outputs

    def multi_attention(self, scope, queries, keys, values, num_units=None, num_heads=6, dropout_rate=0.):

        num_units = queries.get_shape().as_list()[-1] if num_units is None else num_units
        # queries (b, length,struc2vec_size)
        with tf.variable_scope("multiatt" + scope):
            #  线性变换
            #  维度(b,length,num_units)
            Q = tf.layers.dense(queries, num_units, use_bias=True)
            #  维度(b,length,num_units)
            K = tf.layers.dense(keys, num_units, use_bias=True)
            #  维度(b,length,num_units)
            V = tf.layers.dense(values, num_units, use_bias=True)

            # 分解和拼接
            Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (n*b, length, struc2vec_size/n)
            K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (n*b, length, struc2vec_size/n)
            V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (n*b, length, struc2vec_size/n)

            # Attention
            outputs = self.scal_attention(scope, Q_, K_, V_, dropout_rate)

            # 恢复拼接
            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

            # 残差链接
            outputs += queries

            # 正则化
            outputs = self.normalize(outputs)

        return outputs

    def feed_forward(self, scope, inputs, num_units):

        with tf.variable_scope("forward" + scope):
            # 全连接full connection输出
            outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)
            outputs = tf.layers.dense(outputs, num_units[1])
            # 残差链接
            outputs += inputs
            # 正则化
            outputs = self.normalize(outputs)

        return outputs

    def build(self, struc2vecs):

        self.Embedding = tf.Variable(tf.to_float(struc2vecs), trainable=False, name='Embedding')

        # 维度(?, 200, 300)
        c_pos_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.Embedding, self.code_pos), self.dropout_keep_prob)

        # 维度(?, 20, 300)
        q_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.Embedding, self.query), self.dropout_keep_prob)

        # 维度(?, 15, 300)
        c_neg_embed = tf.nn.dropout(tf.nn.embedding_lookup(self.Embedding, self.code_neg), self.dropout_keep_prob)

        # Blocks
        for i in range(self.num_blocks):
            with tf.variable_scope("num_blocks_%d" % i):
                # self-attention 自注意力机制
                with tf.variable_scope('multihead_attention') as scope0:

                    att_c_pos = self.multi_attention("code", queries=c_pos_embed,
                                                     keys=c_pos_embed,
                                                     values=c_pos_embed,
                                                     num_heads=self.num_heads,
                                                     dropout_rate=self.dropout_keep_prob
                                                     )

                    att_q = self.multi_attention("query", queries=q_embed,
                                                 keys=q_embed,
                                                 values=q_embed,
                                                 num_heads=self.num_heads,
                                                 dropout_rate=self.dropout_keep_prob
                                                 )

                    scope0.reuse_variables()
                    att_c_neg = self.multi_attention("code", queries=c_neg_embed,
                                                     keys=c_neg_embed,
                                                     values=c_neg_embed,
                                                     num_heads=self.num_heads,
                                                     dropout_rate=self.dropout_keep_prob
                                                     )

                #  前项传播
                with tf.variable_scope('feed_forward') as scope1:
                    # 维度 (?,200,300)
                    ff_c_pos = self.feed_forward("code", att_c_pos,  num_units=[7 * self.struc2vec_size, self.struc2vec_size])
                    # 维度 (?,20,300)
                    ff_q = self.feed_forward("query", att_q, num_units=[7 * self.struc2vec_size, self.struc2vec_size])
                    # 维度 (?,200,300)
                    scope1.reuse_variables()
                    ff_c_neg = self.feed_forward("code", att_c_neg,num_units=[7 * self.struc2vec_size, self.struc2vec_size])

        with tf.variable_scope('attentive_pooling') :

            # 中间矩阵 (512,512)
            U = tf.Variable(
                tf.truncated_normal(shape=[self.struc2vec_size,self.struc2vec_size], dtype=tf.float32, stddev=0.01),
                name='U')

            # 交互结果 (?,20,200)   (?,20,300) * (?,300,300) * (?,300,200)
            G_pos = tf.tanh(
                tf.matmul(tf.matmul(ff_q, tf.tile(tf.expand_dims(U, 0), [self.batch_size, 1, 1])), ff_c_pos,
                          transpose_b=True))

            # 维度 (?, 20,1)
            delta_q_pos = tf.nn.softmax(tf.reduce_max(G_pos, 2, keep_dims=True))
            # 维度 (?,1,200)
            delta_c_pos = tf.nn.softmax(tf.reduce_max(G_pos, 1, keep_dims=True))

            # 维度 (?,512)  (?,300,15)  * (?,20,1)
            feat_q_pos = tf.squeeze(tf.matmul(ff_q, delta_q_pos, transpose_a=True), axis=2)
            # 维度 (?,512)  (?,1,200) * (?,200,300)
            feat_c_pos = tf.squeeze(tf.matmul(delta_c_pos, ff_c_pos), axis=1)
            #########################################################################################################

            # 交互结果 (?,20,200)  (?,20,512) *(?,300,300) * (?,300,200)
            G_neg = tf.tanh(
                tf.matmul(tf.matmul(ff_q, tf.tile(tf.expand_dims(U, 0), [self.batch_size, 1, 1])), ff_c_neg,
                          transpose_b=True))

            # 维度 (?, 20,1)
            delta_q_neg = tf.nn.softmax(tf.reduce_max(G_neg, 2, keep_dims=True))
            # 维度 (?,1,200)
            delta_c_neg = tf.nn.softmax(tf.reduce_max(G_neg, 1, keep_dims=True))

            # 维度 (?,512)  (?,512,20)  * (?,20,1)
            feat_q_neg = tf.squeeze(tf.matmul(ff_q, delta_q_neg, transpose_a=True), axis=2)
            # 维度 (?,512)  (?,1,200) * (?,200,512)
            feat_c_neg = tf.squeeze(tf.matmul(delta_c_neg, ff_c_neg), axis=1)


        # todo 2：COS式
        q_pos_cosine = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(feat_q_pos, dim=1), tf.nn.l2_normalize(feat_c_pos, dim=1)), axis=1)

        q_neg_cosine = tf.reduce_sum(tf.multiply(tf.nn.l2_normalize(feat_q_neg, dim=1), tf.nn.l2_normalize(feat_c_neg, dim=1)), axis=1)

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
