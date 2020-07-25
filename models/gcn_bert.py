# -*- coding: utf-8 -*-
#表示GCN输出后，每一个target对应一个结果。
import tensorflow as tf
import numpy as np
from IPython import embed
from models.nn_layer import dynamic_rnn, softmax_layer, bi_dynamic_rnn, reduce_mean_with_len,WXA_Relu,WXbA_Relu
from models.att_layer import dot_produce_attention_layer, bilinear_attention_layer, mlp_attention_layer, Mlp_attention_layer


class GCN_BERT(object):
    def __init__(self, sequence_length, target_sequence_length,targets_num_max, num_classes, word_embedding_dim, l2_reg_lambda=0.0,
                 num_hidden=100):
        #tf.set_random_seed(-1)
        # PLACEHOLDERS
        rand_base = 0.01
        self.input_x = tf.compat.v1.placeholder(tf.float32, [None, sequence_length,word_embedding_dim], name="input_x")  # X - The Data
        self.input_target = tf.compat.v1.placeholder(tf.float32, [None, target_sequence_length,word_embedding_dim], name="input_x")  # The target
        self.input_targets_all = tf.compat.v1.placeholder(tf.float32, [None,targets_num_max, target_sequence_length,word_embedding_dim], name="input_x")  #All the targets

        self.sen_len = tf.compat.v1.placeholder(tf.int32, None, name='sen_len')#lens of sentence
        self.target_len = tf.compat.v1.placeholder(tf.int32, None, name='target_len')#lens of target
        with tf.name_scope('targets_all_len'):
            self.targets_all_len_a = tf.compat.v1.placeholder(tf.int32, [None,targets_num_max],name="targets_all_len")
            batch_size = tf.shape(self.input_x)[0]
            self.targets_all_len = []
            for i in range(targets_num_max):
                targets_i_len = tf.slice(self.targets_all_len_a, [0, i], [batch_size, 1])
                # targets_i_len = self.targets_all_len_a[:,i]
                self.targets_all_len.append(tf.squeeze(targets_i_len))              #lens of every target
        self.targets_num = tf.compat.v1.placeholder(tf.int32, None, name='targets_num')     #The number os targets
        self.relate_cross = tf.compat.v1.placeholder(tf.float32, [None,targets_num_max, targets_num_max], name='relate_cross')  #the relation between targets
        self.relate_self = tf.compat.v1.placeholder(tf.float32, [None, targets_num_max, targets_num_max], name='relate_self')
        self.target_which = tf.compat.v1.placeholder(tf.float32, [None, targets_num_max, ], name='which_position')
        self.target_position = tf.compat.v1.placeholder(tf.float32, [None, sequence_length], name='target_position')
        with tf.name_scope('targets_all_position'):
            self.targets_all_position_a = tf.compat.v1.placeholder(tf.float32, [None,targets_num_max,sequence_length],name="targets_all_position")
            self.targets_all_position = []
            for i in range(targets_num_max):
                targets_i_len = self.targets_all_position_a[:, i,:]
                self.targets_all_position.append(tf.squeeze(targets_i_len))
        self.input_y = tf.compat.v1.placeholder(tf.float32, [None, num_classes], name="input_y")  # Y - The Lables
        self.dropout_keep_prob = tf.compat.v1.placeholder(tf.float32, name="dropout_keep_prob")  # Dropout

        l2_loss = tf.constant(0.0)  # Keeping track of l2 regularization loss

        # 1. EMBEDDING LAYER ################################################################

        # Embedding for the context
        with tf.name_scope("embedded_sen"):
            self.embedded_sen = self.input_x    #(?,78,768)
            self.embedded_sen = tf.compat.v1.nn.dropout(self.embedded_sen, keep_prob=self.dropout_keep_prob)
            embedding_size = word_embedding_dim
            print('embedding_size {}'.format(embedding_size))
            num_hidden = word_embedding_dim
        # Embedding for the target
        with tf.name_scope("embedding_target"):
            self.embedded_target = self.input_target    #(?,21,768)
            self.embedded_target = tf.compat.v1.nn.dropout(self.embedded_target, keep_prob=self.dropout_keep_prob)
        # Embedding for all targets
        with tf.name_scope("embedding_targets"):
            self.embedded_targets_all = list(range(targets_num_max))
            for i in range(targets_num_max):
                #get a target
                self.embedded_target_i = self.input_targets_all[:,i,:,:]
                self.embedded_target_i = tf.compat.v1.nn.dropout(self.embedded_target_i, keep_prob=self.dropout_keep_prob)
                self.embedded_targets_all[i] = self.embedded_target_i   #13*(?,21,300)

        num_hidden = 300
        #2. LSTM LAYER ######################################################################
        # Bi-LSTM for the context
        with tf.name_scope("Bi-LSTM_sentence"):
            cell = tf.compat.v1.nn.rnn_cell.LSTMCell
            self.LSTM_Hiddens_sen = bi_dynamic_rnn(cell, self.embedded_sen, num_hidden, self.sen_len,
                                           sequence_length, 'bi-lstm-sentence' ,'all',
                                           dropout = True, dropout_prob= self.dropout_keep_prob) #(?,78,600)
            pool_sen = reduce_mean_with_len(self.LSTM_Hiddens_sen, self.sen_len)

        # Bi-LSTM for the targets
        with tf.compat.v1.variable_scope("Bi-LSTM_targets") as scope:
            self.LSTM_targets_all = list(range(targets_num_max))
            poor_targets_all = list(range(targets_num_max))
            for i in range(targets_num_max):
                cell = tf.compat.v1.nn.rnn_cell.LSTMCell
                self.LSTM_targets_all[i] = bi_dynamic_rnn(cell, self.embedded_targets_all[i], num_hidden, self.targets_all_len[i],
                                                target_sequence_length, 'bi-lstm-targets', 'all',
                                                dropout=True, dropout_prob=self.dropout_keep_prob)  # (?,21,600)
                poor_targets_all[i] = reduce_mean_with_len(self.LSTM_targets_all[i], self.targets_all_len[i])
                scope.reuse_variables()


        # 3. Attention LAYER ######################################################################
        # all targets to sentence attention
        with tf.compat.v1.variable_scope("Attention-targets_all2sentence") as scope:
            self.outputs_ss = list(range(targets_num_max))        #all the target attention for the sentence
            self.outputs_ts = list(range(targets_num_max))
            for i in range(targets_num_max):
                att_s_i = bilinear_attention_layer(self.LSTM_targets_all[i], pool_sen, self.targets_all_len[i], 2 * num_hidden ,l2_reg_lambda,
                                             random_base = rand_base, layer_id = 'tar')
                self.outputs_ss[i] = tf.squeeze(tf.matmul(att_s_i, self.LSTM_targets_all[i]),axis=1)   #13*(?,600)


                #position
                target_position_i = tf.expand_dims(self.targets_all_position[i], 2)  # (?,78,1)
                LSTM_Hiddens_sen_i = tf.multiply(self.LSTM_Hiddens_sen, target_position_i)

                att_s_i = bilinear_attention_layer(LSTM_Hiddens_sen_i, self.outputs_ss[i], self.sen_len, 2 * num_hidden ,l2_reg_lambda,
                                             random_base = rand_base, layer_id = 'sen')
                self.outputs_ts[i] = tf.squeeze(tf.matmul(att_s_i, self.LSTM_Hiddens_sen), axis=1)
                scope.reuse_variables()


        with tf.name_scope("targets_gather"):
            self.targets_concat = tf.concat([tf.expand_dims (i ,axis = 2) for i in self.outputs_ts], axis=2) #(?,600,13)

        # 4. GCN LAYER ######################################################################
        with tf.name_scope('GCN_layer1'):
            W_cross = tf.Variable(tf.random.uniform([2 * num_hidden, 2 * num_hidden],-rand_base,rand_base),name = 'W_cross')
            b_cross = tf.Variable(tf.random.uniform([2 * num_hidden],-rand_base,rand_base),name = 'b_cross')
            W_self = tf.Variable(tf.random.uniform([2 * num_hidden, 2 * num_hidden],-rand_base,rand_base),name = 'W_self')
            b_self = tf.Variable(tf.random.uniform([2 * num_hidden], -rand_base, rand_base), name='b_self')
            GCN1_cross = WXbA_Relu(self.targets_concat,self.relate_cross,W_cross,b_cross)
            GCN1_self = WXbA_Relu(self.targets_concat,self.relate_self,W_self,b_self)
            GCN1_out = GCN1_cross+GCN1_self     #(?,600,13)
        with tf.name_scope('GCN_layer2'):
            W_cross = tf.Variable(tf.random.uniform([2 * num_hidden, 2 * num_hidden],-rand_base,rand_base),name = 'W_cross')
            b_cross = tf.Variable(tf.random.uniform([2 * num_hidden],-rand_base,rand_base),name = 'b_cross')
            W_self = tf.Variable(tf.random.uniform([2 * num_hidden, 2 * num_hidden],-rand_base,rand_base),name = 'W_self')
            b_self = tf.Variable(tf.random.uniform([2 * num_hidden], -rand_base, rand_base), name='b_self')
            GCN2_cross = WXbA_Relu(GCN1_out,self.relate_cross,W_cross,b_cross)
            GCN2_self = WXbA_Relu(GCN1_out,self.relate_self,W_self,b_self)
            GCN2_out = GCN2_cross+GCN2_self        #(?,600,13)
        # with tf.name_scope('GCN_layer3'):
        #     # W_cross = tf.Variable(tf.random.uniform([2 * num_hidden, 3 ],-rand_base,rand_base),name = 'W_cross')
        #     # W_self = tf.Variable(tf.random.uniform([2 * num_hidden, 3 ],-rand_base,rand_base),name = 'W_self')
        #     W_cross = tf.Variable(tf.random_normal([2 * num_hidden, num_hidden ]),name = 'W_cross')
        #     W_self = tf.Variable(tf.random_normal([2 * num_hidden, num_hidden ]),name = 'W_self')
        #     GCN3_cross = WXA_Relu(W_cross,GCN2_out,self.relate_cross)
        #     GCN3_self = WXA_Relu(W_self,GCN2_out,self.relate_self)
        #     GCN3_out = GCN3_cross+GCN3_self        #(?,600,13)
        # with tf.name_scope('GCN_layer4'):
        #     W_cross = tf.Variable(tf.random.uniform([2 * num_hidden, 2 * num_hidden],-rand_base,rand_base),name = 'W_cross')
        #     W_self = tf.Variable(tf.random.uniform([2 * num_hidden, 2 * num_hidden],-rand_base,rand_base),name = 'W_self')
        #     GCN2_cross = WXA_Relu(W_cross,GCN2_out,self.relate_cross)
        #     GCN2_self = WXA_Relu(W_self,GCN2_out,self.relate_self)
        #     GCN2_out = GCN2_cross+GCN2_self        #(?,600,13)

        # GCN2_out = tf.concat([GCN1_out,GCN2_out],1)

        # 和 self.target_which 矩阵相乘，求出对应的矩阵
        target_which = tf.expand_dims(self.target_which,1) # (?,1,13)
        self.GCN2_out = tf.multiply(GCN2_out, target_which)  #(?,600,13)*(?,1,13) = (?,600,13)
        self.targets_representation = tf.reduce_sum(self.GCN2_out, 2)  # (?,600)

        W = tf.Variable(tf.random.normal([2 * num_hidden, num_classes]))
        b = tf.Variable(tf.random.normal([num_classes]))
        with tf.name_scope("output"):
            self.scores = tf.compat.v1.nn.xw_plus_b(self.targets_representation, W,b, name="scores")
            # self.scores = self.targets_representation
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
            self.true_y = tf.argmax(self.input_y, 1, name="true_y")
            self.softmax = tf.nn.softmax(self.scores, name="softmax")

        with tf.name_scope("loss"):
            self.losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores,labels=self.input_y)
            self.loss = tf.reduce_mean(self.losses, name="loss") + l2_reg_lambda * l2_loss

        with tf.name_scope("accuracy"):
            self.correct_pred = tf.equal(self.predictions,self.true_y)
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, "float"),name="accuracy")
        print ("LOADED LSTM-Att-GCN2!")





