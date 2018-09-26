import tensorflow  as tf
import numpy as np
import random
import pdb

def ln(inp, bias=0.0, gain=1.0):
    '''
    layer norm
    '''
    inp_shape = inp.get_shape()

    mean, var = tf.nn.moments(inp, -1)
    var += 1e-20
    if len(inp_shape)==1:
        return gain * (inp - mean) / tf.sqrt(var) + bias
    elif len(inp_shape)==2:
        mean = tf.tile(tf.expand_dims(mean, -1), [1, int(inp_shape[-1])])
        var  = tf.tile(tf.expand_dims(var,  -1), [1, int(inp_shape[-1])])
        return gain * (inp - mean) / tf.sqrt(var) + bias    
    elif len(inp_shape)==3:
        mean = tf.tile(tf.expand_dims(mean, -1), [1, 1, int(inp_shape[-1])])
        var  = tf.tile(tf.expand_dims(var,  -1), [1, 1, int(inp_shape[-1])])
        return gain * (inp - mean) / tf.sqrt(var) + bias    

    return inp



def softmax_monte_carlo_sample(logits, shape, ground=None, sample_rate=0.25, select_max=False):
    p = tf.nn.softmax(logits)
    if ground is None:
        ground = tf.argmax(p, -1)
    else:
        ground = ground

    mask_tmp = tf.sign(tf.random_uniform(shape=[shape[0]], minval=sample_rate, maxval=sample_rate-1))
    samp_mask_p = tf.cast(1 - (tf.abs(mask_tmp) - mask_tmp) / 2, tf.int64)

    res = samp_mask_p * tf.squeeze(tf.multinomial(tf.log(p),1)) + (1 - samp_mask_p) * ground

    return res, tf.cast(samp_mask_p, tf.float32)

class model():
    def __init__(self, cfg, data):
        self.cfg = cfg

        self.prepare_weights()
        self.prepare_data(data)

        self.train_xe()
        self.train_rl()
        self.predict()
        self.predict_step()


    def feat_embedding(self, feat):
       embed_feat = tf.reshape(tf.nn.bias_add(tf.matmul(tf.reshape(feat, [-1, self.cfg.feat_dim]), self.w_feat), self.b_feat), [-1, self.cfg.feat_num, self.cfg.embedding_size])
       embed_feat = tf.nn.relu(embed_feat)

       return embed_feat

    #def prepare_weights(self, init=tf.random_uniform_initializer(minval=-0.1,maxval=0.1)):
    def prepare_weights(self, init=tf.contrib.layers.xavier_initializer()):
        # encode
        with tf.variable_scope('encode'):
            self.w_feat = tf.get_variable('w_feat', shape=[self.cfg.feat_dim, self.cfg.embedding_size], dtype=tf.float32, initializer=init)
            self.b_feat = tf.get_variable('b_feat', shape=[self.cfg.embedding_size], dtype=tf.float32, initializer=init)
        
            self.word_embedding = tf.get_variable('word_embedding', shape=[self.cfg.label_size, self.cfg.embedding_size], dtype=tf.float32, initializer=init)

            self.encode_loss = tf.nn.l2_loss(self.w_feat) + tf.nn.l2_loss(self.word_embedding)

        # lstm
        with tf.variable_scope('lstm_0'):
            self.lstm_cell_0 = tf.nn.rnn_cell.LSTMCell(self.cfg.hidden_state_size, forget_bias=0.0, state_is_tuple=True, initializer=init)
            if self.cfg.drop_out > 0:
                self.lstm_cell_0 = tf.nn.rnn_cell.DropoutWrapper(self.lstm_cell_0, output_keep_prob=self.cfg.drop_out)

        with tf.variable_scope('lstm_1', reuse=tf.AUTO_REUSE):
            self.lstm_cell_1 = tf.nn.rnn_cell.LSTMCell(self.cfg.hidden_state_size, forget_bias=0.0, state_is_tuple=True,  initializer=init)
            if self.cfg.drop_out > 0:
                self.lstm_cell_1 = tf.nn.rnn_cell.DropoutWrapper(self.lstm_cell_1, output_keep_prob=self.cfg.drop_out)


        # decode
        with tf.variable_scope('decode'):
            self.w_softmax = tf.get_variable('w_softmax', shape=[2 * self.cfg.hidden_state_size, self.cfg.label_size], dtype=tf.float32, initializer=init)
            self.b_softmax = tf.get_variable('b_softmax', shape=[self.cfg.label_size], dtype=tf.float32, initializer=init)

            self.decode_loss = tf.nn.l2_loss(self.w_softmax)


        # adap
        with tf.variable_scope('adap'):
            # sent gate
            self.w_sx = tf.get_variable('w_sx', shape=[self.cfg.hidden_state_size, self.cfg.hidden_state_size], dtype=tf.float32, initializer=init)
            self.w_sh = tf.get_variable('w_sh', shape=[self.cfg.hidden_state_size, self.cfg.hidden_state_size], dtype=tf.float32, initializer=init)
            self.b_s  = tf.get_variable('b_s', shape=[self.cfg.hidden_state_size], dtype=tf.float32, initializer=init)

            # att
            self.w_zh = tf.get_variable('w_zh', shape=[2 * self.cfg.hidden_state_size, self.cfg.hidden_state_size], dtype=tf.float32, initializer=init)
            self.w_zi = tf.get_variable('w_zi', shape=[self.cfg.hidden_state_size, self.cfg.hidden_state_size], dtype=tf.float32, initializer=init)
            self.b_zh = tf.get_variable('b_zh', shape=[self.cfg.hidden_state_size], dtype=tf.float32, initializer=init)

            self.w_z = tf.get_variable('w_z', shape=[self.cfg.hidden_state_size, 1], dtype=tf.float32, initializer=init)
            self.b_z = tf.get_variable('b_z', shape=[self.cfg.feat_num], dtype=tf.float32, initializer=init)

            self.adap_loss = tf.nn.l2_loss(self.w_sx) + tf.nn.l2_loss(self.w_sh) + tf.nn.l2_loss(self.w_z) + tf.nn.l2_loss(self.w_zh) + tf.nn.l2_loss(self.w_zi)

        # l2_loss
        self.l2_loss = self.encode_loss + self.decode_loss + self.adap_loss

    def prepare_data(self, data):
        # cfgs
        self.anneal_rate = tf.placeholder(dtype=tf.float32, shape=[1])
        self.weight_decay_xe = tf.constant(self.cfg.weight_decay_xe)
        self.weight_decay_rl = tf.constant(self.cfg.weight_decay_rl)

        self.reward       = tf.placeholder(dtype=tf.float32, shape=[self.cfg.sentence_length, 5 * self.cfg.batch_size])
        self.mask_reward  = tf.tile(tf.expand_dims(self.reward, -1), [1, 1, self.cfg.label_size])

        # ktrain
        #self.file_name_ktrain, self.feat_ktrain, self.label_ktrain, self.refs_ktrain, self.refs_raw_ktrain, self.ref_words_ktrain = data.samp_ktrain_file_name, data.samp_ktrain_feat, data.samp_ktrain_label, data.samp_ktrain_refs, data.samp_ktrain_refs_raw, data.samp_ktrain_ref_words
        file_name_ktrain, feat_ktrain, label_ktrain, refs_ktrain, refs_raw_ktrain, ref_words_ktrain = data.samp_ktrain_file_name, data.samp_ktrain_feat, data.samp_ktrain_label, data.samp_ktrain_refs, data.samp_ktrain_refs_raw, data.samp_ktrain_ref_words
        self.file_name_ktrain = tf.reshape(tf.tile(tf.expand_dims(file_name_ktrain, 1), [1, 5, 1]), [5 * self.cfg.batch_size, 1])
        self.feat_ktrain      = tf.reshape(tf.tile(tf.expand_dims(feat_ktrain, 1), [1, 5, 1, 1]), [5 * self.cfg.batch_size, self.cfg.feat_num, self.cfg.feat_dim])
        self.label_ktrain     = tf.reshape(label_ktrain, [5 * self.cfg.batch_size, self.cfg.sentence_length+1])
        self.refs_ktrain      = tf.reshape(tf.tile(tf.expand_dims(refs_ktrain, 1), [1, 5, 1]), [5 * self.cfg.batch_size, 5])
        self.refs_raw_ktrain  = tf.reshape(tf.tile(tf.expand_dims(refs_raw_ktrain, 1), [1, 5, 1]), [5 * self.cfg.batch_size, 5])
        self.ref_words_ktrain = tf.reshape(tf.tile(tf.expand_dims(ref_words_ktrain, 1), [1, 5, 1]), [5 * self.cfg.batch_size, 2 * self.cfg.sentence_length])

        # kval
        self.file_name_kval, self.feat_kval, self.refs_kval, self.refs_raw_kval     = data.samp_kval_file_name, data.samp_kval_feat, data.samp_kval_refs, data.samp_kval_refs_raw
        
        # ktest
        self.file_name_ktest, self.feat_ktest, self.refs_ktest, self.refs_raw_ktest = data.samp_ktest_file_name, data.samp_ktest_feat, data.samp_ktest_refs, data.samp_ktest_refs_raw

        # val
        self.file_name_val, self.feat_val, self.refs_val, self.refs_raw_val = data.samp_val_file_name, data.samp_val_feat, data.samp_val_refs, data.samp_val_refs_raw

        # embed_feat
        self.embed_feat_ktrain = self.feat_embedding(self.feat_ktrain)
        self.embed_feat_kval   = self.feat_embedding(self.feat_kval)
        self.embed_feat_ktest  = self.feat_embedding(self.feat_ktest)
        self.embed_feat_val   = self.feat_embedding(self.feat_val)
        

    def forward(self, cell_state_0, cell_state_1, embed_feat, embed_word, batch_size=100, ftype='train'):
        drop_out_cfg = (self.cfg.drop_out > 0) and (ftype == 'train')

        # language LSTM
        cell_input_0 = embed_word
        with tf.variable_scope('lstm_0') as scope:
            (cell_output_0, cell_state_0_new) = self.lstm_cell_0(cell_input_0, cell_state_0)
        #cell_input_0 = tf.concat([embed_word, cell_state_1[0], embed_feat[:,0,:]], 1)

        if not drop_out_cfg:
            cell_output_0 = cell_state_0_new[1]
        # cell_state:[c_bew, h_new]


        # attention
        with tf.variable_scope('adap') as scope:
            scope.reuse_variables()

            # att
            tmp1 = tf.matmul(embed_feat, tf.tile(tf.expand_dims(self.w_zi, 0), [batch_size, 1, 1]))
            tmp2 = tf.tile(tf.expand_dims(tf.matmul(tf.concat([cell_state_0_new[1], cell_state_1[1]], -1), self.w_zh), 1), [1, self.cfg.feat_num, 1])
            z_t = tf.matmul(tf.tanh(tf.nn.bias_add(tmp1 + tmp2, self.b_zh)), tf.tile(tf.expand_dims(self.w_z, 0), [batch_size, 1, 1])) + tf.tile(tf.expand_dims(tf.expand_dims(self.b_z, 0), -1), [batch_size, 1, 1])
            self.a_t = tf.nn.softmax(z_t, 1)
            c_t = tf.reduce_sum(tf.tile(self.a_t, [1, 1, self.cfg.embedding_size]) * embed_feat, axis=1)


            # sent
            g_t = tf.nn.sigmoid(tf.matmul(cell_input_0, self.w_sx) + tf.matmul(cell_state_0[1], self.w_sh) + tf.tile(tf.expand_dims(self.b_s, 0), [batch_size, 1]))
            s_t = g_t * cell_state_0_new[0]
        
        # image LSTM
        cell_input_1 = tf.concat([c_t, s_t], -1)
        with tf.variable_scope('lstm_1') as scope:
            (cell_output_1, cell_state_1_new) = self.lstm_cell_1(cell_input_1, cell_state_1)

        if not drop_out_cfg:
            cell_output_1 = cell_state_1_new[1]
        adap_out = tf.concat([cell_output_1, cell_output_0], -1)

        # decode
        with tf.variable_scope('decode') as scope:
            logits = tf.nn.bias_add(tf.matmul(adap_out, self.w_softmax), self.b_softmax)
        
        return logits, cell_state_0_new , cell_state_1_new
        


    def train_xe(self):
        #  loop
        output = []
        cell_state_0  = self.lstm_cell_0.zero_state(5 * self.cfg.batch_size, tf.float32)
        cell_state_1  = self.lstm_cell_1.zero_state(5 * self.cfg.batch_size, tf.float32)
        
        sequence = [self.label_ktrain[:,0]]
        samp_mask = [tf.zeros(5 * self.cfg.batch_size)]
        for t in range(self.cfg.sentence_length):
            # annealing
            embed_word_tmp = tf.nn.embedding_lookup(self.word_embedding, sequence[-1])
            # forward
            logits, cell_state_0, cell_state_1 = self.forward(cell_state_0, cell_state_1, self.embed_feat_ktrain, embed_word_tmp, batch_size=5 * self.cfg.batch_size)
            output.append(logits)

            input_idxs, samp_mask_tmp = softmax_monte_carlo_sample(logits=logits, shape=[5 * self.cfg.batch_size, self.cfg.label_size], ground=self.label_ktrain[:, t+1], sample_rate=self.anneal_rate[0], select_max=self.cfg.select_max)
            sequence.append(input_idxs)
            samp_mask.append(samp_mask_tmp)
        self.sequence_xe  = tf.transpose(tf.stack(sequence), perm=[1,0])
        self.samp_mask_xe = tf.transpose(tf.stack(samp_mask), perm=[1,0])


        # logits, labels, loss
        self.logits_xe = tf.concat([tf.expand_dims(_, 0) for _ in output], 0)
        self.label_xe = tf.one_hot(tf.transpose(self.label_ktrain[:,1:]), self.cfg.label_size)

        self.sent_length_mask = tf.tile(tf.expand_dims(tf.transpose(tf.cast(tf.concat([self.label_ktrain[:,1:2], self.label_ktrain[:,1:-1]], -1)>0, tf.float32)), -1), [1, 1, self.cfg.label_size])
        self.label_xe_masked = (0.9 * self.label_xe + 0.1 * tf.nn.softmax(self.logits_xe)) * self.sent_length_mask

        self.loss_xe = tf.reduce_mean(-1.0 * self.label_xe_masked * tf.log(tf.nn.softmax(self.logits_xe)+1e-8)) * self.cfg.label_size + self.weight_decay_xe * self.l2_loss



    def train_rl(self):
        # input
        self.sequence_rl  = tf.placeholder(tf.int32, shape=[5 * self.cfg.batch_size, self.cfg.sentence_length+1])
        self.feat_rl      = tf.placeholder(tf.float32, shape=[5 * self.cfg.batch_size, self.cfg.feat_num, self.cfg.feat_dim])

        # loop
        embed_feat_rl     = self.feat_embedding(self.feat_rl)
        output = []
        cell_state_0  = self.lstm_cell_0.zero_state(5 * self.cfg.batch_size, tf.float32)
        cell_state_1  = self.lstm_cell_1.zero_state(5 * self.cfg.batch_size, tf.float32)
        for t in range(self.cfg.sentence_length):
            embed_word_tmp = tf.nn.embedding_lookup(self.word_embedding, self.sequence_rl[:,t])
            logits, cell_state_0, cell_state_1 = self.forward(cell_state_0, cell_state_1, embed_feat_rl, embed_word_tmp, batch_size=5 * self.cfg.batch_size, ftype='train')
            output.append(logits)

        # logits and labels
        self.logits_rl = tf.stack(output)
        self.label_rl  = tf.one_hot(self.sequence_rl[:,1:], self.cfg.label_size)
        self.label_rl_masked = tf.transpose(self.label_rl, perm=[1,0,2])# * self.sent_length_mask

        self.loss_rl = tf.reduce_mean(-1.0 * self.mask_reward * self.label_rl_masked * tf.log(tf.nn.softmax(self.logits_rl)+1e-8)) * self.cfg.label_size + self.weight_decay_rl * self.l2_loss


#############################################################################################
    def predict_batch(self, embed_feat, idxs=None, predict_type='MAX'):
        # load input

        # LSTM
        cell_state_0  = self.lstm_cell_0.zero_state(5 * self.cfg.batch_size, tf.float32)
        cell_state_1  = self.lstm_cell_1.zero_state(5 * self.cfg.batch_size, tf.float32)

        # result
        res = []
        res.append(tf.constant(np.zeros(5 * self.cfg.batch_size), dtype=tf.int64))

        # predict
        for t in range(self.cfg.sentence_length):
            # pre word
            embed_word = tf.nn.embedding_lookup(self.word_embedding, tf.cast(res[-1], tf.int64))

            # lstm
            logits, cell_state_0, cell_state_1 = self.forward(cell_state_0, cell_state_1, embed_feat, embed_word, batch_size=5 * self.cfg.batch_size, ftype='predict')

            # res
            if predict_type=='MAX' or (predict_type=='SCST' and idxs is None):
                res.append(tf.argmax(logits, -1))
            elif predict_type=='SCST' and idxs is not None:
                res.append(softmax_monte_carlo_sample(logits=logits, shape=[5 * self.cfg.batch_size, self.cfg.label_size], ground=idxs[:, t+1],  sample_rate=self.anneal_rate[0])[0])

        return tf.concat([tf.expand_dims(_, -1) for _ in res], -1)

    def predict(self):
        # ktrain
        self.res_max_ktrain  = self.predict_batch(self.embed_feat_ktrain, self.label_ktrain, predict_type='MAX')
        #self.res_scst_ktrain = self.predict_batch(self.embed_feat_ktrain, self.label_ktrain, predict_type='SCST')

        # kval
        self.res_max_kval  = self.predict_batch(self.embed_feat_kval, idxs=None, predict_type='MAX')

        # ktest
        self.res_max_ktest  = self.predict_batch(self.embed_feat_ktest, idxs=None, predict_type='MAX')

        # val
        self.res_max_val  = self.predict_batch(self.embed_feat_val, idxs=None, predict_type='MAX')


    def predict_step(self):
        self.feat_single = tf.placeholder(dtype=tf.float32, shape=[1, self.cfg.feat_num, self.cfg.feat_dim])
        self.idxs_single = tf.placeholder(dtype=tf.int64, shape=[1])

        self.embed_feat_single = self.feat_embedding(self.feat_single)
        self.embed_word_single = tf.nn.embedding_lookup(self.word_embedding, self.idxs_single)

        self.cell_h_0 = tf.placeholder(dtype=tf.float32, shape=[1,self.cfg.hidden_state_size])
        self.cell_h_1 = tf.placeholder(dtype=tf.float32, shape=[1,self.cfg.hidden_state_size])
        self.cell_c_0 = tf.placeholder(dtype=tf.float32, shape=[1,self.cfg.hidden_state_size])
        self.cell_c_1 = tf.placeholder(dtype=tf.float32, shape=[1,self.cfg.hidden_state_size])

        self.logits_single, self.cell_state_0_single, self.cell_state_1_single = self.forward((self.cell_c_0, self.cell_h_0), (self.cell_c_1, self.cell_h_1), self.embed_feat_single, self.embed_word_single, batch_size=1, ftype='predict')

        self.prob_single = tf.nn.softmax(self.logits_single)

        self.p_top, self.idx_top = tf.nn.top_k(self.prob_single[0], k=self.cfg.beam_size)
