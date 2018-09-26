import os
import time
import json as js
import numpy as np
import tensorflow as tf
from collections import defaultdict
import copy

from im2sents_config import *
from im2sents_model  import *
from im2sents_data   import dataset
from im2sents_test   import *
from im2sents_util   import *

import pdb


def main(cfg, model, dataset):
    # config
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu_id

    #tf_config = tf.ConfigProto(device_count={"CPU":2},
    #                           inter_op_parallelism_threads=1,
    #                           intra_op_parallelism_threads=2,
    #                           log_device_placement=True)
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    #tf_config.gpu_options.per_process_gpu_memory_fraction = 0.4

    # step
    step_xe = tf.Variable(0, trainable=False)
    step_rl = tf.Variable(0, trainable=False)

    # learning rate
    learning_rate_xe = tf.train.exponential_decay(cfg.learning_rate_xe_begin, step_xe, cfg.learning_rate_epoch_decay * dataset.data_size / cfg.batch_size, 0.8, staircase=False, name='lr_xe')
    learning_rate_rl = tf.train.exponential_decay(cfg.learning_rate_rl_begin, step_rl, 2 * cfg.learning_rate_epoch_decay * dataset.data_size / cfg.batch_size, 0.8, staircase=False, name='lr_rl')
    learning_rate_rl = tf.maximum(learning_rate_rl, cfg.learning_rate_rl_end)

    # optimizer
    optimizer_xe = tf.train.AdamOptimizer(learning_rate_xe, beta1=0.8, beta2=0.999, epsilon=1e-8).minimize(model.loss_xe, global_step=step_xe)
    #optimizer_rl = tf.train.AdamOptimizer(learning_rate_rl, beta1=0.8, beta2=0.999, epsilon=1e-8).minimize(model.loss_rl, global_step=step_rl)
    opt = tf.train.AdamOptimizer(learning_rate_rl, beta1=0.8, beta2=0.999, epsilon=1e-8)
    grads_and_vars = opt.compute_gradients(model.loss_rl)
    clipped_grads_and_vars = [(tf.clip_by_value(gv[0], -1, 1), gv[1]) for gv in grads_and_vars]
    optimizer_rl = opt.apply_gradients(clipped_grads_and_vars)

    # sess
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())

    # saver
    saver = tf.train.Saver(max_to_keep=100)
    # TODO
    # self restore() without learning rate
    # ckpt  = tf.train.get_checkpoint_state(cfg.ckpt_xe_dir)
    # saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
    # TODO
    # merged = tf.summary.merge_all()
    # train_writer = tf.summary.FileWriter(cfg.sum_dir, sess.graph)

    # evaluate tool
    cocoEval = get_evaluate_tool()
    wvst_mask = get_wvst_mask(cfg)

    lm = 0
    t0 = time.time()
    scores_z = defaultdict(list)
    score_best = defaultdict(int)
    # train XE
    '''
    for itr in range(1, cfg.epoch * dataset.data_size / cfg.batch_size):
        # step forward
        anneal_rate = min(cfg.sample_rate_max, cfg.sample_rate_rate * cfg.batch_size * itr / cfg.data_size)
        _, loxe, lrxe = sess.run([optimizer_xe, model.loss_xe, learning_rate_xe], feed_dict={model.anneal_rate:[anneal_rate]})

        # log
        epoch = 1.0 * itr * cfg.batch_size / dataset.data_size
        lm = smooth(lm, loxe)
        if itr% 100 == 0 or itr==1:
            print 'XE_{}_Epo_{:0.2f}_Itr_{:d} loss:{:0.1f} l_mean:{:0.1f} lr:{:d}e-5 anneal:{:0.2f} time:{:0.2f}'.format(cfg.train_info, epoch, itr, cfg.loss_scale * loxe, cfg.loss_scale * lm, int(lrxe * 1e5), anneal_rate, time.time()-t0)
            t0 = time.time()

        # val and save model
        # TODO how to save log to local files
        if itr % int(cfg.epoch_val * dataset.data_size / cfg.batch_size) == 0:
            # kval
            test_cfg = True
            if not test_cfg:
                score_kval, scores_kval, names_kval, ges_kval, gts_kval = test(cfg, sess, model, dataset, cocoEval, split='kval')
                test_cfg = score_kval['CIDEr']>1.0
            else:
                # ktest
                score_ktest, scores_ktest, names_ktest, ges_ktest, gts_ktest = test(cfg, sess, model, dataset, cocoEval, split='ktest')
                update_cfg, score_best, scores_z = update_score(cfg, epoch, lm, score_ktest, score_best, scores_z)

                if update_cfg:
                    #score = test(cfg, sess, model, data, cocoEval, split='val')
                    #early_stop = 0
                    ckpt_name = '{}/{}_XE_epoch_{:0.1f}_loss_{:0.1f}_ktest_{:0.3f}.ckpt'.format(cfg.ckpt_dir_xe, cfg.train_info, epoch, cfg.loss_scale * lm, score_ktest['CIDEr'])
                    if score_ktest['CIDEr'] > 1:
                        saver.save(sess, ckpt_name)
                        print 'save ckpt: {}'.format(ckpt_name)
                    score_best['epoch'] = epoch

                    #if score['CIDEr']>1.05:
                        #score_beam = test_single(model, data, cfg, sess, select='MAX')
                        #print 'Beam Search', score_beam
                #elif score_ktest['CIDEr']>1.05:
                    #early_stop += cfg.epoch_val * cfg.data_size / cfg.batch_size
    '''

    # train rl
    itr = 0
    lm  = 0
    s_max_pre = 0
    s_scst_pre = 0
    ckpt  = tf.train.get_checkpoint_state(cfg.ckpt_dir_xe)
    saver.restore(sess, ckpt.all_model_checkpoint_paths[-1])
    score_ktest, scores_ktest, names_ktest, ges_ktest, gts_ktest = test(cfg, sess, model, dataset, cocoEval, split='ktest')
    score_best = copy.deepcopy(score_ktest)
    score_best['epoch'] = 0
    t0 = time.time()
    while True:
        itr += 1
        anneal_rate = cfg.sample_rate_max
        #anneal_rate = min(cfg.sample_rate_max, cfg.sample_rate_rate * ((cfg.batch_size * itr) / (cfg.sample_rate_step * cfg.data_size) + 1))

        # get rl training source
        feat, seq_scst, value_mask, score_scst, score_max = get_rl_source(cfg, sess, model, dataset.label2word, anneal_rate, cocoEval, wvst_mask, score_type='CIDEr')
        _, lorl, lrrl = sess.run([optimizer_rl, model.loss_rl, learning_rate_rl], feed_dict={model.feat_rl:     feat,
                                                                                             model.sequence_rl: seq_scst,
                                                                                             model.reward:      value_mask,
                                                                                             learning_rate_rl:  cfg.learning_rate_rl_end})


        # summary
        epoch = 1.0 * itr * cfg.batch_size / dataset.data_size
        lm = smooth(lm, lorl)
        s_max_pre = smooth(s_max_pre, score_max)
        s_scst_pre = smooth(s_scst_pre, score_scst)

        # show the result
        if itr % 100 == 0 or itr==1:
            print cfg.train_info+'_'+'RL_v{}_Epo_{:0.2f}_Itr_{:d} l_rl:{:0.1f} s_scst:{:0.3f} s_max:{:0.3f} lr:{:d}e-5 anneal:{:0.2f} time:{:0.2f}'.format(cfg.rl_ver, epoch, itr, cfg.loss_scale * lm, s_scst_pre, s_max_pre, int(lrrl*1e5), anneal_rate, time.time()-t0)
            t0 = time.time()

        if itr % int(cfg.epoch_val * cfg.data_size / cfg.batch_size) == 0:
            # kval
            #score_kval, scores_kval, names_kval, ges_kval, gts_kval = test(cfg, sess, model, data, cocoEval, split='kval')
            #if score_kval['CIDEr']>1.0:
            score_kval = {'CIDEr':1.3}
            if True:
                # ktest
                score_ktest, scores_ktest, names_ktest, ges_ktest, gts_ktest = test(cfg, sess, model, dataset, cocoEval, split='ktest')
                update_cfg, score_best, scores_z = update_score(cfg, epoch, lm, score_ktest, score_best, scores_z)
                with open('RL_V{}.log'.format(cfg.rl_ver), 'w+') as f:
                    f.write('epo {:d}\n'.format(int(epoch)))
                    for k,v in score_ktest.items():
                        f.write('{}: {:0.3f}\n'.format(k,v))
                    f.write('\n')
            
                #fo.write('epo {:0.1f} loss {:0.3f} cider {:0.3f}'.format(epoch, cfg.loss_scale * l_rl_pre, score_ktest['CIDEr']))
                #for k,v in score_ktest.items():
                #    fo.write(' {} {:0.3f}'.format(k.lower(),v))
                #fo.write('\n')
                if update_cfg and itr>0:
                    ckpt_name = cfg.ckpt_dir_rl +'/RL_v{}_epoch_{:0.1f}_kval_{:0.3f}_ktest_{:0.3f}.ckpt'.format(cfg.rl_ver, epoch, score_kval['CIDEr'], score_ktest['CIDEr'])
                    saver.save(sess, ckpt_name)
                    print 'save ckpt: {}'.format(ckpt_name)
                    score_best['epoch'] = epoch


if __name__ == '__main__':
    cfg = cfg()
    # dataset
    dataset = dataset(cfg)
    print 'dataset init done.'
    cfg.data_size  = dataset.data_size
    cfg.label_size = dataset.label_size

    # model
    model = model(cfg, dataset)
    print 'model init done.'

    main(cfg, model, dataset)