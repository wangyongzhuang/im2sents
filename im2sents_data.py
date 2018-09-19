import tensorflow as tf
import numpy as np
import pandas as pd
import csv
import json as js
import random
import pdb

class dataset():
    def __init__(self, cfg):
        self.cfg = cfg
        self.patch = js.load(open('{}/{}'.format(self.cfg.data_dir, self.cfg.patch_name), 'r'))
        self.label2word   = self.patch['label2word']
        self.word2label   = self.patch['word2label']

        self.train_parts  = ['train_ktrain', 'val_ktrain', 'val_kval']

        # ksplit
        self.ktrain_file_names = []
        for sks in self.train_parts:
            self.ktrain_file_names += self.patch['split_ksplit'][sks]
        self.ktrain_size       = len(self.ktrain_file_names)

        self.kval_file_names   = self._batch(self.patch['split_ksplit']['val_kval'])
        self.kval_size         = len(self.kval_file_names)

        self.ktest_file_names  = self._batch(self.patch['split_ksplit']['val_ktest'])
        self.ktest_size        = len(self.ktest_file_names)

        # split
        self.val_file_names  = self._batch(self.patch['split_ksplit']['val_ktrain'] + self.patch['split_ksplit']['val_kval'] + self.patch['split_ksplit']['val_ktest'])
        self.val_size        = len(self.val_file_names)

        self.test_file_names = self._batch(self.patch['split_ksplit']['test'])
        self.test_size       = len(self.test_file_names)

        # cfg
        self.data_size  = 5 * self.ktrain_size
        self.label_size = len(self.label2word)

        self.load_split()
        self.get_samp()

    def _batch(self, files):
        if len(files)%self.cfg.batch_size > 0:
            files.extend(files[:self.cfg.batch_size - len(files) % self.cfg.batch_size])
        return files

    def load_feat(self, feat_path):
        feat = np.load(feat_path)
        feat = feat.reshape([self.cfg.feat_num-1, self.cfg.feat_dim])
        feat_ave = np.mean(feat, axis=0)
        feat = np.concatenate([feat_ave[np.newaxis, :], feat], axis=0)

        return feat

    def gen_ktrain(self):
        i = 0
        num = len(self.ktrain_file_names)
        while True:
            i = (i+1)%num
            file_name = self.ktrain_file_names[i]
            data = self.patch['data'][file_name]
            # feat
            feat = self.load_feat(data['feat_path'])

            # refs
            refs      = np.array([_['sent'] for _ in data['sents'].values()])
            refs_raw  = np.array([_['sent_raw'] for _ in data['sents'].values()])
            ref_words = np.array(data['common_words'])

            for sent_id in data['sents'].keys():
                label = [0] + data['sents'][sent_id]['label'][:self.cfg.sentence_length]
                yield [file_name], feat, label, refs, refs_raw, ref_words


    def gen_kval(self):
        for file_name in self.kval_file_names:
            data = self.patch['data'][file_name]
            # feat
            feat = self.load_feat(data['feat_path'])

            # refs
            refs     = np.array([_['sent'] for _ in data['sents'].values()])
            refs_raw = np.array([_['sent_raw'] for _ in data['sents'].values()])
            #labels   = np.array([_['label'] for _ in data['sents'].values()])

            yield [file_name], feat, refs, refs_raw#, labels.reshape(5*self.cfg.sentence_length+5)


    def gen_ktest(self):
        for file_name in self.ktest_file_names:
            data = self.patch['data'][file_name]
            # feat
            feat = self.load_feat(data['feat_path'])

            # refs
            refs   = np.array([_['sent'] for _ in data['sents'].values()])
            refs_raw = np.array([_['sent_raw'] for _ in data['sents'].values()])
            #labels   = np.array([_['label'] for _ in data['sents'].values()])

            yield [file_name], feat, refs, refs_raw#, labels.reshape(5*self.cfg.sentence_length+5)


    def gen_val(self):
        for file_name in self.val_file_names:
            data = self.patch['data'][file_name]
            # feat
            feat = self.load_feat(data['feat_path'])

            # refs
            refs   = np.array([_['sent'] for _ in data['sents'].values()])
            labels = np.array([_['label'] for _ in data['sents'].values()])

            yield [file_name], feat, refs, refs_raw#, labels.reshape(5*self.cfg.sentence_length+5)
            
    def gen_test(self):
        for file_name in self.test_file_names:
            feat = self.load_feat(self.patch['data'][file_name]['feat_path'])
            yield [file_name], feat


    def load_split(self):
        # ktrain
        data_ktrain = tf.data.Dataset.from_generator(self.gen_ktrain,
                        output_types=(tf.string, tf.float32, tf.int64, tf.string, tf.string, tf.string),
                        output_shapes=(tf.TensorShape([1]),
                            tf.TensorShape([self.cfg.feat_num, self.cfg.feat_dim]),
                            tf.TensorShape([self.cfg.sentence_length + 1]),
                            tf.TensorShape([5]),
                            tf.TensorShape([5]),
                            tf.TensorShape([2 * self.cfg.sentence_length])))
        self.data_ktrain = data_ktrain.repeat().shuffle(buffer_size=self.cfg.batch_size).batch(self.cfg.batch_size)
        self.iter_ktrain = self.data_ktrain.make_one_shot_iterator()

        # kval
        data_kval = tf.data.Dataset.from_generator(self.gen_kval,
                        output_types=(tf.string, tf.float32, tf.string, tf.string),
                        output_shapes=(tf.TensorShape([1]),
                            tf.TensorShape([self.cfg.feat_num, self.cfg.feat_dim]),
                            tf.TensorShape([5]),
                            tf.TensorShape([5])))
        self.data_kval = data_kval.repeat().batch(self.cfg.batch_size)
        self.iter_kval = self.data_kval.make_one_shot_iterator()

        # ktest
        data_ktest = tf.data.Dataset.from_generator(self.gen_ktest,
                        output_types=(tf.string, tf.float32, tf.string, tf.string),
                        output_shapes=(tf.TensorShape([1]),
                            tf.TensorShape([self.cfg.feat_num, self.cfg.feat_dim]),
                            tf.TensorShape([5]),
                            tf.TensorShape([5])))
        self.data_ktest = data_ktest.repeat().batch(self.cfg.batch_size)
        self.iter_ktest = self.data_ktest.make_one_shot_iterator()

        # val
        data_val = tf.data.Dataset.from_generator(self.gen_val,
                        output_types=(tf.string, tf.float32, tf.string, tf.string),
                        output_shapes=(tf.TensorShape([1]),
                            tf.TensorShape([self.cfg.feat_num, self.cfg.feat_dim]),
                            tf.TensorShape([5]),
                            tf.TensorShape([5])))
        self.data_val = data_val.repeat().batch(self.cfg.batch_size)
        self.iter_val = self.data_val.make_one_shot_iterator()

        # val_single
        data_val_single = tf.data.Dataset.from_generator(self.gen_val,
                            output_types=(tf.string, tf.float32, tf.string, tf.string),
                            output_shapes=(tf.TensorShape([1]),
                                tf.TensorShape([self.cfg.feat_num, self.cfg.feat_dim]),
                                tf.TensorShape([5]),
                                tf.TensorShape([5])))
        self.data_val_single = data_val_single.repeat().batch(1)
        self.iter_val_single = self.data_val_single.make_one_shot_iterator()

        # ktest
        data_ktest_single = tf.data.Dataset.from_generator(self.gen_ktest,
                            output_types=(tf.string, tf.float32, tf.string, tf.string),
                            output_shapes=(tf.TensorShape([1]),
                                tf.TensorShape([self.cfg.feat_num, self.cfg.feat_dim]),
                                tf.TensorShape([5]),
                                tf.TensorShape([5])))
        self.data_ktest_single = data_ktest_single.repeat().batch(1)
        self.iter_ktest_single = self.data_ktest_single.make_one_shot_iterator()

        # test_single
        data_test_single = tf.data.Dataset.from_generator(self.gen_test,
                            output_types=(tf.string, tf.float32),
                            output_shapes=(tf.TensorShape([1]),
                                tf.TensorShape([self.cfg.feat_num, self.cfg.feat_dim])))
        self.data_test_single = data_test_single.repeat().batch(1)
        self.iter_test_single = self.data_test_single.make_one_shot_iterator()
        

    def get_samp(self):
        # ksplit
        self.samp_ktrain_file_name, self.samp_ktrain_feat, self.samp_ktrain_label, self.samp_ktrain_refs, self.samp_ktrain_refs_raw, self.samp_ktrain_ref_words = self.iter_ktrain.get_next()

        self.samp_kval_file_name, self.samp_kval_feat, self.samp_kval_refs, self.samp_kval_refs_raw = self.iter_kval.get_next()

        self.samp_ktest_file_name, self.samp_ktest_feat, self.samp_ktest_refs, self.samp_ktest_refs_raw = self.iter_ktest.get_next()

        # split
        self.samp_val_file_name, self.samp_val_feat, self.samp_val_refs, self.samp_val_refs_raw = self.iter_val.get_next()

        # beam
        self.samp_val_single_file_name, self.samp_val_single_feat, self.samp_val_single_refs, self.samp_val_single_refs_raw = self.iter_val_single.get_next()

        self.samp_ktest_single_file_name, self.samp_ktest_single_feat, self.samp_ktest_single_refs, self.samp_ktest_single_refs_raw = self.iter_ktest_single.get_next()

        self.samp_test_single_file_name, self.samp_test_single_feat = self.iter_test_single.get_next()
