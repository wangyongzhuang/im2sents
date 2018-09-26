import json as js
from collections import defaultdict
import numpy as np
import pandas as pd
import re
import os
import pdb

class coco():
    def __init__(self):
        self.dataset_name = 'coco'
        self.year         = 2014
        self.data_dir     = '/data/{}'.format(self.dataset_name)

        self.target_dir   = './data'
        if not os.path.exists(self.target_dir):
            os.mkdir(self.target_dir)
        self.sentence_length = 18
        self.word_threshold  = 5

        self.feat_dir  = '{}/feats_updown'.format(self.data_dir)
        self.feat_num  = 36
        self.feat_dim  = 2048

        # process
        self.load_raw_annotations()
        self.load_karpathy_split()
        self.form_vocabulary()
        self.create_patch()
        js.dump(self.patch, open('{}/patch_coco.json'.format(self.target_dir), 'w'))

        
    def _parse(self, sentence, voc_k=None):
        tokens = [token.strip().lower() for token in re.compile('(\W+)').split(sentence.strip().strip('.')) if len(token.strip())>0]
        for token in tokens:
            if not token.isalpha():
                tokens.remove(token)
        if voc_k is not None:
            for i,token in enumerate(tokens):
                if token not in voc_k:
                    tokens[i] = 'unk'
        tokens = tokens[:self.sentence_length]
        return tokens

    def load_raw_annotations(self):
        # init
        self.data = defaultdict(dict)
        self.split_ksplit = defaultdict(list)
        self.cocoid2filename = {}
        self.filename2cocoid = {}

        # raw data [train, val, test]
        print 'loading raw data ...'
        data_train_raw = js.load(open('{}/annotations/captions_train{}.json'.format(self.data_dir, self.year)))
        data_val_raw   = js.load(open('{}/annotations/captions_val{}.json'.format(self.data_dir, self.year)))
        imgs_test_raw  = js.load(open('{}/annotations/image_info_test{}.json'.format(self.data_dir, self.year)))['images']
        print 'load raw data done.'


        # imgs [test]
        print 'start to process test data'
        for i,img in enumerate(imgs_test_raw):
            file_name = img['file_name']
            feat_path = '{}/test{}/{}'.format(self.feat_dir, self.year, file_name).replace('jpg','npy')
            # data
            self.data[file_name] = {'file_name': file_name,
                                    'coco_id': img['id'],
                                    'split_ksplit': 'test',
                                    'feat_path': feat_path}
            self.split_ksplit['test'].append(file_name)
            self.cocoid2filename[img['id']] = file_name
            self.filename2cocoid[file_name] = img['id']


        # imgs & anns [train, val]
        imgs = data_train_raw['images']
        imgs.extend(data_val_raw['images'])
        anns = data_train_raw['annotations']
        anns.extend(data_val_raw['annotations'])
        for img in imgs:
            file_name = img['file_name']
            self.data[file_name] = {'file_name':file_name,
                                    'coco_id':img['id'],
                                    'sents':{}}
            self.cocoid2filename[img['id']] = file_name
            self.filename2cocoid[file_name] = img['id']

        for ann in anns:
            file_name = self.cocoid2filename[ann['image_id']]
            self.data[file_name]['sents'][ann['id']] = {'sent_raw':ann['caption'].lower().strip('.\n\r'),
                                                        'sent_id':ann['id']}


    def load_karpathy_split(self):
        print 'loading and process karpathy data ...'
        imgs = js.load(open('{}/dataset.json'.format(self.data_dir),'r'))['images']
        for i,img in enumerate(imgs):
            if (i+1)%10000==0:
                print 'pre-process {}/{} images for karpathy data.'.format(i+1, len(imgs))
            file_name = img['filename']
            feat_path = '{}/{}/{}'.format(self.feat_dir, img['filepath'], img['filename']).replace('jpg','npy')#.replace('jpg','mat')

            # filepath->feat_path
            self.data[file_name]['feat_path'] = feat_path

            # split->split_ksplit
            split  = img['filepath'].strip(str(self.year))
            ksplit = img['split'].replace('restval', 'train')
            sks    = '{}_k{}'.format(split, ksplit)
            self.data[file_name]['split_ksplit'] = sks
            self.split_ksplit[sks].append(file_name)


            # sentences and words
            common_words = defaultdict(int)
            for j,sent in enumerate(img['sentences']):
                sent_id = sent['sentid']

                tokens = self._parse(self.data[file_name]['sents'][sent_id]['sent_raw'])
                # raw tokens
                self.data[file_name]['sents'][sent_id]['tokens'] = tokens
                for token in tokens:
                    common_words[token] += 1

                if j >= 5:
                    self.data[file_name]['sents'].pop(sent_id)

            common_words = [_[0] for _ in sorted(common_words.items(), key=lambda x:x[1], reverse=True)] # {'a':1, 'b':3, 'c':2}=>['b','c','a']
            self.data[file_name]['common_words'] = common_words[:2*self.sentence_length]
            words_num = len(common_words)
            if words_num < 2 * self.sentence_length:
                self.data[file_name]['common_words'] += ['a'] * (2 * self.sentence_length - words_num)
        print 'load karpathy data done.'


    def form_vocabulary(self):
        # count words
        self.vocab = defaultdict(int)
        for file_name in self.data.keys():
            if self.data[file_name]['split_ksplit'] not in ['train_ktrain', 'val_ktrain', 'val_kval']:
                continue
            sents = self.data[file_name]['sents']
            for sent_id in sents.keys():
                
                tokens = sents[sent_id]['tokens'][:self.sentence_length]
                for token in tokens:
                    self.vocab[token] += 1
                '''
                tokens = self.parse(sent_id]['sent_raw'])
                #tokens = sents[sent_id]['sent_raw'].split()[:self.sentence_length]
                for token in tokens:
                    self.vocab[token] += 1
                '''
        print 'vocab_init has {} words.'.format(len(self.vocab))

        # prune
        words_num = 0
        for k in self.vocab.keys():
            words_num += self.vocab[k]
            if self.vocab[k] < self.word_threshold or (not k.isalpha()):
                self.vocab['unk'] += self.vocab[k]
                self.vocab.pop(k)
        print 'vocab has {} words (threshold={}).'.format(len(self.vocab), self.word_threshold)
        print 'all {} words, unk has {}/{:.3f}'.format(words_num, self.vocab['unk'], 1.0*self.vocab['unk']/words_num)

        # word2label, label2word
        voc_k = self.vocab.keys()
        voc_k.sort()
        self.word2label = {'.':0, 'unk':1}
        self.label2word = {0:'.', 1:'unk'}
        for token in ['.', 'unk']:
            if token in voc_k:
                voc_k.remove(token)
        for i,k in enumerate(voc_k):
            self.word2label[k] = i+2
            self.label2word[i+2] = k

        print 'create [word2label, label2word, idf_mask] done.'


    def create_patch(self):
        columns = ['file_name', 'coco_id', 'split_ksplit', 'sent_0', 'sent_1', 'sent_2', 'sent_3', 'sent_4'] + \
                  ['label_0_{}'.format(i) for i in range(self.sentence_length)] + \
                  ['label_1_{}'.format(i) for i in range(self.sentence_length)] + \
                  ['label_2_{}'.format(i) for i in range(self.sentence_length)] + \
                  ['label_3_{}'.format(i) for i in range(self.sentence_length)] + \
                  ['label_4_{}'.format(i) for i in range(self.sentence_length)] + \
                  ['common_words_{}'.format(i) for i in range(2 * self.sentence_length)] + \
                  np.arange(self.feat_num * self.feat_dim).tolist()

        print 'start to process [ktrain, ktest] data'
        file_names = self.data.keys()
        voc_k      = self.vocab.keys()
        for i,file_name in enumerate(file_names):
            # file_name, coco_id, split_ksplit
            series = [file_name, self.data[file_name]['coco_id'], self.data[file_name]['split_ksplit']]
            if (i+1)%10000 == 0:
                print 'create patch  for {}/{} images done.'.format(i+1, len(file_names))
            if self.data[file_name]['split_ksplit'] == 'test':
                continue
            #sents = self.data[file_name]['sents']

            # sentences & labels $ common_words
            sentences = []
            labels    = []
            for sent_id in self.data[file_name]['sents'].keys():
                tokens = self.data[file_name]['sents'][sent_id]['tokens']
                for j,token in enumerate(tokens):
                    if token not in voc_k:
                        tokens[j] = 'unk'
                sentence_tmp = ' '.join(tokens)

                label = map(lambda x:self.word2label[x], tokens)
                if len(label) < self.sentence_length:
                    label += [0] * (self.sentence_length - len(label))


                self.data[file_name]['sents'][sent_id]['tokens'] = tokens
                self.data[file_name]['sents'][sent_id]['sent']   = sentence_tmp
                self.data[file_name]['sents'][sent_id]['label']  = label
                sentences.append(sentence_tmp)
                labels += label
            common_words = self.data[file_name]['common_words']



        self.patch = {}
        self.patch['vocab'] = self.vocab
        self.patch['word2label'] = self.word2label
        self.patch['label2word'] = self.label2word
        self.patch['cocoid2filename'] = self.cocoid2filename
        self.patch['filename2cocoid'] = self.filename2cocoid
        self.patch['split_ksplit'] = self.split_ksplit
        self.patch['data'] = self.data



if __name__ == '__main__':
    coco = coco()