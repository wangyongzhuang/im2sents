import json as js
import numpy as np
import scipy.spatial.distance as dist

#from nlgeval import *
from collections import defaultdict
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import pdb

from im2sents_config import *

def get_evaluate_tool():
    class COCOEvalCap:
        def __init__(self):
            self.evalImgs = []
            self.eval = {}
            self.imgToEval = {}

        def evaluate(self, refs, ress, show_info=False, stype='ALL'):
            gts = refs
            res = ress

            # =================================================
            # Set up scorers
            # =================================================
            #print 'setting up scorers...'
            if stype=='ALL':
                scorers = [
                    (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                    (Meteor(),"METEOR"),
                    (Rouge(), "ROUGE_L"),
                    (Cider(), "CIDEr")
                    ]
            elif stype=='CIDEr':
                scorers = [(Cider(), "CIDEr")]


            # =================================================
            # Compute scores
            # =================================================
            for scorer, method in scorers:
                score, scores = scorer.compute_score(gts, res)
                if type(method) == list:
                    for sc, scs, m in zip(score, scores, method):
                        self.eval[m] = sc
                        self.setImgToEvalImgs(scs, gts.keys(), m)
                        if show_info:
                            print "%s: %0.3f"%(m, sc)
                else:
                    self.eval[method] = score
                    self.setImgToEvalImgs(scores, gts.keys(), method)
                    if show_info:
                        print "%s: %0.3f"%(method, score)
            return self.eval, self.imgToEval


        def setImgToEvalImgs(self, scores, imgIds, method):
            for imgId, score in zip(imgIds, scores):
                if not imgId in self.imgToEval:
                    self.imgToEval[imgId] = {}
                    self.imgToEval[imgId]["image_id"] = imgId
                self.imgToEval[imgId][method] = score
    cocoEval = COCOEvalCap()
    return cocoEval


def get_wvst_mask(cfg):
    wvst_mask = np.zeros((cfg.sentence_length, cfg.sentence_length))
    for i in range(cfg.sentence_length):
        for j in range(i, cfg.sentence_length):
            wvst_mask[i,j] = cfg.reward_ratio ** (j-i)
    return wvst_mask


def seq2sent(seq, label2word, get_length=False):
    if seq[0]==0:
        seq = seq[1:]
    seq.append(0)
    sent = ' '.join([label2word[str(_)] for _ in seq[:seq.index(0)]])
    if get_length:
        return sent, seq.index(0)
    else:
        return sent

def get_each_score(cfg, cocoEval, res, refs, label2word, score_type='CIDEr', ref_words=None):
    ges = {}
    gts  = defaultdict(list)
    common_words_mask = []
    for i in range(res.shape[0]):
        # res
        candidate, l = seq2sent(res[i,:].tolist(), label2word, get_length=True)
        ges[i] = [candidate]
        gts[i] = refs[i].tolist()

        # mask
        if ref_words is not None:
            # p: 1, n: -1, none: 0
            ref_word_mask_tmp = np.array(map(lambda x: float(x in ref_words[i]), candidate.split())) * 2 - 1
            ref_word_mask_tmp = ref_word_mask_tmp.tolist()
            if len(ref_word_mask_tmp) < cfg.sentence_length:
                ####!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!############# the end '.' need reward, or will not converge.
                ref_word_mask_tmp += [1]
                ref_word_mask_tmp += [0] * (cfg.sentence_length - len(ref_word_mask_tmp))
            common_words_mask.append(ref_word_mask_tmp)

    score, scores = cocoEval.evaluate(gts, ges, stype=score_type)
    scores = [scores[_][score_type] for _ in range(res.shape[0])]

    if ref_words is None:
        return score[score_type], scores
    else:
        return score[score_type], scores, np.array(common_words_mask).T


def get_rl_source(cfg, sess, model, label2word, anneal_rate, cocoEval, wvst_mask, score_type='CIDEr'):
    feat, refs, common_words, res_max, res_scst, samp_mask = sess.run([model.feat_ktrain, model.refs_ktrain, model.ref_words_ktrain, model.res_max_ktrain, model.sequence_xe, model.samp_mask_xe], feed_dict={model.anneal_rate:[anneal_rate]})

    # scores_mask,
    score_scst, scores_scst, common_words_mask = get_each_score(cfg, cocoEval, res_scst, refs, label2word, score_type=score_type, ref_words=common_words)
    score_max,  scores_max  = get_each_score(cfg, cocoEval, res_max, refs, label2word, score_type=score_type, ref_words=None)


    scores_scst = np.tile(np.array(scores_scst), (cfg.sentence_length,1))
    scores_max  = np.tile(np.array(scores_max), (cfg.sentence_length,1))

    # reward base
    scores_base = scores_scst - scores_max
    
    #scores_base_group = scores_base.reshape([cfg.sentence_length, 5, cfg.batch_size])
    #scores_base_group_norm = scores_base_group - np.mean(scores_base_group, axis=1, keepdims=True)
    #scores_base = scores_base_group_norm.reshape([cfg.sentence_length, 5 * cfg.batch_size])

    #scores_up = np.maximum(scores_max, scores_scst)

    max2up = (scores_max / scores_scst)
    max2up[np.isnan(max2up)] = 1
    max2up = np.minimum(max2up, 10)
    max2up = np.maximum(max2up, 0.1)
    #pdb.set_trace()

    samp_mask = samp_mask[:,1:].T

    
    if cfg.rl_ver==0:
        reward_mask_ratio =  ((1 - samp_mask) * np.abs(common_words_mask) * (1) + \
                              (samp_mask * common_words_mask > 0) * (1) + \
                              (samp_mask * common_words_mask < 0) * (1))

                          
        reward_mask = reward_mask_ratio * scores_base
        value_mask  = reward_mask#np.dot(wvst_mask, reward_mask)

    elif cfg.rl_ver==1:
        reward_mask_ratio =  ((1 - samp_mask) * np.abs(common_words_mask) * (1) + \
                              (samp_mask * common_words_mask > 0) * np.maximum(1, max2up) + \
                              (samp_mask * common_words_mask < 0) * (max2up))

                          
        reward_mask = reward_mask_ratio * scores_base
        value_mask  = np.dot(wvst_mask, reward_mask)

    elif cfg.rl_ver==2:
        reward_mask_ratio =  ((1 - samp_mask) * np.abs(common_words_mask) * (1) + \
                              (samp_mask * common_words_mask > 0) * (1+max2up) + \
                              (samp_mask * common_words_mask < 0) * (max2up))

                          
        reward_mask = reward_mask_ratio * scores_base
        value_mask  = np.dot(wvst_mask, reward_mask)

    elif cfg.rl_ver==3:
        reward_mask_ratio =  ((1 - samp_mask) * np.abs(common_words_mask) * (1) + \
                              (samp_mask * common_words_mask > 0) * (1) + \
                              (samp_mask * common_words_mask < 0) * (1+max2up))

                          
        reward_mask = reward_mask_ratio * scores_base
        value_mask  = np.dot(wvst_mask, reward_mask)

    elif cfg.rl_ver==4:
        reward_mask_ratio =  ((1 - samp_mask) * np.abs(common_words_mask) * (1) + \
                              (samp_mask * common_words_mask > 0) * (1+max2up) + \
                              (samp_mask * common_words_mask < 0) * (1))

                          
        reward_mask = reward_mask_ratio * scores_base
        value_mask  = np.dot(wvst_mask, reward_mask)

    elif cfg.rl_ver==5:
        reward_mask_ratio =  ((1 - samp_mask) * np.abs(common_words_mask) * (1) + \
                              (samp_mask * common_words_mask > 0) * (1) + \
                              (samp_mask * common_words_mask < 0) * (max2up))

                          
        reward_mask = reward_mask_ratio * scores_base
        value_mask  = np.dot(wvst_mask, reward_mask)

    elif cfg.rl_ver==6:
        reward_mask_ratio =  ((1 - samp_mask) * np.abs(common_words_mask) * (1) + \
                              (samp_mask * common_words_mask > 0) * (max2up) + \
                              (samp_mask * common_words_mask < 0) * (1))

                          
        reward_mask = reward_mask_ratio * scores_base
        value_mask  = np.dot(wvst_mask, reward_mask)
    #pdb.set_trace()

    '''
    rl_ver = cfg.rl_ver
    '''
    return feat, res_scst, value_mask, score_scst, score_max#, scores_scst[0,:].tolist(), scores_max[0,:].tolist(), score_scst, score_max

# kval, ktest, val
def test(cfg, sess, model, data, cocoEval, split='val'):# split in ['kval', 'ktest', 'val']
    ges = []
    gts = []
    names = []
    if split=='kval':
        num = data.kval_size
    elif split=='ktest':
        num = data.ktest_size
    elif split=='val':
        num = data.val_size

    for itr in range(num / (5 * cfg.batch_size)):
        if split=='kval':
            name, ress, refs = sess.run([model.file_name_kval, model.res_max_kval, model.refs_kval])
        elif split=='ktest':
            name, ress, refs = sess.run([model.file_name_ktest, model.res_max_ktest, model.refs_ktest])
        elif split=='val':
            name, ress, refs = sess.run([model.file_name_val, model.res_max_val, model.refs_val])

        # label2word
        ges.append(ress)
        gts.append(refs)
        names.append(name[:,0])

    # concat
    ges = np.vstack(ges).tolist()
    gts = np.vstack(gts).tolist()
    names = np.hstack(names).tolist()
    gts = {names[_]:gts[_] for _ in range(num)}
    ges = {names[_]:[seq2sent(ges[_], data.label2word)] for _ in range(num)}
    print '{} has {} imgs'.format(split, len(ges))

    # get score
    score, scores = cocoEval.evaluate(gts, ges, show_info=True, stype='ALL')
    
    if split in ['ktest', 'kval']:
        #js.dump({'gts':gts,'ges':ges,'scores':scores,'score':score}, open('result/{}/{}_result_{:0.3f}.json'.format(split, split, score['CIDEr']), 'w'))
        print '{} B-1:{:0.3f} B-4:{:0.3f} METEOR:{:0.3f} ROUGE_L:{:0.3f} CIDEr:{:0.3f}'.format(split, score['Bleu_1'], score['Bleu_4'], score['METEOR'], score['ROUGE_L'], score['CIDEr'])

    return score, scores, names, ges, gts

# web (or test)
def test_web(model, data, cfg, sess):
    ges = []
    names = []
    cocoids = []
    num = data.web_size
    for itr in range(num / (5 * cfg.batch_size)):
        name, cocoid, ress = sess.run([model.name_test, model.cocoid_test, model.res_max_test])
        names.append(name[:,0])
        cocoids.append(cocoid[:,0])
        ges.append(ress)

    ges = np.vstack(ges).tolist()
    names = np.vstack(names).tolist()
    cocoids = np.vstack(cocoids).tolist()
    ges = {names[_]:[seq2sent(ges[_], data.label2word)] for _ in range(num)}

    return names, cocoids, res





def test_single(model, data, cfg, sess, split='ktest'):
    refs = {}
    ress = {}
    

    blob = []
    if split=='ktest':
        filenames = data.ktest_file_names
    elif split=='kval':
        filenames = data.kval_file_names
    for itr in range(len(filenames)):
        if split=='kval':
            name_single, feat_single, refs_single =  sess.run([data.samp_kval_single_file_name, data.samp_kval_single_feat, data.samp_kval_single_refs])
        elif split=='ktest':
            name_single, feat_single, refs_single =  sess.run([data.samp_ktest_single_file_name, data.samp_ktest_single_feat, data.samp_ktest_single_refs])

        cell_cs_0 = [np.zeros([1,cfg.hidden_state_size])] * cfg.beam_size
        cell_hs_0 = [np.zeros([1,cfg.hidden_state_size])] * cfg.beam_size
        cell_cs_1 = [np.zeros([1,cfg.hidden_state_size])] * cfg.beam_size
        cell_hs_1 = [np.zeros([1,cfg.hidden_state_size])] * cfg.beam_size

        res = [[0]] * cfg.beam_size
        log_p = [0.0] * cfg.beam_size

        is_all = [False] * cfg.beam_size
        for t in range(cfg.sentence_length):
            res_tmp = []
            logp_tmp = []
            _is      = []
            cell_cs_0_tmp = []
            cell_hs_0_tmp = []
            cell_cs_1_tmp = []
            cell_hs_1_tmp = []

            for i in range(cfg.beam_size):
                if not is_all[i]:
                    #print res[i][-1]
                    prob, p_top, idx_top, (cell_c_0, cell_h_0), (cell_c_1, cell_h_1) = sess.run([model.prob_single, model.p_top, model.idx_top, model.cell_state_0_single, model.cell_state_1_single], feed_dict={model.feat_single:feat_single, model.idxs_single: res[i][-1:], model.cell_c_0:cell_cs_0[i], model.cell_h_0:cell_hs_0[i], model.cell_c_1:cell_cs_1[i], model.cell_h_1:cell_hs_1[i]})

                    _is = _is + [i] * cfg.beam_size
                    res_tmp = res_tmp + idx_top.tolist()
                    logp_tmp = logp_tmp + [log_p[i]+np.log(_) for _ in p_top.tolist()]
                else:
                    _is = _is + [i] * cfg.beam_size
                    res_tmp = res_tmp + (0 * idx_top).tolist()
                    logp_tmp = logp_tmp + [log_p[i] for _ in p_top.tolist()]

                cell_cs_0_tmp.append(cell_c_0)
                cell_hs_0_tmp.append(cell_h_0)
                cell_cs_1_tmp.append(cell_c_1)
                cell_hs_1_tmp.append(cell_h_1)

            log_p = []
            res_new = []
            cell_cs_0 = []
            cell_hs_0 = []
            cell_cs_1 = []
            cell_hs_1 = []
            for i in range(cfg.beam_size):
                tmp = max(logp_tmp)
                tmp_idx = logp_tmp.index(tmp)
                _i = _is[tmp_idx]
                _ii = res_tmp[tmp_idx]

                log_p.append(tmp)
                res_new.append(res[_i] + [res_tmp[tmp_idx]])
                cell_cs_0.append(cell_cs_0_tmp[_i])
                cell_hs_0.append(cell_hs_0_tmp[_i])
                cell_cs_1.append(cell_cs_1_tmp[_i])
                cell_hs_1.append(cell_hs_1_tmp[_i])
                if res_tmp[tmp_idx]==0:
                    is_all[i]=True

                if is_all[i]:
                    while _i in _is:
                        ii = _is.index(_i)
                        logp_tmp.pop(ii)
                        res_tmp.pop(ii)
                        _is.pop(ii)
                elif t==0:
                    while _ii in res_tmp:
                        ii = res_tmp.index(_ii)
                        logp_tmp.pop(ii)
                        res_tmp.pop(ii)
                        _is.pop(ii)
                else:
                    logp_tmp.pop(tmp_idx)
                    res_tmp.pop(tmp_idx)
                    _is.pop(tmp_idx)
            #pdb.set_trace()
            res = res_new
            #print res, '\n'
        for i in range(cfg.beam_size):
            r = res[i]
            #r.append(0)
            #print 'logp: ', log_p[i], '\t',' '.join([data.label2word[str(ix)] for ix in r[1:r[1:].index(0)+1]])
        res = res[log_p.index(max(log_p))]
        res.append(0)
        candidate = ' '.join([data.label2word[str(ix)] for ix in res[1:res[1:].index(0)+1]])
        print itr, name_single, ': ', candidate
        #pdb.set_trace()
        #f_res.write(candidate + '\n')
        ress[name_single[0][0]] = [candidate]
        refs[name_single[0][0]] = refs_single[0].tolist()
        '''
        refs[name_single[0][0]] = []
        txts = []
        for i in range(5):
            ref = refs_single[0][i]#.tolist()
            refs[name_single[0][0]].append(ref)
        '''
    js.dump(ress, open('result/ress_beam.json','w'))
    js.dump(refs, open('result/refs_beam.json','w'))

    score, scores = cocoEval.evaluate(refs=refs, ress=ress, show_info=True)
    print 'score:'
    print score
    return score
