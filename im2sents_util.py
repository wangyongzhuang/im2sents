
def update_score(flag, epoch, l_pre, score, score_best, scores_z, utype='CIDEr'):
    update_flag = False
    if (utype=='SUM' and (score['Bleu_1']>score_best['Bleu_1'] or score['Bleu_4']>score_best['Bleu_4'] or score['CIDEr']>score_best['CIDEr']+0.005)) or (utype=='CIDEr' and score['CIDEr']>score_best['CIDEr']+0.002):
        update_flag = True
        # best
        score_best['epoch'] = epoch
        for k in score.keys():
            score_best[k] = max(score[k], score_best[k])

        # history
        scores_z['epoch'].append(epoch)
        for k in score.keys():
            scores_z[k].append(score[k])

        if len(scores_z['epoch']) > 10:
            for k in scores_z.keys():
                scores_z[k].pop(0)

    if flag.dataset=='coco':
        print 'Targ B-1:0.798 B-4:0.363 METEOR:0.277 ROUGE_L:0.569 CIDEr:1.201'
        #print 'Targ B-1:0.811 B-4:0.386 METEOR:0.277 ROUGE_L:0.587 CIDEr:1.254'
    elif flag.dataset=='flickr8k':
        print 'Targ B-1:0.677 B-4:0.251 METEOR:0.204 ROUGE_L:----- CIDEr:0.531'
    elif flag.dataset=='flickr30k':
        print 'Targ B-1:0.677 B-4:0.251 METEOR:0.204 ROUGE_L:----- CIDEr:0.531'
    print 'Best B-1:%0.3f B-4:%0.3f METEOR:%0.3f ROUGE_L:%0.3f CIDEr:%0.3f Epoch:%0.1f'%(score_best['Bleu_1'], score_best['Bleu_4'], score_best['METEOR'], score_best['ROUGE_L'], score_best['CIDEr'], score_best['epoch'])
    print 'Now  B-1:%0.3f B-4:%0.3f METEOR:%0.3f ROUGE_L:%0.3f CIDEr:%0.3f Epoch:%0.1f'%(score['Bleu_1'], score['Bleu_4'], score['METEOR'], score['ROUGE_L'], score['CIDEr'], epoch)

    return update_flag, score_best, scores_z

def smooth(x_pre, x):
    if x_pre==0:
        return x
    else:
        if x < 10:
            t = 1e-4
        else:
            t = 1e-3
        if x / 2 > x_pre or x * 2 < x_pre:
            t *= 100
        elif x_pre < 1.5:
            t = 5e-3
        x_pre = (1 - t) * x_pre + t * x
        return x_pre
