import json as js
import argparse
import os

def cfg():
    parser = argparse.ArgumentParser()

    # data
    parser.add_argument('--dataset', default='coco', type=str)

    parser.add_argument('--data_dir', default='data', type=str)
    parser.add_argument('--patch_name', default='patch_coco.json', type=str)

    #parser.add_argument('--label_size', default=0, type=int)
    #parser.add_argument('--data_size', default=0*5, type=int)
    parser.add_argument('--feat_num', default=37, type=int) #50
    parser.add_argument('--feat_dim', default=2048, type=int)
    parser.add_argument('--sentence_length', default=18, type=int)
    #parser.add_argument('--unk', default=False, type=bool)


    # train
    parser.add_argument('--gpu_id', default='1', type=str)
    parser.add_argument('--train_info', default='init', type=str)
    parser.add_argument('--train_approach', default='XE', type=str)#'XE', 'SCST', 'WVST', 'WVSTB', 'WVSTBS'
    parser.add_argument('--is_fintune', default=False, type=bool)

    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--loss_scale', default=1000,type=int)


    # rl_approach
    parser.add_argument('--reward_ratio', default=0.2, type=float)
    parser.add_argument('--reward_base', default='r-b', type=str)
    parser.add_argument('--reward_positite', default='r+b', type=str)
    parser.add_argument('--reward_negative', default='b', type=str)
    #parser.add_argument('--reward_avg', default=False, type=bool)
    #parser.add_argument('--reward_len', default=False, type=bool)
    parser.add_argument('--rl_ver', default=1, type=int)


    # learning_rate_xe
    parser.add_argument('--learning_rate_xe_begin', default=5e-4, type=float)
    parser.add_argument('--weight_decay_xe', default=1e-5, type=float)


    # learning_rate_rl
    parser.add_argument('--learning_rate_rl_begin', default=5e-5, type=float)
    parser.add_argument('--learning_rate_rl_end', default=5e-5, type=float)
    parser.add_argument('--learning_rate_decay', default=0.8, type=float)
    parser.add_argument('--learning_rate_epoch_decay', default=3, type=int)
    parser.add_argument('--weight_decay_rl', default=0.0, type=float)


    # sample rate
    parser.add_argument('--sample_rate_max', default=0.25, type=float)
    parser.add_argument('--sample_rate_rate', default=0.05, type=float)
    parser.add_argument('--sample_rate_step', default=5, type=int)
    parser.add_argument('--select_max', default=False, type=bool)


    # epoch
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--epoch_val', default=1, type=float)

    # model
    parser.add_argument('--hidden_state_size', default=512, type=int)
    parser.add_argument('--embedding_size', default=512, type=int)
    parser.add_argument('--drop_out', default=0.5, type=float)
    #parser.add_argument('--att_ver', default='wmie', type=str)

    # log
    parser.add_argument('--ckpt_dir', default='ckpt', type=str)
    #parser.add_argument('--sum_dir', default='sum', default=str)

    # decode
    parser.add_argument('--beam_size', default=1, type=int)

    cfg, unparsed = parser.parse_known_args()

    # mkdir
    cfg.ckpt_dir_xe = '{}/ckpt_xe'.format(cfg.ckpt_dir)
    cfg.ckpt_dir_rl = '{}/ckpt_rl_v{}'.format(cfg.ckpt_dir, cfg.rl_ver)
    if not os.path.exists(cfg.ckpt_dir_xe):
        os.mkdir(cfg.ckpt_dir_xe)
        #os.mkdir(cfg.ckpt_dir_xe.replace('xe', 'rl'))
    if not os.path.exists(cfg.ckpt_dir_rl):
        os.mkdir(cfg.ckpt_dir_rl)

    return cfg