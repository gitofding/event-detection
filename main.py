import time
from argparse import ArgumentParser
from preprocessing import Text2Digit
from train import trainFunc

import torch

Tab = "\t"

def get_args():

    parser = ArgumentParser(description='CNN-baseline')
    parser.add_argument('--no_cuda', action='store_false', help='do not use CUDA', dest='gpu')
    parser.add_argument('--batch_size', type=int, default=100)

    parser.add_argument('--epoch_num', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.4) # 原0.5
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--loss_flag', type=str, default='cross-entropy')
    parser.add_argument('--opti_flag', type=str, default='sgd') # or use 'adadelta', 'adam', 'sgd'
    parser.add_argument('--dropout', type=float, default=0.5)
    # transformer内部参数
    parser.add_argument('--ninp', type=int, default=300)
    parser.add_argument('--nhid', type=int, default=200)
    parser.add_argument('--nhead', type=int, default=30)# 多头注意力机制
    parser.add_argument('--trans_dropout', type=float, default=0.2)
    parser.add_argument('--nlayers', type=int, default=3)# 3个Encoder Layer

    data_dir = '../data/'

    parser.add_argument('-evtentarg_train', type=str, default=data_dir+'bt_data_de_ru.txt', help="trig&ent&arg train file path")
    parser.add_argument('-evtentarg_dev', type=str, default=data_dir+'trig_ner_arg.dev.txt', help="trig&ent&arg dev file path")
    parser.add_argument('-evtentarg_test', type=str, default=data_dir+'trig_ner_arg.test.txt', help="trig&ent&arg test file path")
    parser.add_argument('-pretrain_embed', type=str, default=data_dir+'glove.6B.300d.txt.ace', help="pretrain embed path")
    parser.add_argument('-model_path', type=str, default='../ni_data/checkpoints/model.cnn', help="saved model path")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print("Program starts", time.asctime())
    args = get_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

    #######################
    ## load datasets from data file
    args.train, args.dev, args.test = args.evtentarg_train, args.evtentarg_dev, args.evtentarg_test
    ace_data = Text2Digit(args)

    #######################
    ## store and output all parameters
    _, args.embed_dim = ace_data.pretrain_embedding.shape

    param_str = "_".join(["ed%s"%(args.embed_dim),"epochnum%s"%(args.epoch_num), "lr%s"%(args.lr*10), "layer%s"%(args.nlayers),"nhid%s"%(args.nhid),"nhead%s"%(args.nhead)])
    args.model_path += param_str
    args.vocab_size = len(ace_data.vocab)
    args.tagset_size = len(ace_data.atag_dict)
    print(args)

    # begin to train
    trainFunc(args, ace_data)
