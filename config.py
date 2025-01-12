import argparse


def config_algorithm():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='msrvtt', help='indicator to dataset')
    parser.add_argument('--result_path', type=str, default='/mnt/ecrl', help='path to the folder to save results')
    parser.add_argument('--stage', type=int, default=1,help='1:visual information;')
    parser.add_argument('--mode', type=str, default='train',
                        help='select from train, val, test. Used in dataset creation')
    parser.add_argument('--img_net', type=str, default='uniformerv2',
                        help='choose network backbone for video channel')
    
    parser.add_argument('--num_class', type=int, default='20',help='msrvtt:20,anet:200')
    parser.add_argument('--batch_size', type=int, default=8 ,help='batch size')
    parser.add_argument('--num_segments', type=int, default=8,help='num_segments')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--lr_m1', type=float, default=1e-5, help='learning rate for module 1')
    parser.add_argument('--lr_m2', type=float, default=1e-4, help='learning rate for module 2')
    parser.add_argument('--lr_m3', type=float, default=1e-4, help='learning rate for module 3')
    parser.add_argument('--lr_finetune', type=float, default=5e-5, help='fine-tune learning rate')
    parser.add_argument('--lrd_rate', type=float, default=0.1, help='decay rate of learning rate') 
    parser.add_argument('--lrd_rate_finetune', type=float, default=0.05, help='decay rate of fine-tune learning rate')
    parser.add_argument('--lr_decay', type=int, default=4, help=' decay rate of learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
    parser.add_argument('--frozen_blks', type=int, default=12, help='select i in 12 to freeze (0-i)blks')
    parser.add_argument('--exp', type=int, default=4, help='experiment idx')
    parser.add_argument('--test_only', type=bool, default=False, help='test only')
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--useddp', default=False, type=bool, help='use distributed training')
    parser.add_argument('--useapex', default=False, type=bool, help='use apex training')
    parser.add_argument('--divided', default=False, type=bool, help='use different lr training')
    parser.add_argument('--causal', type=bool, default=False, help='use causal learning')
    parser.add_argument('--threshold', type=int, default=200, help='threshold')
    parser.add_argument('--confounder_set', type=int, default=64, help='confounder set')

    args = parser.parse_args()
    return args