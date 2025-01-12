import torch
import torch.nn as nn
import shutil
import os
import numpy as np
import matplotlib

matplotlib.use('Agg')



def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)  # batch_size

        _, pred = output.topk(maxk, 1, True, True)  # pred: (batch_size, maxk)
        pred = pred.t()  # pred: (maxk, batch_size)
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def precision_recall_matrix(predicts, labels, n=10):
    final_pre = torch.zeros([0]).cuda()
    final_recall = torch.zeros([0]).cuda()
    for cur in range(n):
        pre_topk = predicts.topk(cur + 1)[1]
        pre_topk_onehot = torch.zeros(labels.shape).cuda()
        pre_topk_onehot = pre_topk_onehot.scatter_(1, pre_topk, 1)

        hit = torch.sum(pre_topk_onehot * labels, dim=1)
        # precision
        precision = hit / (cur + 1)
        recall = hit * (1. / torch.sum(labels, dim=1))
        final_pre = torch.cat((final_pre, precision.unsqueeze(1)), dim=1)
        final_recall = torch.cat((final_recall, recall.unsqueeze(1)), dim=1)

    return final_pre, final_recall


def precision_recall(predicts, labels, n=10):
    predicts = predicts.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    # from top to bottom
    sorted_predicts = (-predicts).argsort()
    top_n_inds = sorted_predicts[:, :n]

    # compute top-n hits for each sample
    hit = np.zeros([len(labels), n])
    for i in range(len(labels)):

        for j in range(1, n + 1):
            for k in range(j):
                if labels[i, top_n_inds[i, k]] - 1 == 0:
                    hit[i, j - 1] += 1
    # compute precision
    denominator = np.arange(n) + 1  # 10
    denominator = np.tile(denominator, [len(labels), 1])
    # get precision
    precision = hit / denominator  # (128,10)

    denominator = np.sum(labels, axis=1)  # (128)

    denominator = np.tile(np.expand_dims(denominator, axis=1), [1, n])  # (128,10)
    # get recall
    recall = hit / denominator  # (128,10)

    return precision, recall


def frozen_params(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def para_name(args):
    if args.stage == 1:
        name_para = 'datset={}~stage={}~img_net={}~num_segments={}~bs={}~lr_m1={}~lr_m2={}~lrd={}~wd={}~lrd_rate={}'.format(
            args.dataset,
            args.stage,
            args.img_net,
            #args.method,
            args.num_segments,
#             args.frozen_blks,
            args.batch_size,
            args.lr_m1,
            args.lr_m2,
            args.lr_decay,
            args.weight_decay,
            args.lrd_rate,
        )
    elif args.stage == 2:
        name_para = 'datset={}~stage={}~img_net={}~bs={}~lr_m1={}~threshold={}~lrd={}~wd={}~lrd_rate={}'.format(
            args.dataset,
            args.stage,
            #args.word_net,
            args.img_net,
            args.batch_size,
            args.lr_m1,
            args.threshold,
            args.lr_decay,
            args.weight_decay,
            args.lrd_rate,
        )

    elif args.stage ==3:
        name_para = 'datset={}~stage={}~img_net={}~num_segments={}~bs={}~lr_m1={}~lr_m2={}~lrd={}~wd={}~lrd_rate={}'.format(
            args.dataset,
            args.stage,
            args.img_net,
            args.num_segments,
#             args.frozen_blks,
            #args.method,
            args.batch_size,
            args.lr_m1,
            args.lr_m2,
            args.lr_decay,
            args.weight_decay,
            args.lrd_rate,
        )
    elif args.stage in [4,5,6,7]:
        name_para = 'datset={}~stage={}~img_net={}~bs={}~lr_m1={}~lr_m2={}~lr_m3={}~lrd={}~wd={}~lrd_rate={}'.format(
            args.dataset,
            args.stage,
            args.img_net,
            #args.method,
            args.batch_size,
            args.lr_m1,
            args.lr_m2,
            args.lr_m3,
            args.lr_decay,
            args.weight_decay,
            args.lrd_rate,
        )
    return name_para


class AverageMeter(object):

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)