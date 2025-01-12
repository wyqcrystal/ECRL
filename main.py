# -*- coding: utf-8 -*-
import time
from einops import rearrange
import torch.utils.data
#useddp
import torch.distributed as dist
from prepare_datas.transforms import *
from torch import optim
from torchvision import transforms
from prepare_datas.build_dataset import build_dataset
from utils import *
import numpy as np
from collections import OrderedDict

# load environmental settings
# --------------------------------------------------------------------settings------------------------------------------------
# basic
CUDA = 1  # 1 for True; 0 for False
SEED = 1  #
measure_best = 0  # best measurement
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 


kwargs = {'num_workers': 5, 'pin_memory': True} if CUDA else {}
if CUDA:
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

import config
opt = config.config_algorithm()
#------------------------------------------distributed training--------------------------------------------------#

if opt.useddp == True:
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(opt.local_rank)
    print("<----{}---->".format(opt.local_rank)) 
    
# log and model paths
result_path = os.path.join(opt.result_path, para_name(opt))

if not os.path.exists(result_path):
    if opt.useddp == True and dist.get_rank()!=0:
        pass
    else:
        os.makedirs(result_path)

# train settings
mode = opt.mode
EPOCHS = opt.lr_decay * 3 + 1
if opt.test_only == True:
    EPOCHS = 1

# # -----------------------------------------------------------------dataset information--------------------------------------
if opt.dataset=='msrvtt':
    opt.root_path = './msrvttdataset/'  # path to root folder
    opt.data_path ='./msrvttdataset/frame_30/'
    opt.img_path = './frame_30/'  # path to frames folder
    opt.num_class = 20
    global_feat_dict = np.load('./global_dict/msrvtt_global_dict_512.npy', allow_pickle=True)
    global_feat_dict=torch.from_numpy(global_feat_dict).to('cuda')
    bg_confounder_dictionary=np.load('bg_dict/msrvtt_class_avg_bg_features.npy')
    bg_confounder_dictionary=torch.from_numpy(bg_confounder_dictionary).to('cuda')
    bg_prior = np.load('bg_dict/msrvtt_class_bg_probabilities.npy')
    bg_prior=torch.from_numpy(bg_prior).to('cuda')

elif opt.dataset=='activitynet':
    opt.root_path = '/mnt/fuxiancode/datasets/anetdataset/'  # path to root folder
    opt.data_path ='/mnt/fuxiancode/datasets/anetdataset/frame_30/'
    opt.img_path = './frame_30/'  # path to frames folder
    opt.num_class = 200
    global_feat_dict = np.load('./global_dict/anet_global_dict_512.npy', allow_pickle=True)
    global_feat_dict=torch.from_numpy(global_feat_dict).to('cuda')
    bg_confounder_dictionary=np.load('bg_dict/anet_class_avg_bg_features.npy')
    bg_prior = np.load('bg_dict/anet_class_bg_probabilities.npy')
    bg_confounder_dictionary=torch.from_numpy(bg_confounder_dictionary).to('cuda')
    bg_prior=torch.from_numpy(bg_prior).to('cuda')
else:
    print('please input right dataset!')

# -------------------------------------------------------------dataset & dataloader-------------------------------------------

normalize = transforms.Normalize(mean=(0.486, 0.457, 0.407),std=(0.229, 0.224, 0.225))  #vit-transformer

transform_img_train=None
transform_img_test=None 

dataset_train=build_dataset(opt.stage,opt.img_path,opt.data_path,transform_img_train,mode,opt.dataset, 'train')
dataset_test=build_dataset(opt.stage,opt.img_path,opt.data_path,transform_img_test,mode,opt.dataset,'test')

if opt.useddp == True:
    #distributed
    train_sample = torch.utils.data.distributed.DistributedSampler(dataset_train)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, sampler=train_sample,**kwargs)
else:
    #dataloader
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=opt.batch_size, shuffle=True, **kwargs)
      
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=16, shuffle=False, **kwargs)


# -----------------------------------------------------------------update-Model-------------------------------------------
def get_updateModel(model, path):
    pretrained_dict = torch.load(path, map_location='cpu')['state_dict']
    
    model_dict = model.state_dict() 
    print(len(model_dict.keys()))

    new_state_dict = OrderedDict()
    for k, v in pretrained_dict.items():
        if 'backbone' in k:
            name = k[9:]
            new_state_dict[name] = v 
            
    new_state_dict_1 = OrderedDict()
    for k, v in new_state_dict.items():
            name1 = 'module1.'+k
            new_state_dict_1[name1] = v

    new_state_dict_2 = OrderedDict()
    for k, v in new_state_dict.items():
            name1 = 'module2.'+k
            new_state_dict_2[name1] = v
            
    new_state_dict_3 = OrderedDict()
    for k, v in new_state_dict.items():
            name1 = 'module3.'+k
            new_state_dict_3[name1] = v
  
    print(len(new_state_dict_1.keys()))  
    shared_dict_1 = {k: v for k, v in new_state_dict_1.items() if k in model_dict}
    model_dict.update(shared_dict_1)
    
    print(len(new_state_dict_2.keys()))  
    shared_dict_2 = {k: v for k, v in new_state_dict_2.items() if k in model_dict}
    model_dict.update(shared_dict_2)
    
    print(len(new_state_dict_3.keys()))  
    shared_dict_3 = {k: v for k, v in new_state_dict_3.items() if k in model_dict}
    model_dict.update(shared_dict_3)
    
    lens = len(shared_dict_1.keys())+len(shared_dict_2.keys())+len(shared_dict_3.keys())
    
    print("ckpt key lens{}".format(lens))
    
    model.load_state_dict(model_dict, strict=False)

    return model


from model.ECRL import ECRL
model = ECRL(num_class=opt.num_class)
load_path = "/mnt/weights/uniformerv2-base-p16-res224_clip-kinetics710-pre_8xb32-u8_kinetics400-rgb.pth"
model = get_updateModel(model,load_path)


# -----------------------------------------------------------------optimizer--------------------------------------------
for k,v in model.named_parameters():
    print(k)
    print(v.requires_grad)

if opt.img_net =='uniformerv2':
     optimizer = optim.Adam(filter(lambda p: p.requires_grad,model.parameters()), weight_decay=opt.weight_decay, lr=opt.lr_m1)
else:  
     optimizer = torch.optim.SGD(params=model.parameters(),lr=opt.lr_m1,weight_decay=opt.weight_decay,momentum=0.9)

#------------------------------------------------------distributed training--------------------------------------------------------

if opt.useddp == True :
    model = torch.nn.parallel.DistributedDataParallel(model.cuda(), device_ids=[opt.local_rank], find_unused_parameters=True)
else:
    model.cuda()

# ----------------------------------------------------------------Train-----------------------------------------------------------------------------

def train_epoch(epoch, decay, optimizer, train_stage, train_log):
    # vireo measurement
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    losses = AverageMeter('Loss', ':.4e')
    lossAB = AverageMeter('LossAB', ':.4e')
    losses_con = AverageMeter('Loss', ':.4e')
    losses_cls1 = AverageMeter('Loss', ':.4e')
    losses_cls2 = AverageMeter('Loss', ':.4e')

    model.train()
    total_time = time.time()

#     ipdb.set_trace()
    for batch_idx, (data,label) in enumerate(train_loader):
        if opt.useddp == True and dist.get_rank()!=0:
            pass
        else:
            print("- - -- - - - - - -batch_idx : {}".format(batch_idx))
        start_time = time.time()
#         ipdb.set_trace()
  
        # load data
        batch_size_cur=data[0].size(0)
        if CUDA:
            if opt.useddp == True:
                videoframes = data[0].cuda(non_blocking=True)
                object_imgs = data[1].cuda(non_blocking=True)
                bg_imgs = data[2].cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
            else:
                videoframes=data[0].cuda().float()
                object_imgs = data[1].cuda().float()
                bg_imgs = data[2].cuda().float()
                label=label.cuda()
        
        else:
            assert 1 < 0, 'Please fill train_stage!'

        # prediction and loss

        video_imgs = videoframes
        object_imgs = object_imgs
        bg_imgs = bg_imgs
        
        input1 = rearrange(video_imgs, 'b t c h w ->b c t h w')
        input1 = input1.cuda().float()
            
        input2 = rearrange(object_imgs,'b t c h w ->b c t h w')
        input2 = input2.cuda().float()
            
        input3 = rearrange(bg_imgs,'b t c h w ->b c t h w')
        input3 = input3.cuda().float()
            
        output = model(input1,input2,input3,bg_confounder_dictionary,bg_prior,global_feat_dict,batch_size_cur)
            
        criterion = nn.CrossEntropyLoss()
        loss_cls = criterion(output, label)
        final_loss = loss_cls 


        #compute loss
        losses.update(final_loss.item(), batch_size_cur)

        optimizer.zero_grad()
        final_loss.requires_grad_(True)
        final_loss.backward()
        optimizer.step()

        acc1, acc5 = accuracy(output, label, topk=(1, 5))
        top1.update(acc1[0], batch_size_cur)
        top5.update(acc5[0], batch_size_cur)

       
        optimizer_cur = optimizer
#             log_out = ('Epoch: [{0}][{1}/{2}], lr_m1: {lr_m1:.5f}\t lr_m2: {lr_m2:.5f}\t'
#                            'Time {data_time:.3f}\t'
#                            'Loss {loss.val:.4f}({loss.avg:.4f})\t'
#                            'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
#                            'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
#                     epoch, batch_idx, len(train_loader), data_time=round((time.time() - total_time), 4), loss=losses,
#                     top1=top1, top5=top5, lr_m1=optimizer_m1.param_groups[-1]['lr'],lr_m2=optimizer_m2.param_groups[-1]['lr']))
        log_out = ('Epoch: [{0}][{1}/{2}], lr: {lr:.5f}\t'
                           'Time {data_time:.3f}\t'
                           'Loss {loss.val:.4f}({loss.avg:.4f})\t'
                           'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                           'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    epoch, batch_idx, len(train_loader), data_time=round((time.time() - total_time), 4), loss=losses,
                    top1=top1, top5=top5, lr=optimizer_cur.param_groups[-1]['lr']))  
            
        if opt.useddp == True and dist.get_rank()!=0:
            pass
        else:
            train_log.write(log_out + '\n')
            train_log.flush()
        
# ----------------------------------------------------------------Test--------------------------------------------------------
def test_epoch(epoch, stage, test_log):
    # vireo measurement
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    top1_word = AverageMeter('Acc@1', ':6.2f')
    top5_word = AverageMeter('Acc@5', ':6.2f')
    losses = AverageMeter('Loss', ':.4e')
    losses_con = AverageMeter('Loss', ':.4e')
    losses_cls1 = AverageMeter('Loss', ':.4e')
    losses_cls2 = AverageMeter('Loss', ':.4e')
    class_correct = [0. for _ in range(opt.num_class)]
    class_total = [0. for _ in range(opt.num_class)]
    
#     class_acc = {}
    model.eval()
#     ipdb.set_trace()
    start_time = time.time()
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(test_loader):
            if opt.useddp == True and dist.get_rank()!=0:
                pass
            else:
                print("- - -- - - - - - -test_batch_idx : {}".format(batch_idx))
            # load data
            
            
            batch_size_cur = data[0].size(0)
            if CUDA:
                videoframes=data[0].cuda().float()
                object_imgs = data[1].cuda().float()
                bg_imgs = data[2].cuda().float()
                label=label.cuda()
                
                
            video_imgs = videoframes
            object_imgs = object_imgs
            bg_imgs = bg_imgs
        
            input1 = rearrange(video_imgs, 'b t c h w ->b c t h w')
            input1 = input1.cuda().float()

            input2 = rearrange(object_imgs,'b t c h w ->b c t h w')
            input2 = input2.cuda().float()

            input3 = rearrange(bg_imgs,'b t c h w ->b c t h w')
            input3 = input3.cuda().float()
                
            output = model(input1,input2,input3,bg_confounder_dictionary,bg_prior,global_feat_dict,batch_size_cur)
            criterion = nn.CrossEntropyLoss()
            loss_cls = criterion(output, label)
            acc1, acc5 = accuracy(output, label, topk=(1, 5))
            top1.update(acc1[0], batch_size_cur)
            top5.update(acc5[0], batch_size_cur)
            losses.update(loss_cls.item(), batch_size_cur)
     
        log_out = (
                    'Epoch: {epoch} Results: Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} Loss {loss.avg:.5f} Time {time:.3f}'
                    .format(epoch=epoch, top1=top1, top5=top5, loss=losses, time=round((time.time() - start_time), 4)))
        print(log_out)
            
        if opt.useddp == True and dist.get_rank()!=0:
            pass
        else:
            test_log.write(log_out + '\n')
            test_log.flush()
        return top1.avg

def lr_scheduler(epoch, optimizer, lr_decay_iter, decay_rate):
    if not (epoch % lr_decay_iter):
        for i in range(len(optimizer.param_groups)):
            optimizer.param_groups[i]['lr'] = optimizer.param_groups[i]['lr'] * decay_rate


if __name__ == '__main__':
    log_training = open(os.path.join(result_path, 'log_train.csv'), 'w')
    log_testing = open(os.path.join(result_path, 'log_test.csv'), 'w')

    for epoch in range(1, EPOCHS + 1):
        lr_scheduler(epoch, optimizer, opt.lr_decay, opt.lrd_rate)
        if opt.test_only == False:
            train_epoch(epoch, opt.lr_decay, optimizer, opt.stage, log_training)
        measure_cur = test_epoch(epoch, opt.stage, log_testing)  
        # save current model
        if measure_cur > measure_best:
            if opt.useddp == True :
                if dist.get_rank()==0:
                    torch.save(model.module.state_dict(), result_path + '/model_best.pt')
                    measure_best = measure_cur
                    print("rank0 saves ckpt")
                else:
                    print("rank1 don't save ckpt")
            else:
                torch.save(model.state_dict(), result_path + '/model_best.pt')
                measure_best = measure_cur
                print("only one gpu ,and save ckpt")

                torch.save(model.state_dict(), result_path + '/model_{}.pt'.format(epoch))
        torch.save(model.state_dict(), result_path + '/model_{}.pt'.format(epoch)) 
