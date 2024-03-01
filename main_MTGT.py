import cv2
import os
import random
import torch
import copy
import time
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torch.utils import data
import numpy as np
import pandas as pd
# from tools_mine.produse_label import produce_label
from fit_MTGT import fit,set_seed,write_options
from sklearn import metrics
from dataset.Load_Dataset import RandomGenerator, ValGenerator, ImageToImage2D, LV2D
from dataset.create_dataset_rgb import for_train_transform,test_transform
import argparse
import warnings
import segmentation_models_pytorch as smp
import torch.backends.cudnn as cudnn
import Config as config
from utils import read_text
from model.MTGT import MTGT
from torchvision import transforms
import torch.distributed as dist
warnings.filterwarnings("ignore")
# parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--train_dataset', type=str,default='/mnt/ai2022/zlx/dataset/MosMeDataPlus/Train Set/', )
parser.add_argument('--val_dataset', type=str,default='/mnt/ai2022/zlx/dataset/MosMeDataPlus/Val Set/', )
parser.add_argument('--batch_size', default=8,type=int,help='batchsize')
parser.add_argument('--workers', default=4,type=int,help='batchsize')
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--start_epoch', '-s', default=0, type=int, )
parser.add_argument('--warm_epoch', '-w', default=0, type=int, )
parser.add_argument('--end_epoch', '-e', default=100, type=int, )
parser.add_argument('--num_class', '-t', default=2, type=int,)
parser.add_argument('--device', default='cuda', type=str, )
parser.add_argument('--checkpoint', type=str, default='MTGT/MosMeDataPlus/', )
parser.add_argument('--save_name', type=str, default= 'MTGt', )
parser.add_argument('--devicenum', default='0', type=str, )

parser.add_argument(
    "--config", metavar="FILE", help="Path to a pretraining config file."
)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES']=args.devicenum
begin_time = time.time()

set_seed(seed=2021)
device = args.device
if not os.path.exists(args.checkpoint):os.mkdir(args.checkpoint)
model_savedir = args.checkpoint + args.save_name.replace('0',args.name) + '/'#+'lr'+ str(args.lr)+ 'bs'+str(args.batch_size)+'/'
save_name =model_savedir +'ckpt'
print(model_savedir)
if not os.path.exists(model_savedir):os.mkdir(model_savedir)
epochs = args.warm_epoch + args.end_epoch

# train_imgs = [cv2.resize(np.load(i), (args.resize,args.resize))[:,:,::-1] for i in train_imgs]
train_text = read_text(config.train_dataset + 'Train_text.xlsx')
val_text = read_text(config.val_dataset + 'Val_text.xlsx')
train_tf = transforms.Compose([RandomGenerator(output_size=[config.img_size, config.img_size])])
val_tf = ValGenerator(output_size=[config.img_size, config.img_size])

best_acc_final = []
def main():
    cudnn.benchmark = False
    cudnn.deterministic = True

 
    config_vit = config.get_CTranS_config()
    model = LViT0(MTGT, n_channels=config.n_channels, n_classes=2)


    # model.encoder.load_state_dict(torch.load('tools_seg/resnet34-333f7ec4.pth'))
    model= model.to('cuda')

    train_dataset = ImageToImage2D(args.train_dataset, config.task_name, train_text, train_tf,
                                   image_size=config.img_size)
    val_dataset = ImageToImage2D(args.val_dataset, config.task_name, val_text, val_tf, image_size=config.img_size)

    criterion = nn.CrossEntropyLoss(weight=None).to('cuda') #weight=torch.tensor([1,10]

    best_model_wts = None
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-8)
    train_dl = DataLoader(train_dataset,shuffle=True,batch_size=args.batch_size,pin_memory=False,num_workers=0,drop_last=True,)
    val_dl = DataLoader(val_dataset,batch_size=args.batch_size,pin_memory=False,num_workers=0,)
    best_acc = 0
    with tqdm(total=epochs, ncols=60) as t:
        for epoch in range(epochs):
            epoch_loss, epoch_iou, epoch_val_loss, epoch_val_iou = \
                fit(epoch,epochs,model,train_dl,val_dl,device,criterion,optimizer,CosineLR)

            f = open(model_savedir + 'log'+'.txt', "a")
            f.write('epoch' + str(float(epoch)) +
                    '  _train_loss'+ str(epoch_loss)+'  _val_loss'+str(epoch_val_loss)+
                    ' _epoch_acc'+str(epoch_iou)+' _val_iou'+str(epoch_val_iou)+   '\n')

            if epoch_val_iou > best_acc:
                f.write( '\n' + 'here' + '\n')
                best_model_wts = copy.deepcopy(model.state_dict())
                best_acc = epoch_val_iou
                torch.save(best_model_wts, ''.join([save_name,  '.pth']))
            torch.save(best_model_wts, ''.join([save_name, 'last.pth']))
            f.close()
            # torch.cuda.empty_cache()
            t.update(1)
    write_options(model_savedir,args,best_acc)


    # dice,pre,recall,f1_score ,pa = test_mertric_here(model,test_imgs,test_masks,save_name)
    # f = open('./checkpoint/result_txt/' + 'r34u_base_train'+'.txt', "a")
    # f.write(str(model_savedir)+'  dice'+str(dice)+'  pre'+str(pre)+'  recall'+str(recall)+
    #         '  f1_score'+str(f1_score)+'  pa'+str(pa)+'\n')
    # f.close()
    # print('test_acc',acc)
    # print('best_acc','%.4f'%best_acc)

if __name__ == '__main__':
    main()


# print(save_name)
