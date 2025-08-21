# This is a sample Python script.
import os

import torch
import time

from model.R3Dicnet import DIC
from Util.util import save_checkpoint, AverageMeter

import torchvision.transforms as transforms
from tqdm import tqdm
from losses import  sequence_loss
import sys
from Dataset.R3DicDataset import R3DicDataset,Normalization
import torch.nn.functional as F
best_EPE = -1
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def realEPE(output, target, sparse=False):
    b, _, h, w = target.size()
    upsampled_output = F.interpolate(output, (h,w), mode='bilinear', align_corners=False)
    return EPE(upsampled_output, target, sparse, mean=True)

def EPE(input_flow, target_flow, sparse=False, mean=True):
    EPE_map = torch.norm(target_flow-input_flow,2,1)
    batch_size = EPE_map.size(0)
    if sparse:
        # invalid flow is defined with both flow coordinates to be exactly 0
        mask = (target_flow[:,0] == 0) & (target_flow[:,1] == 0)
        EPE_map = EPE_map[~mask]
    if mean:
        return EPE_map.mean()
    else:
        return EPE_map.sum()/batch_size
def lossfun(output, target):
    return EPE(output, target),realEPE(output, target)
def train(train_loader, model, optimizer, epoch, scheduler):
    losslist=[]
    epelist=[]
    model.train()
    # model.training = True
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    EPEs = AverageMeter()
    epoch_size = len(train_loader)
    # switch to train mode
    end = time.time()
    data_loader = tqdm(train_loader, file=sys.stdout,ncols=200)
    for i, batch in enumerate(data_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        target_x = batch['Dispx'].to(device)
        target_y = batch['Dispy'].to(device)
        target = torch.cat([target_x, target_y], 1).to(device)
        Ref = batch['Ref'].float().to(device)
        Def = batch['Def'].float().to(device)
        # Ref = torch.cat([Ref, Ref, Ref], 1).to(device)  # torch.Size([16, 3, 256, 256])
        # Def = torch.cat([Def, Def, Def], 1).to(device)  # torch.Size([16, 3, 256, 256])
        input = torch.cat([Ref , Def], 1).to(device)  # torch.Size([16, 6, 256, 256])

        # compute output
        output = model(Ref,Def,8)
        # 字典中值的长度为2 分别对应光流，size都为torch.Size([2, 2, 384, 512])

        # 字典中值的长度为2 分别对应光流，size都为torch.Size([2, 2, 384, 512])
        loss, epe = sequence_loss(output, target)
        losslist.append(loss.item())
        epelist.append(epe.item())
        losses.update(loss.item(), target_x.size(0))
        # losses.update(loss.item(), target_x.size(0))
        # EPE_ = realEPE(output[0], target)
        EPEs.update(epe.item(), target_x.size(0))
        # compute gradient and do optimization step
        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
        scheduler.step()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 10== 0:
            data_loader.desc = '[Train] Epoch: [{0}][{1}/{2}] Time {3} Data {4} Loss {5} EPE {6}'.format(epoch, i, epoch_size, batch_time, data_time, losses, EPEs)

    infodic={"loss":losslist,"epe":epelist,"lossavg":losses.avg,"epeavg":EPEs.avg}
    # return losslist,epelist, losses.avg, EPEs.avg
    return infodic

def validate(val_loader, model, epoch):
    losslist = []
    epelist = []
    infodic = {}
    batch_time = AverageMeter()
    losses= AverageMeter()
    EPEs = AverageMeter()
    # switch to evaluate mode
    model.eval()
    epoch_size = len(val_loader)
    end = time.time()
    data_loader = tqdm(val_loader, file=sys.stdout,ncols=200)
    for i, batch in enumerate(data_loader):

        target_x = batch['Dispx'].to(device)
        target_y = batch['Dispy'].to(device)

        target= torch.cat([target_x, target_y], 1).to(device)
        # target['target_occ1'] = batch['mask'].to(device)
        Ref = batch['Ref'].float().to(device)
        Def = batch['Def'].float().to(device)

        # compute output
        output = model(Ref,Def,8)

        loss, epe = sequence_loss(output, target)
        losslist.append(loss.item())
        epelist.append(epe.item())

        losses.update(loss.item(), target_x.size(0))

        EPEs.update(epe.item(), target_x.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            data_loader.desc = '[Test] Epoch: [{0}][{1}/{2}]\t Time {3}\t  Loss {4}\t EPE {5}'.format(epoch, i, epoch_size ,batch_time,
                                                                                                         losses, EPEs)
        infodic = {"loss": losslist, "epe": epelist, "lossavg": losses.avg, "epeavg": EPEs.avg}
    return infodic, EPEs.avg

def main():
    best_EPE = -1
    batch_size =1
    num_workers = 1
    save_path = './result/'

    transform = transforms.Compose([Normalization()])
    train_data =R3DicDataset(csv_file='F:/Sp/Train_Annotations.csv',
                           root_dir='F:/Sp/Train/', transform=transform)
    test_data = R3DicDataset(csv_file='F:/Sp/Test_Annotations.csv',
                          root_dir='F:/Sp/Test/', transform=transform)

    print('{} samples found, {} train samples and {} test samples '.format(len(test_data) + len(train_data),
                                                                           len(train_data),
                                                                           len(test_data)))
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,num_workers=num_workers, pin_memory=True, shuffle=True)

    val_loader = torch.utils.data.DataLoader( test_data, batch_size=batch_size,num_workers=num_workers, pin_memory=True, shuffle=True)

    model=DIC(max_disp=4)
    model.training = True
    # device_ids = [0, 1]  # id为0和1的两块显卡
    device_ids = [0]
    # model = torch.nn.DataParallel(model, device_ids=device_ids).to(device)
    model = torch.nn.DataParallel(model, device_ids=device_ids).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=.00005, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 0.0002, 412500,
                                                 pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')
    infodict = {}
    traindict = {}
    vaildict = {}
    # # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40,80,120,160,200,240], gamma=0.5)
    for epoch in range(0, 300):
        # train for one epoch
        dict1 = train(train_loader, model, optimizer, epoch, scheduler)
        index = "Epoch" + str(epoch)
        traindict[index] = dict1
        # evaluate on test dataset
        with torch.no_grad():
            dict2, EPE = validate(val_loader, model, epoch)
        vaildict[index] = dict2
        if best_EPE < 0:
            best_EPE = EPE
        infodict["train"] = traindict
        infodict["vaild"] = vaildict
        is_best = EPE < best_EPE
        best_EPE = min(EPE, best_EPE)
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': 'R3DICnet',
            'state_dict': model.module.state_dict(),
            'best_EPE': best_EPE,
        }, infodict, is_best, save_path, 'R3DICnet-big-Final')


if __name__ == '__main__':
    main()


