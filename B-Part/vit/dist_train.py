import os
import time
import torch
from torch import optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from metric import Accuracy, TopKAccuracy
from vit import ViT
from config import cases

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='B1', help='B1, B2 or BX')
parser.add_argument('--log_interval', type=int, default=1, help='number of iterations between displaying training loss')
args = parser.parse_args()


class ImageNet(ImageFolder):
    def __init__(self, root, train=True, transform=None):
        split = 'train' if train else 'val'
        root = os.path.join(root, split)
        super(ImageNet, self).__init__(root=root, transform=transform)


def only_rank0(fn):
    def wrapper(*args, **kwargs):
        rank = dist.get_rank()
        if rank == 0:
            fn(f'Rank {rank} |', *args, **kwargs)
    return wrapper

print = only_rank0(print)


def train():
    rank = dist.get_rank()
    rank_size = dist.get_world_size()

    if args.model not in ['B1', 'B2', 'BX']:
        if rank == 0:
            parser.print_help()
        exit(1)

    config = cases[args.model]

    g_batch_size = config['batch_size']
    batch_size = g_batch_size // rank_size
    lr = config['lr']
    epoch_num = config['epoch']
    log_interval = args.log_interval

    model = ViT(
        image_size=256,
        patch_size=32,
        num_classes=100,
        dim=config['dim'],
        depth=config['depth'],
        heads=config['heads'],
        mlp_dim=config['mlp_dim'],
    )

    model = DistributedDataParallel(model)

    print(f'model: {args.model}')
    print(f'training with {rank_size} procs')
    print(f'batchsize: {g_batch_size} ({batch_size} samples/rank)')
    print(f'learning rate: {lr}')

    data_path = '/home/export/online1/share/wxsc/data/data100'
    print(f'load data from: {data_path}')

    st = time.perf_counter()

    # train data
    train_transforms = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    train_dataset = ImageNet(root=data_path, train=True, transform=train_transforms)
    train_sample = DistributedSampler(train_dataset)
    train_dataloader = DataLoader(dataset = train_dataset,
            batch_size=batch_size,
            shuffle=(train_sample is None),
            sampler=train_sample,
            drop_last=True)
    print('train dataset loaded done')

    # test data
    valid_transforms = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    valid_dataset = ImageNet(root=data_path, train=False, transform=valid_transforms)
    valid_sample = DistributedSampler(valid_dataset)
    valid_dataloader = DataLoader(dataset=valid_dataset,
            batch_size=batch_size,
            shuffle=(valid_sample is None),
            sampler=valid_sample,
            drop_last=True)
    print('test dataset loaded done')

    ed = time.perf_counter()
    print(f'load data timing: {ed -st}')

    optimizer = optim.AdamW(params = model.parameters(),lr = lr,weight_decay = 0.0001)
    loss_fn = torch.nn.CrossEntropyLoss()

    train_acc = Accuracy()
    acc_top1 = Accuracy()
    acc_top5 = TopKAccuracy(top_k=5)

    for epoch_idx in range(epoch_num):
        print(f'[epoch {epoch_idx}] start training..')
        train_acc.reset()
        model.train()
        tic = time.perf_counter()
        ttic = time.time()
        for step_idx, (data, target) in enumerate(train_dataloader):
            bst = time.perf_counter()

            optimizer.zero_grad()

            st = time.perf_counter()
            output = model(data)
            ed = time.perf_counter()
            print(f'[epoch {epoch_idx}] model forward timing: {ed -st}')

            loss = loss_fn(output,target)

            st = time.perf_counter()
            loss.backward()
            ed = time.perf_counter()
            print(f'[epoch {epoch_idx}] model backward timing: {ed -st}')

            st = time.perf_counter()
            optimizer.step()
            ed = time.perf_counter()
            print(f'[epoch {epoch_idx}] optimizer.step timing: {ed -st}')


            bed = time.perf_counter()
            print(f'[epoch {epoch_idx}] batch {step_idx} timing: {bed - bst}')

            train_acc.update(target, output)

            if log_interval and not (step_idx + 1) % log_interval:
                train_acc_name, train_acc_score = train_acc.get()
                step_throughput = g_batch_size * log_interval / (time.perf_counter() - tic)
                statistics = torch.tensor([train_acc_score, loss.item()])
                dist.reduce(statistics, 0)
                statistics = statistics / rank_size
                tic = time.perf_counter()
                print(f'Epoch[{epoch_idx}] Batch [{step_idx+1}]\t'
                      f'Speed: {step_throughput} samples/sec\t'
                      f'{train_acc_name}: {statistics[0]:.4f}\t'
                      f'loss: {statistics[1]:.4f}')

        epoch_time = time.time() - ttic
        print(f"Epoch[{epoch_idx}] Time:{epoch_time}")

        # evaluation
        print(f'[epoch {epoch_idx}] start testing..')
        acc_top1.reset()
        acc_top5.reset()
        model.eval()
        with torch.no_grad():
            for i, (images, target) in enumerate(valid_dataloader):
                output = model(images)
                acc_top1.update(target, output)
                acc_top5.update(target, output)

        acc_top1_name, acc_top1_score = acc_top1.get()
        acc_top5_name, acc_top5_score = acc_top5.get()
        mean_acc = torch.tensor([acc_top1_score, acc_top5_score])
        dist.reduce(mean_acc, 0)
        mean_acc = mean_acc / rank_size
        print(f'[epoch {epoch_idx}] valid: {acc_top1_name}={mean_acc[0]}')
        print(f'[epoch {epoch_idx}] valid: {acc_top5_name}={mean_acc[1]}')



if __name__ == '__main__':
    dist.init_process_group(backend='mpi')
    train()
