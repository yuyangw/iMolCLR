import os
import shutil
import builtins
import yaml
import numpy as np
from copy import deepcopy
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler

from utils.nt_xent import NTXentLoss
from utils.weighted_nt_xent import WeightedNTXentLoss
from models.ginet import GINet
from data_aug.dataset import read_smiles, collate_fn, MoleculeDataset


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


def get_dataset(batch_size, num_workers, valid_size, data_path):
    data_path = data_path
    batch_size = batch_size
    num_workers = num_workers
    valid_size = valid_size

    smiles_data = read_smiles(data_path)

    # obtain training indices that will be used for validation
    num_train = len(smiles_data)
    indices = list(range(num_train))

    np.random.shuffle(indices)

    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_smiles = [smiles_data[i] for i in train_idx]
    valid_smiles = [smiles_data[i] for i in valid_idx]

    del smiles_data

    train_dataset = MoleculeDataset(train_smiles)
    valid_dataset = MoleculeDataset(valid_smiles)

    return train_dataset, valid_dataset


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    mp.spawn(main_worker, nprocs=config['world_size'], args=(config['world_size'], config))


def main_worker(rank, world_size, config):
    gpu = deepcopy(rank)
    print("Use GPU: {} for training".format(gpu))

    if rank == 0:
        dir_name = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('runs', dir_name)
        log_writer = SummaryWriter(log_dir=log_dir)
        model_checkpoints_folder = os.path.join(log_writer.log_dir, 'checkpoints')
        _save_config_file(model_checkpoints_folder)
    else:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    dist.init_process_group(
        backend=config['backend'], world_size=world_size, rank=rank)
    torch.distributed.barrier()

    model = GINet(**config["model"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    torch.cuda.set_device(gpu)

    # clean up the cache in GPU
    torch.cuda.empty_cache()

    model.cuda(gpu)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    config['batch_size'] = int(config['batch_size'] / world_size)
    config['dataset']['num_workers'] = int((config['dataset']['num_workers'] + world_size - 1) / world_size)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    nt_xent_criterion = NTXentLoss(gpu, **config['loss'])
    weighted_nt_xent_criterion = WeightedNTXentLoss(gpu, **config['loss'])

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config['optim']['lr'], 
        weight_decay=config['optim']['weight_decay'], 
    )

    start_epoch = 0
    if config['resume_from'] == 'None':
        print("=> train from scratch, no resume checkpoint")
    else:
        if os.path.isfile(config['resume_from']):
            print("=> loading checkpoint '{}'".format(config['resume_from']))
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(gpu)
            checkpoint = torch.load(config['resume_from'], map_location=loc)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(config['resume_from'], checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(config['resume_from']))

    cudnn.benchmark = True

    train_dataset, valid_dataset = get_dataset(config['batch_size'], **config['dataset'])

    # define samplers for obtaining training batches
    train_sampler = DistributedSampler(train_dataset, shuffle=True)

    train_loader = DataLoader(
        train_dataset, batch_size=config['batch_size'], sampler=train_sampler,
        num_workers=config['dataset']['num_workers'], drop_last=True, 
        pin_memory=True, collate_fn=collate_fn
    )

    # validation loader on a single GPU 
    valid_loader = DataLoader(
        valid_dataset, batch_size=256, shuffle=True,
        num_workers=config['dataset']['num_workers'], drop_last=True, 
        pin_memory=True, collate_fn=collate_fn
    )

    scheduler = CosineAnnealingLR(optimizer, 
        T_max=len(train_loader)-config['warmup']+1, eta_min=0, last_epoch=-1
    )

    n_iter = 0
    valid_n_iter = 0
    best_valid_loss = np.inf

    for epoch_counter in range(start_epoch, config['epochs']):

        train_sampler.set_epoch(epoch_counter)

        for bn, (g1, g2, mols) in enumerate(train_loader):
            g1 = g1.cuda(gpu, non_blocking=True)
            g2 = g2.cuda(gpu, non_blocking=True)

            # get the representations and the projections
            __, z1_global, z1_sub = model(g1)  # [N,C]
            __, z2_global, z2_sub = model(g2)  # [N,C]

            # normalize projection feature vectors
            z1_global = F.normalize(z1_global, dim=1)
            z2_global = F.normalize(z2_global, dim=1)
            loss_global = weighted_nt_xent_criterion(z1_global, z2_global, mols)

            # normalize projection feature vectors
            z1_sub = F.normalize(z1_sub, dim=1)
            z2_sub = F.normalize(z2_sub, dim=1)
            loss_sub = nt_xent_criterion(z1_sub, z2_sub)

            loss = loss_global + config['loss']['lambda_2'] * loss_sub

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if rank == 0 and n_iter % config['log_every_n_steps'] == 0:
                log_writer.add_scalar('train_loss', loss, global_step=n_iter)
                log_writer.add_scalar('train_loss_global', loss_global, global_step=n_iter)
                log_writer.add_scalar('train_loss_sub', loss_sub, global_step=n_iter)
                log_writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                print(epoch_counter, bn, loss_global.item(), loss_sub.item(), loss.item())
            
            n_iter += 1

        if rank == 0:
            valid_loss_global, valid_loss_sub = validate(
                gpu, valid_loader, [weighted_nt_xent_criterion, nt_xent_criterion], model
            )
            valid_loss = valid_loss_global + config['loss']['lambda_2'] * valid_loss_sub
            print('Valid |', epoch_counter, valid_loss_global, valid_loss_sub, valid_loss)
            if valid_loss < best_valid_loss:
                # save the best model weights
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
            
            # save the model weights at each epoch
            torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(str(epoch_counter))))

            log_writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
            log_writer.add_scalar('validation_loss_global', valid_loss_global, global_step=valid_n_iter)
            log_writer.add_scalar('validation_loss_sub', valid_loss_sub, global_step=valid_n_iter)

            valid_n_iter += 1

        # warmup for the first few epochs
        if epoch_counter >= config['warmup'] - 1:
            scheduler.step()


def validate(gpu, valid_loader, criterion, model):
    model.eval()

    global_criterion, sub_criterion = criterion
    # valid_sampler.set_epoch(0)

    valid_loss_global, valid_loss_sub = 0, 0
    counter = 0

    for bn, (g1, g2, mols) in enumerate(valid_loader):
        g1 = g1.cuda(gpu, non_blocking=True)
        g2 = g2.cuda(gpu, non_blocking=True)

        # get the representations and the projections
        __, z1_global, z1_sub = model(g1)  # [N,C]
        __, z2_global, z2_sub = model(g2)  # [N,C]

        # normalize projection feature vectors
        z1_global = F.normalize(z1_global, dim=1)
        z2_global = F.normalize(z2_global, dim=1)
        loss_global = global_criterion(z1_global, z2_global, mols)

        # normalize projection feature vectors
        z1_sub = F.normalize(z1_sub, dim=1)
        z2_sub = F.normalize(z2_sub, dim=1)
        loss_sub = sub_criterion(z1_sub, z2_sub)

        valid_loss_global += loss_global.item()
        valid_loss_sub += loss_sub.item()

        if counter % 1 == 0:
            print('validation bn:', counter)

        counter += 1

    valid_loss_global /= counter
    valid_loss_sub /= counter
    
    model.train()

    return valid_loss_global, valid_loss_sub


if __name__ == '__main__':
    main()

