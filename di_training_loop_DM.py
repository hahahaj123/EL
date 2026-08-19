# Copyright (c) 2023, Weijian Luo, Peking University <pkulwj1994@icloud.com>. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Train one-step diffusion-based generative model using the techniques described in the
paper "Diff-Instruct: A Universal Approach for Transferring Knowledge From Pre-trained Diffusion Models"
https://github.com/pkulwj1994/diff_instruct

Code was modified from paper ""Elucidating the Design Space of Diffusion-Based Generative Models""
https://github.com/NVlabs/edm
"""

"""Main training loop."""

import os
import time
import copy
import json
import pickle
import psutil
import PIL.Image
import numpy as np
import torch
import dnnlib
from torch_utils import distributed as dist
from torch_utils import training_stats
from torch_utils import misc

from metrics import di_metric_main as metric_main

#----------------------------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import random



class DistributionMatchingLoss(nn.Module):
    def __init__(self, input_shape, num_networks_per_type=1, embed_dim=64):
        super().__init__()
        self.networks = nn.ModuleList()
        
        # 4种不同的网络架构
        network_types = ['simple_cnn', 'multi_scale', 'residual', 'attention']
        init_strategies = ['xavier', 'kaiming', 'normal', 'orthogonal']
        
        # 为每种类型创建网络
        for net_type, init_strategy in zip(network_types, init_strategies):
            for _ in range(num_networks_per_type):
                net = self._create_network(input_shape, embed_dim, net_type)
                self._initialize_network(net, init_strategy)
                net.eval()
                for p in net.parameters():
                    p.requires_grad_(False)
                self.networks.append(net)
    
    def _create_network(self, input_shape, embed_dim, network_type):
        """创建4种不同的轻量级网络"""
        C = input_shape[0]  # 输入通道数
        
        if network_type == 'simple_cnn':
            # 简单CNN - 最轻量
            return nn.Sequential(
                nn.Conv2d(C, 32, 3, padding=1),
                nn.GroupNorm(8, 32),
                nn.ReLU(),
                nn.AvgPool2d(2),  # 32×32
                
                nn.Conv2d(32, 64, 3, padding=1),
                nn.GroupNorm(8, 64),
                nn.ReLU(),
                nn.AvgPool2d(2),  # 16×16
                
                nn.AdaptiveAvgPool2d((4, 4)),
                nn.Flatten(),
                nn.Linear(64 * 16, embed_dim)
            )
        
        elif network_type == 'multi_scale':
            # 多尺度特征提取
            return MultiScaleNetworkLite(C, embed_dim)
        
        elif network_type == 'residual':
            # 残差网络
            return ResidualNetworkLite(C, embed_dim)
        
        elif network_type == 'attention':
            # 注意力网络
            return AttentionNetworkLite(C, embed_dim)
        
        else:
            raise ValueError(f"Unknown network type: {network_type}")
    
    def _initialize_network(self, net, strategy='kaiming'):
        """不同的初始化策略"""
        for m in net.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                if strategy == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                elif strategy == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif strategy == 'normal':
                    nn.init.normal_(m.weight, 0, 0.02)
                elif strategy == 'orthogonal':
                    nn.init.orthogonal_(m.weight)
                
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, real_images, generated_images):
        total_loss = 0.0
        
        for net in self.networks:
            real_emb = net(real_images)
            gen_emb = net(generated_images)
            
            mmd = self._compute_mmd(real_emb, gen_emb)
            total_loss += mmd
        
        return total_loss / len(self.networks)
    
    def _compute_mmd(self, x, y):
        """优化的MMD计算 - 多尺度RBF kernel"""
        def rbf_kernel(x, y, sigma=1.0):
            # 高效计算欧氏距离
            xx = (x ** 2).sum(1, keepdim=True)
            yy = (y ** 2).sum(1, keepdim=True)
            xy = torch.mm(x, y.t())
            dist = xx + yy.t() - 2 * xy
            return torch.exp(-dist / (2 * sigma ** 2))
        
        mmd = 0
        # 多尺度kernel
        for sigma in [0.5, 1.0, 2.0]:
            xx = rbf_kernel(x, x, sigma).mean()
            yy = rbf_kernel(y, y, sigma).mean()
            xy = rbf_kernel(x, y, sigma).mean()
            mmd += (xx + yy - 2 * xy)
        
        return torch.clamp(mmd / 3, min=0.0)


# ============ 4种网络架构的具体实现 ============

class MultiScaleNetworkLite(nn.Module):
    """多尺度特征提取 - 适合64×64"""
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        # 3个不同尺度的卷积
        self.conv_3x3 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv_5x5 = nn.Conv2d(in_channels, 32, 5, padding=2)
        self.conv_7x7 = nn.Conv2d(in_channels, 32, 7, padding=3)
        
        self.gn = nn.GroupNorm(8, 96)  # 32*3=96
        self.fusion = nn.Conv2d(96, 64, 1)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 16, embed_dim)
    
    def forward(self, x):
        # 多尺度特征
        f1 = F.relu(self.conv_3x3(x))
        f2 = F.relu(self.conv_5x5(x))
        f3 = F.relu(self.conv_7x7(x))
        
        # 特征融合
        fused = torch.cat([f1, f2, f3], dim=1)
        fused = F.relu(self.fusion(self.gn(fused)))
        
        # 池化和分类
        pooled = self.pool(fused).flatten(1)
        return self.fc(pooled)


class ResidualNetworkLite(nn.Module):
    """轻量级残差网络 - 适合64×64"""
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        
        # 2个残差块
        self.res1 = ResidualBlockLite(32, 64)
        self.res2 = ResidualBlockLite(64, 64)
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 16, embed_dim)
    
    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.avg_pool2d(x, 2)  # 32×32
        
        x = self.res1(x)
        x = F.avg_pool2d(x, 2)  # 16×16
        
        x = self.res2(x)
        x = self.pool(x).flatten(1)
        return self.fc(x)


class ResidualBlockLite(nn.Module):
    """轻量级残差块"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.GroupNorm(8, out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += residual
        return F.relu(out)


class AttentionNetworkLite(nn.Module):
    """轻量级注意力网络 - 适合64×64"""
    def __init__(self, in_channels, embed_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, 32)
        
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)
        
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(64, 16, 1),
            nn.ReLU(),
            nn.Conv2d(16, 64, 1),
            nn.Sigmoid()
        )
        
        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(64, 1, 1),
            nn.Sigmoid()
        )
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Linear(64 * 16, embed_dim)
    
    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.avg_pool2d(x, 2)  # 32×32
        
        x = F.relu(self.gn2(self.conv2(x)))
        x = F.avg_pool2d(x, 2)  # 16×16
        
        # 应用注意力
        channel_att = self.channel_attention(x)
        x = x * channel_att
        
        spatial_att = self.spatial_attention(x)
        x = x * spatial_att
        
        x = self.pool(x).flatten(1)
        return self.fc(x)

def setup_snapshot_image_grid(training_set, random_seed=0):
    rnd = np.random.RandomState(random_seed)
    gw = np.clip(7680 // training_set.image_shape[2], 7, 32)
    gh = np.clip(4320 // training_set.image_shape[1], 4, 32)

    # No labels => show random subset of training samples.
    if not training_set.has_labels:
        all_indices = list(range(len(training_set)))
        rnd.shuffle(all_indices)
        grid_indices = [all_indices[i % len(all_indices)] for i in range(gw * gh)]

    else:
        # Group training samples by label.
        label_groups = dict() # label => [idx, ...]
        for idx in range(len(training_set)):
            label = tuple(training_set.get_details(idx).raw_label.flat[::-1])
            if label not in label_groups:
                label_groups[label] = []
            label_groups[label].append(idx)

        # Reorder.
        label_order = sorted(label_groups.keys())
        for label in label_order:
            rnd.shuffle(label_groups[label])

        # Organize into grid.
        grid_indices = []
        for y in range(gh):
            label = label_order[y % len(label_order)]
            indices = label_groups[label]
            grid_indices += [indices[x % len(indices)] for x in range(gw)]
            label_groups[label] = [indices[(i + gw) % len(indices)] for i in range(len(indices))]

    # Load data.
    images, labels = zip(*[training_set[i] for i in grid_indices])
    return (gw, gh), np.stack(images), np.stack(labels)
    

#----------------------------------------------------------------------------

def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape(gh, gw, C, H, W)
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape(gh * H, gw * W, C)

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], 'L').save(fname)
    if C == 3:
        PIL.Image.fromarray(img, 'RGB').save(fname)

def training_loop(
    run_dir             = '.',      # Output directory.
    dataset_kwargs      = {},       # Options for training set.
    data_loader_kwargs  = {},       # Options for torch.utils.data.DataLoader.
    network_kwargs      = {},       # Options for model and preconditioning.
    loss_kwargs         = {},       # Options for loss function.
    sg_optimizer_kwargs    = {},       # Options for optimizer.
    g_optimizer_kwargs    = {},       # Options for optimizer.
    augment_kwargs      = None,     # Options for augmentation pipeline, None = disable.
    seed                = 0,        # Global random seed.
    batch_size          = 512,      # Total batch size for one training iteration.
    batch_gpu           = None,     # Limit batch size per GPU, None = no limit.
    total_kimg          = 200000,   # Training duration, measured in thousands of training images.
    ema_halflife_kimg   = 500,      # Half-life of the exponential moving average (EMA) of model weights.
    ema_rampup_ratio    = 0.05,     # EMA ramp-up coefficient, None = no rampup.
    lr_rampup_kimg      = None,    # Learning rate ramp-up duration.
    loss_scaling        = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    sgls                = 1,        # Loss scaling factor for reducing FP16 under/overflows.
    kimg_per_tick       = 50,       # Interval of progress prints.
    snapshot_ticks      = 50,       # How often to save network snapshots, None = disable.
    resume_pkl          = None,     # Start from the given network snapshot, None = random initialization.
    resume_state_dump   = None,     # Start from the given training state, None = reset training state.
    resume_kimg         = 0,        # Start from the given training progress.
    cudnn_benchmark     = True,     # Enable torch.backends.cudnn.benchmark?
    device              = torch.device('cuda'),
    metrics = None,
    init_sigma = None,
    ema_mu = None, 
    use_fp16 = None,
    transfer_pkl = None, 
):

    # Initialize.
    start_time = time.time()
    np.random.seed((seed * dist.get_world_size() + dist.get_rank()) % (1 << 31))
    torch.manual_seed(np.random.randint(1 << 31))
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False

    # Select batch size per GPU.
    batch_gpu_total = batch_size // dist.get_world_size()
    if batch_gpu is None or batch_gpu > batch_gpu_total:
        batch_gpu = batch_gpu_total
    num_accumulation_rounds = batch_gpu_total // batch_gpu
    assert batch_size == batch_gpu * num_accumulation_rounds * dist.get_world_size()

    # Load dataset.
    dist.print0('Loading dataset...')
    dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # subclass of training.dataset.Dataset
    dataset_sampler = misc.InfiniteSampler(dataset=dataset_obj, rank=dist.get_rank(), num_replicas=dist.get_world_size(), seed=seed)
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, sampler=dataset_sampler, batch_size=batch_gpu, **data_loader_kwargs))

    # Construct network.
    dist.print0('Constructing network...')
    interface_kwargs = dict(img_resolution=dataset_obj.resolution, img_channels=dataset_obj.num_channels, label_dim=dataset_obj.label_dim)
    net = dnnlib.util.construct_class_by_name(**network_kwargs, **interface_kwargs) # subclass of torch.nn.Module
    net.eval().requires_grad_(False).to(device)
    Sg = copy.deepcopy(net).eval().requires_grad_(False).to(device)
    G = copy.deepcopy(net).eval().requires_grad_(False).to(device)
    Sg.train().requires_grad_(True).to(device)
    G.train().requires_grad_(True).to(device)
    
    if dist.get_rank() == 0:
        with torch.no_grad():
            images = torch.zeros([batch_gpu, G.img_channels, G.img_resolution, G.img_resolution], device=device)
            sigma = torch.ones([batch_gpu], device=device)
            labels = torch.zeros([batch_gpu, G.label_dim], device=device)
            misc.print_module_summary(G, [images, sigma, labels], max_nesting=2)

    # Setup optimizer.
    dist.print0('Setting up optimizer...')
    loss_fn = dnnlib.util.construct_class_by_name(**loss_kwargs) # training.loss.(VP|VE|EDM)Loss
    optimizer = dnnlib.util.construct_class_by_name(params=Sg.parameters(), **sg_optimizer_kwargs) # subclass of torch.optim.Optimizer
    g_optimizer = dnnlib.util.construct_class_by_name(params=G.parameters(), **g_optimizer_kwargs) # subclass of torch.optim.Optimizer
    augment_pipe = dnnlib.util.construct_class_by_name(**augment_kwargs) if augment_kwargs is not None else None # training.augment.AugmentPipe
    Sgddp = torch.nn.parallel.DistributedDataParallel(Sg, device_ids=[device], broadcast_buffers=False,find_unused_parameters=False)
    Gddp = torch.nn.parallel.DistributedDataParallel(G, device_ids=[device], broadcast_buffers=False,find_unused_parameters=False)
    Gema = copy.deepcopy(G).eval().requires_grad_(False)
    
    # Resume training from previous snapshot.
    if resume_pkl is not None:
        dist.print0(f'Loading network weights from "{resume_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(resume_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=net, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=G, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=Sg, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=Gema, require_all=False)
        del data # conserve memory

    # Resume training from previous snapshot.
    if transfer_pkl is not None:
        dist.print0(f'Loading network weights from "{transfer_pkl}"...')
        if dist.get_rank() != 0:
            torch.distributed.barrier() # rank 0 goes first
        with dnnlib.util.open_url(transfer_pkl, verbose=(dist.get_rank() == 0)) as f:
            data = pickle.load(f)
        if dist.get_rank() == 0:
            torch.distributed.barrier() # other ranks follow
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=G, require_all=False)
        misc.copy_params_and_buffers(src_module=data['ema'], dst_module=Gema, require_all=False)
        del data # conserve memory

    # Export sample images.
    grid_size = None
    grid_z = None
    grid_c = None
        
    if dist.get_rank() == 0:
        print('Exporting sample images...')
        grid_size, images, labels = setup_snapshot_image_grid(training_set=dataset_obj)
        save_image_grid(images, os.path.join(run_dir, 'reals.png'), drange=[0,255], grid_size=grid_size)
        
        grid_z = init_sigma*torch.randn([labels.shape[0], Gema.img_channels, Gema.img_resolution, Gema.img_resolution], device=device)
        grid_z = grid_z.split(batch_gpu)
        
        grid_c = torch.from_numpy(labels).to(device)
        grid_c = grid_c.split(batch_gpu)
                        
        images = torch.cat([Gema(z, (init_sigma*torch.ones(z.shape[0],1,1,1)).to(z.device), c, augment_labels=torch.zeros(z.shape[0], 9).to(z.device)).cpu() for z, c in zip(grid_z, grid_c)]).numpy()
        save_image_grid(images, os.path.join(run_dir, 'fakes_init.png'), drange=[-1,1], grid_size=grid_size)
        del images

    # Train.
    dist.print0(f'Training for {total_kimg} kimg...')
    dist.print0()
    cur_nimg = resume_kimg * 1000
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    maintenance_time = tick_start_time - start_time
    dist.update_progress(cur_nimg // 1000, total_kimg)
    stats_jsonl = None
    stats_metrics = dict()
        
    iter_cnt = 0

    while True:
        iter_cnt += 1        
        
        # Accumulate gradients.
        optimizer.zero_grad(set_to_none=True)
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(Sgddp, (round_idx == num_accumulation_rounds - 1)):                    
                images, labels = next(dataset_iterator)
                images = images.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)

                with torch.no_grad():
                    G.eval()
                    z = init_sigma*torch.randn_like(images)
                    gen_images = G(z, init_sigma*torch.ones(z.shape[0],1,1,1).to(z.device), labels, augment_labels=torch.zeros(z.shape[0], 9).to(z.device))
                    G.train()

                loss = loss_fn(net=Sgddp, images=gen_images, labels=labels, augment_pipe=augment_pipe)
                training_stats.report('SgLoss/loss', loss)
                loss.sum().mul(sgls / batch_gpu_total).backward()

        # Update weights.
        if lr_rampup_kimg > 0:
            for g in optimizer.param_groups:
                g['lr'] = sg_optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)   

        for param in Sg.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)      

        optimizer.step()
        g_optimizer.zero_grad(set_to_none=True)
        
        
        
        
        for round_idx in range(num_accumulation_rounds):
            with misc.ddp_sync(Gddp, (round_idx == num_accumulation_rounds - 1)):
                real_image, labels = next(dataset_iterator)
                real_image = real_image.to(device).to(torch.float32) / 127.5 - 1
                labels = labels.to(device)

                z = init_sigma*torch.randn_like(images)
                distribution_loss_fn = DistributionMatchingLoss(
                    input_shape=(3, 32, 32),
                    num_networks_per_type=2,
                    embed_dim=64,
                ).to(device)
                gen_images = Gddp(z, init_sigma*torch.ones(z.shape[0],1,1,1).to(z.device), labels, augment_labels=torch.zeros(z.shape[0], 9).to(z.device))
                
                Sg.eval()
                loss = loss_scaling*loss_fn.gloss(Sd=net, Sg=Sg, images=gen_images, labels=labels, augment_pipe=None)
                Sg.train()

                
                
                EL_loss = distribution_loss_fn(real_image,gen_images)
                # loss = loss.sum([1,2,3])
                
                training_stats.report('GLoss/loss', loss)
                loss = loss.sum().mul(1.0 / batch_gpu_total)

                loss =  loss + EL_loss * 10
                # print(EL_loss)
                # print(loss)
                loss *= 100
                
                loss.backward()

        # Update weights.
        if lr_rampup_kimg > 0:
            for g in g_optimizer.param_groups:
                g['lr'] = g_optimizer_kwargs['lr'] * min(cur_nimg / max(lr_rampup_kimg * 1000, 1e-8), 1)   

        for param in G.parameters():
            if param.grad is not None:
                torch.nan_to_num(param.grad, nan=0, posinf=1e5, neginf=-1e5, out=param.grad)

        g_optimizer.step()

        # Update EMA.
        if ema_mu > 0.0:
            ema_beta = ema_mu
        else:
            ema_halflife_nimg = ema_halflife_kimg * 1000
            if ema_rampup_ratio is not None:
                ema_halflife_nimg = min(ema_halflife_nimg, cur_nimg * ema_rampup_ratio)
            ema_beta = 0.5 ** (batch_size / max(ema_halflife_nimg, 1e-8))
                    
        for p_ema, p_net in zip(Gema.parameters(), G.parameters()):
            p_ema.copy_(p_net.detach().lerp(p_ema, ema_beta))

        # Perform maintenance tasks once per tick.
        cur_nimg += batch_size
        done = (cur_nimg >= total_kimg * 1000)
        if (not done) and (cur_tick != 0) and (cur_nimg < tick_start_nimg + kimg_per_tick * 1000):
            continue

        # Print status line, accumulating the same information in training_stats.
        tick_end_time = time.time()
        fields = []
        fields += [f"tick {training_stats.report0('Progress/tick', cur_tick):<5d}"]
        fields += [f"kimg {training_stats.report0('Progress/kimg', cur_nimg / 1e3):<9.1f}"]
        fields += [f"time {dnnlib.util.format_time(training_stats.report0('Timing/total_sec', tick_end_time - start_time)):<12s}"]
        fields += [f"sec/tick {training_stats.report0('Timing/sec_per_tick', tick_end_time - tick_start_time):<7.1f}"]
        fields += [f"sec/kimg {training_stats.report0('Timing/sec_per_kimg', (tick_end_time - tick_start_time) / (cur_nimg - tick_start_nimg) * 1e3):<7.2f}"]
        fields += [f"maintenance {training_stats.report0('Timing/maintenance_sec', maintenance_time):<6.1f}"]
        fields += [f"cpumem {training_stats.report0('Resources/cpu_mem_gb', psutil.Process(os.getpid()).memory_info().rss / 2**30):<6.2f}"]
        fields += [f"gpumem {training_stats.report0('Resources/peak_gpu_mem_gb', torch.cuda.max_memory_allocated(device) / 2**30):<6.2f}"]
        fields += [f"reserved {training_stats.report0('Resources/peak_gpu_mem_reserved_gb', torch.cuda.max_memory_reserved(device) / 2**30):<6.2f}"]
        torch.cuda.reset_peak_memory_stats()
        dist.print0(' '.join(fields))

        # Check for abort.
        if (not done) and dist.should_stop():
            done = True
            dist.print0()
            dist.print0('Aborting...')

        # Save network snapshot.
        if (snapshot_ticks is not None) and (done or (cur_tick % snapshot_ticks == 0 and cur_tick > 0)):
            data = dict(ema=Gema, Sg=Sg)
            for key, value in data.items():
                if isinstance(value, torch.nn.Module):
                    value = copy.deepcopy(value).eval().requires_grad_(False)
                    misc.check_ddp_consistency(value)
                    data[key] = value.cpu()
                del value # conserve memory
                
            if dist.get_rank() == 0:
                with open(os.path.join(run_dir, f'network-snapshot-{cur_nimg//1000:06d}.pkl'), 'wb') as f:
                    pickle.dump(data, f)
            del data # conserve memory

            # ## save training stats 
            # torch.save(dict(net=net, optimizer_state=optimizer.state_dict()), os.path.join(run_dir, f'training-state-{cur_nimg//1000:06d}.pt'))

            pass 
 
            if dist.get_rank() == 0:
                print('Exporting sample images...')
                images = torch.cat([Gema(z, init_sigma*torch.ones(z.shape[0],1,1,1).to(z.device).to(z.dtype), c, augment_labels=torch.zeros(z.shape[0], 9).to(z.device).to(z.dtype)).cpu() for z, c in zip(grid_z, grid_c)]).numpy()
                save_image_grid(images, os.path.join(run_dir, f'fakes{cur_nimg//1000:06d}.png'), drange=[-1,1], grid_size=grid_size)
                del images
                
                print('Evaluating metrics...')
                
            for metric in metrics:
                result_dict = metric_main.calc_metric(metric=metric, G=Gema, init_sigma=init_sigma,
                    dataset_kwargs=dataset_kwargs, num_gpus=dist.get_world_size(), rank=dist.get_rank(), device=device)
                if dist.get_rank() == 0:
                    metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=f'fakes{cur_nimg//1000:06d}.png')                        
                stats_metrics.update(result_dict.results)

        # Update logs.
        training_stats.default_collector.update()
        if dist.get_rank() == 0:
            if stats_jsonl is None:
                stats_jsonl = open(os.path.join(run_dir, 'stats.jsonl'), 'at')
            stats_jsonl.write(json.dumps(dict(training_stats.default_collector.as_dict(), timestamp=time.time())) + '\n')
            stats_jsonl.flush()
        dist.update_progress(cur_nimg // 1000, total_kimg)

        # Update state.
        cur_tick += 1
        tick_start_nimg = cur_nimg
        tick_start_time = time.time()
        maintenance_time = tick_start_time - tick_end_time
        if done:
            break

    # Done.
    dist.print0()
    dist.print0('Exiting...')

#----------------------------------------------------------------------------
