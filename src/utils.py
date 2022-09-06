# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import contextlib
import math
import os
import sys
import numpy as np
import logging
import torch
import errno
from typing import Union, Tuple, List, Dict
from collections import defaultdict

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from src import dist_utils

Number = Union[float, int]

logger = logging.getLogger(__name__)

def init_logger(args, stdout_only=False):
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    stdout_handler = logging.StreamHandler(sys.stdout)
    handlers = [stdout_handler]
    if not stdout_only:
        file_handler = logging.FileHandler(filename=os.path.join(args.output_dir, 'run.log'))
        handlers.append(file_handler)
    logging.basicConfig(
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if dist_utils.is_main() else logging.WARN,
        format="[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        handlers=handlers,
    )
    return logger


def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


def save(model, optimizer, scheduler, step, opt, dir_path, name):
    model_to_save = model.module if hasattr(model, "module") else model
    path = os.path.join(dir_path, "checkpoint")
    epoch_path = os.path.join(path, name) #"step-%s" % step)
    os.makedirs(epoch_path, exist_ok=True)
    cp = os.path.join(path, "latest")
    fp = os.path.join(epoch_path, "checkpoint.pth")
    checkpoint = {
        "step": step,
        "model": model_to_save.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "opt": opt,
    }
    torch.save(checkpoint, fp)
    symlink_force(epoch_path, cp)
    if not name == 'lastlog':
        logger.info(f'Saving model to {epoch_path}')


def load(model_class, dir_path, opt, reset_params=False):
    epoch_path = os.path.realpath(dir_path)
    checkpoint_path = os.path.join(epoch_path, "checkpoint.pth")
    logger.info(f"loading checkpoint {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    opt_checkpoint = checkpoint["opt"]
    state_dict = checkpoint["model"]

    model = model_class(opt_checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model = model.cuda()
    step = checkpoint["step"]
    if not reset_params:
        optimizer, scheduler = set_optim(opt_checkpoint, model)
        scheduler.load_state_dict(checkpoint["scheduler"])
        optimizer.load_state_dict(checkpoint["optimizer"])
    else:
        optimizer, scheduler = set_optim(opt, model)

    return model, optimizer, scheduler, opt_checkpoint, step

############ OPTIM

class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(WarmupLinearScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup:
            return (1 - self.ratio)*step/float(max(1, self.warmup))

        return max(0.0,
            1.0 + (self.ratio - 1) * (step - self.warmup)/float(max(1.0, self.total - self.warmup)),
        )

class CosineScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup, total, ratio=0.1, last_epoch=-1):
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        super(CosineScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup:
            return float(step) / self.warmup
        s = float(step - self.warmup) / (self.total - self.warmup)
        return self.ratio + (1.0 - self.ratio) * math.cos(0.5 * math.pi * s)


def get_cosine_with_hard_decayed_restarts_schedule_with_warmup(
        optimizer: Optimizer, num_warmup_steps: int, num_training_steps: int, num_cycles: int = 1,
        last_epoch: int = -1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        if progress >= 1.0:
            return 0.0
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * ((float(num_cycles) * progress) % 1.0)))
                   * float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def set_optim(opt, model):
    if opt.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2), eps=opt.eps, weight_decay=opt.weight_decay)
    else:
        raise NotImplementedError('optimizer class not implemented')

    scheduler_args = {
        'warmup': opt.warmup_steps,
        'total': opt.total_steps,
        'ratio': opt.lr_min_ratio,
    }
    if opt.scheduler == 'linear':
        scheduler_class = WarmupLinearScheduler
    elif opt.scheduler == 'cosine':
        scheduler_class = CosineScheduler
    else:
        raise ValueError
    scheduler = scheduler_class(optimizer, **scheduler_args)
    return optimizer, scheduler

def get_parameters(net, verbose=False):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    message = "[Network] Total number of parameters : %.6f M" % (num_params / 1e6)
    return message


class WeightedAvgStats:
    """provides an average over a bunch of stats"""

    def __init__(self):
        self.raw_stats: Dict[str, float] = defaultdict(float)
        self.total_weights: Dict[str, float] = defaultdict(float)

    def update(self, vals: Dict[str, Tuple[Number, Number]]) -> None:
        for key, (value, weight) in vals.items():
            self.raw_stats[key] += value * weight
            self.total_weights[key] += weight

    @property
    def stats(self) -> Dict[str, float]:
        return {
            x: self.raw_stats[x] / self.total_weights[x] for x in self.raw_stats.keys()
        }

    @property
    def tuple_stats(self) -> Dict[str, Tuple[float, float]]:
        return {
            x: (self.raw_stats[x] / self.total_weights[x], self.total_weights[x])
            for x in self.raw_stats.keys()
        }

    def reset(self) -> None:
        self.raw_stats = defaultdict(float)
        self.total_weights = defaultdict(float)

    @property
    def average_stats(self) -> Dict[str, float]:
        local_dict =  {x: self.raw_stats[x] / self.total_weights[x] for x in self.raw_stats.keys()}
        global_dict = {}
        for k, v in local_dict.items():
            if not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            v = dist_utils.average_main(v.cuda())
            global_dict[k] = v.item()
        return global_dict


def load_hf(object_class, model_name, config, moco_config):
    try:
        obj = object_class.from_pretrained(model_name, config=config, moco_config=moco_config, local_files_only=True)
    except:
        obj = object_class.from_pretrained(model_name, config=config, moco_config=moco_config, local_files_only=False)
    return obj


@contextlib.contextmanager
def numpy_seed(seed, *addl_seeds):
    """Context manager which seeds the NumPy PRNG with the specified seed and
    restores the state afterward"""
    if seed is None:
        yield
        return
    if len(addl_seeds) > 0:
        seed = int(hash((seed, *addl_seeds)) % 1e6)
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)
