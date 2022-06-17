import click
import os, sys
import numpy as np
import random
from setproctitle import setproctitle
import inspect
import pdb

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import Parameter
from torch.optim import SGD, Adam
from torch.nn.modules.loss import CrossEntropyLoss

from task import OmniglotTask, MNISTTask
from dataset import Omniglot, MNIST
from inner_loop import InnerLoop
from omniglot_net import OmniglotNet
from score import *
from data_loading import *

class MetaLearner(object):
    def __init__(self,
                dataset,
                num_classes,
                num_inst,
                meta_batch_size, 
                meta_step_size, 
                inner_batch_size, 
                inner_step_size,
                num_updates, 
                num_inner_updates,
                loss_fn):
        super(self.__class__, self).__init__()
        self.dataset = dataset
        self.num_classes = num_classes
        self.num_inst = num_inst
        self.meta_batch_size = meta_batch_size
        self.meta_step_size = meta_step_size
        self.inner_batch_size = inner_batch_size
        self.inner_step_size = inner_step_size
        self.num_updates = num_updates
        self.num_inner_updates = num_inner_updates
        self.loss_fn = loss_fn

        # make the nets
        num_input_channels = 1 if self.dataset == 'mnist' else 3
        self.net = OmniglotNet(num_classes, self.loss_fn, num_input_channels)
        self.net.cuda()
        self.fast_net = InnerLoop(num_classes, self.loss_fn, self.num_inner_updates, self.inner_step_size, self.inner_batch_size, self.meta_batch_size, num_input_channels)
        self.fast_net.cuda()
        self.opt = Adam(self.net.parameters(), lr=meta_step_size)
    
    def get_task(self, root, n_cl, n_inst, split='train'):
        if 'mnist' in root:
            return MNISTTask(root, n_cl, n_inst, split)
        elif 'omniglot' in root:
            return OmniglotTask(root, n_cl, n_inst, split)
        else:
            print('Unknown dataset')
            raise(Exception)

    def meta_update(self, task, ls):
        print("\n Meta update \n")
        loader = get_data_loader(task, self.inner_batch_size, split='val')
        in_, target = loader.__iter__().next()
        # use a dummy forward / backward pass to get the correct grads into self.net
        loss, out = forward_pass(self.net, in_, target)
        # unpack the list of grad dicts
        gradients = {k: sum(d[k] for d in ls) for k in ls[0].keys()}
        # Register a hook on each parameter in the net that replaces the current dummy grad
        # with our grads accumulated across the meta-batch
        hooks = []
        for (k,v) in self.net.named_parameters():
            def get_closue():
                key = k
                def replace_grad(grad):
                    return gradients[key]
                return replace_grad(grad)
            hooks.append(v.register_hook(get_closue()))
        # Compute grads for current step, replace with summed gradients as defined by hook
        self.opt.zero_grad()
        loss.backward()
        # Update the net parameters with the accumulated gradient according to optimizer
        self.opt.step()
        # Remove the hooks before next training phase
        for h in hooks:
            h.remove()