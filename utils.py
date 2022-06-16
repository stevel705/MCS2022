import os
import shutil
from collections import namedtuple
from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def convert_dict_to_tuple(dictionary):
    for key, value in dictionary.items():
        if isinstance(value, dict):
            dictionary[key] = convert_dict_to_tuple(value)
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


def clear_folder(directory: str):
    """Clear file in folder if old best model exist"""
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                # os.unlink(file_path)
                os.remove(file_path)
            # elif os.path.isdir(file_path):
            #     shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))

def save_checkpoint(model, optimizer, scheduler, epoch, outdir):
    """Saves checkpoint to disk"""
    filename = "model_{:04d}.pth".format(epoch)
    directory = outdir
    filename = os.path.join(directory, filename)
    weights = model.state_dict()
    state = OrderedDict([
        ('state_dict', weights),
        ('optimizer', optimizer.state_dict()),
        ('scheduler', scheduler.state_dict()),
        ('epoch', epoch),
    ])
    # Save only best model
    clear_folder(directory)
    torch.save(state, filename)


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def get_optimizer(config, net):
    lr = config.train.learning_rate

    print("Opt: ", config.train.optimizer)

    if config.train.optimizer == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(),
                                    lr=lr,
                                    momentum=config.train.momentum,
                                    weight_decay=config.train.weight_decay)
    elif config.train.optimizer == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(),
                                    lr=lr)
    elif config.train.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(net.parameters(),
                                    lr=lr)
    elif config.train.optimizer == 'RMSprop':
        optimizer = torch.optim.RMSprop(net.parameters(),
                                    lr=lr,
                                    momentum=config.train.momentum,
                                    weight_decay=config.train.weight_decay)

    else:
        raise Exception("Unknown type of optimizer: {}".format(config.train.optimizer))
    return optimizer


def get_scheduler(config, optimizer):
    if config.train.lr_schedule.name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.train.n_epoch)
    elif config.train.lr_schedule.name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=config.train.lr_schedule.step_size,
                                                    gamma=config.train.lr_schedule.gamma)
    elif config.train.lr_schedule.name == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, threshold = 0.9)
    else:
        raise Exception("Unknown type of lr schedule: {}".format(config.train.lr_schedule))
    return scheduler


def reduce_loss(loss, reduction='mean'):
    return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss


def linear_combination(x, y, epsilon):
    return epsilon * x + (1 - epsilon) * y


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, epsilon: float = 0.1, reduction='mean'):
        super().__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, preds, target):
        n = preds.size()[-1]
        log_preds = F.log_softmax(preds, dim=-1)
        loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
        nll = F.nll_loss(log_preds, target, reduction=self.reduction)
        return linear_combination(loss / n, nll, self.epsilon)

class DenseCrossEntropy(nn.Module):

    def __init__(self):
        super(DenseCrossEntropy, self).__init__()
        
        
    def forward(self, logits, labels):
        logits = logits.float()
        labels = labels.float()
        
        logprobs = F.log_softmax(logits, dim=-1)
        
        loss = -labels * logprobs
        loss = loss.sum(-1)

        return loss.mean()

def get_training_parameters(config, net):
    # criterion = torch.nn.CrossEntropyLoss().to('cuda')
    criterion = LabelSmoothingCrossEntropy().to('cuda')
    # criterion = DenseCrossEntropy().to('cuda')
    optimizer = get_optimizer(config, net)
    scheduler = get_scheduler(config, optimizer)
    return criterion, optimizer, scheduler


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name):
        self.name = name
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

    def __call__(self):
        return self.val, self.avg


def get_max_bbox(bboxes):
    bbox_sizes = [x[2] * x[3] for x in bboxes]
    max_bbox_index = np.argmax(bbox_sizes)
    return bboxes[max_bbox_index]
