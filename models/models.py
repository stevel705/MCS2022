import torch
from torch import nn
from torchvision import models
from collections import OrderedDict
# from vit_pytorch.cct import CCT
# from vit_pytorch.deepvit import DeepViT
# from efficientnet_pytorch import EfficientNet

def load_model(config):
    """
    The function of loading a model by name from a configuration file
    :param config:
    :return:
    """
    arch = config.model.arch
    num_classes = config.dataset.num_of_classes
    if config.train.pretrained:
        print("Pretrained")
        model = models.__dict__[config.model.arch](num_classes=config.dataset.num_of_classes)
        checkpoint = torch.load(config.train.pretrained, map_location='cuda')['state_dict']

        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k.replace("module.", "")
            new_state_dict[name] = v

        model.load_state_dict(new_state_dict)
    else:
        print("From Scratch")
        if arch.startswith("resnet"):
            model = models.__dict__[arch](pretrained=True)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
        elif arch.startswith("efficientnet"):
            # model_name = 'efficientnet-b7'
            # model = EfficientNet.from_pretrained(arch, num_classes=num_classes) 
            model = models.__dict__[arch](pretrained=True)
            model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        else:
            raise Exception("model type is not supported:", arch)
    model.to("cuda")
    return model
