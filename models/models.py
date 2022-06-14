from torch import nn
from torchvision import models
from .cgd_model import CGDModel 
# from vit_pytorch.cct import CCT
# from vit_pytorch.deepvit import DeepViT


def load_model(config):
    """
    The function of loading a model by name from a configuration file
    :param config:
    :return:
    """
    arch = config.model.arch
    num_classes = config.dataset.num_of_classes
    if arch.startswith("resnet"):
        model = models.__dict__[arch](pretrained=True)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif arch.startswith("efficientnet"):
        model = models.__dict__[arch](pretrained=True)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif arch.startswith("cgd"):
        model = CGDModel(config.model.backbone, config.model.gd_config[5], config.model.feature_dim, num_classes=num_classes)
    else:
        raise Exception("model type is not supported:", arch)
    model.to("cuda")
    return model
