import logging
import os
from urllib import request

import torch

from src_files.ml_decoder.ml_decoder import add_ml_decoder_head

logger = logging.getLogger(__name__)

from src_files.models.tresnet import TResnetM, TResnetL, TResnetXL


def create_model(config, model_path, load_head=False):
    """Create a model
    """
    model_params = {'config': config, 'num_classes': config.dataset.num_of_classes}
    config = model_params['config']
    config.model.model_name = config.model.model_name.lower()

    if config.model.model_name == 'tresnet_m':
        model = TResnetM(model_params)
    elif config.model_name == 'tresnet_l':
        model = TResnetL(model_params)
    elif config.model_name == 'tresnet_xl':
        model = TResnetXL(model_params)
    else:
        print("model: {} not found !!".format(config.model.model_name))
        exit(-1)

    ####################################################################################
    # if config.use_ml_decoder:
    #     model = add_ml_decoder_head(model,num_classes=config.num_classes,num_of_groups=config.num_of_groups,
    #                                 decoder_embedding=config.decoder_embedding, zsl=config.zsl)
    ####################################################################################
    # loading pretrain model
    # model_path = checkpoint_path
    if config.model.model_name == 'tresnet_l' and os.path.exists("./tresnet_l.pth"):
        model_path = "./tresnet_l.pth"
    if model_path:  # make sure to load pretrained model
        if not os.path.exists(model_path):
            print("downloading pretrain model...")
            request.urlretrieve(config.model_path, "./tresnet_l.pth")
            model_path = "./tresnet_l.pth"
            print('done')
        state = torch.load(model_path, map_location='cpu')
        if not load_head:
            if 'model' in state:
                key = 'model'
            else:
                key = 'state_dict'
            filtered_dict = {k: v for k, v in state[key].items() if
                             (k in model.state_dict() and 'head.fc' not in k)}
            model.load_state_dict(filtered_dict, strict=False)
        else:
            model.load_state_dict(state[key], strict=True)

    return model
