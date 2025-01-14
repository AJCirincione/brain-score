import torch
import torch.nn as nn
import os
import requests
import torch.distributed
from torch.nn.parallel import DistributedDataParallel
from vonenet import VOneNet
# from .resnet import ResNet18
from torch.nn import Module

FILE_WEIGHTS = {'alexnet': 'vonealexnet_e70.pth.tar', 'resnet50': 'voneresnet50_e70.pth.tar',
                'resnet50_at': 'voneresnet50_at_e96.pth.tar', 'cornets': 'vonecornets_e70.pth.tar',
                'resnet50_ns': 'voneresnet50_ns_e70.pth.tar', 'resnet18': 'resnet18_imagenet_1000.h5'}


class Wrapper(Module):
    def __init__(self, model):
        super(Wrapper, self).__init__()
        self.module = model


def get_model(model_arch='resnet18', pretrained=True, map_location='cuda', div_norm=True, vonenet_on=True, color_channels = 1, use_full_image_net=0, restore_path=None, **kwargs):
    """
    Returns a VOneNet model.
    Select pretrained=True for returning one of the 3 pretrained models.
    model_arch: string with identifier to choose the architecture of the back-end (resnet50, cornets, alexnet)
    """
    if pretrained and model_arch:
        if (model_arch != 'resnet18'):
            url = f'https://vonenet-models.s3.us-east-2.amazonaws.com/{FILE_WEIGHTS[model_arch.lower()]}'
        else:
            url = f'https://github.com/qubvel/classification_models/releases/download/0.0.1/resnet18_imagenet_1000.h5'
        home_dir = os.environ['HOME'] #Change to USERPROFILE if on windows
        vonenet_dir = os.path.join(home_dir, '.vonenet')
        weightsdir_path = os.path.join(vonenet_dir, FILE_WEIGHTS[model_arch.lower()])
        if not os.path.exists(vonenet_dir):
            os.makedirs(vonenet_dir)
        if not os.path.exists(weightsdir_path):
            print('Downloading model weights to ', weightsdir_path)
            r = requests.get(url, allow_redirects=True)
            open(weightsdir_path, 'wb').write(r.content)
        print(model_arch.lower())
        ckpt_data = torch.load(weightsdir_path, map_location=map_location)

        stride = ckpt_data['flags']['stride']
        simple_channels = ckpt_data['flags']['simple_channels']
        complex_channels = ckpt_data['flags']['complex_channels']
        k_exc = ckpt_data['flags']['k_exc']
        div_norm = ckpt_data['flags']['divisive_norm']

        noise_mode = ckpt_data['flags']['noise_mode']
        noise_scale = ckpt_data['flags']['noise_scale']
        noise_level = ckpt_data['flags']['noise_level']

        model_id = ckpt_data['flags']['arch'].replace('_','').lower()

        model = globals()[f'VOneNet'](model_arch=model_id, stride=stride, k_exc=k_exc,
                                      simple_channels=simple_channels, complex_channels=complex_channels,
                                      noise_mode=noise_mode, noise_scale=noise_scale, noise_level=noise_level,
                                      div_norm=div_norm, vonenet_on=vonenet_on, color_channels = color_channels,
                                      use_full_image_net=use_full_image_net)

        if model_arch.lower() == 'resnet50_at':
            ckpt_data['state_dict'].pop('vone_block.div_u.weight')
            ckpt_data['state_dict'].pop('vone_block.div_t.weight')
            model.load_state_dict(ckpt_data['state_dict'])
        else:
            model = Wrapper(model)
            model.load_state_dict(ckpt_data['state_dict'])
            model = model.module
        model = nn.DataParallel(model)

        #model = nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
        #model = nn.parallel.DistributedDataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    else:
        model = globals()[f'VOneNet'](model_arch=model_arch, div_norm=div_norm, vonenet_on=vonenet_on, color_channels=color_channels, use_full_image_net=use_full_image_net,
                                      map_location=map_location, restore_path=restore_path, **kwargs)
        model = nn.DataParallel(model)



    model.to(map_location)
    return model

