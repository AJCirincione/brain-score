
import numpy as np
from brain_model_commitments import get_v1_model, get_model_layers_metrics

V1_LAYERS = {'alexnet': np.arange(8),
             'alexnet-random': np.arange(8), 
             'resnet-18': np.arange(8),
             'resnet18-local_aggregation': np.arange(8),
             'resnet18-autoencoder': np.arange(8),
             'resnet18-contrastive_predictive': np.arange(8),
             'resnet18-simclr': np.arange(8),
             'resnet18-deepcluster': np.arange(8),
             'resnet18-contrastive_multiview': np.arange(8),
             'resnet-34': np.arange(15),
             'resnet-50-pytorch': np.arange(15),
             'resnet50-SIN': np.arange(15),
             'resnet50-SIN_IN_IN': np.arange(15),
             'resnet-50-robust': np.arange(15),
             'resnet-50-random': np.arange(15),
             'CORnet-Z': np.arange(8),
             'CORnet-S': np.arange(21),
             'vgg-16': np.arange(14),
             'vgg-19': np.arange(16), #np.array([10,11]),  #np.arange(16),
             'bagnet17': np.arange(17),
             'bagnet33': np.arange(17),
             }

model_identifier = 'alexnet'
layer_index = 0

area = 'V1'

v1_layer = get_model_layers_metrics(model_identifier)[0][V1_LAYERS[model_identifier]]
if type(v1_layer) is str:
    v1_layer = [v1_layer]

degrees = 8


model = get_v1_model(model_identifier, area, v1_layer[layer_index], degrees)
