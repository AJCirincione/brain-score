
import functools
import numpy as np
from model_tools.activations.pytorch import PytorchWrapper, load_preprocess_images

MODEL_TRAINING = {'alexnet': 'IN_supervised',
             'alexnet-random': 'None',
             'resnet-18': 'IN_supervised',
             'resnet18-local_aggregation': 'ImageNet',
             'resnet18-autoencoder': 'IN_unsupervised',
             'resnet18-contrastive_predictive': 'IN_unsupervised',
             'resnet18-simclr': 'IN_unsupervised',
             'resnet18-deepcluster': 'IN_unsupervised',
             'resnet18-contrastive_multiview': 'IN_unsupervised',
             'resnet-34': 'IN_supervised',
             'resnet-50-pytorch': 'IN_supervised',
             'resnet50-SIN': 'SIN',
             'resnet50-SIN_IN_IN': 'SIN_IN_IN',
             'resnet-50-robust': 'IN_AT',
             'resnet-50-random': 'None',
             'CORnet-Z': 'IN_supervised',
             'CORnet-S': 'IN_supervised',
             'vgg-16': 'IN_supervised',
             'vgg-19': 'IN_supervised',
             'bagnet17': 'IN_supervised',
             'bagnet33': 'IN_supervised',
             }

MODEL_INPUT_SIZE = {'alexnet': 224,
                    'alexnet-random': 224,
                    'resnet-18': 224,
                    'resnet18-local_aggregation': 224,
                    'resnet18-autoencoder': 224,
                    'resnet18-contrastive_predictive': 224,
                    'resnet18-simclr': 224,
                    'resnet18-deepcluster': 224,
                    'resnet18-contrastive_multiview': 224,
                    'resnet-34': 224,
                    'resnet-50-pytorch': 224,
                    'resnet50-SIN': 224,
                    'resnet50-SIN_IN_IN': 224,
                    'resnet-50-robust': 224,
                    'resnet-50-random': 224,
                    'CORnet-Z': 224,
                    'CORnet-S': 224,
                    'vgg-16': 224,
                    'vgg-19': 224,
                    'bagnet17': 224,
                    'bagnet33': 224,
                    }


def activations_wrapper(model):    
    image_size = model.image_size
    preprocessing = functools.partial(load_preprocess_images, image_size=image_size,
                                      normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.5, 0.5, 0.5))
    wrapper = PytorchWrapper(identifier=model.identifier, model=model, preprocessing=preprocessing)
    wrapper.image_size = image_size
    return wrapper

def get_v1_model(base_model_identifier, area, layer, degrees, activations_model=None):
    from model_tools.brain_transformation import ModelCommitment

    if activations_model is None:
        from candidate_models.base_models import base_model_pool, cornet

        if base_model_identifier[:6] == 'CORnet':
            activations_model = cornet(base_model_identifier, separate_time=False)
        else:
            activations_model = base_model_pool[base_model_identifier]
    identifier = f"{base_model_identifier}_{area}-{layer}_deg-{degrees}"
    layer_commitment = {area: layer}
    model = ModelCommitment(identifier, activations_model=activations_model, layers=[layer],
                            region_layer_map=layer_commitment, visual_degrees=degrees)

    return model


def get_model_layers_metrics(model_name, layer_ind=[]):

    layers_fun = {'alexnet': alexnet_layers, 'alexnet-random': alexnet_layers,
                  'vgg-16': functools.partial(vgg_layers, layers=16),
                  'vgg-19': functools.partial(vgg_layers, layers=19),
                  'CORnet-Z': cornetz_layers, 'CORnet-S': cornets_layers,
                  'resnet-18': functools.partial(resnet_layers, n=18),
                  'resnet-34': functools.partial(resnet_layers, n=34),
                  'resnet-50-pytorch': functools.partial(resnet_layers, n=50),
                  'resnet-50-random': functools.partial(resnet_layers, n=50),
                  'resnet50-SIN': functools.partial(resnet_layers, n=50),
                  'resnet50-SIN_IN_IN': functools.partial(resnet_layers, n=50),
                  'resnet-50-robust': functools.partial(resnet_layers, n=50),
                  'bagnet9': functools.partial(bagnet_layers, px=9),
                  'bagnet17': functools.partial(bagnet_layers, px=17),
                  'bagnet33': functools.partial(bagnet_layers, px=33),
                  'resnet18-local_aggregation': resnet18_unsup_layers,
                  'resnet18-autoencoder': resnet18_unsup_layers,
                  'resnet18-contrastive_predictive': resnet18_unsup_layers,
                  'resnet18-contrastive_multiview': resnet18_unsup_alt_layers,
                  'resnet18-simclr': resnet18_unsup_layers,
                  'resnet18-deepcluster': resnet18_unsup_alt_layers,
                  }

    if model_name in list(layers_fun.keys()):
        layers, depth, features_num, spatial_map_size, receptive_field_px, units_num, layer_type = layers_fun[model_name]()

        training = np.array([MODEL_TRAINING[model_name]] * len(layers))
        total_depth = np.array([depth[-1]] * len(layers))
        fov_px = np.array([MODEL_INPUT_SIZE[model_name]] * len(layers))
    else:
        from candidate_models.model_commitments.model_layer_def import layers as candidate_layers
        layers = np.array(candidate_layers[model_name])
        n_layers = len(layers)
        depth = nan_array(n_layers)
        features_num = nan_array(n_layers)
        spatial_map_size = nan_array(n_layers)
        receptive_field_px = nan_array(n_layers)
        units_num = nan_array(n_layers)
        layer_type = nan_array(n_layers)
        training = nan_array(n_layers)
        total_depth = nan_array(n_layers)
        fov_px = nan_array(n_layers)

    if len(layer_ind) > 0:
        return layers[layer_ind], depth[layer_ind], features_num[layer_ind], spatial_map_size[layer_ind], \
               receptive_field_px[layer_ind], units_num[layer_ind], layer_type[layer_ind], training[layer_ind], \
               total_depth[layer_ind], fov_px[layer_ind]
    else:
        return layers, depth, features_num, spatial_map_size, receptive_field_px, units_num, layer_type, training, \
               total_depth, fov_px


def nan_array(n):
    a = np.zeros(n)
    a[:]=np.nan
    return a

def resnet18_unsup_layers():
    layers = np.array(['encode_1.conv', 'encode_1', 'encode_2', 'encode_3', 'encode_4', 'encode_5',
                       'encode_6', 'encode_7', 'encode_8', 'encode_9'])
    depth = np.array([1, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    features = np.array([64, 64, 64, 64, 128, 128, 256, 256, 512, 512])
    size = np.array([55, 55, 55, 55, 28, 28, 14, 14, 7, 7])
    rf = np.array([7, 11, 27, 43, 59, 91, 123, 187, 251, 379])
    units = size * size * features
    layer_type = np.array(
        ['conv', 'maxpool', 'conv', 'conv', 'conv', 'conv', 'conv', 'conv', 'conv', 'conv'])

    return layers, depth, features, size, rf, units, layer_type


def resnet18_unsup_alt_layers():
    layers = np.array(['relu', 'maxpool', 'layer1.0', 'layer1.1', 'layer2.0', 'layer2.1',
                       'layer3.0', 'layer3.1', 'layer4.0', 'layer4.1'])
    depth = np.array([1, 2, 4, 6, 8, 10, 12, 14, 16, 18])
    features = np.array([64, 64, 64, 64, 128, 128, 256, 256, 512, 512])
    size = np.array([55, 55, 55, 55, 28, 28, 14, 14, 7, 7])
    rf = np.array([7, 11, 27, 43, 59, 91, 123, 187, 251, 379, 571])
    units = size * size * features
    layer_type = np.array(
        ['conv', 'maxpool', 'conv', 'conv', 'conv', 'conv', 'conv', 'conv', 'conv'])

    return layers, depth, features, size, rf, units, layer_type


def alexnet_layers():
    layers = np.array(['features.1', 'features.2', 'features.4', 'features.5', 'features.7', 'features.9',
                       'features.11', 'features.12', 'classifier.2', 'classifier.5', 'classifier.6'])
    depth = np.arange(len(layers)) + 1
    features = np.array([64, 64, 192, 192, 384, 256, 256, 256, 4096, 4096, 1000])
    size = np.array([55, 27, 27, 13, 13, 13, 13, 6, 1, 1, 1])
    rf = np.array([11, 19, 51, 67, 99, 131, 163, 195, 355, 355, 355])
    units = size * size * features
    layer_type = np.array(
        ['conv', 'maxpool', 'conv', 'maxpool', 'conv', 'conv', 'conv', 'maxpool', 'fc', 'fc', 'fc'])

    return layers, depth, features, size, rf, units, layer_type


def vgg_layers(layers=19):
    if layers == 16:
        vgg19_block_size = [2, 2, 3, 3, 3]
        rf = np.array(
            [3, 5, 6, 10, 14, 16, 24, 32, 40, 44, 60, 76, 92, 100, 132, 164, 196, 212, 404, 404, 404])
    elif layers == 19:
        vgg19_block_size = [2, 2, 4, 4, 4]
        rf = np.array(
            [3, 5, 6, 10, 14, 16, 24, 32, 40, 48, 52, 68, 84, 100, 116, 124, 156, 188, 220, 252, 268, 460, 460,
             460])

    features_block = [64, 128, 256, 512, 512]
    size_block = [224, 112, 56, 28, 14]
    layers = []
    features = []
    size = []
    layer_type = []
    for b in range(len(vgg19_block_size)):
        for c in range(vgg19_block_size[b]):
            layers.append(f'block{b + 1}_conv{c + 1}')
            features.append(features_block[b])
            size.append(size_block[b])
            layer_type.append('conv')
        layers.append(f'block{b + 1}_pool')
        features.append(features_block[b])
        size.append(size_block[b] / 2)
        layer_type.append('maxpool')

    layers = layers + ['fc1', 'fc2', 'predictions']
    features = features + [4096, 4096, 1000]
    size = size + [1, 1, 1]
    layer_type = layer_type + ['fc', 'fc', 'fc']

    layers = np.array(layers)
    depth = np.arange(len(layers)) + 1
    features = np.array(features)
    size = np.array(size)
    layer_type = np.array(layer_type)
    units = size * size * features

    return layers, depth, features, size, rf, units, layer_type


def cornetz_layers():
    cornetz_areas = ['V1', 'V2', 'V4', 'IT']
    size_areas = [56, 28, 14, 7]
    features_areas = [64, 128, 256, 512]

    layers = []
    size = []
    features = []
    layer_type = []

    for a_ind, a in enumerate(cornetz_areas):
        layers.append(a + '.nonlin-t0')
        size.append(size_areas[a_ind] * 2)
        features.append(features_areas[a_ind])
        layer_type.append('maxpool')

        layers.append(a + '.pool-t0')
        size.append(size_areas[a_ind])
        features.append(features_areas[a_ind])
        layer_type.append('maxpool')

    layers.append('decoder.avgpool-t0')
    size.append(1)
    features.append(512)
    layer_type.append('avgpool')

    layers.append('decoder.output-t0')
    size.append(1)
    features.append(1000)
    layer_type.append('fc')

    depth = np.arange(len(layers)) + 1

    layers = np.array(layers)
    features = np.array(features)
    size = np.array(size)
    layer_type = np.array(layer_type)
    units = size * size * features

    rf = np.array([7, 11, 19, 27, 43, 59, 91, 123, 315, 315])

    return layers, depth, features, size, rf, units, layer_type


def cornets_layers():
    cornets_areas = ['V2', 'V4', 'IT']
    cornets_times = [2, 4, 2]

    size_areas = [28, 14, 7]
    features_areas = [128, 256, 512]

    layers = ['V1.nonlin1-t0', 'V1.pool-t0', 'V1.nonlin2-t0']
    size = [112, 56, 56]
    features = [64, 64, 64]
    layer_type = ['conv', 'maxpool', 'conv']

    for a_ind, a in enumerate(cornets_areas):
        for t in range(cornets_times[a_ind]):
            layers.append(a + f'.nonlin1-t{t}')
            if t == 0:
                size.append(size_areas[a_ind] * 2)
            else:
                size.append(size_areas[a_ind])
            features.append(features_areas[a_ind] * 4)
            layer_type.append('conv')
            layers.append(a + f'.nonlin2-t{t}')
            size.append(size_areas[a_ind])
            features.append(features_areas[a_ind] * 4)
            layer_type.append('conv')
            layers.append(a + f'.nonlin3-t{t}')
            size.append(size_areas[a_ind])
            features.append(features_areas[a_ind])
            layer_type.append('conv')

    layers.append('decoder.avgpool-t0')
    size.append(1)
    features.append(512)
    layer_type.append('avgpool')

    layers.append('decoder.output-t0')
    size.append(1)
    features.append(1000)
    layer_type.append('fc')

    depth = np.arange(len(layers)) + 1

    layers = np.array(layers)
    features = np.array(features)
    size = np.array(size)
    layer_type = np.array(layer_type)
    units = size * size * features

    rf = np.array(
        [7, 11, 19, 19, 27, 27, 27, 43, 43, 43, 59, 59, 59, 91, 91, 91, 123, 123, 123, 155, 155, 155, 187, 187, 187,
         251, 251, 443, 443])

    return layers, depth, features, size, rf, units, layer_type


def resnet_layers(n=50):
    if n == 50:
        resnet_block_size = [3, 4, 6, 3]
        rf = np.array([7, 11, 19, 27, 35,
                       43, 59, 75, 91, 107,
                       139, 171, 203, 235, 267,
                       299, 363, 427, 619, 619])
        delta_depth = 3
        features_block = [256, 512, 1024, 2048]
    elif n == 34:
        resnet_block_size = [3, 4, 6, 3]
        rf = np.array([7, 11, 27, 43, 59,
                       75, 107, 139, 171, 203,
                       267, 331, 395, 459, 523,
                       587, 715, 843, 1035, 1035])
        delta_depth = 2
        features_block = [64, 128, 256, 512]
    elif n == 18:
        resnet_block_size = [2, 2, 2, 2]
        rf = np.array([7, 11, 27, 43, 59, 91, 123, 187, 251, 379, 571, 571])
        delta_depth = 2
        features_block = [64, 128, 256, 512]

    size_block = [56, 28, 14, 7]

    layers = ['relu'] + ['maxpool']
    size = [112, 56]
    features = [64, 64]
    depth = [1, 2]
    layer_type = ['conv', 'maxpool']

    for b in range(len(resnet_block_size)):
        for c in range(resnet_block_size[b]):
            layers.append(f'layer{b + 1}.{c}')
            depth.append(depth[-1] + delta_depth)
            features.append(features_block[b])
            size.append(size_block[b])
            layer_type.append('conv')

    layers.append('avgpool')
    depth.append(depth[-1] + 1)
    size.append(1)
    features.append(features[-1])
    layer_type.append('avgpool')

    layers.append('fc')
    depth.append(depth[-1] + 1)
    size.append(1)
    features.append(1000)
    layer_type.append('fc')

    layers = np.array(layers)
    depth = np.array(depth)
    features = np.array(features)
    size = np.array(size)
    layer_type = np.array(layer_type)
    units = size * size * features

    return layers, depth, features, size, rf, units, layer_type


def bagnet_layers(px=33):
    bagnet_block_size = [3, 4, 6, 3]
    size_block = [110, 54, 26, 24]
    features_block = [256, 512, 1024, 2048]
    if px == 9:
        rf_block = [5, 9, 9, 9]
    elif px == 17:
        rf_block = [5, 9, 17, 17]
    elif px == 33:
        rf_block = [5, 9, 17, 33]
    layers = ['relu']
    size = [222]
    features = [64]
    depth = [1]
    layer_type = ['conv']
    rf = [3]

    for b in range(len(bagnet_block_size)):
        for c in range(bagnet_block_size[b]):
            layers.append(f'layer{b + 1}.{c}')
            depth.append(depth[-1] + 3)
            features.append(features_block[b])
            size.append(size_block[b])
            layer_type.append('conv')
            rf.append(rf_block[b])

    layers.append('avgpool')
    depth.append(depth[-1] + 1)
    size.append(1)
    features.append(2048)
    layer_type.append('avgpool')
    rf.append(217)

    layers.append('fc')
    depth.append(depth[-1] + 1)
    size.append(1)
    features.append(1000)
    layer_type.append('fc')
    rf.append(217)

    layers = np.array(layers)
    depth = np.array(depth)
    features = np.array(features)
    size = np.array(size)
    layer_type = np.array(layer_type)
    units = size * size * features
    rf = np.array(rf)

    return layers, depth, features, size, rf, units, layer_type
