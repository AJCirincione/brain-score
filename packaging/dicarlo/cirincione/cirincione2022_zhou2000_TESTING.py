import numpy as np
import sys
import os
import pandas as pd
from brainscore.model_interface import BrainModel
sys.path.insert(1, '/braintree/home/andrewci/brain-score/')
from candidate_models.model_commitments import brain_translated_pool
#sys.path.insert(1, '/braintree/home/andrewci/brain-score/brainscore/benchmarks')
from cirincione_stim_common import load_stim_info
from brainscore.benchmarks.screen import place_on_screen
from brainscore.benchmarks._properties_common import get_stimulus_position, \
    filter_receptive_fields, firing_rates_affine, record_from_model, get_firing_rates, get_stimulus_position, filter_receptive_fields
import matplotlib.pyplot as plt


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


def modified_record_from_model(model:BrainModel, stimuli, number_of_trials):
    stimulus_set = stimuli
    stimulus_set = place_on_screen(stimulus_set, target_visual_degrees=model.visual_degrees())
    activations = model.look_at(stimulus_set, number_of_trials)
    if 'time_bin' in activations.dims:
        activations = activations.squeeze('time_bin')  # static case for these benchmarks
    if not activations.values.flags['WRITEABLE']:
        activations.values.setflags(write=1)
    return activations

import numpy as np
from candidate_models.base_models import base_model_pool, cornet
from brain_model_commitments import get_v1_model, get_model_layers_metrics, activations_wrapper
sys.path.insert(1, '/braintree/home/andrewci/brain-score/packaging/dicarlo/cirincione/VOneNet_DN_Code')
from __init__ import get_model
from utils import loadWeights

def run_v1block_DN():
    v1_layer = 'dn_block'
    area = 'V1'
    degrees = 8
    
    pytorch_model = get_model(map_location='cuda', model_arch='resnet18', pretrained=False,
                    visual_degrees=8, stride=2, simple_channels=256,
                    complex_channels=256, noise_mode=None,
                    div_norm=True, vonenet_on=True,
                    color_channels=3, image_size=256)
    
    pytorch_model = pytorch_model.module
    pytorch_model = loadWeights(pytorch_model, '/braintree/home/solaiya/vonenetwithdn.pth', 'cpu')
    
    activations_model = activations_wrapper(pytorch_model)
    
    model = get_v1_model(activations_model.identifier, area, v1_layer, degrees, activations_model)
    
    return model




def ks_similarity(p, q):
    z1 = np.zeros_like(p)
    z2 = np.zeros_like(p)
    z1[0] = 1
    z2[-1] = 1

    kolm_raw = np.max(np.abs(np.cumsum(p)-np.cumsum(q)))
    kolm_max1 = np.max(np.abs(np.cumsum(z1)-np.cumsum(q)))
    kolm_max2 = np.max(np.abs(np.cumsum(z2)-np.cumsum(q)))
    kolm_max = np.max(np.array([kolm_max1, kolm_max2]))
    kolm_ceiled = kolm_raw / kolm_max

    return 1 - kolm_ceiled

def modified_get_firing_rates(model_identifier, model, region, number_of_trials, in_rf, stimuli):
    affine_transformation = firing_rates_affine(model_identifier=model_identifier, model=model, region=region)
    affine_transformation = affine_transformation.values

    activations = modified_record_from_model(model, stimuli, number_of_trials)

    activations = activations[in_rf]
    activations.values[activations.values < 0] = 0

    activations = affine_transformation[0] * activations + affine_transformation[1]
    activations.values[activations.values < 0] = 0
    return activations

def load_model_and_BO_stimulus(BO_OPTIM_NAME, optim_test_dir, BO_STIM_NAME, standard_test_dir, model_identifier='alexnet',
                     region= 'V1', time_bins = [(70, 170)], number_of_trials = 20, model=None):
    if model is None:
        model = brain_translated_pool[model_identifier]
    model.start_recording(region, time_bins=time_bins)

    optimization_test_stimuli = load_stim_info(BO_OPTIM_NAME, optim_test_dir)
    standard_test_stimuli = load_stim_info(BO_STIM_NAME, standard_test_dir)
    stim_pos = get_stimulus_position(optimization_test_stimuli)
    #standard_stim_pos = get_stimulus_position(standard_test_stimuli)
    stim_in_rf = filter_receptive_fields(model_identifier=model_identifier, model=model, region=region,
                                        pos=stim_pos)
    print
    #standard_stim_in_rf = filter_receptive_fields(model_identifier=model_identifier, model=model, region=region,
                                        #pos=standard_stim_pos)                                    
    optimization_test_responses = modified_get_firing_rates(model_identifier=model_identifier, model=model, region=region,
                                        stimuli=optimization_test_stimuli, number_of_trials=number_of_trials, in_rf=stim_in_rf)
    standard_test_responses = modified_get_firing_rates(model_identifier=model_identifier, model=model, region=region,
                                        stimuli=standard_test_stimuli, number_of_trials=number_of_trials, in_rf=stim_in_rf)
    return model, optimization_test_responses, standard_test_responses         


def load_model_and_BO_stimulus_LAYERS(BO_OPTIM_NAME, optim_test_dir, BO_STIM_NAME, standard_test_dir, model_identifier='alexnet',
                     region= 'V1', time_bins = [(70, 170)], number_of_trials = 20, layer_index = 0):

    v1_layer = get_model_layers_metrics(model_identifier)[0][V1_LAYERS[model_identifier]]
    if type(v1_layer) is str:
        v1_layer = [v1_layer]

    degrees = 8

    model = get_v1_model(model_identifier, region, v1_layer[layer_index], degrees)
    model_identifier=model.identifier             
    model.start_recording(region, time_bins=time_bins)

    optimization_test_stimuli = load_stim_info(BO_OPTIM_NAME, optim_test_dir)
    standard_test_stimuli = load_stim_info(BO_STIM_NAME, standard_test_dir)
    stim_pos = get_stimulus_position(optimization_test_stimuli)
    #standard_stim_pos = get_stimulus_position(standard_test_stimuli)
    stim_in_rf = filter_receptive_fields(model_identifier=model_identifier, model=model, region=region,
                                        pos=stim_pos)
    #standard_stim_in_rf = filter_receptive_fields(model_identifier=model_identifier, model=model, region=region,
      #                                  pos=standard_stim_pos)                                    
    optimization_test_responses = modified_get_firing_rates(model_identifier=model_identifier, model=model, region=region,
                                        stimuli=optimization_test_stimuli, number_of_trials=number_of_trials, in_rf=stim_in_rf)
    standard_test_responses = modified_get_firing_rates(model_identifier=model_identifier, model=model, region=region,
                                        stimuli=standard_test_stimuli, number_of_trials=number_of_trials, in_rf=stim_in_rf)
    return model, optimization_test_responses, standard_test_responses                           

def BO_optimization_and_standard_test(optimization_test_responses, standard_test_responses, model_identifier, path = None, save_figs_and_data = 0, layer_index=None):
    #sanity_checks: Fix color, run different orientations, include in slides

    opt_response = optimization_test_responses.values
    opt_n_neuroids = opt_response.shape[0]
    BO_responses = pd.DataFrame(
            columns=['neuroid_no', 'response_ratio', 'A', 'B', 'C', 'D'])
    for neuroid_no in range(opt_n_neuroids):
        opt_color = np.array(sorted(set(optimization_test_responses.color.values)))
        opt_orientation = np.array(sorted(set(optimization_test_responses.orientation.values)))
        opt_width = np.array(sorted(set(optimization_test_responses.width.values)))
        opt_length = np.array(sorted(set(optimization_test_responses.length.values)))
        opt_response = opt_response.reshape((opt_n_neuroids, len(opt_color),
                len(opt_width), len(opt_length),  len(opt_orientation)))        
        opt_pref_color, opt_pref_width, opt_pref_length, opt_pref_orientation =\
            np.unravel_index(np.argmax(opt_response[neuroid_no, :, :, :, :]), 
            (len(opt_color),  len(opt_width), len(opt_length), len(opt_orientation)))

        std_response = standard_test_responses.values
        std_n_neuroids = std_response.shape[0]
        std_color = np.array(sorted(set(standard_test_responses.color.values)))
        std_orientation = np.array(sorted(set(standard_test_responses.orientation.values)))
        std_polarity = np.array(sorted(set(standard_test_responses.polarity.values)))
        std_side = np.array(sorted(set(standard_test_responses.side.values)))
        std_response = std_response.reshape((std_n_neuroids, len(std_color),
                len(std_polarity), len(std_side),  len(std_orientation)))     
        std_A = std_response[neuroid_no, opt_pref_color, 0, 0, opt_pref_orientation]
        std_B = std_response[neuroid_no, opt_pref_color, 1, 1, opt_pref_orientation]
        std_C = std_response[neuroid_no, opt_pref_color, 1, 0, opt_pref_orientation]
        std_D = std_response[neuroid_no, opt_pref_color, 0, 1, opt_pref_orientation]
        AC = std_A + std_C
        BD = std_B + std_D
        ratio_numerator = min(AC, BD)
        ratio_denominator = max(AC, BD)
        response_ratio = ratio_numerator / ratio_denominator
        BO_responses = BO_responses.append(
                            {'neuroid_no': neuroid_no, 'response_ratio': response_ratio, 'A': std_A, 
                            'B': std_B, 'C': std_C, 'D':std_D},ignore_index=True)
        if layer_index is not None:
            save_path = os.path.join(path, (model_identifier + '_layer_' + str(layer_index)))
        else:
            save_path = os.path.join(path, model_identifier)
        if save_figs_and_data == 1:

            plt.ioff()

            ABCD_values = BO_responses.loc[neuroid_no][2:]
            keys = BO_responses.keys()[2:]

            if not os.path.isdir(save_path):
                os.mkdir(save_path)

            

            if not os.path.isdir(os.path.join(save_path, 'ABCD')):
                os.mkdir(os.path.join(save_path, 'ABCD'))
            fig = plt.figure(neuroid_no)
            plt.bar(keys, ABCD_values)
            plt.title(f'{model_identifier}: \n Neuroid Number: {neuroid_no}, Response Ratio: {BO_responses.loc[neuroid_no][1]}')
            plt.savefig(os.path.join(save_path, 'ABCD', f'ABCD_neuroid_{neuroid_no}.png'),bbox_inches='tight')
            plt.close(fig)

            if not os.path.isdir(os.path.join(save_path, 'Ori Tuning Curve')):
                os.mkdir(os.path.join(save_path, 'Ori Tuning Curve'))
            fig = plt.figure(opt_n_neuroids+neuroid_no)
            plt.plot(opt_response[neuroid_no, opt_pref_color, opt_pref_width, opt_pref_length])
            plt.title(f'{model_identifier}:\n Orientation Tuning Curve for Neuroid {neuroid_no}')
            plt.savefig(os.path.join(save_path, 'Ori Tuning Curve', f'opt_orientation_tuning_neuroid_{neuroid_no}.png'),bbox_inches='tight')
            plt.close(fig)

            #fig = plt.figure(opt_n_neuroids*2)
            #plt.hist(BO_responses['response_ratio'])
            #plt.title(f'{model_identifier}: \n Response Ratio Distribution')
            #plt.savefig(os.path.join(save_path, f'response_ratio_dist.png'))
            #plt.close(fig)

            BO_responses.to_csv(os.path.join(save_path, 'BO_responses.csv'), index=False)
    return BO_responses

def create_response_ratio_plots(model_identifier, BO_responses, region = 'V1', path = None, 
    save_figs_and_data = 0, V1_experimental_dist = np.array([1,2,3,1,9,13,15,19]), V2_experimental_dist = np.array([7,6,8,9,15,15,10,20]), layer_index=0):

    if save_figs_and_data == 1:
        if layer_index is not None:
            save_path = os.path.join(path, (model_identifier + '_layer_' + str(layer_index)))
        else:
            save_path = os.path.join(path, model_identifier)
        if not os.path.isdir(save_path):
                    os.mkdir(save_path)

    V1_experimental_dist = V1_experimental_dist / np.sum(V1_experimental_dist)
    num_bins = len(V1_experimental_dist)
    V2_experimental_dist = V2_experimental_dist / np.sum(V2_experimental_dist)
    num_bins = len(V2_experimental_dist)

    V1_model_dist = np.array(np.histogram(BO_responses['response_ratio'], bins = 8, range=(0,1)))
    V1_model_dist[0] = V1_model_dist[0] / np.sum(V1_model_dist[0])

    custom_bins = np.arange(1/num_bins, 1 + 1/num_bins, 1/num_bins)

    V1_similarity = ks_similarity(V1_experimental_dist, V1_model_dist[0])
    V2_similarity = ks_similarity(V2_experimental_dist, V1_model_dist[0])

    fig = plt.figure('line')
    V1_dist_bins = np.linspace(1/num_bins, 1, num_bins)
    plt.plot(V1_dist_bins, V1_experimental_dist)
    plt.plot(V1_dist_bins, V1_model_dist[0])
    plt.xticks(custom_bins)
    plt.title(f'{model_identifier}:\n V1 Response Ratio Frequency\n KS: {V1_similarity}')
    plt.legend(['V1 Distribution', 'Model Distribution'])
    if save_figs_and_data == 1:
        plt.savefig(os.path.join(save_path, f'V1 Response_Ratio_Line_Plot.png'),bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure('bar')
    plt.bar(V1_dist_bins, V1_experimental_dist, width=1/(num_bins-1.5))
    plt.bar(V1_dist_bins, V1_model_dist[0], width=1/(num_bins-1.5), alpha=0.66)
    plt.xticks(custom_bins)
    plt.title(f'{model_identifier}:\n V1 Response Ratio Frequency\n KS: {V1_similarity}')
    plt.legend(['V1 Distribution', 'Model Distribution'])
    if save_figs_and_data == 1:
        plt.savefig(os.path.join(save_path, f'V1_Response_Ratio_Bar_Plot.png'),bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure('v2line')
    V1_dist_bins = np.linspace(1/num_bins, 1, num_bins)
    plt.plot(V1_dist_bins, V2_experimental_dist)
    plt.plot(V1_dist_bins, V1_model_dist[0])
    plt.xticks(custom_bins)
    plt.title(f'{model_identifier}:\n V2 Response Ratio Frequency\n KS: {V2_similarity}')
    plt.legend(['V2 Distribution', 'Model Distribution'])
    if save_figs_and_data == 1:
        plt.savefig(os.path.join(save_path, f'V2 Response_Ratio_Line_Plot.png'),bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure('v2bar')
    plt.bar(V1_dist_bins, V2_experimental_dist, width=1/(num_bins-1.5))
    plt.bar(V1_dist_bins, V1_model_dist[0], width=1/(num_bins-1.5), alpha=0.66)
    plt.xticks(custom_bins)
    plt.title(f'{model_identifier}:\n V2 Response Ratio Frequency\n KS: {V2_similarity}')
    plt.legend(['V1 Distribution', 'Model Distribution'])
    if save_figs_and_data == 1:
        plt.savefig(os.path.join(save_path, f'V2_Response_Ratio_Bar_Plot.png'),bbox_inches='tight')
    plt.close(fig)
  


    V1_similarity = ks_similarity(V1_experimental_dist, V1_model_dist[0])
    V2_similarity = ks_similarity(V2_experimental_dist, V1_model_dist[0])
    return(V1_similarity, V2_similarity)

def region_accuracy(V1_similarity_vec, V2_similarity_vec, model_identifier, path):
    fig = plt.figure('v1acc')
    plt.plot(V1_similarity_vec)
    plt.xlabel('Layer of Model')
    plt.ylabel('KS Similarity')
    plt.title(f'{model_identifier} V1 Similarity over Layers')
    plt.savefig(os.path.join(path, f'{model_identifier} V1 Similarity over Layers.png'), bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure('v2acc')
    plt.plot(V2_similarity_vec)
    plt.xlabel('Layer of Model')
    plt.ylabel('KS Similarity')
    plt.title(f'{model_identifier} V2 Similarity over Layers')
    plt.savefig(os.path.join(path, f'{model_identifier} V2 Similarity over Layers.png'), bbox_inches='tight')
    plt.close(fig)