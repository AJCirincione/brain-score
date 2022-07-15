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

def modified_record_from_model(model:BrainModel, stimuli, number_of_trials):
    stimulus_set = stimuli
    stimulus_set = place_on_screen(stimulus_set, target_visual_degrees=model.visual_degrees())
    activations = model.look_at(stimulus_set, number_of_trials)
    if 'time_bin' in activations.dims:
        activations = activations.squeeze('time_bin')  # static case for these benchmarks
    if not activations.values.flags['WRITEABLE']:
        activations.values.setflags(write=1)
    return activations

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
                     region= 'V1', time_bins = [(70, 170)], number_of_trials = 20):
    model = brain_translated_pool[model_identifier]
    model.start_recording(region, time_bins=time_bins)

    optimization_test_stimuli = load_stim_info(BO_OPTIM_NAME, optim_test_dir)
    standard_test_stimuli = load_stim_info(BO_STIM_NAME, standard_test_dir)
    optimization_stim_pos = get_stimulus_position(optimization_test_stimuli)
    standard_stim_pos = get_stimulus_position(standard_test_stimuli)
    optimization_stim_in_rf = filter_receptive_fields(model_identifier=model_identifier, model=model, region=region,
                                        pos=optimization_stim_pos)
    standard_stim_in_rf = filter_receptive_fields(model_identifier=model_identifier, model=model, region=region,
                                        pos=standard_stim_pos)                                    
    optimization_test_responses = modified_get_firing_rates(model_identifier=model_identifier, model=model, region=region,
                                        stimuli=optimization_test_stimuli, number_of_trials=number_of_trials, in_rf=optimization_stim_in_rf)
    standard_test_responses = modified_get_firing_rates(model_identifier=model_identifier, model=model, region=region,
                                        stimuli=standard_test_stimuli, number_of_trials=number_of_trials, in_rf=standard_stim_in_rf)
    return model, optimization_test_responses, standard_test_responses                                

def BO_optimization_and_standard_test(optimization_test_responses, standard_test_responses, model_identifier, path = None, save_figs_and_data = 0):
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
            plt.savefig(os.path.join(save_path, 'ABCD', f'ABCD_neuroid_{neuroid_no}.png'))
            plt.close(fig)

            if not os.path.isdir(os.path.join(save_path, 'Ori Tuning Curve')):
                os.mkdir(os.path.join(save_path, 'Ori Tuning Curve'))
            fig = plt.figure(opt_n_neuroids+neuroid_no)
            plt.plot(opt_response[neuroid_no, opt_pref_color, opt_pref_width, opt_pref_length])
            plt.title(f'{model_identifier}:\n Orientation Tuning Curve for Neuroid {neuroid_no}')
            plt.savefig(os.path.join(save_path, 'Ori Tuning Curve', f'opt_orientation_tuning_neuroid_{neuroid_no}.png'))
            plt.close(fig)

            #fig = plt.figure(opt_n_neuroids*2)
            #plt.hist(BO_responses['response_ratio'])
            #plt.title(f'{model_identifier}: \n Response Ratio Distribution')
            #plt.savefig(os.path.join(save_path, f'response_ratio_dist.png'))
            #plt.close(fig)

            BO_responses.to_csv(os.path.join(save_path, 'BO_responses.csv'), index=False)
    return BO_responses

def create_response_ratio_plots(model_identifier, BO_responses, region = 'V1', path = None, save_figs_and_data = 0, V1_experimental_dist = np.array([1,2,3,1,9,13,15,19])):

    if save_figs_and_data == 1:
        save_path = os.path.join(path, model_identifier)
        if not os.path.isdir(save_path):
                    os.mkdir(save_path)

    V1_experimental_dist = V1_experimental_dist / np.sum(V1_experimental_dist)
    num_bins = len(V1_experimental_dist)

    V1_model_dist = np.array(np.histogram(BO_responses['response_ratio'], bins = 8, range=(0,1)))
    V1_model_dist[0] = V1_model_dist[0] / np.sum(V1_model_dist[0])

    custom_bins = np.arange(1/num_bins, 1 + 1/num_bins, 1/num_bins)

    plt.figure('line')
    V1_dist_bins = np.linspace(1/num_bins, 1, num_bins)
    plt.plot(V1_dist_bins, V1_experimental_dist)
    plt.plot(V1_dist_bins, V1_model_dist[0])
    plt.xticks(custom_bins)
    plt.title(f'{model_identifier}:\n V1 Response Ratio Frequency')
    plt.legend(['V1 Distribution', 'Model Distribution'])
    if save_figs_and_data == 1:
        plt.savefig(os.path.join(save_path, f'{region} Response_Ratio_Line_Plot.png'))
    plt.show()

    plt.figure('bar')
    plt.bar(V1_dist_bins, V1_experimental_dist, width=1/(num_bins-1.5))
    plt.bar(V1_dist_bins, V1_model_dist[0], width=1/(num_bins-1.5), alpha=0.66)
    plt.xticks(custom_bins)
    plt.title(f'{model_identifier}:\n V1 Response Ratio Frequency')
    plt.legend(['V1 Distribution', 'Model Distribution'])
    if save_figs_and_data == 1:
        plt.savefig(os.path.join(save_path, f'{region}_Response_Ratio_Bar_Plot.png'))
    plt.show()
