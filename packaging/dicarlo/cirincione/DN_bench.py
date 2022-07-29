import sys
import os
sys.path.insert(1, '/braintree/home/andrewci/brain-score/')
sys.path.insert(1, '/braintree/home/andrewci/brain-score/packaging/dicarlo/cirincione/VOneNet_DN_Code')
from candidate_models.model_commitments import brain_translated_pool
from cirincione2022_zhou2000_TESTING import load_model_and_BO_stimulus, BO_optimization_and_standard_test, create_response_ratio_plots, load_model_and_BO_stimulus_LAYERS, V1_LAYERS, region_accuracy, run_v1block_DN
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
optim_test_dir = '/braintree/data2/active/users/andrewci/stimulus_sets/dicarlo.Cirincione2022_border_ownership_optimization_test'
standard_test_dir = '/braintree/data2/active/users/andrewci/stimulus_sets/dicarlo.Cirincione2022_border_ownership_standard_test'
BO_STIM_NAME = 'dicarlo.Cirincione2022_border_ownership_standard_test'
BO_OPTIM_NAME = 'dicarlo.Cirincione2022_border_ownership_optimization_test'
#os.environ['CUDA_VISIBLE_DEVICES']= '0, 1, 2, 3'
model = run_v1block_DN()
model_identifier = 'VOneNet_DN'
region = 'V1'
model, optimization_test_responses, standard_test_responses = load_model_and_BO_stimulus(BO_OPTIM_NAME=BO_OPTIM_NAME, optim_test_dir=optim_test_dir,
                    BO_STIM_NAME=BO_STIM_NAME, standard_test_dir=standard_test_dir, model_identifier=model_identifier, region=region, model=model)
save_path = '/braintree/data2/active/users/andrewci/BO_Responses/'
BO_responses = BO_optimization_and_standard_test(optimization_test_responses=optimization_test_responses, standard_test_responses=standard_test_responses,
model_identifier = model_identifier, path=save_path, save_figs_and_data=1)
create_response_ratio_plots(model_identifier=model_identifier, BO_responses=BO_responses, 
                    region=region, save_figs_and_data=1, path=save_path)