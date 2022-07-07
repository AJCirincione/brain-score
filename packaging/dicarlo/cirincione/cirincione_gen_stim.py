import os
import numpy as np
from cirincione_stim_common import gen_BO_stim, load_stim_info
from brainio.packaging import package_stimulus_set


BO_STIM_NAME = 'dicarlo.Cirincione2022_border_ownership_standard_test'
BO_OPTIM_NAME = 'dicarlo.Cirincione2022_border_ownership_optimization_test'
DATA_DIR = '/braintree/data2/active/users/andrewci/stimulus_sets/' #(braintree stimulus folder)
DEGREES = 12
SIZE_PX = 672
ORIENTATION_DIV = 12
POS_X = 0.5
POS_Y = 0.5
SQUARE_SIZE = 4 #degrees

BO_PARAMS = np.array([DEGREES, SIZE_PX, ORIENTATION_DIV, POS_X, POS_Y, SQUARE_SIZE])

#STIM_NAMES = [BO_STIM_NAME, BO_OPTIM_NAME]
STIM_NAMES = [BO_STIM_NAME]
generate_stimulus=0
def main():
    for stim_name in STIM_NAMES:
        stim_dir = os.path.join(DATA_DIR, stim_name)
        if generate_stimulus==1:
            print('Generating...')
            gen_BO_stim(BO_params = BO_PARAMS, save_dir = stim_dir, stim_name=stim_name)
        print(f'Stim name: {stim_name}')
        print(f'Stim directory: {stim_dir}')
        stimuli = load_stim_info(stim_name, stim_dir)
        print('Packaging stimuli:' + stimuli.identifier)
        package_stimulus_set(catalog_name='brainio_brainscore', proto_stimulus_set=stimuli, stimulus_set_identifier=stimuli.identifier, bucket_name='brainio.dicarlo')


if __name__ == '__main__':
    main()