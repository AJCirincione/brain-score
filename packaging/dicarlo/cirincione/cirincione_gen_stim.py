import os
import numpy as np
from cirincione_stim_common import gen_BO_stim, load_stim_info
from brainio.packaging import package_stimulus_set

BO_STIM_NAME = 'dicarlo.Cirincione2022_border_ownership'
DATA_DIR = '/braintree/data2/active/users/andrewci/stimulus_sets/' #(braintree stimulus folder)
DEGREES = 12
SIZE_PX = 672
ORIENTATION_DIV = 12
POS_X = 0.5
POS_Y = 0.5
SQUARE_SIZE = 4 #degrees


BO_PARAMS = np.array([DEGREES, SIZE_PX, ORIENTATION_DIV, POS_X, POS_Y, SQUARE_SIZE])

STIM_NAMES = [BO_STIM_NAME]
STIM_PARAMS = {BO_STIM_NAME: BO_PARAMS}

def main():
    for stim_name in STIM_NAMES:
        stim_dir = DATA_DIR + stim_name
        if not (os.path.isdir(stim_dir)):
            print('TEST')
            gen_BO_stim(BO_params = STIM_PARAMS[stim_name], save_dir = stim_dir)
        stimuli = load_stim_info(stim_name, stim_dir)
        print('Packaging stimuli:' + stimuli.identifier)
        #package_stimulus_set(stimuli, stimulus_set_identifier=stimuli.identifier, bucket_name='brainio.dicarlo')

if __name__ == '__main__':
    main()