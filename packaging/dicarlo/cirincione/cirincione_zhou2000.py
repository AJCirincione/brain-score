import numpy as np
import xarray
from cirincione_utils import gen_sample
from brainio.packaging import package_data_assembly
from brainio.assemblies import DataAssembly

ASSEMBLY_NAME = 'cirincione_zhou2000'
BO_STIM_NAME = 'dicarlo.Cirincione2022_border_ownership'
PROPERTY_NAMES = ['border_ownership']

def collect_data():
    n_neurons = 63
    BO_ratio_bins = np.linspace(0, 1, num=8)
    BO_ratio_hist = np.array([0, 1, 2, 0, 9, 13, 17, 21])
    BO_ratio_index = gen_sample(BO_ratio_hist, BO_ratio_bins, scale='linear')

    assembly = np.concatenate((BO_ratio_index), axis=1)
    assembly = DataAssembly(assembly, coords={'neuroid_id': ('neuroid', range(assembly.shape[0])),
                                              'region': ('neuroid', ['V1'] * assembly.shape[0]),
                                              'neuronal_property': PROPERTY_NAMES},
                            dims=['neuroid', 'neuronal_property'])

    assembly.attrs['number_of_trials'] = 20
    for p in assembly.coords['neuronal_property'].values:
        assembly.attrs[p+'_bins'] = eval(p+'_bins')

    return assembly


def main():
    assembly = collect_data()
    assembly.name = ASSEMBLY_NAME

    print('Packaging assembly')
    package_data_assembly(xarray.DataArray(assembly), assembly_identifier=assembly.name,
                          stimulus_set_identifier=SIZE_STIM_NAME, assembly_class='PropertyAssembly',
                          bucket_name='brainio.contrib')


if __name__ == '__main__':
    main()