# here the needs to be added different functions
import numpy as np
import pandas as pd
import os
from ucimlrepo import fetch_ucirepo

combined_cycle_power_plant = fetch_ucirepo(id=294)
real_estate_valuation = fetch_ucirepo(id=477)
energy_efficiency = fetch_ucirepo(id=242)
air_quality = fetch_ucirepo(id=360)
computer_hardware = fetch_ucirepo(id=29)
forest_fires = fetch_ucirepo(id=162)
gas_turbine_co_and_nox_emission_data_set = fetch_ucirepo(id=551)

frame = pd.read_csv(
    'equations_and_vars.csv'
)

def get_dset(folder_name):
    defs = frame.set_index(frame.Equation)['Vars'].to_dict()
    funs = frame.set_index(frame.Equation)['funs'].to_dict()
    temp_consts = frame.set_index(frame.Equation)['consts'].to_dict()
    consts = {}
    for key, value in temp_consts.items():
        elem = value
        if elem is np.nan:
            consts[key] = []
        else:
            consts[key] = elem.split(',')
    d_sets = {}
    files = os.listdir(folder_name)
    for elem, _ in defs.items():
        try:
            for name in files:
                if elem.lower() == name[8:-4]:
                    d_sets[elem] = (os.path.join(folder_name, name))
                    break
        except Exception as err:
            print(err)
    return defs, d_sets, funs, consts


defs, d_sets, funs, consts = get_dset('srds_input_emb')
source_tasks = {
    k: [defs[k].replace(' ', '').split(','), pd.read_csv(v, header=None, delimiter='\s+'),
        funs[k].split(','), consts[k]] for k, v in d_sets.items()
}

real_estate = {
    'real_estate': [[f'x{i}' for i in range(real_estate_valuation.data.features.shape[1] - 1)],
                    pd.concat([real_estate_valuation.data.features,
                               real_estate_valuation.data.targets], axis=1)]
}

target_airfoil_UCI = {
    'airfoil_noise_UCI': [['x0', 'x1', 'x2', 'x3', 'x4'],
                          pd.read_table('airfoil_noise/airfoil_self_noise.dat',
                                        header=None)]
}
