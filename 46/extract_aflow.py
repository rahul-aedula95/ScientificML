from aflow import *
from tqdm import tqdm
import pandas as pd

keys = ['compound','Bravais_lattice_relax','spacegroup_relax','volume_cell','density','geometry','enthalpy_formation_cell','natoms','energy_atom','eentropy_atom','scintillation_attenuation_length','valence_cell_std','bader_atomic_volumes', 'bader_net_charges']
aflow_dict = {k: [] for k in keys}
data = pd.read_csv('data/Supercon_data.csv')
data = data[data['Tc']<=0]
compound_list = list(data['name'])

for compound in tqdm(compound_list):
    
    result = search(catalog='icsd', batch_size=1000).filter(K.compound==compound)

    try:
        entry = result[0]
        for key in keys:
            aflow_dict[key].append(getattr(entry,key))
        
    except:
        continue
aflow_frame = pd.DataFrame.from_dict(aflow_dict)

print (aflow_frame)
aflow_frame.to_csv('data/Aflow_data_zero.csv')

