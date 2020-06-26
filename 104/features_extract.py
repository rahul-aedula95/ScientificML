
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge, Lasso, LinearRegression, BayesianRidge
import warnings
warnings.filterwarnings("ignore")
from mmltoolkit.CV_tools import *
from mmltoolkit.featurizations import * 
from mmltoolkit.fingerprints import * 
import pickle
from mmltoolkit.test_everything import * 
from sklearn.model_selection import ShuffleSplit 
from mmltoolkit.descriptors import *
from mmltoolkit.featurizations import *




#Read the data
data = pd.read_excel('data/Huang_Massa_data_with_all_SMILES.xlsx', skipfooter=1)

target_prop = 'Explosive energy (kj/cc)'

#Add some new columns
data['Mols'] = data['SMILES'].apply(Chem.MolFromSmiles)


#important - add hydrogens!!
data['Mols'] = data['Mols'].apply(Chem.AddHs)


X_Estate = truncated_Estate_featurizer(list(data['Mols']))


num_mols = len(data)

targets = [
 'Density (g/cm3)',
 'Delta Hf solid (kj/mol)',
 'Explosive energy (kj/cc)',
 'Shock velocity (km/s)',
 'Particle velocity (km/s)',
 'Speed of sound (km/s)',
 'Pressure (Gpa)',
 'T(K)',
 'TNT Equiv (per cc)'
  ]



bond_types, X_LBoB = literal_bag_of_bonds(list(data['Mols'])) 

num_atoms = []
for mol in data['Mols']:
    mol = Chem.AddHs(mol)
    num_atoms += [mol.GetNumAtoms()]
    
max_atoms = int(max(num_atoms))

X_Cmat_as_vec = np.zeros((num_mols, (max_atoms**2-max_atoms)//2 + max_atoms))
X_Cmat_eigs = np.zeros((num_mols, max_atoms))
X_Cmat_unsorted_eigs = np.zeros((num_mols, max_atoms))

X_summedBoB = []
filename_list = []

for i, refcode in enumerate(data['Molecular Name']):
    filename = 'data/HM_all_xyz_files/'+refcode+'.xyz'
    this_Cmat_eigs, this_Cmat_as_vec = coulombmat_and_eigenvalues_as_vec(filename, max_atoms )
    this_Cmat_unsorted_eigs, this_Cmat_as_vec = coulombmat_and_eigenvalues_as_vec(filename, max_atoms, sort=False)

    summed_BoB_feature_names, summedBoB = summed_bag_of_bonds(filename)
    X_summedBoB += [summedBoB]

    filename_list += [filename]
    
    X_Cmat_eigs[i,:] = this_Cmat_eigs
    X_Cmat_unsorted_eigs[i,:] = this_Cmat_eigs
    X_Cmat_as_vec[i,:] = this_Cmat_as_vec

X_summedBoB = np.array(X_summedBoB)

BoB_feature_list, X_BoB = bag_of_bonds(filename_list, verbose=False)


data['Oxygen Balance_100'] = data['Mols'].apply(oxygen_balance_100)
data['Oxygen Balance_1600'] = data['Mols'].apply(oxygen_balance_1600)

data['modified OB'] = data['Mols'].apply(modified_oxy_balance)
data['OB atom counts'] = data['Mols'].apply(return_atom_nums_modified_OB)
data['combined_nums'] =  data['Mols'].apply(return_combined_nums)


X_OB100 = np.array(list(data['Oxygen Balance_100'])).reshape(-1,1)     
X_OB1600 = np.array(list(data['Oxygen Balance_1600'])).reshape(-1,1)     
X_OBmod = np.array(list(data['modified OB'])).reshape(-1,1)   
X_OB_atom_counts = np.array(list(data['OB atom counts']))
X_combined = np.array(list(data['combined_nums']))

X_Estate_combined = np.concatenate((X_Estate, X_combined), axis=1)
X_Estate_combined_Cmat_eigs = np.concatenate((X_Estate_combined, X_Cmat_eigs), axis=1)
X_Estate_combined_lit_BoB = Estate_CDS_LBoB_featurizer(list(data['Mols']))
X_CustDesrip_lit_BoB = np.concatenate(( X_combined, X_LBoB), axis=1)


featurization_dict = {
                 "Estate": X_Estate,
                 "Oxygen balance$_{100}$": X_OB100, 
                 "Oxygen balance$_{1600}$": X_OB1600, 
                 "Oxygen balance atom counts": X_OB_atom_counts,
                 "CDS": X_combined,
                 "SoB" : X_LBoB,
                 'Estate+CDS':   X_Estate_combined,
                 "Coulomb matrices as vec" :   X_Cmat_as_vec,
                 "CM eigs": X_Cmat_eigs,
                 "Bag of Bonds": X_BoB,
                 "Summed Bag of Bonds (sBoB)": X_summedBoB, 
                 "\\footnotesize{Estate+CDS+SoB}":X_Estate_combined_lit_BoB,
                 "C.D.S + LBoB": X_CustDesrip_lit_BoB,
                 "LBoB + OB100": np.concatenate(( X_LBoB, X_OB100), axis=1)
                }

targets = [
 #'Density (g/cm3)',
 #'Delta Hf solid (kj/mol)',
 'Explosive energy (kj/cc)',
 #'Shock velocity (km/s)',
 #'Particle velocity (km/s)',
 #'Speed of sound (km/s)',
 #'Pressure (Gpa)',
 #'T(K)',
 #'TNT Equiv (per cc)'
  ]


(results, best) = test_everything(data, featurization_dict, targets, verbose=True, normalize=True )



pickle.dump( results, open( "data/test_all_results3.pkl", "wb" ) )
pickle.dump( best, open( "data/test_all_best3.pkl", "wb" ) )