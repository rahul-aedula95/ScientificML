##########################################################################################################################################################
# This feature extraction code is constructed from snippets of the https://github.com/delton137/Machine-Learning-Energetic-Molecules-Notebooks repository#
# I do not own any part of this code however I have made changes to make it faster to reproduce the results                                              #
##########################################################################################################################################################




import pandas as pd
import scipy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.model_selection import KFold, ShuffleSplit
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import make_scorer
from sklearn.linear_model import Ridge, Lasso, LinearRegression, BayesianRidge
import warnings
# warnings.filterwarnings("ignore")
from mmltoolkit.CV_tools import *
from mmltoolkit.featurizations import * 
from mmltoolkit.fingerprints import * 
import pickle
from mmltoolkit.test_everything import * 
from sklearn.model_selection import ShuffleSplit 
from mmltoolkit.descriptors import *
from mmltoolkit.featurizations import *
from sklearn.model_selection import RandomizedSearchCV as CV

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred)))
def make_score_dict(x,y,krr):

    

    clf = krr.fit(x,y)
    

    keys = ['mean_test_MAE', 'mean_train_MAE', 'std_test_MAE','mean_test_r2', 'mean_train_r2', 'std_test_r2','mean_test_p', 'mean_train_p', 'std_test_p']
    return ({x:clf.cv_results_[x] for x in keys})

def pearson(y_true,y_pred):
    return (np.corrcoef(y_true, y_pred)[0, 1])


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / (y_true))) * 100


def all_test_everything(data, featurization_dict, targets, inner_cv=KFold(n_splits=5,shuffle=True),
                    outer_cv=ShuffleSplit(n_splits=1, test_size=0.2), verbose=False, normalize=False ):
    '''
        test all combinations of target variable, featurizations, and models
        by performing a gridsearch CV hyperparameter
        optimization for each combination and then CV scoring the best model.
        required args:
            data : a pandas dataframe with data for the different targets in the columns
            featurization_dict : a dictionary of the form {"featurization name" : X_featurization }, where X_featurization is the data array
            targets : a list of target names, corresponding to the columns in data
        important optional args:
            outer_cv : crossvalidation object specifying cross validation strategy for the outer
                     train-test CV loop. Typically we choose ShuffleSplit with a large # of splits.
            inner_cv : crossvalidation object specifying cross validation strategy for the inner train-validation
                     CV loop. K-fold with 5 folds is the standard.
        returns:
            results : a nested dictionary of the form
                     {target: { featurization_name: {model_name: scores_dict{ 'MAE': value, 'r2':value, etc }}}}
            best : a dictionary of the form {target : [best_featurization_name, best_model_name]}
    '''

    results={}
    best={}

    #num_targets = len(targets)
    

    for target in targets:
        if (verbose): print("running target %s" % target)

        y = np.array(data[target].values)

        featurizationresults = {}

        best_value = 1000000000000

        for featurization in featurization_dict.keys():
            if (verbose): print("    testing featurization %s" % featurization)

            x = featurization_dict[featurization]

            x = np.array(x)

            if (x.ndim == 1):
                x = x.reshape(-1,1)
            
            #print (x)
            if (normalize):
                st = StandardScaler()
                x = st.fit_transform(x)

            model_dict = make_models()

            modelresults = {}

            for modelname in model_dict.keys():

                model = model_dict[modelname]['model']
                param_grid = model_dict[modelname]['param_grid']

                scores_dict = nested_grid_search_CV(x, y, model, param_grid,
                                                    inner_cv=KFold(n_splits=5, shuffle=True),
                                                    outer_cv=outer_cv, verbose=verbose)
                # print ("Stuck")



                # a = 1
                # g = 0.01
                # #eta = 0.01
                # param_grid = {'alpha': scipy.stats.expon(scale=a), 'gamma': scipy.stats.expon(scale=g)}
                # scoring = {'MAE':make_scorer(mean_absolute_error), 'r2':'r2','p':make_scorer(pearson),'MAPE':make_scorer(mean_absolute_percentage_error)}
                # krr = CV(KernelRidge(kernel='rbf'), param_distributions=param_grid, cv=5, scoring=scoring, n_iter=1, n_jobs=-1, refit='MAE', return_train_score=True, random_state=0,iid=False)
                
                #scores_dict = make_score_dict(x,y,krr)
                modelresults[modelname] = scores_dict

                if (scores_dict['MAE'] < best_value):
                    best[target]=[featurization, modelname]
                    best_value = scores_dict["MAE"]

            featurizationresults[featurization] = modelresults
        
        results[target] = featurizationresults

    return (dict(results), dict(best),scores_dict)

if __name__ == "__main__":

    
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
    X_Estate_combined_lit_BoB = Estate_CDS_LBoB_featurizer(list(data['Mols']))[1]
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

    
    (results, best,fin) = all_test_everything(data, featurization_dict, targets, verbose=True, normalize=True )
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore',r'LinAlgWarning')



    pickle.dump( results, open( "data/test_all_results3.pkl", "wb" ) )
    pickle.dump( best, open( "data/test_all_best3.pkl", "wb" ) )
    pickle.dump( fin, open( "data/test_all_fin3.pkl", "wb" ) )