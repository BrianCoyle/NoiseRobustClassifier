
import json
import numpy as np
import sys
import os
from pyquil.api import get_qc
import argparse

"""
    Some functions for printing information to file
"""

def make_dir(path):
	'''Makes an directory in the given \'path\', if it does not exist already'''
	if not os.path.exists(path):
		os.makedirs(path)
	return


def make_trial_name_file(optimiser, data):
    '''This function prints out all information generated during the training process for a specified set of parameters'''

    # [n_samples, n_measurements, batch_size] = N_samples
    trial_name = "outputs/Output_data" %(data)

    path_to_output = './%s/' %trial_name
    make_dir(path_to_output)

    return trial_name

def print_params_to_file(trial_name, params, costs):
    '''This function prints out all information generated during the training process for a specified set of parameters'''

    params_path = '%s/params/' %trial_name
    make_dir(params_path)

    for cost in costs:

        cost_path  = '%s/cost/%s' %(trial_name, cost)
        make_dir(cost_path)


    np.savetxt('%s/avg_cost/train' 	%trial_name,  	avg_cost)
    np.savetxt('%s/avg_fidelity/train' %trial_name, avg_fidelity)
    params = np.array(params)

    for step in range(0, len(params)):
        np.savetxt('%s/params/step%s' %(trial_name, step), params[step])

    return

def print_to_file(optimiser, data, costs):
        
    trial_name = make_trial_name_file(optimiser,data) 

    costs = np.zeros((n_runs, n_epochs))
    fidelities = np.zeros((n_runs, n_epochs, 2))

    params = np.zeros((n_runs, n_epochs, n_params))

    for run in range(n_runs):
        params[run], costs[run], fidelities[run] = training_data_from_file(optimiser, qubit_state_type, n_epochs, η_init, qc, n_samples, n_measurements, batch_size, run)
        print(params, '\n', costs, '\n', fidelities)

    cost_mean = np.mean(costs, axis=0)
    cost_std = np.std(costs, axis=0)
    
    fidelity_mean = np.mean(fidelities, axis=0)
    fidelity_std = np.std(fidelities, axis=0)
    print(fidelities)
    print(fidelity_mean)
    params_path = '%s/params/' %avg_trial_name

    cost_mean_path   	= '%s/cost/mean' %avg_trial_name
    cost_std_path   	= '%s/cost/std' %avg_trial_name

    fidelity_mean_path   	= '%s/fidelity/mean' %avg_trial_name
    fidelity_std_path   	= '%s/fidelity/std' %avg_trial_name

    make_dir(cost_mean_path)
    make_dir(cost_std_path)

    make_dir(fidelity_mean_path)
    make_dir(fidelity_std_path)


    np.savetxt('%s/cost/mean/train' 	%avg_trial_name, cost_mean)
    np.savetxt('%s/cost/std/train'     %avg_trial_name, cost_std)

    np.savetxt('%s/fidelity/mean/train'    %avg_trial_name, fidelity_mean)
    np.savetxt('%s/fidelity/std/train'     %avg_trial_name, fidelity_std)

    return


def main():
   
      
    print_averages_to_file(optimiser, qubit_state_type, n_epochs, η_init, qc, n_samples, n_measurements, batch_size, n_params, n_runs)

if __name__ == '__main__':
	main()
