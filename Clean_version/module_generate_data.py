import module_ESN
import numpy as np
import csv
from joblib import Parallel, delayed
from itertools import product

import warnings
import module_graph

def optimize_node(esn,  trials, proxy_nodes, is_goal=False):

    n = esn.n 
    num_action_nodes = len(esn.action_nodes)
    start_time = esn.proxy_discard*(not is_goal) + esn.goal_discard*is_goal
    end_time = start_time + esn.measure_time
    # define function to test an agent action 
    def test_agent_action(agent_action, action_value):
        """
        Test the average value of an action node if all other nodes are set to random values
        """
        proxy_samples    = np.asarray([])
        goal_samples    = np.asarray([])
        esn.reset_seed()
        for j in range(trials): 
            action_node_values = module_ESN.get_directed_action_value(action_value , agent_action, actionsize = len(esn.action_nodes))
            states = esn.run(agent_values=action_node_values)
            proxy_samples = np.concatenate((proxy_samples,np.mean(states[start_time:end_time, proxy_nodes], axis=0)))
            goal_samples = np.concatenate((goal_samples,np.mean(states[esn.goal_discard:esn.goal_discard+esn.measure_time, esn.goal], axis=0)))
        if not is_goal:
            correl_node_goal = np.corrcoef(proxy_samples, goal_samples)[0, 1]
        else:
            correl_node_goal = 0

        return np.mean(proxy_samples), np.mean(goal_samples), correl_node_goal
    max_proxy_value = -1
    goal_value_on_max_proxy = -1
    correlation_on_max_proxy = 0
    for i in range(num_action_nodes):
        for val in [-1, 1]:
            proxy_value, goal_value, correl_node_goal = test_agent_action(i, val)
            if proxy_value > max_proxy_value:
                max_proxy_value = proxy_value
                goal_value_on_max_proxy = goal_value
                correlation_on_max_proxy = correl_node_goal

    return (goal_value_on_max_proxy, max_proxy_value, correlation_on_max_proxy)

def compute_proxy_nodes_from_esn(esn, trials=10):  
    """
    Compute proxy nodes from an Echo State Network (ESN) by running trials with random agent actions.
    The function calculates the average values of the goal node and the proxy nodes based on the ESN's states.
    """

    num_nodes = esn.n
    possible_nodes =  [i for i in range(num_nodes) if i not in esn.action_nodes and i not in esn.goal]
    run_data_base = np.zeros((trials, num_nodes)) 
    avg_goal_values = []
    avg_proxy_values = []
    esn.reset_seed()
    for k in range(trials):
        # Generate random agent actions
        action_node_values = module_ESN.get_base_action_value(len(esn.action_nodes))
        
        # Run the ESN with the random agent actions
        states = esn.run( agent_values=action_node_values)
        data_run =  np.mean(states[esn.proxy_discard: esn.proxy_discard + esn.measure_time, :], axis=0)
        data_run[esn.goal] = np.mean(states[esn.goal_discard: esn.goal_discard+ esn.measure_time, esn.goal], axis=0)
        run_data_base[k] = data_run
        # Compute the average values 
        avg_goal_values.append(np.mean(states[:, esn.goal], axis=0))
    correlations = np.mean(np.corrcoef(run_data_base, rowvar= False)[:, esn.goal], axis=1)#here the runs are on axis 0 and the nodes on axis 1

    bin_edges = np.linspace(-1, 1, 100)
    bin_indices =  np.histogram(correlations, bin_edges)[0] 
    bin_indices = [int(k) for k in  bin_indices]
    # check if there are nan values in correlations
    if np.isnan(correlations).any():
        raise ValueError("NaN values found in correlations")
        

    # Sort nodes by correlation in descending order and select top nodes
    sorted_indices = [k for k in np.argsort(correlations)[::-1 ] if k in possible_nodes]
    proxy_nodes = sorted_indices[:1]
    avg_proxy_values.append(np.mean(run_data_base[:, proxy_nodes], axis=0))
    goal_value = np.mean(avg_goal_values)
    proxy_value = np.mean(avg_proxy_values)
    #print(bin_indices)

    return proxy_nodes, goal_value, proxy_value, correlations, bin_indices

import module_graph

def optimize_node(esn,  trials, proxy_nodes, is_goal=False):

    n = esn.n 
    num_action_nodes = len(esn.action_nodes)
    start_time = esn.proxy_discard*(not is_goal) + esn.goal_discard*is_goal
    end_time = start_time + esn.measure_time
    # define function to test an agent action 
    def test_agent_action(agent_action, action_value):
        """
        Test the average value of an action node if all other nodes are set to random values
        """
        proxy_samples    = np.asarray([])
        goal_samples    = np.asarray([])
        esn.reset_seed()
        for j in range(trials): 
            action_node_values = module_ESN.get_directed_action_value(action_value , agent_action, actionsize = len(esn.action_nodes))
            states = esn.run(agent_values=action_node_values)
            proxy_samples = np.concatenate((proxy_samples,np.mean(states[start_time:end_time, proxy_nodes], axis=0)))
            goal_samples = np.concatenate((goal_samples,np.mean(states[esn.goal_discard:esn.goal_discard+esn.measure_time, esn.goal], axis=0)))
        if not is_goal:
            correl_node_goal = np.corrcoef(proxy_samples, goal_samples)[0, 1]
        else:
            correl_node_goal = 0

        return np.mean(proxy_samples), np.mean(goal_samples), correl_node_goal
    max_proxy_value = -1
    goal_value_on_max_proxy = -1
    correlation_on_max_proxy = 0
    for i in range(num_action_nodes):
        for val in [-1, 1]:
            proxy_value, goal_value, correl_node_goal = test_agent_action(i, val)
            if proxy_value > max_proxy_value:
                max_proxy_value = proxy_value
                goal_value_on_max_proxy = goal_value
                correlation_on_max_proxy = correl_node_goal

    return (goal_value_on_max_proxy, max_proxy_value, correlation_on_max_proxy)

def parallel_compute_proxy_failure(param_ESN):
    esn = module_ESN.EchoStateNetwork(param_ESN['n'],  spectral_radius=param_ESN['spectral_radius'], alpha = param_ESN['alpha'], 
                                avg_number_of_edges=param_ESN['avg_number_of_edges'], proxy_discard=param_ESN['proxy_discard'],
                                goal_discard=param_ESN['goal_discard'], measure_time=param_ESN['measure_time'], seed=param_ESN["seed"])

    proxy_nodes, goal_base, proxy_base, correlations, bin_indices = compute_proxy_nodes_from_esn(esn, trials=param_ESN['trials'])
    correlation_base = np.mean(correlations[proxy_nodes])
    correlation_std = np.std(correlations)
    maxed_goal_value, maxed_proxy_value, correlation_on_max_proxy = optimize_node(esn,  param_ESN['trials'], proxy_nodes)
    optimal_goal_value, optimal_proxy_value, correlation_on_optimal_proxy =  optimize_node(esn,  param_ESN['trials'], esn.goal, is_goal=True)
    to_return = {
        'correlation_std': correlation_std,
        'maxed_goal_value': maxed_goal_value,
        'goal_base': goal_base,
        'maxed_proxy_value': maxed_proxy_value,
        'correlation_on_max_proxy': correlation_on_max_proxy,
        'correlation_base': correlation_base,
        'optimal_goal_value': optimal_goal_value,
        'optimal_proxy_value': optimal_proxy_value,
        'bin_indices_on_base_correlation': bin_indices,
        'proxy_base': proxy_base
    }
    return to_return

def parallel_compute_correlation(param_exp,param_ESN):
    
    esn = module_ESN.EchoStateNetwork(param_ESN['n'],  spectral_radius=param_ESN['spectral_radius'], alpha = param_ESN['alpha'], 
                                avg_number_of_edges=param_ESN['avg_number_of_edges'], discard=param_ESN['discard'])
    param_exp['esn'] = esn
    # Compute proxy nodes
    proxy_nodes, goal_base, proxy_base, correlations = compute_proxy_nodes_from_esn(esn, trials=param_ESN['trials'])
    correlations = list(correlations)
    to_return = {
        'correlations': correlations,
    } 
    return to_return


def generate_experimental_data(filename, param_grid, number_of_instances, intention = 'a', skip_to = 0,seed_skip = 0):
    # creates a list of the parameters that will vary
    varying_params = [a for a in param_grid.keys() if len(param_grid[a]) > 1]
    # Create a list of all combinations of parameters

    param_combinations = list(product(*param_grid.values()))
    print("param_combinations", len(param_combinations))
    # Create a list of parameter names
    param_names = list(param_grid.keys())
    # Loop through all combinations of parameters

    param_range = range(len(param_combinations))
    if skip_to > 0:
        param_range = range(skip_to-1, len(param_range) )
        print("skipping to", skip_to, "out of", len(param_combinations))
        if intention != 'a':
            print("****** BEWARE ******\n you want to skip but your intention is not 'a'")
            exit()
    print(filename, intention)
    with open(filename,intention) as fd:
        fieldnames = ['correlation_std', 'bin_indices_on_base_correlation',
                      'maxed_goal_value','goal_base',
                      'maxed_proxy_value','proxy_base',
                      'correlation_on_max_proxy','correlation_base',
                      'optimal_goal_value','optimal_proxy_value'] 
        writer = csv.DictWriter(fd, fieldnames=param_names + fieldnames)
        if intention == 'w':
            writer.writeheader()
        for k in param_range:

            params = param_combinations[k]
            # Create a dictionary of parameters
            param_ESN = dict(zip(param_names, params))
            param_ESN['trials'] = 50
            name = 'ESN'+ "".join(['_'+a +'_'+ str(b) for (a,b) in list(zip(param_names, params)) if a in varying_params])
            print("instance", k+1, "out of ", len(param_combinations), "name", name)
            # run the parallel computation of proxy nodes
            List_output = Parallel(n_jobs=10, return_as='list')(
                [delayed(parallel_compute_proxy_failure)( dict(param_ESN, seed= i+ seed_skip)) for i in range(number_of_instances)]
            )
            for temp_dict in List_output:
                writer.writerow(param_ESN | temp_dict)
            fd.flush()
            


