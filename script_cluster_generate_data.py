import pandas as pd
import module_generate_data 



data_file_dump = 'data_test_Uni_1_no_S.csv'
number_of_instances = 100
param_grid = {
    'n': [128, 256, 512, 1024], 
    'trials': [50],
    'proxy_discard': [50],
    'goal_discard': [50, 150, 250, 450],
    'measure_time': [50],
    'alpha': [0.1],
    "avg_number_of_edges": [5,10,20,40], 
    "weight_range": [1]
}


module_generate_data.generate_experimental_data(data_file_dump, param_grid, number_of_instances, intention = 'w', skip_to = 0, seed_skip=0, nb_cores=100)