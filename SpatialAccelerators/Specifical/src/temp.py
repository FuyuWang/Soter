import os
import psutil
import yaml
import pickle

import torch
import torch.optim as optim
import argparse
import numpy as np
import random
from datetime import datetime

dnn = 'vgg16'
layers = yaml.load(open('../in_config/{}_problems/layers.yaml'.format(dnn), 'r'), Loader=yaml.SafeLoader)
problem = {'problem': {'shape': {'name': 'CNN-Layer', 'dimensions': ['C', 'K', 'R', 'S', 'N', 'P', 'Q'],
                                 'coefficients': [{'name': 'Wstride', 'default': 1}, {'name': 'Hstride', 'default': 1},
                                                  {'name': 'Wdilation', 'default': 1}, {'name': 'Hdilation', 'default': 1}],
                                 'data-spaces': [{'name': 'Weights', 'projection': [[['C']], [['K']], [['R']], [['S']]]},
                                                 {'name': 'Inputs', 'projection': [[['N']], [['C']],
                                                                                   [['R', 'Wdilation'], ['P', 'Wstride']],
                                                                                   [['S', 'Hdilation'], ['Q', 'Hstride']]]},
                                                 {'name': 'Outputs', 'projection': [[['N']], [['K']], [['Q']], [['P']]],
                                                  'read-write': True}]},
                       'instance': {'C': 256, 'K': 512, 'R': 3, 'S': 3, 'P': 56, 'Q': 56, 'N': 16}}}

layer2chkpt = {}
for i, layer in enumerate(layers[:-1]):
    with open('../in_config/{}_problems/{}.yaml'.format(dnn, layer), 'r') as fd:
        layer_problem = yaml.load(fd, Loader=yaml.SafeLoader)
        problem['problem']['instance']['N'] = 1
        problem['problem']['instance']['K'] = layer_problem['problem']['K']
        problem['problem']['instance']['C'] = layer_problem['problem']['C']
        problem['problem']['instance']['P'] = layer_problem['problem']['P']
        problem['problem']['instance']['Q'] = layer_problem['problem']['P']
        problem['problem']['instance']['R'] = layer_problem['problem']['R']
        problem['problem']['instance']['S'] = layer_problem['problem']['S']
        problem['problem']['instance']['Wstride'] = layer_problem['problem']['Wstride']
        problem['problem']['instance']['Hstride'] = layer_problem['problem']['Hstride']
        problem['problem']['instance']['Wdilation'] = layer_problem['problem']['Wdilation']
        problem['problem']['instance']['Hdilation'] = layer_problem['problem']['Hdilation']
    fd.close()

    layer_to_key = ''
    for key in ['R', 'P', 'C', 'K', 'Wstride', 'Wdilation']:
        layer_to_key += str(problem['problem']['instance'][key]) + '_'

    if layer_to_key in layer2chkpt:
        print(layer_to_key, 'repeated')
        chkpt = layer2chkpt[layer_to_key]
    else:
        layer2chkpt[layer_to_key] = i

print(layer2chkpt.keys(), len(layer2chkpt.keys()), layer2chkpt.values())