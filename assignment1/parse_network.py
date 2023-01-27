from network import *
from dense import Dense
from activation import Activation
from loss import *
from activation_functions import *
import ast
import re

def parse_file(filepath):
    with open(filepath, 'r') as f:
        data = f.read()

    defaults = {}
    layers = []

    # extract defaults
    defaults_match = re.search(r'DEFAULTS\n(.*?)\nDEFAULTS', data, re.DOTALL)
    if defaults_match:
        defaults_data = defaults_match.group(1)
        defaults_lines = defaults_data.strip().split('\n')
        for line in defaults_lines:
            key, value = line.split(':')
            defaults[key] = value

    # extract layers
    layers_match = re.search(r'LAYERS\n(.*?)\nLAYERS', data, re.DOTALL)
    if layers_match:
        layers_data = layers_match.group(1)
        layers_lines = layers_data.strip().split('\n')
        for line in layers_lines:
            parts = line.split()
            layer_type = parts[0]
            layer_data = {}
            for part in parts[1:]:
                key, value = part.split(':')
                layer_data[key] = value
            layers.append((layer_type, layer_data))

    

    default_loss = defaults['loss']
    default_lr = float(defaults['lr'])
    default_wlambda = float(defaults['wlambda'])
    default_wrt = defaults['wrt']

    network = []
    wr = None
    br = None


    for layer in layers:
        if layer[0].lower().capitalize() == 'Dense':
            values = list(layer[1].values())
            keys = list(layer[1].keys())
            in_dim = int(values[0])
            out_dim = int(values[1])
            if 'wr' in keys:
                wr = ast.literal_eval(layer[1]['wr'])
                wr = tuple(map(float, wr))
            if 'br' in keys:
                br = ast.literal_eval(layer[1]['br'])
                br = tuple(map(float, br))
            if 'wrt' in keys:
                wrt = layer[1]['wrt']
            if 'lr' in keys:
                lr = float(layer[1]['lr'])
            network.append(Dense(in_dim, out_dim, wr=wr, br=br, regularization=default_wrt, reg_lambda=default_wlambda, lr=lr))

        elif layer[0].lower().capitalize() == 'Sigmoid':
            network.append(Sigmoid())
        elif layer[0].lower().capitalize() == 'Relu':
            network.append(Relu())
        elif layer[0].lower().capitalize() == 'Tanh':
            network.append(Tanh())
        elif layer[0].lower().capitalize() == 'LeakyRelu':
            network.append(LeakyRelu())
        elif layer[0].lower().capitalize() == 'Linear':
            network.append(Linear())
        elif layer[0].lower().capitalize() == 'Softmax':
            network.append(Softmax())

    if default_loss == 'mse':
        default_loss = mse
        default_dloss = dmse
    elif default_loss == 'cross_entropy':
        default_loss = cross_entropy
        default_dloss = dcross_entropy


    return network, default_loss, default_dloss, default_lr, default_wlambda, default_wrt

