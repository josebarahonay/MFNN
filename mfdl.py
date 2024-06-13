import os
import sys
import numpy as np
import torch
import torch.nn as nn

# Neural network class
class NN(nn.Module):
    def __init__(self, input_dim, output_dim, n_layers, n_neurons, activation, dropout):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(input_dim, n_neurons))
            if activation == 'ReLU':
                #print("Activation: ReLU")
                layers.append(nn.ReLU())
            elif activation == 'Tanh':
                #print("Activation: Tanh")
                layers.append(nn.Tanh())
            elif activation == 'None':
                print("Activation: None")
            if dropout == True:
                layers.append(nn.Dropout(0.0))
            input_dim = n_neurons
        layers.append(nn.Linear(input_dim, output_dim))
        self.layers_stack = nn.Sequential(*layers)

    def forward(self, x):
        y = self.layers_stack(x)
        return y

# Weights initialization
def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)

# Single-fidelity neural network
def singlefidelity_NN(hf, d_H, activation,regularization, n_epochs, learning_rate, device, history):
    """
    Construct and train a single-fidelity neural network.
    
    hf = [layers_H, neurons_H],
    d_H = [X_H, Y_H],
    regularization = [r_H],
    n_epochs = epochs,
    learning_rate = alpha
    """
    
    layers_H = hf[0]
    neurons_H = hf[1]
    X_H = d_H[0]
    Y_H = d_H[1]
    r_H = regularization
    epochs = n_epochs
    alpha = learning_rate
    act = activation

    NN_H = NN(input_dim = X_H.shape[1],
                output_dim = Y_H.shape[1],
                n_layers = layers_H,
                n_neurons = neurons_H,
                activation = act,
                dropout = False).to(device)

    NN_H.apply(weights_init)

    # Training settings
    criterion = nn.MSELoss()
    params_nns = NN_H.parameters()
    optimizer = torch.optim.Adam(params_nns, lr=alpha)

    num_epochs = epochs
    lambda_H = r_H
    
    ## Fit (optimize) the models
    for epoch in range(num_epochs):

        yh = NN_H(X_H)    
        MSE_yh = criterion(yh, Y_H)
        
        reg_H = sum([(p**2).sum() for p in NN_H.parameters()])

        MSE = MSE_yh + lambda_H*reg_H
        loss = MSE

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if history == True:
            if (epoch+1) % 1000 == 0:
                print(f'epoch: {epoch+1},\
                        loss = {loss.item():.4f}')
        
    print('Model optimized')
    
    return NN_H

# Multi-fidelity neural network
def multifidelity_NN(lf, hf, d_L, d_H, activation, regularization, n_epochs, learning_rate, device, history):
    """
    Construct and train a multi-fidelity neural network.
    
    Parameters
    ----------
    lf : list
    	Number of layers and neurons for low-fidelity NN.
    	[layers_L, neurons_L]
    hf : list
    	Number of layers and neurons for low-fidelity NN.
    	[layers_H, neurons_H]
    d_L : list
    	Input and output low-fidelity data.
    	[X_L, Y_L]
    d_H : list
    	Input and output high-fidelity data.
    	[X_H, Y_H]
    activation: str
    	Activation function ('RelU', 'Tanh').
    regularization : list
    	Regularization values for low- and high-fidelity NNs.
    	[r_L, r_H1, r_H2]
    n_epochs : int
    	Number of epochs.  
    	epochs
    learning_rate : float
    	Initial learning rate for Adam optimization. 
    	alpha
    history: boolean
        If True, returns the loss in the training process.
    """
    
    layers_L = lf[0]
    neurons_L = lf[1]
    layers_H = hf[0]
    neurons_H = hf[1]
    X_L = d_L[0]
    Y_L = d_L[1]
    X_H = d_H[0]
    Y_H = d_H[1]
    r_L = regularization[0]
    r_H1 = regularization[1]
    r_H2 = regularization[2]
    epochs = n_epochs
    alpha = learning_rate
    act = activation

    NN_L = NN(input_dim = X_L.shape[1],
                output_dim = Y_L.shape[1],
                n_layers = layers_L,
                n_neurons = neurons_L,
                activation = act,
                dropout = False).to(device)

    NN_H1 = NN(input_dim = X_H.shape[1] + Y_L.shape[1],
                output_dim = Y_H.shape[1],
                n_layers = layers_H,
                n_neurons = neurons_H,
                activation = 'None',
                dropout = False).to(device)

    NN_H2 = NN(input_dim = X_H.shape[1] + Y_L.shape[1],
                output_dim = Y_H.shape[1],
                n_layers = layers_H,
                n_neurons = neurons_H,
                activation = act,
                dropout = False).to(device)

    NN_L.apply(weights_init)
    NN_H1.apply(weights_init)
    NN_H2.apply(weights_init)

    # Training settings
    criterion = nn.MSELoss()
    params_nns = list(NN_L.parameters()) + list(NN_H1.parameters()) + list(NN_H2.parameters())
    optimizer = torch.optim.Adam(params_nns, lr=alpha)

    num_epochs = epochs
    lambda_L = r_L
    lambda_H1 = r_H1
    lambda_H2 = r_H2
    
    ## Fit (optimize) the models
    for epoch in range(num_epochs):

        yl = NN_L(X_L)
        yl_h = NN_L(X_H)
        Fl = NN_H1(torch.concatenate([X_H, yl_h], 1))
        Fnl = NN_H2(torch.concatenate([X_H, yl_h], 1))
        F = Fl + Fnl
        yh = F
 
        MSE_yl = criterion(yl, Y_L)
        MSE_yh = criterion(yh, Y_H)
        
        reg_L = sum([(p**2).sum() for p in NN_L.parameters()])
        reg_H1 = sum([(p**2).sum() for p in NN_H1.parameters()])
        reg_H2 = sum([(p**2).sum() for p in NN_H2.parameters()])

        reg = reg_L + reg_H1 + reg_H2

        MSE = MSE_yl + MSE_yh + lambda_L*reg_L + lambda_H1*reg_H1 + lambda_H2*reg_H2
        loss = MSE

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if history == True:
            if (epoch+1) % 1000 == 0:
                print(f'epoch: {epoch+1},\
                        loss = {loss.item():.4f}')

    print('Model optimized')
    
    return NN_L, NN_H1, NN_H2
