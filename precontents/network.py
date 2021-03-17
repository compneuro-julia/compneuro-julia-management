# -*- coding: utf-8 -*-

#import numpy as xp
import cupy as xp
# import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + xp.exp(-x))

def dsigmoid(x):
    return (1-x)*x

def f(x):
    return sigmoid(x)
    #return xp.maximum(x, 0.0)
    #return xp.tanh(x)

def df(x):
    return (1 - sigmoid(x))*sigmoid(x)
    #return 1. * (x > 0)
    #return 1/xp.cosh(10*xp.tanh(x/10))**2  # the tanh prevents overflow

class RNN:
    def __init__(self, num_neurons=[4096, 512, 128, 32], tau_m=10, 
                 alpha=0.005, beta1=0.9, beta2=0.999, eps=1e-8):
        self.num_neurons = num_neurons
        self.num_layers = len(num_neurons) - 1 # two layer
        self.tau_m = tau_m
        #self.alpha_m = 1 / tau_m
        self.alpha_m = 0.1
        self.grad_thr = 0.1
        self.param_name = ['err', 'rec', 'prd', 'td', 'bu']
        self.state_name = ['u', 'h', 'err']

        # Initialize weights: 左から out_neurons, in_neurons
        self.w_err = [self.init_weights(self.num_neurons[l+1], self.num_neurons[l]) for l in range(self.num_layers)]
        self.w_rec = [self.init_weights(self.num_neurons[l+1], self.num_neurons[l+1]) for l in range(self.num_layers)]
        self.w_prd = [self.init_weights(self.num_neurons[l], self.num_neurons[l+1]) for l in range(self.num_layers)]
        self.w_td = [self.init_weights(self.num_neurons[l+1], self.num_neurons[l+2]) for l in range(self.num_layers-1)]
        self.w_bu = [self.init_weights(self.num_neurons[l+1], self.num_neurons[l]) for l in range(self.num_layers-1)]
            
        # to Adam optimizer
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}
    
    # LeCun Normalization (http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf)
    def init_weights(self, out_neurons, in_neurons):
        return xp.random.randn(out_neurons, in_neurons) / xp.sqrt(in_neurons)
    
    def initialize_states(self):
        # changes to weights
        for pn in self.param_name:
            setattr(self, 'dw_' + pn, [0] * self.num_layers)
        
    def normalize_weight(self, weight):
        weight = weight / xp.maximum(xp.linalg.norm(weight, ord=2, axis=1, keepdims=True), 1e-8)

    # Adam optimizer
    def _update_weights(self, param, grad):
        key = id(param)
        if key not in self.ms:
            self.ms[key] = xp.zeros_like(param)
            self.vs[key] = xp.zeros_like(param)

        m, v = self.ms[key], self.vs[key]

        m += (1 - self.beta1) * (grad - m)
        v += (1 - self.beta2) * (grad * grad - v)
        param -= self.alpha * m / (xp.sqrt(v) + self.eps)
        
    # Main simulation
    def __call__(self, x, learning=True):
        #xp = cp.get_array_module(x)

        # number of timesteps
        num_batch, t_max, _ = x.shape 
        
        # Reset states
        self.initialize_states()

        # Set eligibility traces p         
        p_rec = [xp.zeros_like(self.w_rec[l]) for l in range(self.num_layers)]
        p_err = [xp.zeros_like(self.w_err[l]) for l in range(self.num_layers)]
        p_td = [xp.zeros_like(self.w_td[l]) for l in range(self.num_layers-1)]

        pred = xp.zeros_like(x)  # RNN output

        # Set local states        
        # time minus one variables
        #u_tm1 = [xp.zeros((num_batch, self.num_neurons[l+1])) for l in range(self.num_layers)]
        r_tm1 = [xp.zeros((num_batch, self.num_neurons[l+1])) for l in range(self.num_layers)]
        e_tm1 = [xp.zeros((num_batch, self.num_neurons[l])) for l in range(self.num_layers)]
        
        # current variables
        u_t = [None] * self.num_layers
        r_t = [None] * self.num_layers
        e_t = [None] * self.num_layers
        
        ahat = [None] * self.num_layers
        a = [None] * (self.num_layers - 1)
            
        # Main simulation
        for t in range(t_max):            
            # Update R states
            for l in reversed(range(self.num_layers)):
                if l == self.num_layers-1:
                    u_t[l] = r_tm1[l].dot(self.w_rec[l].T) + e_tm1[l].dot(self.w_err[l].T)
                else:
                    u_t[l] = r_tm1[l].dot(self.w_rec[l].T) + e_tm1[l].dot(self.w_err[l].T) + r_tm1[l+1].dot(self.w_td[l].T)
                    #u_t[l] = r_tm1[l].dot(self.w_rec[l].T) + e_tm1[l].dot(self.w_err[l].T) + r_t[l+1].dot(self.w_td[l].T)

                r_t[l] = r_tm1[l] + self.alpha_m * (-r_tm1[l] + f(u_t[l]))
                #r_t[l] = f(u_t[l])
                
            # Update Ahat, A, E states
            for l in range(self.num_layers):    
                if l == 0:                
                    #ahat[l] = r_t[l].dot(self.w_prd[l].T)
                    ahat[l] = sigmoid(r_t[l].dot(self.w_prd[l].T))
                    e_t[l] = x[:, t] - ahat[l]  # readout error
                else:
                    ahat[l] = r_t[l].dot(self.w_prd[l].T)
                    a[l-1] = e_tm1[l-1].dot(self.w_bu[l-1].T)
                    e_t[l] = a[l-1] - ahat[l]
                
                # Update eligibility traces
                # Note: einsum is batch-wise outer product
                p_rec[l] = (1 - self.alpha_m)*p_rec[l] \
                    + self.alpha_m * xp.einsum('bi,bj->bij', df(u_t[l]), r_tm1[l])
                p_err[l] = (1 - self.alpha_m)*p_err[l] + self.alpha_m * xp.einsum('bi,bj->bij', df(u_t[l]), e_tm1[l])

                if l < self.num_layers - 1:    
                    p_td[l] = (1 - self.alpha_m)*p_td[l] \
                        + self.alpha_m * xp.einsum('bi,bj->bij', df(u_t[l]), r_tm1[l+1])
                        #+ self.alpha_m * xp.einsum('bi,bj->bij', df(u_t[l]), r_t[l+1])
                
            # Update dw
            for l in range(self.num_layers):
                e_signal = (e_t[l]*ahat[l]*(1-ahat[l])).dot(self.w_prd[l])
                #e_signal = e_t[l].dot(self.w_err[l].T)
                #self.dw_prd[l] += xp.einsum('bi,bj->bij', dsigmoid(ahat[l])*e_t[l], r_t[l])
                self.dw_prd[l] += xp.einsum('bi,bj->bij', e_t[l]*ahat[l]*(1-ahat[l]), r_t[l])
                self.dw_rec[l] += xp.einsum('bi,bj->bij', e_signal, 
                                            xp.ones((num_batch, self.num_neurons[l+1]))) * p_rec[l]
                self.dw_err[l] += xp.einsum('bi,bj->bij', e_signal, xp.ones((num_batch, self.num_neurons[l]))) * p_err[l]
                #self.dw_err[l] += xp.einsum('bi,bj->bij', r_t[l], e_t[l])
                
                if l < self.num_layers - 1:
                    self.dw_td[l] += xp.einsum('bi,bj->bij', e_signal, 
                                            xp.ones((num_batch, self.num_neurons[l+2]))) * p_td[l]
                    self.dw_bu[l] += xp.einsum('bi,bj->bij', a[l], e_t[l])
                    
            # Update states
            #u_tm1 = u_t
            r_tm1 = r_t
            e_tm1 = e_t
            
            # Save prediction                        
            pred[:, t] = ahat[0]
        
        # Update weights
        if learning:
            for pn in self.param_name:
                if pn in ['err', 'rec', 'prd']:
                    for l in range(self.num_layers):
                        # self._update_weights(w, -np.sum(dw, axis=0)), summing up with mini-batch
                        grad = -xp.mean(getattr(self, 'dw_' + pn)[l], axis=0)
                        #grad = xp.clip(grad + xp.random.rand(*grad.shape)*1e-2, -self.grad_thr, self.grad_thr)
                        grad = xp.clip(grad, -self.grad_thr, self.grad_thr)
                        self._update_weights(getattr(self, 'w_' + pn)[l], grad)
                        #self.normalize_weight(getattr(self, 'w_' + pn)[l])
                else:
                    for l in range(self.num_layers-1):
                        grad = -xp.mean(getattr(self, 'dw_' + pn)[l], axis=0)
                        #grad = xp.clip(grad + xp.random.rand(*grad.shape)*1e-2, -self.grad_thr, self.grad_thr)
                        grad = xp.clip(grad, -self.grad_thr, self.grad_thr)
                        self._update_weights(getattr(self, 'w_' + pn)[l], grad)
                        #self.normalize_weight(getattr(self, 'w_' + pn)[l])                    
        
        return pred
