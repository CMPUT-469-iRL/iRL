# This file holds the RNN function for the RTU, used for initializing and updating the weights of the model
import sys
sys.path.append('./')
sys.path.append('../')
import flax 
from flax import linen as nn
import jax 
import jax.numpy as jnp 
from typing import Callable, Any, Tuple, Iterable,Optional
from rtu_model.rtus_utils import *
from rtu_model.model import *
## real-time rtus expect inputs of shape (batch_size, n_features)

'''
A Consice interface to Real-Time Linear RTUs
Linear recurrence + non-linear output 
'''
class RTULayer(nn.Module):
    n_hidden: int   # number of hidden features
    activation: str = 'relu'
    @nn.compact
    def __init__(self,carry,x_t):
        update_gate = RTUModel(self.n_hidden)
        carry,(h_t_c1,h_t_c2)  = update_gate(carry,x_t)
        h_t = act_options[self.activation](jnp.concatenate((h_t_c1, h_t_c2), axis=-1))
        return carry,h_t # carry, output
    
    @staticmethod
    def initialize_state(batch_size,d_rec,d_input):
        hidden_init = (jnp.zeros((batch_size,d_rec)),jnp.zeros((batch_size,d_rec)))
        memory_grad_init = (jnp.zeros((batch_size,d_rec)),jnp.zeros((batch_size,d_rec)),
                            jnp.zeros((batch_size,d_rec)),jnp.zeros((batch_size,d_rec)),
                            jnp.zeros((batch_size,d_input, d_rec)),jnp.zeros((batch_size,d_input, d_rec)),
                            jnp.zeros((batch_size,d_input, d_rec)),jnp.zeros((batch_size,d_input, d_rec)))
        return (hidden_init,memory_grad_init)