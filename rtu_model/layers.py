# This file holds the RNN function for the RTU, initializing and updating the weights of the model
import sys
sys.path.append('./')
sys.path.append('../')
from flax import linen as nn
import jax 
import jax.numpy as jnp 
import flax 
from typing import Callable, Any, Tuple, Iterable,Optional
from rtu_model_example.rtus_utils import *
from rtu_model_example.linear_rtus import *
from rtu_model_example.non_linear_rtus import *

class RTUlayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_size, forget_bias=0.):
        super().__init__()

        # define dimensions of input and hidden layers
        d_input = input_dim
        d_rec = hidden_dim

        # define the hidden unit tensor
        hidden_init = (jnp.zeros((batch_size,d_rec)),jnp.zeros((batch_size,d_rec)))
        # define the memory gradient
        memory_grad_init = (jnp.zeros((batch_size,d_rec)),jnp.zeros((batch_size,d_rec)),
                            jnp.zeros((batch_size,d_rec)),jnp.zeros((batch_size,d_rec)),
                            jnp.zeros((batch_size,d_input, d_rec)),jnp.zeros((batch_size,d_input, d_rec)),
                            jnp.zeros((batch_size,d_input, d_rec)),jnp.zeros((batch_size,d_input, d_rec)))