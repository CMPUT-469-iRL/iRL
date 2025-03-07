# This file holds the model definition for the RTU as well as gradient calulations for the RTU.
import sys
sys.path.append('./')
sys.path.append('../')
from flax import linen as nn
import jax 
import jax.numpy as jnp 
import flax 
from typing import Callable, Any, Tuple, Iterable,Optional
from rtu_model_example.rtus_utils import *

# eLSTM only uses a forward layer for prediction, so we will use only the forward function RTU for prediction  (might need backward layer for RL training)
class FwdRealTimeLinearRTUs(nn.Module):
    n_hidden: int                 # number of hidden features
    @nn.compact
    def __init__(hidden_init, memory_grad_unit, hidden_dim):
        # x_t.shape = (batch_size, n_features), 
        # h_tminus1.shape = ((batch_size, n_hiddens),(batch_size, n_hiddens))
        h_tminus1,grad_memory = carry
        h_tminus1_c1,h_tminus1_c2 = h_tminus1
        h_tminus1_c1,h_tminus1_c2 = jax.lax.stop_gradient(h_tminus1_c1),jax.lax.stop_gradient(h_tminus1_c2) 
        
        # these params might not be the actual r and theta, but a transformed version of them based on the param_type
        r_param = self.param('r_param', initialize_exp_exp_r, (1,self.n_hidden))
        theta_param = self.param('theta_param', initialize_theta_log, (1,self.n_hidden))
        
        mlp_xc1 = nn.Dense(hidden_dim,name='wx1',use_bias=False) 
        mlp_xc2 = nn.Dense(hidden_dim,name='wx2',use_bias=False) 
        g,phi,norm = g_phi_params(r_param,theta_param)
        
        w_c1_x_t = mlp_xc1(x_t)
        w_c2_x_t = mlp_xc2(x_t) 

        h_t_c1 = jnp.multiply(g,h_tminus1_c1) - jnp.multiply(phi,h_tminus1_c2) + jnp.multiply(norm,w_c1_x_t)
        h_t_c2 = jnp.multiply(g,h_tminus1_c2) + jnp.multiply(phi,h_tminus1_c1) + jnp.multiply(norm,w_c2_x_t)
        
        ### Needed Gradient information for gradient corrections
        d_g_w_r, d_g_w_theta, d_phi_w_r, d_phi_w_theta, d_norm_w_r = d_g_phi_exp_exp_nu_params(r_param,theta_param,g,phi,norm) 
        
        new_grad_memory_hc1_w_r = d_g_w_r * h_tminus1_c1 + g * grad_memory[0] - d_phi_w_r * h_tminus1_c2 - phi * grad_memory[1] + jnp.multiply(d_norm_w_r,w_c1_x_t)
        new_grad_memory_hc2_w_r = d_g_w_r * h_tminus1_c2 + g * grad_memory[1] + d_phi_w_r * h_tminus1_c1 + phi * grad_memory[0] + jnp.multiply(d_norm_w_r,w_c2_x_t)
        
        new_grad_memory_hc1_w_theta = d_g_w_theta * h_tminus1_c1 + g * grad_memory[2] - d_phi_w_theta * h_tminus1_c2 - phi * grad_memory[3]
        new_grad_memory_hc2_w_theta = d_g_w_theta * h_tminus1_c2 + g * grad_memory[3] + d_phi_w_theta * h_tminus1_c1 + phi * grad_memory[2]
       
         
        new_grad_c1_wx1 = jnp.multiply(g,grad_memory[4]) - jnp.multiply(phi,grad_memory[5]) + jnp.multiply(norm,jnp.repeat(jnp.expand_dims(x_t,2),h_t_c1.shape[-1],axis=2))
        new_grad_c1_wx2 = jnp.multiply(g,grad_memory[6]) - jnp.multiply(phi,grad_memory[7]) 
        
        new_grad_c2_wx1 = jnp.multiply(g,grad_memory[5]) + jnp.multiply(phi,grad_memory[4]) 
        new_grad_c2_wx2 = jnp.multiply(g,grad_memory[7]) + jnp.multiply(phi,grad_memory[6]) + jnp.multiply(norm,jnp.repeat(jnp.expand_dims(x_t,2),h_t_c2.shape[-1],axis=2))
    
        new_grad_memory = (new_grad_memory_hc1_w_r,new_grad_memory_hc2_w_r,new_grad_memory_hc1_w_theta,new_grad_memory_hc2_w_theta,new_grad_c1_wx1,new_grad_c2_wx1,new_grad_c1_wx2,new_grad_c2_wx2)
        
        return ((h_t_c1,h_t_c2),jax.lax.stop_gradient(new_grad_memory)), ((h_t_c1,h_t_c2),jax.lax.stop_gradient(new_grad_memory))