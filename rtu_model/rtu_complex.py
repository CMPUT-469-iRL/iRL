import torch.nn as nn
import torch
import numpy as np
import math
import random

device = torch.device('cuda')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
# from rtu_utils import *

class RTUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_t, h, lamda, theta, B_c1, B_c2, s_r_c1_prev, s_r_c2_prev, s_theta_c1_prev, s_theta_c2_prev, s_B_c1_h_c1_prev, s_B_c2_h_c1_prev,  s_B_c1_h_c2_prev, s_B_c2_h_c2_prev):

        # theta = math.log(6.28 * random.uniform(1, B.shape[0]))   # hidden_size = B.shape[0]
        # r_max = 1
        # r_min = 0
        # r = math.log(0.5 * math.log(random.uniform(1, B.shape[0]) * (r_max**2 - r_min**2) + r_min**2))

        # let r = lamda
        
        r = lamda

        # common variables in gradients
        y = torch.exp(-torch.exp(lamda))
        z = torch.exp(theta)
        gamma = torch.sqrt(1-(y**2))
        
        # common gradients
        y_gradient = -torch.exp(r) * y
        gamma_gradient = (-y / gamma) * y_gradient
        z_gradient = torch.exp(theta)

        # extract the previous h-values
        hidden_size = B_c1.shape[0]
        # print(type(h))
        h_prev_c1 = h[0:hidden_size] # hidden_size = B.shape[0]
        h_prev_c2 = h[hidden_size:]
        
        # print(h().shape)
        # print("h", h())
        # print(h_prev_c1.shape)
        # print(h_prev_c2.shape)

        # c1 and c2 h updates
        h_next_c1 = y * torch.cos(z) * h_prev_c1 - y * torch.sin(z) * h_prev_c2 + gamma * B_c1.mv(input_t.to(dtype=torch.float))
        h_next_c2 = y * torch.cos(z) * h_prev_c2 + y * torch.sin(z) * h_prev_c1 + gamma * B_c2.mv(input_t.to(dtype=torch.float))

        # get gradients of h wrt. r                 #ht                             # delta ht-1 / detta r
        s_r_c1_next = (y_gradient * torch.cos(z) * h_prev_c1) + (y * torch.cos(z) * s_r_c1_prev) - (y_gradient * torch.sin(z) * h_prev_c2) - (y * torch.sin(z) * s_r_c2_prev) + (gamma_gradient * B_c1.mv(input_t.to('cuda', dtype=torch.float)))
        s_r_c2_next = (y_gradient * torch.cos(z) * h_prev_c2) + (y * torch.cos(z) * s_r_c2_prev) + (y_gradient * torch.sin(z) * h_prev_c1) + (y * torch.sin(z) * s_r_c1_prev) + (gamma_gradient * B_c2.mv(input_t.to('cuda', dtype=torch.float)))

        # get gradients of h wrt. theta
        s_theta_c1_next = (-y * torch.sin(z) * z_gradient * h_prev_c1) + (y * torch.cos(z) * s_theta_c1_prev) - (y * torch.cos(z) * z_gradient * h_prev_c2) - (y * torch.sin(z) * s_theta_c2_prev)
        s_theta_c2_next = (-y * torch.sin(z) * z_gradient * h_prev_c2) + (y * torch.cos(z) * s_theta_c2_prev) + (y * torch.cos(z) * z_gradient * h_prev_c1) + (y * torch.sin(z) * s_theta_c1_prev)


        # gradients of sentivity matrices of B
        s_B_c1_h_c1_next = (y.unsqueeze(1) * torch.cos(z).unsqueeze(1) * s_B_c1_h_c1_prev) - (y.unsqueeze(1) * torch.sin(z).unsqueeze(1) * s_B_c1_h_c2_prev) + (torch.outer(gamma, input_t.T))
        s_B_c1_h_c2_next = (y.unsqueeze(1) * torch.cos(z).unsqueeze(1) * s_B_c1_h_c2_prev) + (y.unsqueeze(1) * torch.sin(z).unsqueeze(1) * s_B_c1_h_c1_prev)
        s_B_c2_h_c1_next = (y.unsqueeze(1) * torch.cos(z).unsqueeze(1) * s_B_c2_h_c1_prev) - (y.unsqueeze(1) * torch.sin(z).unsqueeze(1) * s_B_c2_h_c2_prev)
        s_B_c2_h_c2_next = (y.unsqueeze(1) * torch.cos(z).unsqueeze(1) * s_B_c2_h_c2_prev) + (y.unsqueeze(1) * torch.sin(z).unsqueeze(1) * s_B_c2_h_c1_prev) + (torch.outer(gamma, input_t.T))

        ctx.save_for_backward(s_r_c1_next, s_r_c2_next, s_theta_c1_next, s_theta_c2_next, s_B_c1_h_c1_next, s_B_c2_h_c1_next, s_B_c1_h_c2_next, s_B_c2_h_c2_next,  B_c1, B_c2)

        # concatnate h_c1 and h_c2 together
        h = torch.cat((h_next_c1, h_next_c2), dim=0) # h = torch.cat((h_next_c1, h_next_c2), dim=0) # h = torch.cat((h_next_c1, h_next_c2), dim=0)
        return h, s_r_c1_next, s_r_c2_next, s_theta_c1_next, s_theta_c2_next, s_B_c1_h_c1_next, s_B_c2_h_c1_next, s_B_c1_h_c2_next, s_B_c2_h_c2_next

    @staticmethod
    def backward(ctx, grad_output_h_next, grad_output_s_lamda_next_c1,grad_output_s_lamda_next_c2, grad_output_s_theta_next_c1, grad_output_s_theta_next_c2, grad_output_s_B_c1_h_c1_next, grad_output_s_B_c2_h_c1_next, grad_output_s_B_c1_h_c2_next, grad_output_s_B_c2_h_c2_next): # def backward(ctx, grad_output_h_next, grad_output_s_lamda_next_c1,grad_output_s_lamda_next_c2, grad_output_s_theta_next_c1, grad_output_s_theta_next_c2, grad_output_s_B_next_c1, grad_output_s_B_next_c2):
        s_r_c1_next, s_r_c2_next, s_theta_c1_next, s_theta_c2_next,  s_B_c1_h_c1_next, s_B_c2_h_c1_next, s_B_c1_h_c2_next, s_B_c2_h_c2_next, B_c1, B_c2 = ctx.saved_tensors  # s_lamda_next, s_B_next, B = ctx.saved_tensors
        
        hidden_size = B_c1.shape[0]
        grad_output_h_next_c1 = grad_output_h_next[0:hidden_size] # hidden_size = B.shape[0]
        grad_output_h_next_c2 = grad_output_h_next[hidden_size:]

        # dL/dh_c1 = grad_output_h_next_c1
        # dL/dh_c2 = grad_output_h_next_c2
        # dh_c1 / r = s_r_c1_next
        # dh_c2 / r = s_r_c2_next

        grad_lamda_c1 = grad_output_h_next_c1 * s_r_c1_next 
        grad_lamda_c2 = grad_output_h_next_c2 * s_r_c2_next
        grad_lamda = grad_lamda_c1 + grad_lamda_c2

        grad_theta_c1 = grad_output_h_next_c1 * s_theta_c1_next
        grad_theta_c2 = grad_output_h_next_c2 * s_theta_c2_next
        grad_theta = grad_theta_c1 + grad_theta_c2

        # dL/dh_c1 = grad_output_h_next_c1;  dh_c1/dB_c1 = gamma * x_t 
        grad_B_c1 = torch.diag(grad_output_h_next_c1).matmul(s_B_c1_h_c1_next) + torch.diag(grad_output_h_next_c2).matmul(s_B_c1_h_c2_next)  # update for 4 sensitivity matrix gradients
        grad_B_c2 = torch.diag(grad_output_h_next_c1).matmul(s_B_c2_h_c1_next) + torch.diag(grad_output_h_next_c2).matmul(s_B_c2_h_c2_next)

        grad_input_t = torch.diag(grad_output_h_next_c1).matmul(B_c1) + torch.diag(grad_output_h_next_c2).matmul(B_c2)

        grad_h_prev = None
        grad_s_lamda_c1_prev = None
        grad_s_lamda_c2_prev = None
        grad_s_theta_c1_prev = None
        grad_s_theta_c2_prev = None
        # grad_s_B_c1_prev = None
        # grad_s_B_c2_prev = None
        grad_s_B_c1_h_c1_prev = None
        grad_s_B_c2_h_c1_prev = None
        grad_s_B_c1_h_c2_prev = None
        grad_s_B_c2_h_c2_prev = None

        return grad_input_t, grad_h_prev, grad_lamda, grad_theta, grad_B_c1, grad_B_c2, grad_s_lamda_c1_prev, grad_s_lamda_c2_prev, grad_s_theta_c1_prev, grad_s_theta_c2_prev, grad_s_B_c1_h_c1_prev, grad_s_B_c2_h_c1_prev, grad_s_B_c1_h_c2_prev, grad_s_B_c2_h_c2_prev

class RTRLRTU(nn.Module):
    """Linear Diagonal RNN Module with RTRL"""
    def __init__(self, hidden_size: int, in_vocab_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = in_vocab_size

        u1 = torch.rand(self.hidden_size)
        u2 = torch.rand(self.hidden_size)

        self.lamda = nn.Parameter(torch.log(-0.5*torch.log(u1*(1+0)*(1-0) + 0**2))) #nn.Parameter(torch.log(-0.5 * torch.log(torch.rand(hidden_size).float()))) #nn.Parameter(torch.randn(hidden_size) * 0.2) # ** r = lamda **
        self.theta = nn.Parameter(torch.log(6.28 * u2)) #nn.Parameter(torch.log(6.28 * torch.rand(hidden_size).float())) # math.log(6.28 * random.uniform(1, hidden_size)) # ADDED theta definition

        #self.B = nn.Parameter(torch.randn(hidden_size, in_vocab_size) / torch.sqrt(torch.tensor(in_vocab_size)).float())
        self.B_c1 = nn.Parameter(torch.randn(hidden_size, self.input_size) / torch.sqrt(torch.tensor(self.input_size).float()))
        self.B_c2 = nn.Parameter(torch.randn(hidden_size, self.input_size) / torch.sqrt(torch.tensor(self.input_size).float()))
        self.reset_rtrl_state()

    def reset_rtrl_state(self) -> None:
        """Resets RTRL sensitivities to zero."""
        # self.s_lamda = torch.zeros(self.hidden_size)
        self.s_lamda_c1 = torch.zeros(self.hidden_size)
        self.s_lamda_c2 = torch.zeros(self.hidden_size)

        # self.s_B = torch.zeros((self.hidden_size, self.input_size))
        # self.s_B_c1 = torch.zeros((self.hidden_size, self.input_size))
        # self.s_B_c2 = torch.zeros((self.hidden_size, self.input_size))
        self.s_B_c1_h_c1 = torch.zeros((self.hidden_size, self.input_size))
        self.s_B_c2_h_c1 = torch.zeros((self.hidden_size, self.input_size))
        self.s_B_c1_h_c2 = torch.zeros((self.hidden_size, self.input_size))
        self.s_B_c2_h_c2 = torch.zeros((self.hidden_size, self.input_size))

        self.h = torch.zeros(2 * self.hidden_size, dtype=torch.float32) # h is a concanated vector of 2 h values #, requires_grad = True)
        # self.h_c1 = torch.zeros(self.hidden_size, dtype=torch.float32) #, requires_grad = True)
        # self.h_c2 = torch.zeros(self.hidden_size, dtype=torch.float32)
        #self.h = self.h.to(device) # set the self.h to the device to work with cuda

        # ** added new theta gradients **
        self.s_theta_c1 = torch.zeros(self.hidden_size)
        self.s_theta_c2 = torch.zeros(self.hidden_size)

    def forward_step(self, input_t) -> torch.Tensor:
        # Process input from one-hot to hidden dimension
        # self.h, self.s_lamda, self.s_B = RTUFunction.apply(input_t, self.h.detach(), self.lamda, self.B, self.s_lamda.detach(), self.s_B.detach())
        #self.h, self.s_lamda_c1, self.s_lamda_c2, self.s_theta_c1, self.s_theta_c2, self.s_B_c1, self.s_B_c2 = RTUFunction.apply(input_t, self.h.detach(), self.lamda, self.theta, self.B_c1, self.B_c2, self.s_lamda_c1.detach(), self.s_lamda_c2.detach(), self.s_theta_c1.detach(), self.s_theta_c2.detach(), self.s_B_c1.detach(), self.s_B_c2.detach())
        self.h, self.s_lamda_c1, self.s_lamda_c2, self.s_theta_c1, self.s_theta_c2, self.s_B_c1_h_c1, self.s_B_c2_h_c1, self.s_B_c1_h_c2, self.s_B_c2_h_c2 = RTUFunction.apply(input_t, self.h.detach(), self.lamda, self.theta, self.B_c1, self.B_c2, self.s_lamda_c1.detach(), self.s_lamda_c2.detach(), self.s_theta_c1.detach(), self.s_theta_c2.detach(), self.s_B_c1_h_c1.detach(), self.s_B_c2_h_c1.detach(), self.s_B_c1_h_c2.detach(), self.s_B_c2_h_c2.detach())
        return self.h                                                                                                  

    def forward(self, x_sequence):
        outputs = []
        for x_t in x_sequence:
            outputs.append(self.forward_step(x_t))
        return torch.stack(outputs)

class BPTTRTU(nn.Module):
    """Linear Diagonal RNN Module with BPTT"""
    def __init__(self, hidden_size: int, input_size: int):  # input_size = in_vocab_size
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        #self.lamda = nn.Parameter(torch.log(-0.5 * torch.log(torch.rand(hidden_size).float()))) #nn.Parameter(torch.randn(hidden_size) * 0.2)
        u1 = torch.rand(self.hidden_size)
        u2 = torch.rand(self.hidden_size)

        self.lamda = nn.Parameter(torch.log(-0.5*torch.log(u1*(1+0)*(1-0) + 0**2))) #nn.Parameter(torch.log(-0.5 * torch.log(torch.rand(hidden_size).float()))) #nn.Parameter(torch.randn(hidden_size) * 0.2) # ** r = lamda **
        self.theta = nn.Parameter(torch.log(6.28 * u2)) 
        
        #self.B = nn.Parameter(torch.randn(hidden_size, input_size) / torch.sqrt(torch.tensor(input_size)).float())
        self.B_c1 = nn.Parameter(torch.randn(hidden_size, input_size) / torch.sqrt(torch.tensor(input_size).float()))
        self.B_c2 = nn.Parameter(torch.randn(hidden_size, input_size) / torch.sqrt(torch.tensor(input_size).float()))
        # RTU-only parameters


    def forward(self, x_sequence):
        #h = torch.zeros(2 * self.hidden_size) # make h concanated C1 and C2 h vectors (2 * hidden_size) #, self.hidden_size) # h = torch.zeros(self.hidden_size)
        h_c1 = torch.zeros(self.hidden_size)
        h_c2 = torch.zeros(self.hidden_size)

        # common variables in gradients
        y = torch.exp(-torch.exp(self.lamda))
        z = torch.exp(self.theta)
        gamma = torch.sqrt(1-(y**2))

        outputs = []
        for x_t in x_sequence:

            # https://pytorch.org/docs/stable/nn.html
            # mlp_x = nn.Linear(self.input_size, self.hidden_size, bias = False) # nxd  #nn.Softmax(dim=None) #nn.Dense(self.n_hidden,name='wx1',use_bias=False) 
            # W_x = mlp_x(x_t) # Ax + bias (bias = 0 here) # use W_x.transpose later since dimantions are flipped

            # y = torch.exp(-torch.exp(self.lamda))
            # gamma = torch.sqrt(1-(y**2))
            h_c1_prev = h_c1.clone()
            h_c2_prev = h_c2.clone()

            h_c1 = y * torch.cos(z) * h_c1_prev - y * torch.sin(z) * h_c2_prev + gamma * self.B_c1.mv(x_t.to('cuda', dtype=torch.float))
            h_c2 = y * torch.cos(z) * h_c2_prev + y * torch.sin(z) * h_c1_prev + gamma * self.B_c2.mv(x_t.to('cuda', dtype=torch.float))

            h = torch.cat((h_c1, h_c2), dim=0)

            outputs.append(h)
        return torch.stack(outputs)

############################################################################################

def test_gradient_correctness():
    # Setup
    input_size = 3
    hidden_size = 2048 # 2048
    seq_length = 5
    torch.manual_seed(42)

    pre_linear1 = nn.Linear(input_size, input_size, bias=False)
    pre_linear2 = nn.Linear(input_size, input_size, bias=False)
    post_linear1 = nn.Linear(2 * hidden_size, hidden_size, bias=False)
    post_linear2 = nn.Linear(2 * hidden_size, hidden_size, bias=False)

    # Create identical networks
    rtrl_rnn = RTRLRTU(hidden_size, input_size)
    bptt_rnn = BPTTRTU(hidden_size, input_size)

    # Copy parameters for identical initialization.
    bptt_rnn.lamda.data = rtrl_rnn.lamda.data.clone()
    bptt_rnn.theta.data = rtrl_rnn.theta.data.clone()
    bptt_rnn.B_c1.data = rtrl_rnn.B_c1.data.clone()
    bptt_rnn.B_c2.data = rtrl_rnn.B_c2.data.clone()
    pre_linear1.weight.data = pre_linear2.weight.data.clone()
    post_linear1.weight.data = post_linear2.weight.data.clone()

    # Generate random input sequence
    x_sequence = torch.randn(seq_length, input_size)
        
    processed1 = pre_linear1(x_sequence)
    processed2 = pre_linear2(x_sequence)

    # RTRL forward pass
    rtrl_outputs = rtrl_rnn(processed1)
    rtrl_outputs = post_linear1(rtrl_outputs)

    # BPTT forward pass
    bptt_outputs = bptt_rnn(processed2)
    bptt_outputs = post_linear2(bptt_outputs)

    # Create random gradient for backprop
    grad_output = torch.randn_like(rtrl_outputs)

    # Backward passes
    rtrl_outputs.backward(grad_output)
    bptt_outputs.backward(grad_output)

    # Check gradients
    torch.testing.assert_close(rtrl_outputs, bptt_outputs, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(post_linear1.weight.grad, post_linear2.weight.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(rtrl_rnn.theta.grad, bptt_rnn.theta.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(rtrl_rnn.lamda.grad, bptt_rnn.lamda.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(rtrl_rnn.B_c2.grad, bptt_rnn.B_c2.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(rtrl_rnn.B_c1.grad, bptt_rnn.B_c1.grad, rtol=1e-4, atol=1e-4)
    


    # Note that it's expected that any parameter below the recurrent
    # unit (e.g. pre_linear) will have different RTRL and BPTT gradients.
    assert not torch.allclose(pre_linear1.weight.grad, pre_linear2.weight.grad, rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
    test_gradient_correctness()
    print("All gradient tests passed!")



