import torch.nn as nn
import torch

device = 'cuda'
from rtu_utils import *

class SigmoidDiagonalRNNFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_t, h_prev, lamda, B, s_lamda_prev, s_B_prev, n_hidden):

        # put all variables to device and float for training.
        input_t = input_t.to(device, dtype=torch.float)
        h_prev = h_prev.to(device, dtype=torch.float)
        lamda = lamda.to(device, dtype=torch.float)
        B = B.to(device, dtype=torch.float)
        s_lamda_prev = s_lamda_prev.to(device, dtype=torch.float)
        s_B_prev = s_B_prev.to(device, dtype=torch.float)

        # OLD SIGMOID IMPLEMENTATION:
        # sigmoid_lamda = torch.sigmoid(lamda)
        # h_next = sigmoid_lamda * h_prev + B.mv(input_t) # W*xt
        # sigmoid_derivative = sigmoid_lamda * (1 - sigmoid_lamda)
        # s_lamda_next = sigmoid_lamda * s_lamda_prev + sigmoid_derivative * h_prev

        # s_B_next = torch.diag(sigmoid_lamda).matmul(s_B_prev) + torch.outer(torch.ones_like(input_t), input_t)
        # ctx.save_for_backward(s_lamda_next, s_B_next, B)

        # RTU IMPLEMENTATION: (pg. 4-5 of paper) 
        # extract previous h-values
        h_prev_c1,h_prev_c2 = h_prev

        # get r and theta parameters
        r_param = initialize_exp_exp_r, (1,n_hidden)
        #theta_param = initialize_theta_log, (1,n_hidden)

        # Update rows of the lambda matrix in each iteration
        lamda = r_param  * (np.cos(theta_param)) # check if r_param and theta_param are matrices

        # create layers for weight matrices
        mlp_xc1 = nn.Dense(n_hidden,name='wx1',use_bias=False) 
        #mlp_xc2 = nn.Dense(n_hidden,name='wx2',use_bias=False)

        # calulate weight matrices
        w_c1_x_t = mlp_xc1(input_t)
        #w_c2_x_t = mlp_xc2(input_t) 

        # get g, phi, and norm fo h_t_c1 and h_t_c2 initializations 
        g,phi,norm = g_phi_params(r_param,theta_param)


        h_t_c1 = np.multiply(g,h_prev_c1) - np.multiply(phi,h_prev_c2) + np.multiply(norm,w_c1_x_t) #h_t_c1 = r * cos(theta) * h_prev_c1 - r * sin(theta) * h_prev_c2 + W_x_c1 * x_t
        #h_t_c2 = np.multiply(g,h_prev_c2) + np.multiply(phi,h_prev_c1) + np.multiply(norm,w_c2_x_t)
        
        # h_next = [h_t_c1, h_t_c2]

        # calculate gradients for RTU:
        d_g_w_r, d_g_w_theta, d_phi_w_r, d_phi_w_theta, d_norm_w_r = d_g_phi_exp_exp_nu_params(r_param,theta_param,g,phi,norm) 

        new_grad_memory_hc1_w_r = d_g_w_r * h_prev_c1 + g * grad_memory[0] - d_phi_w_r * h_prev_c2 - phi * grad_memory[1] + np.multiply(d_norm_w_r,w_c1_x_t)
        new_grad_memory_hc2_w_r = d_g_w_r * h_prev_c2 + g * grad_memory[1] + d_phi_w_r * h_prev_c1 + phi * grad_memory[0] + np.multiply(d_norm_w_r,w_c2_x_t)
        
        new_grad_memory_hc1_w_theta = d_g_w_theta * h_prev_c1 + g * grad_memory[2] - d_phi_w_theta * h_prev_c2 - phi * grad_memory[3]
        new_grad_memory_hc2_w_theta = d_g_w_theta * h_prev_c2 + g * grad_memory[3] + d_phi_w_theta * h_prev_c1 + phi * grad_memory[2]
       
           
        new_grad_c1_wx1 = np.multiply(g,grad_memory[4]) - np.multiply(phi,grad_memory[5]) + np.multiply(norm,np.repeat(np.expand_dims(input_t,2),h_t_c1.shape[-1],axis=2))
        new_grad_c1_wx2 = np.multiply(g,grad_memory[6]) - np.multiply(phi,grad_memory[7]) 
        
        new_grad_c2_wx1 = np.multiply(g,grad_memory[5]) + np.multiply(phi,grad_memory[4]) 
        new_grad_c2_wx2 = np.multiply(g,grad_memory[7]) + np.multiply(phi,grad_memory[6]) + np.multiply(norm,np.repeat(np.expand_dims(input_t,2),h_t_c2.shape[-1],axis=2))
    
        new_grad_memory = (new_grad_memory_hc1_w_r,new_grad_memory_hc2_w_r,new_grad_memory_hc1_w_theta,new_grad_memory_hc2_w_theta,new_grad_c1_wx1,new_grad_c2_wx1,new_grad_c1_wx2,new_grad_c2_wx2)

        # calulations for s_lambda_next and s_B_next are maybe the same or similar *******************************
        sigmoid_lamda = torch.sigmoid(lamda)
        s_lamda_next = r_k * np.cos(theta_k) #sigmoid_lamda * s_lamda_prev + sigmoid_derivative * h_prev
        sigmoid_derivative = sigmoid_lamda * (1 - sigmoid_lamda)
        s_B_next = sigmoid_lamda.unsqueeze(1) * s_B_prev + torch.outer(torch.ones(B.shape[0]), input_ #s_B_next = torch.diag(sigmoid_lamda).matmul(s_B_prev) + torch.outer(torch.ones_like(input_t), input_t)
        #ctx.save_for_backward(s_lamda_next, s_B_next, B)

        return h_next, s_lamda_next, s_B_next

    @staticmethod
    def backward(ctx, grad_output_h_next, grad_output_s_lamda_next, grad_output_s_B_next):
        s_lamda_next, s_B_next, B = ctx.saved_tensors
        grad_lamda = grad_output_h_next * s_lamda_next
        grad_B = torch.diag(grad_output_h_next).matmul(s_B_next)
        grad_input_t = torch.diag(grad_output_h_next).matmul(B)
        grad_h_prev = None
        grad_s_lamda_prev = None
        grad_s_B_prev = None
        return grad_input_t, grad_h_prev, grad_lamda, grad_B, grad_s_lamda_prev, grad_s_B_prev

class RTRLSigmoidDiagonalRNN(nn.Module):
    """Linear Diagonal RNN Module with RTRL"""
    def __init__(self, hidden_size: int, in_vocab_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.lamda = nn.Parameter(torch.randn(hidden_size) / torch.sqrt(torch.tensor(hidden_size).float()))
        self.B = nn.Parameter(torch.randn(hidden_size, input_size) / torch.sqrt(torch.tensor(input_size).float()))
        self.reset_rtrl_state()
    def reset_rtrl_state(self) -> None:
        """Resets RTRL sensitivities to zero."""
        self.s_lamda = torch.zeros(self.hidden_size)
        self.s_B = torch.zeros((self.hidden_size, self.input_size)) #self.s_B = torch.zeros((self.hidden_size, self.hidden_size))
        self.h = torch.zeros(self.hidden_size, dtype=torch.float32) #self.h = torch.zeros(self.hidden_size, dtype=torch.float32, requires_grad = True)
        self.h = self.h.to(device) # set the self.h to the device to work with cuda

    def forward_step(self, input_t) -> torch.Tensor:
        # Process input from one-hot to hidden dimension
        self.h, self.s_lamda, self.s_B = SigmoidDiagonalRNNFunction.apply(input_t, self.h.detach(), self.lamda, self.B, self.s_lamda.detach(), self.s_B.detach()) #self.h, self.s_lamda, self.s_B = SigmoidDiagonalRNNFunction.apply(input_t, self.h.detach(), self.lamda, self.B, self.s_lamda.detach(), self.s_B.detach(), self.hidden_size) # Added self.hidden_size parameter
        return self.h

    def forward(self, x_sequence):
        outputs = []
        for x_t in x_sequence:
            outputs.append(self.forward_step(x_t))
        return torch.stack(outputs)

class BPTTSigmoidDiagonalRNN(nn.Module):
    """Linear Diagonal RNN Module with BPTT"""
    def __init__(self, hidden_size: int, input_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.lamda = nn.Parameter(torch.randn(hidden_size) / torch.sqrt(torch.tensor(hidden_size).float()))
        self.B = nn.Parameter(torch.randn(hidden_size, input_size) /
                              torch.sqrt(torch.tensor(input_size).float()))

    def forward(self, x_sequence):
        h = torch.zeros(self.hidden_size)
        outputs = []
        for x_t in x_sequence:
            h = torch.sigmoid(self.lamda) * h + self.B.mv(x_t)
            outputs.append(h)
        return torch.stack(outputs)

############################################################################################

def test_gradient_correctness():
    # Setup
    input_size = 10
    hidden_size = 8
    seq_length = 5
    torch.manual_seed(42)

    pre_linear1 = nn.Linear(input_size, input_size, bias=False)
    pre_linear2 = nn.Linear(input_size, input_size, bias=False)
    post_linear1 = nn.Linear(hidden_size, hidden_size, bias=False)
    post_linear2 = nn.Linear(hidden_size, hidden_size, bias=False)

    # Create identical networks
    rtrl_rnn = RTRLSigmoidDiagonalRNN(hidden_size, input_size)
    bptt_rnn = BPTTSigmoidDiagonalRNN(hidden_size, input_size)

    # Copy parameters for identical initialization.
    bptt_rnn.lamda.data = rtrl_rnn.lamda.data.clone()
    bptt_rnn.B.data = rtrl_rnn.B.data.clone()
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
    torch.testing.assert_close(rtrl_rnn.B.grad, bptt_rnn.B.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(rtrl_rnn.lamda.grad, bptt_rnn.lamda.grad, rtol=1e-4, atol=1e-4)

    # Note that it's expected that any parameter below the recurrent
    # unit (e.g. pre_linear) will have different RTRL and BPTT gradients.
    assert not torch.allclose(pre_linear1.weight.grad, pre_linear2.weight.grad, rtol=1e-4, atol=1e-4)

if __name__ == "__main__":
    test_gradient_correctness()
    print("All gradient tests passed!")

