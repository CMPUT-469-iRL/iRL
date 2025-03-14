import torch.nn as nn
import torch

device = 'cuda'
class DiagonalRNNFunction(torch.autograd.Function):
    """
    Implements the Diagonal RNN cell as a custom autograd function using RTRL.
    Equation:
    h_t = λ ⊙ h_{t-1} + B * x_t
    sensitivity_lamda_t = lamda * sensitivity_lamda_{t-1} + h_{t-1}
    sensitivity_B_t = diag(lamda) * sensitivity_B_{t-1} + 1 ⊗ x_{t-1}
    """
    @staticmethod
    def forward(ctx, input_t, h_prev, lamda, B, s_lamda_prev, s_B_prev):
        # ctx = ctx.to(device)
        input_t = input_t.to(device, dtype=torch.float)
        h_prev = h_prev.to(device, dtype=torch.float)
        lamda = lamda.to(device, dtype=torch.float)
        B = B.to(device, dtype=torch.float)
        s_lamda_prev = s_lamda_prev.to(device, dtype=torch.float)
        s_B_prev = s_B_prev.to(device, dtype=torch.float)


        h_next = lamda * h_prev + B.mv(input_t)
        s_lamda_next = lamda * s_lamda_prev + h_prev
        s_B_next = torch.diag(lamda).matmul(s_B_prev) + torch.outer(torch.ones_like(input_t), input_t)
        ctx.save_for_backward(s_lamda_next, s_B_next, B)
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

class RTRLDiagonalRNN(nn.Module):
    """Linear Diagonal RNN Module with RTRL"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.lamda = nn.Parameter(torch.randn(hidden_size) * 0.2)
        self.B = nn.Parameter(torch.randn(hidden_size, hidden_size) / torch.sqrt(torch.tensor(hidden_size).float()))
        self.reset_rtrl_state()

    def reset_rtrl_state(self) -> None:
        """Resets RTRL sensitivities to zero."""
        self.s_lamda = torch.zeros(self.hidden_size)
        self.s_B = torch.zeros((self.hidden_size, self.hidden_size))
        self.h = torch.zeros(self.hidden_size, dtype=torch.float32)

    def forward_step(self, input_t) -> torch.Tensor:
        # Process input from one-hot to hidden dimension
        self.h, self.s_lamda, self.s_B = DiagonalRNNFunction.apply(input_t, self.h.detach(), self.lamda, self.B, self.s_lamda.detach(), self.s_B.detach())
        return self.h

    def forward(self, x_sequence):
        outputs = []
        for x_t in x_sequence:
            outputs.append(self.forward_step(x_t))
        return torch.stack(outputs)

class BPTTDiagonalRNN(nn.Module):
    """Linear Diagonal RNN Module with BPTT"""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.lamda = nn.Parameter(torch.randn(hidden_size) * 0.2)
        self.B = nn.Parameter(torch.randn(hidden_size, hidden_size) /
                              torch.sqrt(torch.tensor(hidden_size).float()))

    def forward(self, x_sequence):
        h = torch.zeros(self.hidden_size)
        outputs = []
        for x_t in x_sequence:
            h = self.lamda * h + self.B.mv(x_t)
            outputs.append(h)
        return torch.stack(outputs)

############################################################################################

def test_gradient_correctness():
    # Setup
    input_size = 10
    hidden_size = 8
    seq_length = 5
    torch.manual_seed(42)

    pre_linear1 = nn.Linear(input_size, hidden_size, bias=False)
    pre_linear2 = nn.Linear(input_size, hidden_size, bias=False)
    post_linear1 = nn.Linear(hidden_size, hidden_size, bias=False)
    post_linear2 = nn.Linear(hidden_size, hidden_size, bias=False)

    # Create identical networks
    rtrl_rnn = RTRLDiagonalRNN(hidden_size)
    bptt_rnn = BPTTDiagonalRNN(hidden_size)

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
