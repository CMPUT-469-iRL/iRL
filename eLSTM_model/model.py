# Contains model implementations.

import torch
import torch.nn as nn


from eLSTM_model.layers import QuasiLSTMlayer
from eLSTM_model.rtrl_layers import RTRLQuasiLSTMlayer


class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    # return number of parameters
    def num_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_grad(self):
        # More efficient than optimizer.zero_grad() according to:
        # Szymon Migacz "PYTORCH PERFORMANCE TUNING GUIDE" at GTC-21.
        # - doesn't execute memset for every parameter
        # - memory is zeroed-out by the allocator in a more efficient way
        # - backward pass updates gradients with "=" operator (write) (unlike
        # zero_grad() which would result in "+=").
        # In PyT >= 1.7, one can do `model.zero_grad(set_to_none=True)`
        for p in self.parameters():
            p.grad = None

    def print_params(self):
        for p in self.named_parameters():
            print(p)


# Pure PyTorch LSTM model
class LSTMModel(BaseModel):
    def __init__(self, emb_dim, hidden_size, in_vocab_size, out_vocab_size,
                 dropout=0.0, num_layers=1, no_embedding=False):
        super().__init__()
        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size
        self.hidden_size = hidden_size

        self.no_embedding = no_embedding
        rnn_input_size = in_vocab_size

        if not no_embedding:
            self.embedding = nn.Embedding(
                num_embeddings=in_vocab_size, embedding_dim=emb_dim)
            rnn_input_size = emb_dim
        else:
            self.num_classes = in_vocab_size

        self.rnn_func = nn.LSTM(
            input_size=rnn_input_size, hidden_size=hidden_size,
            num_layers=num_layers)

        self.dropout = dropout
        if dropout > 0.:
            self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Linear(hidden_size, out_vocab_size)

    def forward(self, x):
        if self.no_embedding:
            out = torch.nn.functional.one_hot(x, self.num_classes).permute(1, 0, 2).float()
        else:
            out = self.embedding(x).permute(1, 0, 2)  # seq dim first

        # if self.dropout:
        #     out = self.dropout(out)
        out, _ = self.rnn_func(out)

        if self.dropout:
            out = self.dropout(out)
        logits = self.out_layer(out).permute(1, 0, 2)

        return logits


class QuasiLSTMModel(BaseModel):
    def __init__(self, emb_dim, hidden_size, in_vocab_size, out_vocab_size,
                 dropout=0.0, num_layers=1, no_embedding=False):
        super().__init__()
        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size
        self.hidden_size = hidden_size

        self.no_embedding = no_embedding
        rnn_input_size = in_vocab_size
        if not no_embedding:
            self.embedding = nn.Embedding(
                num_embeddings=in_vocab_size, embedding_dim=emb_dim)
            rnn_input_size = emb_dim
        else:
            self.num_classes = in_vocab_size

        self.rnn_func = QuasiLSTMlayer(
            input_dim=rnn_input_size, hidden_dim=hidden_size)

        self.output_gate = nn.Linear(
            rnn_input_size + hidden_size, hidden_size)

        self.dropout = dropout
        if dropout > 0.:
            self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Linear(hidden_size, out_vocab_size)

    def forward(self, x):
        if self.no_embedding:
            out = torch.nn.functional.one_hot(x, self.num_classes).permute(1, 0, 2).float()
        else:
            out = self.embedding(x).permute(1, 0, 2)  # seq dim first

        # if self.dropout:
        #     out = self.dropout(out)
        cell_out = self.rnn_func(out)

        gate_out = self.output_gate(torch.cat([out, cell_out], dim=-1))
        gate_out = torch.sigmoid(gate_out)
        gate_out = cell_out * gate_out

        if self.dropout:
            gate_out = self.dropout(gate_out)
        logits = self.out_layer(gate_out).permute(1, 0, 2)

        return logits


class RTRLQuasiLSTMModel(BaseModel):
    def __init__(self, emb_dim, hidden_size, in_vocab_size, out_vocab_size,
                 dropout=0.0, num_layers=1, no_embedding=False):
        super().__init__()
        self.in_vocab_size = in_vocab_size
        self.out_vocab_size = out_vocab_size
        self.hidden_size = hidden_size

        self.no_embedding = no_embedding
        rnn_input_size = in_vocab_size
        self.num_classes = in_vocab_size
        if not no_embedding:
            self.embedding = nn.Embedding(
                num_embeddings=in_vocab_size, embedding_dim=emb_dim)
            rnn_input_size = emb_dim

        self.rnn_func = RTRLQuasiLSTMlayer(
            input_dim=rnn_input_size, hidden_dim=hidden_size)

        self.output_gate = nn.Linear(
            rnn_input_size + hidden_size, hidden_size)

        self.dropout = dropout
        if dropout > 0.:
            self.dropout = nn.Dropout(dropout)
        self.out_layer = nn.Linear(hidden_size, out_vocab_size)

    def forward(self, x, state):
        if self.no_embedding:
            out = torch.nn.functional.one_hot(x, self.num_classes).float()
        else:
            out = self.embedding(x)  # seq dim first

        # RTRLQuasiLSTMlayer can take inputs of shape (B, dim)
        cell_out, state = self.rnn_func(out, state)
        cell_out.requires_grad_()
        cell_out.retain_grad()

        gate_out = self.output_gate(torch.cat([out, cell_out], dim=-1))
        gate_out = torch.sigmoid(gate_out)
        gate_out = cell_out * gate_out

        logits = self.out_layer(gate_out)

        return logits, cell_out, state

    def compute_gradient_rtrl(self, top_grad_, rtrl_state):
        Z_state, F_state, wz_state, wf_state, bz_state, bf_state = rtrl_state

        self.rnn_func.wm_z.grad += (top_grad_.unsqueeze(-1) * Z_state).sum(dim=0)
        self.rnn_func.wm_f.grad += (top_grad_.unsqueeze(-1) * F_state).sum(dim=0)

        self.rnn_func.wv_z.grad += (top_grad_ * wz_state).sum(dim=0)
        self.rnn_func.wv_f.grad += (top_grad_ * wf_state).sum(dim=0)

        self.rnn_func.bias_z.grad += (top_grad_ * bz_state).sum(dim=0)
        self.rnn_func.bias_f.grad += (top_grad_ * bf_state).sum(dim=0)

    def get_init_states(self, batch_size, device):
        return self.rnn_func.get_init_states(batch_size, device)

    def rtrl_reset_grad(self):
        self.rnn_func.wm_z.grad = torch.zeros_like(self.rnn_func.wm_z)
        self.rnn_func.wm_f.grad = torch.zeros_like(self.rnn_func.wm_f)

        self.rnn_func.wv_z.grad = torch.zeros_like(self.rnn_func.wv_z)
        self.rnn_func.wv_f.grad = torch.zeros_like(self.rnn_func.wv_f)

        self.rnn_func.bias_z.grad = torch.zeros_like(self.rnn_func.bias_z)
        self.rnn_func.bias_f.grad = torch.zeros_like(self.rnn_func.bias_f)

class BPTT(nn.module):
    def __init__(self, config, observation_space, action_space_shape):
        """Model setup

        Arguments:
            config {dict} -- Configuration and hyperparameters of the environment, trainer and model.
            observation_space {box} -- Properties of the agent's observation space
            action_space_shape {tuple} -- Dimensions of the action space
        """
        super().__init__()
        self.hidden_size = config["hidden_layer_size"]
        self.recurrence = config["recurrence"]
        self.observation_space_shape = observation_space.shape

        # Observation encoder
        if len(self.observation_space_shape) > 1:
            # Case: visual observation is available
            # Visual encoder made of 3 convolutional layers
            self.conv1 = nn.Conv2d(observation_space.shape[0], 32, 8, 4,)
            self.conv2 = nn.Conv2d(32, 64, 4, 2, 0)
            self.conv3 = nn.Conv2d(64, 64, 3, 1, 0)
            nn.init.orthogonal_(self.conv1.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv2.weight, np.sqrt(2))
            nn.init.orthogonal_(self.conv3.weight, np.sqrt(2))
            # Compute output size of convolutional layers
            self.conv_out_size = self.get_conv_output(observation_space.shape)
            in_features_next_layer = self.conv_out_size
        else:
            # Case: vector observation is available
            in_features_next_layer = observation_space.shape[0]

        # Recurrent layer (GRU or LSTM)
        if self.recurrence["layer_type"] == "gru":
            self.recurrent_layer = nn.GRU(in_features_next_layer, self.recurrence["hidden_state_size"], batch_first=True)
        elif self.recurrence["layer_type"] == "lstm":
            self.recurrent_layer = nn.LSTM(in_features_next_layer, self.recurrence["hidden_state_size"], batch_first=True)
        # Init recurrent layer
        for name, param in self.recurrent_layer.named_parameters():
            if "bias" in name:
                nn.init.constant_(param, 0)
            elif "weight" in name:
                nn.init.orthogonal_(param, np.sqrt(2))
        
        # Hidden layer
        self.lin_hidden = nn.Linear(self.recurrence["hidden_state_size"], self.hidden_size)
        nn.init.orthogonal_(self.lin_hidden.weight, np.sqrt(2))

        # Decouple policy from value
        # Hidden layer of the policy
        self.lin_policy = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_policy.weight, np.sqrt(2))

        # Hidden layer of the value function
        self.lin_value = nn.Linear(self.hidden_size, self.hidden_size)
        nn.init.orthogonal_(self.lin_value.weight, np.sqrt(2))

        # Outputs / Model heads
        # Policy (Multi-discrete categorical distribution)
        self.policy_branches = nn.ModuleList()
        for num_actions in action_space_shape:
            actor_branch = nn.Linear(in_features=self.hidden_size, out_features=num_actions)
            nn.init.orthogonal_(actor_branch.weight, np.sqrt(0.01))
            self.policy_branches.append(actor_branch)

        # Value function
        self.value = nn.Linear(self.hidden_size, 1)
        nn.init.orthogonal_(self.value.weight, 1)

    def forward(self, obs:torch.tensor, recurrent_cell:torch.tensor, device:torch.device, sequence_length:int=1):
        """Forward pass of the model

        Arguments:
            obs {torch.tensor} -- Batch of observations
            recurrent_cell {torch.tensor} -- Memory cell of the recurrent layer
            device {torch.device} -- Current device
            sequence_length {int} -- Length of the fed sequences. Defaults to 1.

        Returns:
            {Categorical} -- Policy: Categorical distribution
            {torch.tensor} -- Value Function: Value
            {tuple} -- Recurrent cell
        """
        # Set observation as input to the model
        h = obs
        # Forward observation encoder
        if len(self.observation_space_shape) > 1:
            batch_size = h.size()[0]
            # Propagate input through the visual encoder
            h = F.relu(self.conv1(h))
            h = F.relu(self.conv2(h))
            h = F.relu(self.conv3(h))
            # Flatten the output of the convolutional layers
            h = h.reshape((batch_size, -1))

        # Forward reccurent layer (GRU or LSTM)
        if sequence_length == 1:
            # Case: sampling training data or model optimization using sequence length == 1
            h, recurrent_cell = self.recurrent_layer(h.unsqueeze(1), recurrent_cell)
            h = h.squeeze(1) # Remove sequence length dimension
        else:
            # Case: Model optimization given a sequence length > 1
            # Reshape the to be fed data to batch_size, sequence_length, data
            h_shape = tuple(h.size())
            h = h.reshape((h_shape[0] // sequence_length), sequence_length, h_shape[1])

            # Forward recurrent layer
            h, recurrent_cell = self.recurrent_layer(h, recurrent_cell)

            # Reshape to the original tensor size
            h_shape = tuple(h.size())
            h = h.reshape(h_shape[0] * h_shape[1], h_shape[2])

        # The output of the recurrent layer is not activated as it already utilizes its own activations.

        # Feed hidden layer
        h = F.relu(self.lin_hidden(h))

        # Decouple policy from value
        # Feed hidden layer (policy)
        h_policy = F.relu(self.lin_policy(h))
        # Feed hidden layer (value function)
        h_value = F.relu(self.lin_value(h))
        # Head: Value function
        value = self.value(h_value).reshape(-1)
        # Head: Policy
        pi = [Categorical(logits=branch(h_policy)) for branch in self.policy_branches]

        return pi, value, recurrent_cell

    def get_conv_output(self, shape:tuple) -> int:
        """Computes the output size of the convolutional layers by feeding a dummy tensor.

        Arguments:
            shape {tuple} -- Input shape of the data feeding the first convolutional layer

        Returns:
            {int} -- Number of output features returned by the utilized convolutional layers
        """
        o = self.conv1(torch.zeros(1, *shape))
        o = self.conv2(o)
        o = self.conv3(o)
        return int(np.prod(o.size()))
 
    def init_recurrent_cell_states(self, num_sequences:int, device:torch.device) -> tuple:
        """Initializes the recurrent cell states (hxs, cxs) as zeros.

        Arguments:
            num_sequences {int} -- The number of sequences determines the number of the to be generated initial recurrent cell states.
            device {torch.device} -- Target device.

        Returns:
            {tuple} -- Depending on the used recurrent layer type, just hidden states (gru) or both hidden states and
                     cell states are returned using initial values.
        """
        hxs = torch.zeros((num_sequences), self.recurrence["hidden_state_size"], dtype=torch.float32, device=device).unsqueeze(0)
        cxs = None
        if self.recurrence["layer_type"] == "lstm":
            cxs = torch.zeros((num_sequences), self.recurrence["hidden_state_size"], dtype=torch.float32, device=device).unsqueeze(0)
        return hxs, cxs