# This file contains the code for training an rtu RTRL model.
import os
import time
from datetime import datetime
import argparse
import logging
import random

import torch
import torch.nn as torch_nn # changed from "nn" so that nn doesnt conflict with flax nn in RTU model
from torch.utils.data import DataLoader

from copy_task_data import CopyTaskDataset
# from eLSTM_model.model import QuasiLSTMModel, RTRLQuasiLSTMModel
from eval_utils import compute_accuracy

from rtu_model.rtus_utils import *
from rtu_model.model import *
from rtu_model.layers import *

#from jax.experimental import optimizers as jax_opt # import jax optimizers
import jaxopt

DEVICE = 'cuda'


parser = argparse.ArgumentParser(description='Learning to execute')
parser.add_argument('--data_dir', type=str,
                    default='utils/data/',
                    help='location of the data corpus')
parser.add_argument('--level', type=int, default=500,
                    choices=[50, 500],
                    help='Number of variables (3, 5 or 10)')
parser.add_argument('--work_dir', default='save_models', type=str,
                    help='where to save model ckpt.')
parser.add_argument('--full_sequence', action='store_true', 
                    help='Print training loss after each batch.')
parser.add_argument('--model_type', type=int, default=0,
                    choices=[11])
parser.add_argument('--seed', default=1, type=int, help='Seed.')
parser.add_argument('--debug', action='store_true', 
                    help='Print training loss after each batch.')
parser.add_argument('--no_embedding', action='store_true', 
                    help='no embedding layer in the LSTM/Q-LSTM.')
# model hyper-parameters:
parser.add_argument('--num_layer', default=1, type=int,
                    help='number of layers. for both LSTM and Trafo.')
parser.add_argument('--hidden_size', default=512, type=int,
                    help='hidden size. for both LSTM and Trafo.')
parser.add_argument('--emb_size', default=128, type=int,
                    help='emb size. for LSTM.')
parser.add_argument('--n_head', default=8, type=int,
                    help='Transformer number of heads.')
parser.add_argument('--ff_factor', default=4, type=int,
                    help='Transformer ff dim to hidden dim ratio.')
parser.add_argument('--remove_pos_enc', action='store_true',
                    help='Remove postional encoding in Trafo.')
parser.add_argument('--dropout', default=0.0, type=float,
                    help='dropout rate.')
# training hyper-parameters:
parser.add_argument('--batch_size', default=64, type=int,
                    help='batch size.')
parser.add_argument('--learning_rate', default=1e-4, type=float,
                    help='batch size.')
parser.add_argument('--grad_cummulate', default=1, type=int,
                    help='number of gradient accumulation steps.')
parser.add_argument('--num_epoch', default=200, type=int,
                    help='number of training epochs.')
parser.add_argument('--report_every', default=200, type=int,
                    help='Report valid acc every this steps (not used).')
parser.add_argument('--clip', default=0.0, type=float,
                    help='global norm clipping threshold.')
# for wandb
parser.add_argument('--project_name', type=str, default=None,
                    help='project name for wandb.')
parser.add_argument('--job_name', type=str, default=None,
                    help='job name for wandb.')
parser.add_argument('--use_wandb', action='store_true',
                    help='use wandb.')

args = parser.parse_args()

# Set seed
torch.manual_seed(args.seed)
random.seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# Set work directory
args.work_dir = os.path.join(args.work_dir, time.strftime('%Y%m%d-%H%M%S'))
if not os.path.exists(args.work_dir):
    os.makedirs(args.work_dir)

# logging
log_file_name = f"{args.work_dir}/log.txt"
handlers = [logging.FileHandler(log_file_name), logging.StreamHandler()]
logging.basicConfig(
    level=logging.INFO, format='%(message)s', handlers=handlers)

loginf = logging.info

loginf(f"torch version: {torch.__version__}")
# loginf(f"Last commit: {subprocess.check_output(['git', 'rev-parse', 'HEAD'])}")
loginf(f"Work dir: {args.work_dir}")

model_name = 'rtrl_elstm'

# wandb settings
if args.use_wandb:  # configure wandb.
    import wandb
    use_wandb = True

    if args.project_name is None:
        project_name = (os.uname()[1]
                        + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    else:
        project_name = args.project_name

    wandb.init(project=project_name)

    if args.job_name is None:
        # wandb.run.name = (os.uname()[1]
        #                   + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        #                   + args.work_dir)
        wandb.run.name = f"{os.uname()[1]}//{model_name}-{args.model_type}//" \
                         f"level{args.level}//seed{args.seed}/" \
                         f"L{args.num_layer}/h{args.hidden_size}/" \
                         f"e{args.emb_size}/" \
                         f"n{args.n_head}/ff{args.ff_factor}/" \
                         f"d{args.dropout}/b{args.batch_size}/" \
                         f"lr{args.learning_rate}/pos{args.remove_pos_enc}/" \
                         f"g{args.grad_cummulate}/ep{args.num_epoch}"
    else:
        wandb.run.name = f"{os.uname()[1]}//{args.job_name}"

    config = wandb.config
    config.host = os.uname()[1]  # host node name
    config.data_dir = args.data_dir
    config.seed = args.seed
    config.level = args.level
    config.work_dir = args.work_dir
    config.model_type = args.model_type
    config.hidden_size = args.hidden_size
    config.emb_size = args.emb_size
    config.n_head = args.n_head
    config.ff_factor = args.ff_factor
    config.dropout = args.dropout
    config.remove_pos_enc = args.remove_pos_enc
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.grad_cummulate = args.grad_cummulate
    config.num_epoch = args.num_epoch
    config.report_every = args.report_every
    config.no_embedding = args.no_embedding
else:
    use_wandb = False


# Set paths
data_path = args.data_dir

src_pad_idx = 2  # src_pad does not matter; as padding is aligned.
tgt_pad_idx = 2  # to be passed to the loss func.

# train_batch_size = 64
train_batch_size = args.batch_size
valid_batch_size = train_batch_size
test_batch_size = train_batch_size

train_file_src = f"{data_path}/train_{args.level}.src"
train_file_tgt = f"{data_path}/train_{args.level}.tgt"

valid_file_src = f"{data_path}/valid_{args.level}.src"
valid_file_tgt = f"{data_path}/valid_{args.level}.tgt"

test_file_src = f"{data_path}/test_{args.level}.src"
test_file_tgt = f"{data_path}/test_{args.level}.tgt"

# Construct dataset
train_data = CopyTaskDataset(src_file=train_file_src, tgt_file=train_file_tgt,
                        src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx,
                        src_vocab=None, tgt_vocab=None)

src_vocab = train_data.src_vocab
tgt_vocab = train_data.tgt_vocab

no_print_idx = 2  # Used to compute print accuracy.

valid_data = CopyTaskDataset(src_file=valid_file_src, tgt_file=valid_file_tgt,
                        src_pad_idx=src_pad_idx, tgt_pad_idx=tgt_pad_idx,
                        src_vocab=src_vocab, tgt_vocab=tgt_vocab)

# Set dataloader
train_data_loader = DataLoader(
    dataset=train_data, batch_size=train_batch_size, shuffle=True)
valid_data_loader = DataLoader(
    dataset=valid_data, batch_size=valid_batch_size, shuffle=False)

model_type = args.model_type  # 0 for LSTM, 1 for regular Trafo

assert model_type == 11 

# LSTM params:
emb_dim = args.emb_size
hidden_size = args.hidden_size
num_layers = args.num_layer
dropout = args.dropout

# Common params:
in_vocab_size = src_vocab.size()
out_vocab_size = tgt_vocab.size()

loginf(f"Input vocab size: {in_vocab_size}")
loginf(f"Output vocab size: {out_vocab_size}")

# model

loginf("Model: RTU")
#************************************************************************************************
# TODO: Replace this model with RTRL RTU definition
# model = RTRLQuasiLSTMModel(emb_dim=emb_dim, hidden_size=hidden_size,  
#                     num_layers=num_layers, in_vocab_size=in_vocab_size,
#                     out_vocab_size=out_vocab_size, dropout=dropout,
#                     no_embedding=args.no_embedding)

# define the layer the RNN is defined and weights are initialized
layer = RTULayer(hidden_size)
# define the model where gradients are calulated
model = RTUModel()
print(model)
# initial_states = model.initialize_state(train_batch_size,hidden_size,in_vocab_size)  # model.initialize_state(batch_size,d_rec,d_input) # what is d_rec??
# loginf(f"Number of trainable params: {model.num_params()}")
loginf(f"{model}")

# model = model.to(DEVICE)

# TODO: Replace this with batch RTU implementation
# eval_model = QuasiLSTMModel(emb_dim=emb_dim, hidden_size=hidden_size,
#                     num_layers=num_layers, in_vocab_size=in_vocab_size,
#                     out_vocab_size=out_vocab_size, dropout=dropout,
#                     no_embedding=args.no_embedding)
# eval_model = eval_model.to(DEVICE)
#************************************************************************************************

# Optimization settings:
num_epoch = args.num_epoch
grad_cummulate = args.grad_cummulate
loginf(f"Batch size: {train_batch_size}")
loginf(f"Gradient accumulation for {grad_cummulate} steps.")
loginf(f"Seed: {args.seed}")
learning_rate = args.learning_rate

loss_fn = torch_nn.CrossEntropyLoss(ignore_index=tgt_pad_idx) 

# TODO: change to have the correct model parameters
#************************************************************************************************
# optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate,  
#                              betas=(0.9, 0.995), eps=1e-9)
# initial_states = layer.initialize_state(train_batch_size,hidden_size,in_vocab_size)
# optimizer = torch.optim.Adam(params=initial_states[1] ,lr=learning_rate,   # TODO: find ay to turn jax array into torch tensor
#                              betas=(0.9, 0.995), eps=1e-9)

# opt_init, opt_update, get_params = jaxopt.adam(1e-3)
optimizer = jaxopt
#************************************************************************************************
clip = args.clip

loginf(f"Learning rate: {learning_rate}")
loginf(f"clip at: {clip}")

# Training
acc_loss = 0.
steps = 0
stop_acc = 100
best_val_acc = 0.0
best_epoch = 1
check_between_epochs = False
report_every = args.report_every

best_model_path = os.path.join(args.work_dir, 'best_model.pt')
lastest_model_path = os.path.join(args.work_dir, 'lastest_model.pt')

loginf(f"[{datetime.now().strftime('%Y/%m/%d %H:%M:%S')}] Start training")
start_time = time.time()
interval_start_time = time.time()

# Re-seed so that the order of data presentation
# is determined by the seed independent of the model choice.
torch.manual_seed(args.seed)
random.seed(args.seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)

# TODO: reset gradient in RTU implementation
#********************************************************************************************
# model.reset_grad() # change this function name to be the reset gradient of RTU
# model.rtrl_reset_grad() # change this function name to be the reset gradient of RTU

# initialize the states, AKA restart gradients
initial_states = layer.initialize_state(train_batch_size,hidden_size,in_vocab_size)
#********************************************************************************************

# TODO: Create new training loop for streaming RTRL
for ep in range(num_epoch):
    # loop through dataset
    for idx, batch in enumerate(train_data_loader):
        # print(batch)

        # get dataset information and transform it to be compatible with model
        src, tgt = batch
        bsz, _ = src.shape

        # reset states at the beginning of the sequence
        # TODO: get correct state initialization for RTU model
        # state = model.get_init_states(batch_size=bsz, device=src.device)

        src = src.permute(1, 0)
        tgt = tgt.permute(1, 0)

        # loop in online setting
        for src_token, tgt_token in zip(src, tgt):
            # TODO: calculate the gradients here:
            pass