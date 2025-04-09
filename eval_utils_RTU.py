# Utils
import torch
import torch.nn as nn

def compute_accuracy(hidden_size, model, data_iterator, loss_fn, no_print_idx, pad_value=-1,  # ADDED HIDDEN_SIZE
                     show_example=False, only_nbatch=-1,):
    """Compute accuracies and loss.

    :param str, split_name: for printing the accuracy with the split name.
    :param bool, show_example: if True, print some decoding output examples.
    :param int, only_nbatch: Only use given number of batches. If -1, use all
      data (default).
    returns loss, accucary char-level accuracy, print accuracy
    """
    model.eval()

    total_loss = 0.0
    corr = 0
    corr_char = 0
    corr_print = 0

    step = 0
    total_num_seqs = 0
    total_char = 0
    total_print = 0

    layer = nn.Linear(hidden_size, 3)  # in_vocab_size = 3

    for idx, batch in enumerate(data_iterator):
        src, tgt = batch
        bsz, _ = src.shape
        model.reset_rtrl_state() #state = model.get_init_states(batch_size=bsz, device=src.device) # ADDED
        # src = src.permute(1, 0)
        # tgt = tgt.permute(1, 0)
        for sample in range(bsz): # loop through the batch to get each sample
            for src_token, tgt_token in zip(src[sample], tgt[sample]): # ADDED
                step += 1

                # logits, cell_out, state = model(src_token, state)
                #logits = model(src, state) # logits = model(src) # src, state
                labels = tgt_token.view(-1)  #.view(-1) # tgt  # (B, len)
                target = labels

                src_token = torch.nn.functional.one_hot(src_token.view(-1), 3) # in_vocab_size = 3
                src_token = src_token.squeeze(0)

                h_next = model.forward_step(src_token)
                output = layer(h_next) #.to(DEVICE)

                # to compute accuracy
                output = torch.argmax(output, dim=-1).squeeze()

                # compute loss
                #output = output.contiguous().view(-1, output.shape[-1])
                #output = tgt_token.flatten() # TODO: .flatten() ?????????????? #tgt.view(-1)
                loss = loss_fn(output, labels)
                total_loss += loss

                # sequence level accuracy
                # seq_match = (torch.eq(target, output) | (target == pad_value)  #           seq_match = (torch.eq(target, output) | (target == pad_value)).all(1).sum().item()
                #             ).all(1).sum().item()
                seq_match = (torch.eq(target, output)| (target == pad_value)).sum().item()
                corr += seq_match
                total_num_seqs += src_token.size()[0]  # src.size()[0] 

                # padded part should not be counted as correct
                char_match = torch.logical_and(
                    torch.logical_and(torch.eq(target, output), target != pad_value),
                    target == no_print_idx).sum().item()
                corr_char += char_match
                total_char += torch.logical_and(
                    target != pad_value, target == no_print_idx).sum().item()

                # Ignore non-print outputs
                print_match = torch.logical_and(
                    torch.logical_and(torch.eq(target, output), target != pad_value),
                    target != no_print_idx).sum().item()
                corr_print += print_match

                total_print += torch.logical_and(
                    target != pad_value, target != no_print_idx).sum().item()

                if only_nbatch > 0:
                    if idx > only_nbatch:
                        break

    res_loss = total_loss.item() / float(step)
    acc = corr / float(total_num_seqs) * 100
    if total_char > 0:
      no_op_acc = corr_char / float(total_char) * 100
    else:
      no_op_acc = -0
    print_acc = corr_print / float(total_print) * 100

    return res_loss, acc, no_op_acc, print_acc
