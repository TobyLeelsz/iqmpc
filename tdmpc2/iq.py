import pdb

import torch
import pdb


def iq_loss(agent, current_Q, current_v, next_v, z, gamma=0.99, expert_only=True, alpha=0.5, is_expert=None, v0=None, batch_size=512, grad_pen=0):

    loss_dict = {}
    y = gamma * next_v
    reward = (current_Q - y)[is_expert.bool()]   # TODO: modify for combined training with non-expert data
    # reward = (current_Q - y)

    with torch.no_grad():
        # biased dual form for kl divergence
        phi_grad = 1

    loss = -(phi_grad * reward).mean()
    loss_dict['softq_loss'] = loss.item()

    if v0 is not None:
        value_loss = (1 - gamma) * v0
    else:
        value_loss = (current_v - y)[~is_expert.bool()].mean()

    loss += value_loss
    loss_dict['value_loss'] = value_loss.item()

    reward = current_Q - y
    chi2_loss = 1/(4 * alpha) * (reward**2).mean()
    loss += chi2_loss
    loss_dict['regularize_loss'] = chi2_loss.item()
    loss += grad_pen

    loss_dict['total_loss'] = loss.item()
    return loss, reward, loss_dict
