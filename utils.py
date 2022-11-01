import torch
import numpy as np
import torch.nn.functional as F


def reset_grad(opt_G, opt_C1, opt_C2):
    opt_G.zero_grad()
    opt_C1.zero_grad()
    opt_C2.zero_grad()


def ent(output):
    return - torch.mean(output * torch.log(output + 1e-6))


def discrepancy(out1, out2):
    return torch.mean(torch.abs(F.softmax(out1, dim=-1) - F.softmax(out2,dim=-1)))

def dense_to_one_hot(labels_dense):
    """Convert class labels from scalars to one-hot vectors."""
    labels_one_hot = np.zeros((len(labels_dense), 10))
    labels_dense = list(labels_dense)
    for i, t in enumerate(labels_dense):
        if t == 10:
            t = 0
            labels_one_hot[i, t] = 1
        else:
            labels_one_hot[i, t] = 1
    return labels_one_hot