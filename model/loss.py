import torch
import torch.nn as nn
import torch.nn.functional as F

def cross_entropy(logits, targets):
    """Compute the cross-entropy loss between input logits and target

    Args: 
        logits (Tensor) : Predicted unnormalized batched logits (batch, seq_len, vocab_size)
        target (Tensor) : Ground truth class indices or class probabilities; (batch_size, seq_len)

    use logsumexp trick for numerical stability
    """
    max_values, _ = torch.max(logits, dim=-1, keepdim=True)
    # calculate logsumexp
    logsumexp = max_values + torch.log(torch.sum(torch.exp(logits - max_values), dim=-1, keepdim=True)) # (B, T, 1)

    # compute log probs
    logprobs = logits - logsumexp

    # gather model predictions of correct class
    pred_logporbs = torch.gather(logprobs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1) # (B, T)

    # compute loss as NLL over seq and over batch
    loss = -pred_logporbs.mean()

    return loss