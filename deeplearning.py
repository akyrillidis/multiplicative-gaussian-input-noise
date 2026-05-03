import sys
sys.dont_write_bytecode = True

import numpy as np
import torch

from classifier import train_model, iterate_minibatches  # your PyTorch classifier module

# keep same numpy seed as original for reproducibility of any numpy ops
np.random.seed(21312)
torch.manual_seed(21312)


def _probs_from_batches(model, inputs, batch_size, device):
    """
    Helper: run `model` (nn.Module) on `inputs` (numpy array) in batches (no shuffle),
    return an (N, num_classes) numpy array of softmax probabilities (float32).
    """
    model.eval()
    probs_parts = []
    if batch_size > len(inputs):
        batch_size = len(inputs)
    with torch.no_grad():
        for xb_np, _ in iterate_minibatches(inputs, np.zeros(len(inputs), dtype=np.int32), batch_size, shuffle=False):
            xb = torch.from_numpy(xb_np).to(device=device, dtype=torch.float32)
            logits = model(xb)  # model should return logits
            probs = torch.softmax(logits, dim=1)
            probs_parts.append(probs.cpu().numpy())
    if len(probs_parts) == 0:
        return np.zeros((0, 0), dtype=np.float32)
    return np.vstack(probs_parts).astype('float32')


def train_target_model(dataset,
                       epochs=100,
                       batch_size=100,
                       learning_rate=0.01,
                       l2_ratio=1e-7,
                       n_hidden=50,
                       model='nn',
                       # NEW: MG + early-stop controls (all optional)
                       mg_kappa=0.0,          # 0.0 disables multiplicative noise
                       mg_mode='gauss',       # 'gauss' or 'lognorm'
                       target_test_acc=None,  # e.g., 0.35 to stop at 35% test acc
                       eval_every=1):
    """
    Train a target model and build an attack dataset from it.

    Args:
      dataset: tuple (train_x, train_y, test_x, test_y) with numpy arrays.
      epochs, batch_size, learning_rate, l2_ratio, n_hidden, model: passed to train_model.
      mg_kappa: multiplicative Gaussian strength (0 disables).
      mg_mode: 'gauss' (1 + kappa*Z) or 'lognorm' (exp(kappa*Z - 0.5*kappa^2)).
      target_test_acc: early-stop when test accuracy >= this value.
      eval_every: evaluate test accuracy every N epochs for early-stop.

    Returns:
      attack_x: float32 (N_total, num_classes) softmax probs
      attack_y: int32 (N_total,) 1=member (train), 0=non-member (test)
      trained_model: nn.Module
    """
    train_x, train_y, test_x, test_y = dataset

    # Train the model (now forwarding MG + early-stop to classifier.train_model)
    trained_model = train_model(
        (train_x, train_y, test_x, test_y),
        n_hidden=n_hidden,
        batch_size=batch_size,
        epochs=epochs,
        learning_rate=learning_rate,
        model=model,
        l2_ratio=l2_ratio,
        mg_kappa=mg_kappa,
        mg_mode=mg_mode,
        target_test_acc=target_test_acc,
        eval_every=eval_every
    )

    try:
        device = next(trained_model.parameters()).device
    except StopIteration:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # build attack dataset:
    attack_x_parts, attack_y_parts = [], []

    # members (label 1)
    members_probs = _probs_from_batches(trained_model, train_x, batch_size, device)
    if members_probs.size > 0:
        attack_x_parts.append(members_probs)
        attack_y_parts.append(np.ones(len(members_probs), dtype=np.int32))

    # non-members (label 0)
    nonmembers_probs = _probs_from_batches(trained_model, test_x, batch_size, device)
    if nonmembers_probs.size > 0:
        attack_x_parts.append(nonmembers_probs)
        attack_y_parts.append(np.zeros(len(nonmembers_probs), dtype=np.int32))

    if len(attack_x_parts) == 0:
        attack_x = np.zeros((0, 0), dtype=np.float32)
        attack_y = np.zeros((0,), dtype=np.int32)
    else:
        attack_x = np.vstack(attack_x_parts).astype('float32')
        attack_y = np.concatenate(attack_y_parts).astype('int32')

    return attack_x, attack_y, trained_model

