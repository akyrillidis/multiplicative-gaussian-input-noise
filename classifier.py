# - iterate_minibatches works the same (shuffles once per call, yields remainder)
# - model='cnn' | 'nn' | anything-else -> softmax (logistic regression)
# - channels-first expected for CNN: (N, C, H, W)
# - inputs for NN/softmax: (N, D)
# - uses Adam, CrossEntropyLoss, and weight_decay to mimic L2
# - prints Accuracy and classification_report like the original

import sys
sys.dont_write_bytecode = True

import numpy as np
from sklearn.metrics import classification_report, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as tv_models
import torch.nn.functional as F

from sklearn.metrics import precision_score, recall_score


def iterate_minibatches(inputs, targets, batch_size, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    start_idx = None
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

    if start_idx is not None and start_idx + batch_size < len(inputs):
        excerpt = indices[start_idx + batch_size:] if shuffle else slice(start_idx + batch_size, len(inputs))
        yield inputs[excerpt], targets[excerpt]


# ---------------------------
#           Models
# ---------------------------
class CNNNet(nn.Module):
    """
    Mirrors get_cnn_model:
    conv5x5 (pad='same') -> maxpool2 -> conv5x5 (valid) -> maxpool2 -> tanh FC -> softmax(out)
    Note: For training with CrossEntropyLoss we return logits (no softmax in forward).
    """
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        # n_in is the shape tuple of train_x: (N, C, H, W)
        _, C, H, W = n_in
        # conv1: pad='same' for 5x5 -> pad=2
        self.conv1 = nn.Conv2d(C, 32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(2)

        # conv2: valid (no padding)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2)

        # compute flatten size after conv/pool stack
        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            x = self.pool1(self.relu1(self.conv1(dummy)))
            x = self.pool2(self.relu2(self.conv2(x)))
            flat_dim = x.numel()

        self.fc = nn.Linear(flat_dim, n_hidden)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.tanh(self.fc(x))
        logits = self.out(x)  # no softmax here; CrossEntropyLoss expects logits
        return logits

class CIFARResNet18(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32x32):
      - conv1: 3x3, stride=1, padding=1
      - remove initial maxpool
      - fc: num_classes
    """
    def __init__(self, n_in, n_out):
        super().__init__()
        # n_in is (N, C, H, W); we only need num_classes
        base = tv_models.resnet18(weights=None)
        # tweak stem for CIFAR
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        # replace classifier
        base.fc = nn.Linear(base.fc.in_features, n_out)
        self.net = base

    def forward(self, x):
        return self.net(x)  # logits


class CIFARResNet34(nn.Module):
    """
    ResNet-34 adapted for CIFAR-10 (32x32):
      - conv1: 3x3, stride=1, padding=1
      - remove initial maxpool
      - fc: num_classes
    """
    def __init__(self, n_in, n_out):
        super().__init__()
        base = tv_models.resnet34(weights=None)
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        base.fc = nn.Linear(base.fc.in_features, n_out)
        self.net = base

    def forward(self, x):
        return self.net(x)  # logits


# class MLPNet(nn.Module):
#     """
#     Mirrors get_nn_model: Input (N, D) -> Dense(tanh, n_hidden) -> Dense(n_out)
#     """
#     def __init__(self, n_in, n_hidden, n_out):
#         super().__init__()
#         # n_in is train_x.shape: (N, D)
#         D = n_in[1]
#         self.fc = nn.Linear(D, n_hidden)
#         self.tanh = nn.Tanh()
#         self.out = nn.Linear(n_hidden, n_out)

#     def forward(self, x):
#         x = self.tanh(self.fc(x))
#         logits = self.out(x)
#         return logits


# class SoftmaxNet(nn.Module):
#     """
#     Mirrors get_softmax_model: Input (N, D) -> Dense(n_out)
#     """
#     def __init__(self, n_in, n_out):
#         super().__init__()
#         D = n_in[1]
#         self.out = nn.Linear(D, n_out)

#     def forward(self, x):
#         logits = self.out(x)
#         return logits
class MLPNet(nn.Module):
    """
    Mirrors get_nn_model: Input (N, D) -> Dense(tanh, n_hidden) -> Dense(n_out)
    Robust: always flattens input to (N, D) inside forward().
    """
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        # n_in is train_x.shape: (N, D) after preprocessing/flattening in train_model
        D = n_in[1]
        self.fc = nn.Linear(D, n_hidden)
        self.tanh = nn.Tanh()
        self.out = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        # Defensive: if x has spatial dims (N, C, H, W) flatten to (N, D)
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        x = self.tanh(self.fc(x))
        logits = self.out(x)
        return logits


class SoftmaxNet(nn.Module):
    """
    Mirrors get_softmax_model: Input (N, D) -> Dense(n_out)
    Robust: always flattens input to (N, D) inside forward().
    """
    def __init__(self, n_in, n_out):
        super().__init__()
        D = n_in[1]
        self.out = nn.Linear(D, n_out)

    def forward(self, x):
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        logits = self.out(x)
        return logits


# ---------------------------
#          Training
# ---------------------------
@torch.no_grad()
def _predict_batches(model, inputs, targets, batch_size, device, is_cnn):
    # Returns predicted class indices (numpy)
    model.eval()
    preds = []
    if batch_size > len(targets):
        batch_size = len(targets)
    for xb_np, _ in iterate_minibatches(inputs, targets, batch_size, shuffle=False):
        xb = torch.from_numpy(xb_np).to(device=device, dtype=torch.float32)
        if is_cnn and xb.ndim == 4:
            # channels-first expected; no change
            pass
        elif not is_cnn and xb.ndim == 2:
            # fine
            pass
        else:
            raise ValueError("Input shape does not match model type (cnn vs nn/softmax).")
        logits = model(xb)
        pred = torch.argmax(logits, dim=1)
        preds.append(pred.cpu().numpy())
    return np.concatenate(preds, axis=0) if len(preds) > 0 else np.array([], dtype=np.int64)

@torch.no_grad()
def eval_accuracy(model, inputs, targets, batch_size, device, is_cnn) -> float:
    """
    Compute accuracy on (inputs, targets) without modifying training code.
    """
    if inputs is None or targets is None or len(targets) == 0:
        return float('nan')
    preds = _predict_batches(model, inputs, targets, batch_size, device, is_cnn)
    return float(accuracy_score(targets, preds))


def train_model(
    dataset,
    n_hidden=50,
    batch_size=100,
    epochs=100,
    learning_rate=0.01,
    model='cnn',
    l2_ratio=1e-7,
    *,
    mg_kappa: float = 0.0,            # 0.0 disables MG noise (default = old behavior)
    mg_mode: str = "gauss",           # "gauss" or "lognorm"
    target_test_acc= None,            # e.g., 0.35 to stop at 35% test accuracy
    eval_every: int = 1               # evaluate every N epochs for early-stop
):
    """
    dataset: tuple (train_x, train_y, test_x, test_y) with NumPy arrays
      - CNN expects train_x/test_x: (N, C, H, W) channels-first
      - NN/softmax expect train_x/test_x: (N, D)
      - train_y/test_y: integer labels (N,)
    Returns: trained PyTorch model (nn.Module)
    """
    train_x, train_y, test_x, test_y = dataset

    # Flatten for non-CNN models (so MLP/Softmax accept images too)
    is_cnn = model in {'cnn', 'cnn2', 'Droppcnn', 'Droppcnn2', 'resnet18', 'resnet34'}
    if not is_cnn:
        print('Flattening inputs...')
        if train_x.ndim > 2:
            train_x = train_x.reshape(train_x.shape[0], -1)
        if test_x is not None and getattr(test_x, "ndim", 0) > 2:
            test_x = test_x.reshape(test_x.shape[0], -1)

    n_in = train_x.shape
    n_out = len(np.unique(train_y))

    if batch_size > len(train_y):
        batch_size = len(train_y)

    print(f'Building model with {len(train_x)} training data, {n_out} classes...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Build net
    if is_cnn:
        if model =='resnet18':
            print('Using a ResNet-18 model...')
            net = CIFARResNet18(n_in=n_in, n_out=n_out)
        elif model == 'resnet34':
            print('Using a ResNet-34 model...')
            net = CIFARResNet34(n_in=n_in, n_out=n_out)
        else:
            print('Using a multilayer convolution neural network based model...')
            net = CNNNet(n_in=n_in, n_hidden=n_hidden, n_out=n_out)
    elif model == 'nn':
        print('Using a multilayer neural network based model...')
        net = MLPNet(n_in=n_in, n_hidden=n_hidden, n_out=n_out)
    else:
        print('Using a single layer softmax based model...')
        net = SoftmaxNet(n_in=n_in, n_out=n_out)

    net.to(device)

    # Loss & optimizer
    criterion = nn.CrossEntropyLoss()
    if model in {'resnet18', 'resnet34'}:
        optimizer = optim.SGD(net.net.parameters(), lr=0.1, momentum=0.9, weight_decay=l2_ratio)
    else:
        optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=l2_ratio)

    # ---- Training ----
    print('Training...')
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0

        for xb_np, yb_np in iterate_minibatches(train_x, train_y, batch_size, shuffle=True):
            xb = torch.from_numpy(xb_np).to(device=device, dtype=torch.float32)
            yb = torch.from_numpy(yb_np).to(device=device, dtype=torch.long)

            # --- Multiplicative Gaussian noise (train-time only) ---
            if mg_kappa and mg_kappa > 0.0:
                if mg_mode == "lognorm":
                    # mean-one log-normal: exp(kappa*Z - 0.5*kappa^2)
                    mask = torch.exp(torch.randn_like(xb) * mg_kappa - 0.5 * (mg_kappa ** 2))
                else:
                    # zero-mean multiplicative around 1: 1 + kappa*Z
                    mask = 1.0 + mg_kappa * torch.randn_like(xb)
                xb = xb * mask

            optimizer.zero_grad()
            logits = net(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item())

        if epoch % 10 == 0:
            print(f'Epoch {epoch+1}, train loss {round(running_loss, 3)}')

        # ---- Early-stop check (on clean test set) ----
        if target_test_acc is not None and (epoch % max(1, eval_every) == 0):
            acc = eval_accuracy(net, test_x, test_y, batch_size, device, is_cnn)
            print(f"[eval] epoch {epoch} test acc = {acc*100:.2f}%")
            if acc >= target_test_acc:
                print(f"Reached target test accuracy {target_test_acc*100:.1f}%. Stopping.")
                break

    # ---- optional final test report  ----
    if test_x is not None and test_y is not None and len(test_y) > 0:
        print('Testing...')
        pred_y = _predict_batches(net, test_x, test_y, batch_size, device, is_cnn=is_cnn)
        if len(pred_y) > 0:
            print('Testing Accuracy: {}'.format(accuracy_score(test_y, pred_y)))
            print('More detailed results:')
            print(classification_report(test_y, pred_y))
        else:
            print('No test batches were produced (check batch_size vs dataset size).')

    return net

