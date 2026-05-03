import sys
sys.dont_write_bytecode = True

import os
import pickle
import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle as sk_shuffle

from sklearn.metrics import precision_score, recall_score

import deeplearning as dp
import classifier

# -------------------------
# CLI
# -------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--adv', default='1', choices=['1','2','3'], help='Which adversary 1,2,3')
parser.add_argument('--dataset', default='CIFAR10', choices=['CIFAR10','News'], help='Which dataset for target/shadow')
parser.add_argument('--classifierType', default='cnn', choices=['cnn','nn','softmax','resnet18', 'resnet34'], help='Classifier type for dataset1')
parser.add_argument('--dataset2', default='News', choices=['CIFAR10','News'], help='Second dataset for adversary2')
parser.add_argument('--classifierType2', default='nn', choices=['cnn','nn','softmax','resnet18', 'resnet34'], help='Classifier type for dataset2')
parser.add_argument('--dataFolderPath', default='/content/drive/MyDrive/mlleaks_data', help='Base path to save preprocessed data')
parser.add_argument('--pathToLoadData', default='/content/drive/MyDrive/datasets/cifar-10-batches-py', help='Path to CIFAR pickles')
parser.add_argument('--num_epoch', type=int, default=50, help='Epochs for training shadow/target models')
parser.add_argument('--preprocessData', action='store_true', help='Run preprocessing (otherwise load saved .npz)')
parser.add_argument('--trainTargetModel', action='store_true', help='Train target model (otherwise load saved)')
parser.add_argument('--trainShadowModel', action='store_true', help='Train shadow model (otherwise load saved)')
parser.add_argument('--top_k', type=int, default=3, help='Top-k probabilities to keep for attack features')
parser.add_argument('--attack_clf_sklearn', action='store_true', help='Train attack classifier with sklearn LogisticRegression (default True)')
parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
#### MG #####
parser.add_argument('--trainer', default='standard', choices=['standard','mg','dp'],
                    help='Training mode for target/shadow models')
parser.add_argument('--mg_kappa', type=float, default=0.0,
                    help='Multiplicative Gaussian noise strength (0 disables)')
parser.add_argument('--mg_mode', default='gauss', choices=['gauss','lognorm'],
                    help="MG noise: 'gauss' => (1 + kappa*Z), 'lognorm' => exp(kappa*Z - 0.5*kappa^2)")
parser.add_argument('--target_test_acc', type=float, default=None,
                    help='Early-stop when test accuracy >= this value (e.g., 0.35)')
parser.add_argument('--eval_every', type=int, default=1,
                    help='Evaluate test accuracy every N epochs for early-stop')

#### DP #####
parser.add_argument('--dp_noise_multiplier', type=float, default=1.0,
                    help='DP-SGD noise multiplier (sigma).')
parser.add_argument('--dp_max_grad_norm', type=float, default=1.0,
                    help='Per-sample gradient clip norm C.')
parser.add_argument('--dp_target_epsilon', type=float, default=None,
                    help='Stop early when epsilon <= this (optional).')
parser.add_argument('--dp_target_delta', type=float, default=1e-5,
                    help='Delta for (epsilon, delta)-DP accounting.')
parser.add_argument('--dp_max_epochs', type=int, default=None,
                    help='Optional hard cap on DP epochs (overrides --num_epoch if set).')
# letting shadow use its own trainer; if omitted, it mirrors target
parser.add_argument('--shadow_trainer', default=None, choices=[None,'standard','mg','dp'],
                    help='If set, overrides --trainer for the shadow model.')

#opt = parser.parse_args()
opt, _ = parser.parse_known_args()


np.random.seed(opt.random_seed)
random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)


def clip_top_k(data, top=3):
    """Keep only the top-k values of each row (descending). Returns (N,top)."""
    if top is None:
        return data
    # data: (N, C) or (N, >=1)
    res = [sorted(row, reverse=True)[:top] for row in data]
    return np.array(res, dtype=np.float32)

def read_cifar10(data_path):
    """Load original CIFAR-10 Python pickles (works in Python3). Returns X (N,3072), y (N,)."""
    X_parts = []
    y_parts = []
    for i in range(1, 6):
        p = Path(data_path) / f"data_batch_{i}"
        with open(p, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
        X_parts.append(np.array(batch['data']))
        y_parts.append(np.array(batch['labels']))
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    # test batch
    with open(Path(data_path) / 'test_batch', 'rb') as f:
        test_batch = pickle.load(f, encoding='latin1')
    Xtest = np.array(test_batch['data'])
    ytest = np.array(test_batch['labels'])
    return X, y, Xtest, ytest

def reshape_and_normalize_cifar(train_flat, test_flat):
    """From (N,3072) -> (N,3,32,32) channels-first, normalized by train mean/std."""
    def reshape(raw):
        raw = np.dstack((raw[:, :1024], raw[:, 1024:2048], raw[:, 2048:]))
        raw = raw.reshape((raw.shape[0], 32, 32, 3)).transpose(0,3,1,2)
        return raw.astype(np.float32)
    train_img = reshape(train_flat)
    test_img = reshape(test_flat)
    mean = np.mean(train_img, axis=0)
    std = np.std(train_img, axis=0).clip(min=1.0)
    train_scaled = (train_img - mean) / std
    test_scaled = (test_img - mean) / std
    return train_scaled.astype(np.float32), test_scaled.astype(np.float32)

def preprocess_news(all_texts_train, all_texts_test, max_features=None):
    """TF-IDF vectorize (train+test combined to share vocab), then normalize columns."""
    vectorizer = TfidfVectorizer(max_features=max_features)
    combined = np.concatenate([all_texts_train, all_texts_test], axis=0)
    X = vectorizer.fit_transform(combined).toarray()
    # split back
    n_train = len(all_texts_train)
    train = X[:n_train]
    test = X[n_train:]
    # normalize
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0).clip(min=1.0)
    return ((train - mean) / std).astype(np.float32), ((test - mean) / std).astype(np.float32)

def list_shuffle_split(X, y, cluster):
    """Shuffle (stable) and split into 4 groups each size cluster.
       Returns 8 arrays: toTrain, toTrainLabel, shadowTrain, shadowTrainLabel, toTest, toTestLabel, shadowTest, shadowTestLabel
    """
    arr = list(zip(X, y))
    random.shuffle(arr)
    Xs, ys = zip(*arr)
    Xs = np.array(Xs)
    ys = np.array(ys)
    # ensure enough length
    total_needed = cluster * 4
    if len(Xs) < total_needed:
        raise ValueError(f"Not enough data for cluster={cluster}; need {total_needed}, got {len(Xs)}")
    a = Xs
    b = ys
    toTrainData, toTrainLabel = a[:cluster], b[:cluster]
    shadowData, shadowLabel = a[cluster:2*cluster], b[cluster:2*cluster]
    toTestData, toTestLabel = a[2*cluster:3*cluster], b[2*cluster:3*cluster]
    shadowTestData, shadowTestLabel = a[3*cluster:4*cluster], b[3*cluster:4*cluster]
    return toTrainData, toTrainLabel, shadowData, shadowLabel, toTestData, toTestLabel, shadowTestData, shadowTestLabel

def save_npz(path, *arrays):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, *arrays)

def load_npz(path):
    with np.load(path) as f:
        return [f[f"arr_{i}"] for i in range(len(f.files))]


def initialize_data(dataset, origin_path, data_folder='./data/'):
    data_folder = Path(data_folder)
    path_out = data_folder / dataset / 'Preprocessed'
    path_out.mkdir(parents=True, exist_ok=True)

    if dataset == 'CIFAR10':
        print("Loading CIFAR-10 from", origin_path)
        X, y, Xtest, ytest = read_cifar10(origin_path)
        cluster = 10520
        # we combine both train+test from official into a single pool like original code did
        X_all = np.concatenate([X, Xtest], axis=0)
        y_all = np.concatenate([y, ytest], axis=0)
        toTrain, toTrainLabel, shadow, shadowLabel, toTest, toTestLabel, shadowTest, shadowTestLabel = list_shuffle_split(X_all, y_all, cluster)
        toTrainSave, toTestSave = reshape_and_normalize_cifar(toTrain, toTest)
        shadowSave, shadowTestSave = reshape_and_normalize_cifar(shadow, shadowTest)
    else:  # News
        from sklearn.datasets import fetch_20newsgroups
        print("Fetching 20 newsgroups")
        newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers','footers','quotes'))
        newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers','footers','quotes'))
        texts_train = newsgroups_train.data
        texts_test = newsgroups_test.data
        labels_train = newsgroups_train.target
        labels_test = newsgroups_test.target
        X_all_texts = np.concatenate([texts_train, texts_test], axis=0)
        y_all = np.concatenate([labels_train, labels_test], axis=0)
        cluster = 4500
        toTrain, toTrainLabel, shadow, shadowLabel, toTest, toTestLabel, shadowTest, shadowTestLabel = list_shuffle_split(X_all_texts, y_all, cluster)
        # vectorize using TF-IDF on the union of train/test for stable vocab
        toTrainSave, toTestSave = preprocess_news(toTrain, toTest, max_features=None)
        shadowSave, shadowTestSave = preprocess_news(shadow, shadowTest, max_features=None)


    save_npz(str(path_out / 'targetTrain.npz'), toTrainSave, toTrainLabel)
    save_npz(str(path_out / 'targetTest.npz'), toTestSave, toTestLabel)
    save_npz(str(path_out / 'shadowTrain.npz'), shadowSave, shadowLabel)
    save_npz(str(path_out / 'shadowTest.npz'), shadowTestSave, shadowTestLabel)
    print("Saved preprocessed data to:", path_out)

# -------------------------
# Train + save target/shadow models + build attack datasets
# -------------------------
def initialize_target_model(dataset, num_epoch, data_folder='./data/', model_folder='./model/', classifier_type='cnn', batch_size=100, learning_rate=1e-3, top_k=3):
    data_path = Path(data_folder) / dataset / 'Preprocessed'
    attacker_out = Path(data_folder) / dataset / 'attackerModelData'
    model_out = Path(model_folder) / dataset
    attacker_out.mkdir(parents=True, exist_ok=True)
    model_out.mkdir(parents=True, exist_ok=True)

    print(f"Training target model for {num_epoch} epochs...")
    target_train, target_train_label = load_npz(str(data_path / 'targetTrain.npz'))
    target_test, target_test_label = load_npz(str(data_path / 'targetTest.npz'))

    # # train_target_model returns attack_x, attack_y, trained_model
    # attack_x, attack_y, model = dp.train_target_model(
    #     dataset=(target_train.astype(np.float32), target_train_label.astype(np.int32),
    #              target_test.astype(np.float32), target_test_label.astype(np.int32)),
    #     epochs=num_epoch, batch_size=batch_size, learning_rate=learning_rate, n_hidden=8192, model=classifier_type
    # )
    # #n_hidden=128

    attack_x, attack_y, model = dp.train_target_model(
    dataset=(target_train.astype(np.float32), target_train_label.astype(np.int32),
             target_test.astype(np.float32),  target_test_label.astype(np.int32)),
    epochs=num_epoch,
    batch_size=batch_size,
    learning_rate=learning_rate,
    l2_ratio=1e-7,
    n_hidden=100,
    model=classifier_type,
    # --- pass MG + early-stop flags through ---
    mg_kappa=opt.mg_kappa if opt.trainer == 'mg' else 0.0,
    mg_mode=opt.mg_mode,
    target_test_acc=opt.target_test_acc,
    eval_every=opt.eval_every
    )


    # save attack data and model state_dict
    attack_x = attack_x.astype(np.float32)
    attack_y = attack_y.astype(np.int32)
    save_npz(str(attacker_out / 'targetModelData.npz'), attack_x, attack_y)
    # save PyTorch model
    try:
        torch.save(model.state_dict(), str(model_out / 'targetModel.pth'))
    except Exception as e:
        print("Warning: unable to save model state_dict:", e)

    # optionally clip top-k later in pipeline
    return attack_x, attack_y, model

def initialize_shadow_model(dataset, num_epoch, data_folder='./data/', model_folder='./model/', classifier_type='cnn', batch_size=100, learning_rate=1e-3, top_k=3):
    data_path = Path(data_folder) / dataset / 'Preprocessed'
    attacker_out = Path(data_folder) / dataset / 'attackerModelData'
    model_out = Path(model_folder) / dataset
    attacker_out.mkdir(parents=True, exist_ok=True)
    model_out.mkdir(parents=True, exist_ok=True)

    print(f"Training shadow model for {num_epoch} epochs...")
    shadow_train, shadow_train_label = load_npz(str(data_path / 'shadowTrain.npz'))
    shadow_test, shadow_test_label = load_npz(str(data_path / 'shadowTest.npz'))

    # attack_x, attack_y, model = dp.train_target_model(
    #     dataset=(shadow_train.astype(np.float32), shadow_train_label.astype(np.int32),
    #              shadow_test.astype(np.float32), shadow_test_label.astype(np.int32)),
    #     epochs=num_epoch, batch_size=batch_size, learning_rate=learning_rate, n_hidden=8192, model=classifier_type
    # )

    ######## attack option A: Make shadow mimic the target
    attack_x, attack_y, model = dp.train_target_model(
    dataset=(shadow_train.astype(np.float32), shadow_train_label.astype(np.int32),
             shadow_test.astype(np.float32),  shadow_test_label.astype(np.int32)),
    epochs=num_epoch,
    batch_size=batch_size,
    learning_rate=learning_rate,
    l2_ratio=1e-7,
    n_hidden=100,
    model=classifier_type,
    mg_kappa=opt.mg_kappa if opt.trainer == 'mg' else 0.0,
    mg_mode=opt.mg_mode,
    target_test_acc=opt.target_test_acc,
    eval_every=opt.eval_every
    )

    # ###### Attack option B: Keep shadow standard (no MG), while target uses MG
    # attack_x, attack_y, model = dp.train_target_model(
    # dataset=(shadow_train.astype(np.float32), shadow_train_label.astype(np.int32),
    #          shadow_test.astype(np.float32),  shadow_test_label.astype(np.int32)),
    # epochs=num_epoch,
    # batch_size=batch_size,
    # learning_rate=learning_rate,
    # l2_ratio=1e-7,
    # n_hidden=8192,
    # model=classifier_type,
    # mg_kappa=0.0,                 # <- force off for shadow
    # mg_mode=opt.mg_mode,
    # target_test_acc=opt.target_test_acc,
    # eval_every=opt.eval_every
    # )


    attack_x = attack_x.astype(np.float32)
    attack_y = attack_y.astype(np.int32)
    save_npz(str(attacker_out / 'shadowModelData.npz'), attack_x, attack_y)
    try:
        torch.save(model.state_dict(), str(model_out / 'shadowModel.pth'))
    except Exception as e:
        print("Warning: unable to save model state_dict:", e)

    return attack_x, attack_y, model

# -------------------------
# Load precomputed attack data (and optionally model)
# -------------------------
def load_attack_data(dataset, kind='target', data_folder='./data/'):
    data_path = Path(data_folder) / dataset / 'attackerModelData'
    arr = load_npz(str(data_path / f'{kind}ModelData.npz'))
    return arr[0].astype(np.float32), arr[1].astype(np.int32)

# -------------------------
# Attack classifier training + evaluation
# -------------------------
def train_attack_classifier_and_eval(train_X, train_y, test_X, test_y, balance=True):
    """Train sklearn LogisticRegression on train_X/train_y and evaluate on test_X/test_y."""
    # optionally balance (downsample majority)
    if balance:
        pos = np.where(train_y == 1)[0]
        neg = np.where(train_y == 0)[0]
        m = min(len(pos), len(neg))
        if m == 0:
            raise ValueError("One of classes empty in attack training data.")
        sel = np.concatenate([np.random.choice(pos, m, replace=False), np.random.choice(neg, m, replace=False)])
        train_X_bal, train_y_bal = train_X[sel], train_y[sel]
    else:
        train_X_bal, train_y_bal = train_X, train_y

    train_X_bal, train_y_bal = sk_shuffle(train_X_bal, train_y_bal, random_state=opt.random_seed)

    clf = LogisticRegression(max_iter=2000, solver='lbfgs')
    clf.fit(train_X_bal, train_y_bal)

    preds = clf.predict(test_X)
    probs = clf.predict_proba(test_X)[:, 1] if hasattr(clf, "predict_proba") else None

    acc = accuracy_score(test_y, preds)
    auc = roc_auc_score(test_y, probs) if probs is not None else None
    print("Attack classifier results — accuracy: {:.4f} AUC: {}".format(acc, auc))
    print(classification_report(test_y, preds))

    preds = clf.predict(test_X)
    probs = clf.predict_proba(test_X)[:, 1] if hasattr(clf, "predict_proba") else None

    acc = accuracy_score(test_y, preds)
    auc = roc_auc_score(test_y, probs) if probs is not None else None
    prec = precision_score(test_y, preds, average="binary", zero_division=0)
    rec  = recall_score(test_y, preds,    average="binary", zero_division=0)

    print("Attack classifier — acc: {:.4f}  AUC: {}  Prec: {:.4f}  Rec: {:.4f}".format(acc, auc, prec, rec))
    print(classification_report(test_y, preds))


    ####
    return clf, {"acc": acc, "auc": auc, "prec": prec, "rec": rec}


# ----------------------------------------------------------------
# Orchestration: build/generate attack data, run attacker
# ----------------------------------------------------------------
def generate_attack_data(dataset, classifierType, dataFolderPath, pathToLoadData, num_epoch, preprocessData, trainTargetModel, trainShadowModel, top_k=3):
    attackerModelDataPath = Path(dataFolderPath) / dataset / 'attackerModelData'
    # Preprocess (once)
    if preprocessData:
        initialize_data(dataset, pathToLoadData, data_folder=dataFolderPath)

    # target
    if trainTargetModel:
        tX, tY, tmodel = initialize_target_model(dataset, num_epoch, data_folder=dataFolderPath, classifier_type=classifierType)
    else:
        tX, tY = load_attack_data(dataset, 'target', data_folder=dataFolderPath)
        tmodel = None

    # shadow
    if trainShadowModel:
        sX, sY, smodel = initialize_shadow_model(dataset, num_epoch, data_folder=dataFolderPath, classifier_type=classifierType)
    else:
        sX, sY = load_attack_data(dataset, 'shadow', data_folder=dataFolderPath)
        smodel = None

    # clip top-k
    tX_clipped = clip_top_k(tX, top=top_k)
    sX_clipped = clip_top_k(sX, top=top_k)
    return tX_clipped, tY, sX_clipped, sY, tmodel, smodel


def attacker_one(dataset='CIFAR10', classifierType='cnn', dataFolderPath='/content/drive/MyDrive/mlleaks_data', pathToLoadData='/content/drive/MyDrive/datasets/cifar-10-batches-py', num_epoch=50, preprocessData=True, trainTargetModel=True, trainShadowModel=True, top_k=3):
    tX, tY, sX, sY, tmodel, smodel = generate_attack_data(dataset, classifierType, dataFolderPath, pathToLoadData, num_epoch, preprocessData, trainTargetModel, trainShadowModel, top_k=top_k)
    print("Training attack classifier (train on SHADOW, evaluate on TARGET).")
    # IMPORTANT: train on shadow (sX,sY), test on target (tX,tY)
    clf, acc, auc = train_attack_classifier_and_eval(sX, sY, tX, tY, balance=True)
    return clf, acc, auc

def attacker_two(dataset1='CIFAR10', dataset2='News', classifierType1='cnn', classifierType2='nn', dataFolderPath='./data/', pathToLoadData='./data/cifar-10-batches-py-official', num_epoch=50, preprocessData=True, trainTargetModel=True, trainShadowModel=True, top_k=3):
    # shadow from dataset1, target from dataset2 (or vice versa depending on experiment)
    print("Generating attack data for dataset1 (shadow) and dataset2 (target).")
    _, _, sX, sY, _, _ = generate_attack_data(dataset1, classifierType1, dataFolderPath, pathToLoadData, num_epoch, preprocessData, trainTargetModel, trainShadowModel, top_k=top_k)
    tX, tY, _, _, tmodel, _ = generate_attack_data(dataset2, classifierType2, dataFolderPath, pathToLoadData, num_epoch, preprocessData, trainTargetModel, trainShadowModel, top_k=top_k)
    print("Training attack classifier (train on SHADOW from dataset1, evaluate on TARGET from dataset2).")
    clf, acc, auc = train_attack_classifier_and_eval(sX, sY, tX, tY, balance=True)
    return clf, acc, auc

def attacker_three(dataset='CIFAR10', classifierType='cnn', dataFolderPath='./data/', pathToLoadData='./data/cifar-10-batches-py-official', num_epoch=50, preprocessData=True, trainTargetModel=True, top_k=1):
    # Compute AUC of top-1 probability as a simple detector on the target model (diagnostic)
    tX, tY, _, _, tmodel, _ = generate_attack_data(dataset, classifierType, dataFolderPath, pathToLoadData, num_epoch, preprocessData, trainTargetModel, trainShadowModel=False, top_k=top_k)
    # tX shape (N, 1) if top_k==1
    if tX.ndim == 2 and tX.shape[1] == 1:
        scores = tX.squeeze()
        auc = roc_auc_score(tY, scores)
        print(f"AUC of top-1 probability on target model (diagnostic) = {auc:.4f}")
        return auc
    else:
        raise ValueError("attacker_three expects top_k=1 to compute AUC of single score per sample.")

def attack_epochs_curve(
    epochs_list=(10,20,30,40,50,60,80,100),
    dataset="CIFAR10",
    classifierType="cnn",
    dataFolderPath="/content/drive/MyDrive/mlleaks_data",
    batch_size=100,
    learning_rate=1e-3,
    n_hidden=100,
    top_k=3,
    trainer="mg",          # "mg" or "standard"
    mg_kappa=0.0,
    mg_mode="gauss",
):
    """
    For each E in epochs_list:
      - train TARGET for E epochs
      - train SHADOW for E epochs
      - build attack data (shadow->train, target->test), train attack
      - collect precision/recall
    Returns: dict {"epochs": [...], "precision": [...], "recall": [...]}
    """
    import numpy as np
    from pathlib import Path

    # Load preprocessed splits once
    data_path = Path(dataFolderPath) / dataset / 'Preprocessed'
    target_train, target_train_label = load_npz(str(data_path / 'targetTrain.npz'))
    target_test,  target_test_label  = load_npz(str(data_path / 'targetTest.npz'))
    shadow_train, shadow_train_label = load_npz(str(data_path / 'shadowTrain.npz'))
    shadow_test,  shadow_test_label  = load_npz(str(data_path / 'shadowTest.npz'))

    xs, precisions, recalls = [], [], []

    for E in epochs_list:
        print(f"\n=== Epoch budget: {E} ===")

        # Train TARGET
        tX, tY, _tmodel = dp.train_target_model(
            dataset=(target_train.astype(np.float32), target_train_label.astype(np.int32),
                     target_test.astype(np.float32),  target_test_label.astype(np.int32)),
            epochs=E, batch_size=batch_size, learning_rate=learning_rate,
            l2_ratio=1e-7, n_hidden=n_hidden, model=classifierType,
            mg_kappa=(mg_kappa if trainer == "mg" else 0.0),
            mg_mode=mg_mode, target_test_acc=None, eval_every=1
        )

        # Train SHADOW
        sX, sY, _smodel = dp.train_target_model(
            dataset=(shadow_train.astype(np.float32), shadow_train_label.astype(np.int32),
                     shadow_test.astype(np.float32),  shadow_test_label.astype(np.int32)),
            epochs=E, batch_size=batch_size, learning_rate=learning_rate,
            l2_ratio=1e-7, n_hidden=n_hidden, model=classifierType,
            mg_kappa=(mg_kappa if trainer == "mg" else 0.0),
            mg_mode=mg_mode, target_test_acc=None, eval_every=1
        )

        # top-k clip like ML-Leaks
        sXc = clip_top_k(sX, top=top_k)
        tXc = clip_top_k(tX, top=top_k)

        # Train attack on SHADOW, test on TARGET
        _, metrics = train_attack_classifier_and_eval(sXc, sY, tXc, tY, balance=True)

        xs.append(E)
        precisions.append(metrics["prec"])
        recalls.append(metrics["rec"])

    return {"epochs": xs, "precision": precisions, "recall": recalls}


if __name__ == "__main__":
    if opt.adv == '1':
        attacker_one(dataset=opt.dataset, classifierType=opt.classifierType, dataFolderPath=opt.dataFolderPath,
                     pathToLoadData=opt.pathToLoadData, num_epoch=opt.num_epoch, preprocessData=opt.preprocessData,
                     trainTargetModel=opt.trainTargetModel, trainShadowModel=opt.trainShadowModel, top_k=opt.top_k)
    elif opt.adv == '2':
        attacker_two(dataset1=opt.dataset, dataset2=opt.dataset2,
                     classifierType1=opt.classifierType, classifierType2=opt.classifierType2,
                     dataFolderPath=opt.dataFolderPath, pathToLoadData=opt.pathToLoadData,
                     num_epoch=opt.num_epoch, preprocessData=opt.preprocessData,
                     trainTargetModel=opt.trainTargetModel, trainShadowModel=opt.trainShadowModel, top_k=opt.top_k)
    elif opt.adv == '3':
        attacker_three(dataset=opt.dataset, classifierType=opt.classifierType, dataFolderPath=opt.dataFolderPath,
                       pathToLoadData=opt.pathToLoadData, num_epoch=opt.num_epoch, preprocessData=opt.preprocessData,
                       trainTargetModel=opt.trainTargetModel, top_k=opt.top_k)
