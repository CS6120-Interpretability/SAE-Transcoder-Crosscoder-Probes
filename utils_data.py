import os, glob
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
try:
    from IPython import get_ipython  # type: ignore

    ipython = get_ipython()
    assert ipython is not None
    ipython.run_line_magic("load_ext", "autoreload")
    ipython.run_line_magic("autoreload", "2")

    is_notebook = True
except:
    is_notebook = False

BASEPATH = '../SAE-Probing'
# we use this to point towards a directory where we host model activations

def resolve_model_id(model_name: str, model_id_override: Optional[str] = None) -> str:
    """
    Resolve a TransformerLens-compatible model id from a path-friendly name.

    Args:
        model_name: Path-friendly identifier used in filenames.
        model_id_override: Explicit model id (HF or TransformerLens) if provided.
    """
    if model_id_override:
        return model_id_override
    model_aliases = {
        "gemma-2-9b": "google/gemma-2-9b",
        "llama-3.1-8b": "meta-llama/Llama-3.1-8B",
        "gemma-2-2b": "google/gemma-2-2b",
    }
    return model_aliases.get(model_name, model_name)

# DATA UTILS
def get_binary_df() -> pd.DataFrame:
    """Return the subset of the master metadata CSV that corresponds to binary tasks."""
    df = pd.read_csv('data/probing_datasets_MASTER.csv')
    binary_datasets = df[df['Data type'] == 'Binary Classification']
    return binary_datasets

def get_numbered_binary_tags() -> List[str]:
    """Return dataset identifiers in ``{id}_{name}`` form for all binary tasks."""
    df = get_binary_df()
    return [name.split('/')[-1].split('.')[0] for name in df['Dataset save name']]

def read_dataset_df(dataset_tag: str) -> pd.DataFrame:
    """
    Load a dataset dataframe based on its human-readable tag.

    Args:
        dataset_tag: The tag listed in ``Dataset Tag`` column of the master CSV.
    """
    df = get_binary_df()
    dataset_save_name = df[df['Dataset Tag'] == dataset_tag]['Dataset save name'].iloc[0]
    return pd.read_csv(f'{BASEPATH}/data/{dataset_save_name}')

def read_numbered_dataset_df(numbered_dataset_tag: str) -> pd.DataFrame:
    """
    Load a dataset using its numbered identifier (``{id}_{dataset_tag}``).

    Args:
        numbered_dataset_tag: Dataset key such as ``1_hist_fig_ismale``.
    """
    dataset_tag = '_'.join(numbered_dataset_tag.split('_')[1:])
    return read_dataset_df(dataset_tag)

def get_yvals(numbered_dataset_tag: str) -> np.ndarray:
    """Return integer-encoded labels for a numbered dataset."""
    df = read_numbered_dataset_df(numbered_dataset_tag)
    le = LabelEncoder()
    yvals = le.fit_transform(df['target'].values)
    return yvals

def get_xvals(
    numbered_dataset_tag: str,
    layer: Union[int, str, None],
    model_name: str = 'gemma-2-9b',
    hook_name: Optional[Union[str, Sequence[str]]] = None,
) -> torch.Tensor:
    """
    Load cached model activations for a dataset at the requested hook.

    Args:
        numbered_dataset_tag: Dataset key such as ``1_hist_fig_ismale``.
        layer: Residual layer index or ``"embed"`` hook. Required when
            ``hook_name`` is not provided.
        model_name: Model identifier used when generating activations.
        hook_name: Optional full hook name (e.g., ``blocks.20.hook_resid_post``).
    """
    if hook_name is not None:
        if isinstance(hook_name, (list, tuple)):
            activations = []
            for name in hook_name:
                fname = f'{BASEPATH}/data/model_activations_{model_name}/{numbered_dataset_tag}_{name}.pt'
                activations.append(torch.load(fname, weights_only = False))
            return torch.cat(activations, dim=-1)
        fname = f'{BASEPATH}/data/model_activations_{model_name}/{numbered_dataset_tag}_{hook_name}.pt'
    elif layer is None:
        raise ValueError("layer is required when hook_name is not provided.")
    elif layer == 'embed':
        fname = f'{BASEPATH}/data/model_activations_{model_name}/{numbered_dataset_tag}_hook_embed.pt'
    else:
        fname = f'{BASEPATH}/data/model_activations_{model_name}/{numbered_dataset_tag}_blocks.{layer}.hook_resid_post.pt'
    activations = torch.load(fname, weights_only = False)
    return activations

def get_xyvals(
    numbered_dataset_tag: str,
    layer: Union[int, str, None],
    model_name: str,
    MAX_AMT: int = 1500,
    hook_name: Optional[Union[str, Sequence[str]]] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Load (X, y) pairs for a dataset and truncate to ``MAX_AMT`` examples."""
    xvals = get_xvals(numbered_dataset_tag, layer, model_name, hook_name=hook_name)
    yvals = get_yvals(numbered_dataset_tag)
    xvals = xvals[:MAX_AMT]
    yvals = yvals[:MAX_AMT]
    return xvals, yvals

def get_train_test_indices(
    y: np.ndarray,
    num_train: int,
    num_test: int,
    pos_ratio: float = 0.5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratify indices into train and test splits while enforcing a target
    positive ratio for both sets.

    Args:
        y: Integer labels array.
        num_train: Number of training samples to draw.
        num_test: Number of test samples to draw.
        pos_ratio: Desired fraction of positive examples in each split.
        seed: Random seed used for reproducibility.
    """
    np.random.seed(seed)
    
    pos_indices = np.where(y == 1)[0]
    neg_indices = np.where(y == 0)[0]
    
    pos_train_size = int(np.ceil(pos_ratio * num_train))
    neg_train_size = num_train - pos_train_size
    
    pos_test_size = int(np.ceil(pos_ratio * num_test))
    neg_test_size = num_test - pos_test_size
    
    train_pos = np.random.choice(pos_indices, size=pos_train_size, replace=False)
    train_neg = np.random.choice(neg_indices, size=neg_train_size, replace=False)
    
    remaining_pos = np.setdiff1d(pos_indices, train_pos)
    remaining_neg = np.setdiff1d(neg_indices, train_neg)
    
    test_pos = np.random.choice(remaining_pos, size=pos_test_size, replace=False)
    test_neg = np.random.choice(remaining_neg, size=neg_test_size, replace=False)
    
    train_indices = np.random.permutation(np.concatenate([train_pos, train_neg]))
    test_indices = np.random.permutation(np.concatenate([test_pos, test_neg]))
    
    return train_indices, test_indices

def get_xy_traintest_specify(
    num_train: int,
    numbered_dataset_tag: str,
    layer: Union[int, str, None],
    model_name: str,
    pos_ratio: float = 0.5,
    MAX_AMT: int = 5000,
    seed: int = 42,
    num_test: Optional[int] = None,
    hook_name: Optional[Union[str, Sequence[str]]] = None,
) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray]:
    """
    Return stratified train/test splits for a dataset with a configurable class
    balance and train/test sizes.

    Args:
        num_train: Number of training samples to draw.
        numbered_dataset_tag: Dataset key such as ``1_hist_fig_ismale``.
        layer: Residual layer index or ``"embed"`` hook.
        model_name: Model identifier used when generating activations.
        pos_ratio: Desired fraction of positive examples in each split.
        MAX_AMT: Maximum number of examples to load from disk.
        seed: Random seed for reproducibility.
        num_test: Optional test set size; defaults to the remaining available
            examples.
    """
    X, y = get_xyvals(numbered_dataset_tag, layer, model_name, MAX_AMT=MAX_AMT, hook_name=hook_name)
    if num_test is None:
        num_test = X.shape[0] - num_train - 1
    if num_train + min(100, num_test) > X.shape[0]:
        raise ValueError(f"Requested {num_train + 100} total samples (train={num_train}, test={100}) but only {X.shape[0]} samples available in dataset {numbered_dataset_tag}")
        
    train_indices, test_indices = get_train_test_indices(y, num_train, num_test, pos_ratio, seed)
    return X[train_indices], y[train_indices], X[test_indices], y[test_indices]

def get_xy_traintest(
    num_train: int,
    numbered_dataset_tag: str,
    layer: Union[int, str, None],
    model_name: str,
    MAX_AMT: int = 5000,
    seed: int = 42,
    hook_name: Optional[Union[str, Sequence[str]]] = None,
) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray]:
    """Convenience wrapper for ``get_xy_traintest_specify`` with balanced classes."""
    X_train, y_train, X_test, y_test  = get_xy_traintest_specify(
        num_train,
        numbered_dataset_tag,
        layer,
        model_name,
        pos_ratio=0.5,
        MAX_AMT=MAX_AMT,
        seed=seed,
        hook_name=hook_name,
    )
    return X_train, y_train, X_test, y_test 

def get_dataset_sizes() -> Dict[str, int]:
    """Return a mapping from numbered dataset tag to total number of samples."""
    dataset_tags = get_numbered_binary_tags()
    dataset_sizes = {}
    for i,dataset_tag in enumerate(dataset_tags):
        df = read_numbered_dataset_df(dataset_tag)
        num_samples = len(df)
        dataset_sizes[dataset_tag] = num_samples
    return dataset_sizes

def get_training_sizes() -> np.ndarray:
    """Log-spaced training set sizes used in the data-scarcity regime."""
    min_size, max_size, num_points = 1, 10, 20
    points = np.unique(np.round(np.logspace(min_size, max_size, num=num_points, base=2)).astype(int))
    return points

def get_class_imbalance() -> np.ndarray:
    """Set of positive-class ratios explored in the class-imbalance regime."""
    min_size, max_size, num_points = 0.05, 0.95, 19
    points = np.linspace(min_size, max_size, num=num_points)
    return points

def get_classimabalance_num_train(numbered_dataset: str, min_num_test: int = 100) -> Tuple[int, int]:
    """
    Compute the largest feasible train/test sizes for class-imbalance sweeps
    while respecting a minimum test size.
    """
    y = get_yvals(numbered_dataset)
    points = get_class_imbalance()
    min_p, max_p = min(points), max(points)
    num_pos = np.sum(y)
    num_neg = len(y) - num_pos
    max_total_neg = num_neg / (1-min_p)
    max_total_pos = num_pos / max_p 
    max_total = int(min(max_total_neg, max_total_pos))
    num_train = min(max_total - min_num_test, 1024)
    num_test = max(100,max_total - num_train -1)
    return num_train, num_test

def corrupt_ytrain(ytrain: np.ndarray, frac: float) -> np.ndarray:
    """Flip a fraction of training labels in-place to simulate annotation noise."""
    assert 0<=frac<=0.5
    np.random.seed(42)
    num_to_flip = int(len(ytrain) * frac)
    flip_indices = np.random.choice(len(ytrain), size=num_to_flip, replace=False)
    ytrain_corrupted = ytrain.copy()
    ytrain_corrupted[flip_indices] = 1 - ytrain_corrupted[flip_indices]
    return ytrain_corrupted

def get_corrupt_frac() -> np.ndarray:
    """Fraction values explored in the label corruption regime."""
    min_size, max_size, num_points = 0, 0.5, 11
    points = np.linspace(min_size, max_size, num=num_points)
    return points

def get_OOD_datasets(translation: bool = True) -> List[str]:
    """
    List datasets used in the OOD regime.

    Args:
        translation: When False, drop translation-augmented variants.
    """
    dataset_names = glob.glob("data/OOD data/*.csv")
    if translation:
        datasets = [os.path.basename(path).replace('_OOD.csv', '') for path in dataset_names]
    else:
        datasets = [os.path.basename(path).replace('_OOD.csv', '') for path in dataset_names if 'translation' not in path]
    return datasets

def get_xy_OOD(
    dataset: str,
    model_name: str = 'gemma-2-9b',
    layer: int = 20,
    hook_name: Optional[str] = None,
) -> Tuple[torch.Tensor, np.ndarray]:
    """Load OOD activations and labels for a dataset."""
    if hook_name is not None:
        fname = f'{BASEPATH}/data/model_activations_{model_name}_OOD/{dataset}_OOD_{hook_name}.pt'
    else:
        fname = f'{BASEPATH}/data/model_activations_{model_name}_OOD/{dataset}_OOD_blocks.{layer}.hook_resid_post.pt'
    X = torch.load(fname, weights_only = False)
    df = pd.read_csv(f'{BASEPATH}/data/OOD data/{dataset}_OOD.csv')
    le = LabelEncoder()
    y = le.fit_transform(df['target'].values)
    return X,y

def get_OOD_traintest(
    dataset: str,
    model_name: str = 'gemma-2-9b',
    layer: int = 20,
    hook_name: Optional[str] = None,
) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray]:
    """Return in-distribution training splits and OOD test splits for a dataset."""
    X_train, y_train, _, _ = get_xy_traintest_specify(
        num_train = 1024,
        numbered_dataset_tag = dataset,
        layer = layer,
        model_name = model_name,
        MAX_AMT = 1500,
        pos_ratio = 0.5,
        num_test = 0,
        hook_name=hook_name,
    )
    X_test, y_test = get_xy_OOD(dataset, model_name, layer, hook_name=hook_name)
    return X_train, y_train, X_test, y_test

def get_xy_glue(toget: str = 'ensemble') -> Tuple[torch.Tensor, np.ndarray]:
    """Load GLUE-CoLA activations and the requested label column."""
    X = torch.load(f'data/dataset_investigate/87_glue_cola_blocks.20.hook_resid_post.pt', weights_only = False)
    df = pd.read_csv(f'results/investigate/87_glue_cola_investigate.csv')
    le = LabelEncoder()
    y = le.fit_transform(df[toget].values)
    return X,y

def get_disagree_glue(path_beginning: str = '') -> np.ndarray:
    """Return indices where original and ensemble GLUE labels disagree."""
    df = pd.read_csv(f'{path_beginning}/results/investigate/87_glue_cola_investigate.csv')
    original = np.array(df['original_target'], dtype=int)
    ensemble = np.array(df['ensemble'], dtype=int)
    disagree_idx = np.where(original != ensemble)[0]
    return disagree_idx

def get_glue_traintest(toget: str = 'ensemble', model_name: str = 'gemma-2-9b', layer: int = 20) -> Tuple[torch.Tensor, np.ndarray, torch.Tensor, np.ndarray]:
    """Return GLUE-CoLA train/test splits using the specified label variant."""
    X_train, y_train, _, _ = get_xy_traintest_specify(num_train = 1024, numbered_dataset_tag = '87_glue_cola', layer = layer, model_name = model_name, MAX_AMT = 1500, pos_ratio = 0.5, num_test = 0)
    X_test, y_test = get_xy_glue(toget)
    return X_train, y_train, X_test, y_test

def get_datasets(model_name: str = 'llama-3.1-8b') -> List[str]:
    """
    Enumerate datasets that have cached model activations for the given model.
    Filters to binary probing datasets.
    """
    dataset_sizes = get_dataset_sizes()
    files = os.listdir(f'{BASEPATH}/data/model_activations_{model_name}')
    block_files = [f for f in files if 'blocks' in f]
    datasets = set()
    for file in block_files:
        dataset = file.split('_blocks')[0]
        if dataset in dataset_sizes.keys():
            datasets.add(dataset)
    return sorted(list(datasets))

def get_layers(model_name: str = 'gemma-2-9b') -> List[Union[str, int]]:
    """Map model names to the layers that are probed in the experiments."""
    if model_name == 'gemma-2-9b':
        layers = ['embed', 9,20,31,41]
    elif model_name == 'llama-3.1-8b':
        layers = ['embed', 8,16,24,31]
    elif model_name == 'gemma-2-2b':
        layers = [12]
    else:
        raise ValueError('model not accepted')
    return layers


def get_avg_test_size() -> np.ndarray:
    """
    Estimate the default test sizes (max of 100 or remainder after a 1024-train
    split) for each dataset.
    """
    sizes = get_dataset_sizes()
    test = []
    for dataset in sizes.keys():
        size = sizes[dataset]
        test.append(max(100, size-1024))
    test = np.array(test)
    return test
