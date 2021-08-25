
import random
from typing import Dict, Any, List
from configparser import ConfigParser, ExtendedInterpolation

from numpy import random as np_random
import torch
from torch.utils.data import Dataset, random_split
from sklearn.metrics import confusion_matrix


def set_seed(seed):
    random.seed(seed)
    np_random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)


def to_cuda(data):
    if isinstance(data, tuple):
        return [d.cuda() for d in data]
    elif isinstance(data, torch.Tensor):
        return data.cuda()
    raise RuntimeError


def load_config(config_file: str) -> dict:
    '''
    config example:
        # This is a comment
        [section]  # section
        a = 5  # int
        b = 3.1415  # float
        s = 'abc'  # str
        lst = [3, 4, 5]  # list
    
    it will ouput:
        (dict) {'section': {'a': 5, 'b':3.1415, 's': 'abc', 'lst':[3, 4, 5]}
    '''
    config = ConfigParser(interpolation=ExtendedInterpolation())
    # fix the problem of automatic lowercase
    config.optionxform = lambda option: option  # type: ignore
    config.read(config_file)

    config_dct: Dict[str, Dict] = dict()
    for section in config.sections():
        tmp_dct: Dict[str, Any] = dict()

        for key, value in config.items(section):
            if value == '':  # allow no value
                tmp_dct[key] = None
                continue
            try:
                tmp_dct[key] = eval(value)  # It may be unsafe
            except NameError:
                print("Note the configuration file format!")

        config_dct[section] = tmp_dct
    
    return config_dct


def compute_acc(logit, y_gt):
    predicts = torch.max(logit, 1)[1]
    corrects = (predicts.view(y_gt.size()).data == y_gt.data).float().sum()
    accuracy = 100.0 * float(corrects/len(y_gt))

    return accuracy


def seq_mask_by_lens(lengths:torch.Tensor, 
                     maxlen=None, 
                     dtype=torch.bool):
    """
    giving sequence lengths, return mask of the sequence.
    example:
        input: 
        lengths = torch.tensor([4, 5, 1, 3])

        output:
        tensor([[ True,  True,  True,  True, False],
                [ True,  True,  True,  True,  True],
                [ True, False, False, False, False],
                [ True,  True,  True, False, False]])
    """
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(start=0, end=maxlen.item(), step=1)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix

    mask.type(dtype)
    return mask


def train_val_split(train_dataset: Dataset, val_ratio: float, shuffle: bool = True) -> List: # Tuple[Subset] actually
    size = len(train_dataset)  # type: ignore
    val_size = int(size * val_ratio)
    train_size = size - val_size
    if shuffle:
        return random_split(train_dataset, (train_size, val_size), None)
    else:
        return [train_dataset[:train_size], train_dataset[train_size:]]


def compute_measures(logit, y_gt):
    predicts = torch.max(logit, 1)[1]

    tmp = confusion_matrix(y_true=y_gt.cpu().numpy(), y_pred=predicts.cpu().numpy(), labels=[0, 1]).ravel()
    tn, fp, fn, tp = [float(x) for x in tmp]

    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)
    
    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)
    
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    if precision + recall  == 0:
        f1 = 0
    else:
        f1 = 2*precision*recall/(precision+recall)

    measures = {"precision":precision, "accuracy":accuracy, "recall":recall, "f1":f1}
    return measures