import torch
from torch.utils.data import Dataset

class DHF1K_(Dataset):
    def __init__(self, args):
        super().__init__()
        self.args = args