import torch
from utils import dense_to_one_hot


class PairedData(object):
    def __init__(self, data_loader_A, data_loader_B):
        self.data_loader_A = data_loader_A
        self.data_loader_B = data_loader_B
        self.stop_A = False
        self.stop_B = False
        

    def __iter__(self):
        self.stop_A = False
        self.stop_B = False
        self.data_loader_A_iter = iter(self.data_loader_A)
        self.data_loader_B_iter = iter(self.data_loader_B)
        self.iter = 0
        return self

    def __next__(self):
        A, A_paths = next(self.data_loader_A_iter)
        B, B_paths = next(self.data_loader_B_iter)
        self.iter += 1

        A_paths = dense_to_one_hot(A_paths)
        A_paths = torch.from_numpy(A_paths)

        #B_paths = dense_to_one_hot(B_paths)
        #B_paths = torch.from_numpy(B_paths)
        #print(A_paths)
            

        return {'S': A, 'S_label': A_paths,
                'T': B, 'T_label': B_paths}
        


class UnalignedDataLoader():
    def initialize(self, dataset_source, dataset_target, batch_size1, batch_size2):
        data_loader_s = torch.utils.data.DataLoader(
            dataset_source,
            batch_size=batch_size1,
            shuffle=True)

        data_loader_t = torch.utils.data.DataLoader(
            dataset_target,
            batch_size=batch_size2,
            shuffle=True)
        self.dataset_s = dataset_source
        self.dataset_t = dataset_target
        self.paired_data = PairedData(data_loader_s, data_loader_t)

    def name(self):
        return 'UnalignedDataLoader'

    def load_data(self):
        return self.paired_data

    def get_len(self):
        return min(max(len(self.dataset_s), len(self.dataset_t)), float("inf"))
