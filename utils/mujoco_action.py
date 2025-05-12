import torch
from utils.build_datasets import NNDataset


class NNDatasetMujoco(NNDataset):
    """
    tailored mini_series for missing-frame cases,
    data stored in this class are clipped when constructing data_dict
    """
    def __init__(
            self, data_dict: dict, series_shape=torch.Size([]), data_device='cpu', search_device: str = None
    ):
        super(NNDatasetMujoco, self).__init__(
            data_dict, series_shape, missing=True, return_full=False, data_device=data_device,
            H=None  # nn_utils already set in the entire dataset
        )
        self.search_device = search_device


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from utils.lazy_datasets_misfrms import LazySeriesDatasetMisFrms

    torch.manual_seed(0)
    f = '../data/mujoco/mujoco_missing_1000.npz'

    mujoco_set = LazySeriesDatasetMisFrms(file_path=f, train=True, H=10, build_sequential_first=True)
    print(f"dataset length: ", len(mujoco_set))

    print(f"==== Take one element of the dataset ====")
    s, t, m, id = mujoco_set[0]
    print(f"Each element in the dataset: {s.shape}, {t.shape}, {m.shape}, {id}, series_length: {m.sum()}")
    # print(f"NN structure over all series({nn[0].shape}):\n{nn[0]}")

    mujoco_loader = DataLoader(mujoco_set, batch_size=2, shuffle=True)
    for sq, t, mask, id in mujoco_loader:
        print(f"minibatch shape:\n{sq.shape}, {t.shape}, {mask.shape}, {id}, series_length: {mask.sum(dim=(-1, -2))}")
        print(f"t_all: \n{t}")

        print(f"==== find NN ====")
        t_query = torch.randint(low=0, high=2000, size=(5, 1), dtype=torch.float32)
        print(f"t_query: \n{t_query}")
        nn_idx = mujoco_set.nnutil_all.find_nn_idx_from_index_subsets(t_query, id)
        print(f"nn_idx ({nn_idx.shape}):\n{nn_idx}")  # the idx can't be greater than series length
        break



