"""
PyTorch dataset specifications.
"""

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate

def get_data_loaders(name, batch_size, n_workers=0, **data_args):
    """This may replace the datasets function above"""
    collate_fn = default_collate
    if name == 'pixel_graph':
        from torch_geometric.data import Batch
        from . import pixel_graph
        train_dataset, valid_dataset = pixel_graph.get_datasets(**data_args)
        collate_fn = Batch.from_data_list
    else:
        raise Exception('Dataset %s unknown' % name)

    # Construct the data loaders
    loader_args = dict(batch_size=batch_size, collate_fn=collate_fn,
                       num_workers=n_workers)
    train_sampler, valid_sampler = None, None
    train_data_loader = DataLoader(train_dataset, sampler=train_sampler,
                                   shuffle=(train_sampler is None), **loader_args)
    valid_data_loader = (DataLoader(valid_dataset, sampler=valid_sampler, **loader_args)
                         if valid_dataset is not None else None)
    return train_data_loader, valid_data_loader
