from ..data.datasets import FRCNN_MYCN, DEFAULT_FRCNN_TRANSFORMS
from ..utils.data import collate
from ..ModelZoo import RetinaNet
from ..training.training import train
import torch
from torch.utils.data import DataLoader
from ..data.custom_transforms import ToTensorBoxes
from ..data.sampler import custom_sampler

if __name__ == "__main__":
        
    model = RetinaNet()

    train_transform = DEFAULT_FRCNN_TRANSFORMS
    val_transform = [ToTensorBoxes()]

    train_set = FRCNN_MYCN("/data_isilon_main/isilon_images/10_MetaSystems/MetaSystemsData/MYCN_SpikeIn/SAMPLE_TIFS/H5/BOXES.h5", dataset="train", transform=train_transform)
    val_set = FRCNN_MYCN("/data_isilon_main/isilon_images/10_MetaSystems/MetaSystemsData/MYCN_SpikeIn/SAMPLE_TIFS/H5/BOXES.h5", dataset="val", transform=val_transform)

    #train_sampler = custom_sampler(train_set)

    train_loader = DataLoader(train_set, batch_size=16, collate_fn=collate, shuffle=True)
    validation_loader = DataLoader(val_set, batch_size=16, collate_fn=collate, shuffle=True)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9999)
    loss_fn = sum

    ret_dict = train(1000, model, train_loader, validation_loader, loss_fn, optimizer, save_dir="/home/simon_g/MICCAI/results/RetinaNet", patients=50, use_gpu=True)