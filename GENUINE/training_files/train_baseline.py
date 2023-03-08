from MICCAI.ModelZoo import Baseline
from MICCAI.training.training import train
import torch
from torch.utils.data import DataLoader
from MICCAI.data.datasets import PatchDataset
from MICCAI.data.custom_transforms import ToTensor, RandomNoise

if __name__ == "__main__":
    
    model = Baseline()

    train_transform = [ToTensor()]
    val_transform = [ToTensor()]
    
    train_set = PatchDataset("/data_isilon_main/isilon_images/10_MetaSystems/MetaSystemsData/MYCN_SpikeIn/SAMPLE_TIFS/H5/DATASET.h5", transform=train_transform, dataset="train")
    val_set = PatchDataset("/data_isilon_main/isilon_images/10_MetaSystems/MetaSystemsData/MYCN_SpikeIn/SAMPLE_TIFS/H5/DATASET.h5", transform=val_transform, dataset="val")

    train_loader = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=4)
    validation_loader = DataLoader(val_set, batch_size=512, shuffle=True, num_workers=4)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9999)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    ret_dict = train(50, model, train_loader, validation_loader, loss_fn, optimizer, patients=10, save_dir="/home/simon_g/MICCAI/results/Baseline", use_gpu=True)