from MICCAI.ModelZoo import ResNet
from MICCAI.training.training import train
import torch
from torch.utils.data import DataLoader
from MICCAI.data.datasets import PatchDataset, DEFAULT_TRANSFORMS
from torchvision.transforms import ToTensor

if __name__ == "__main__":
    
    model = ResNet()

    train_transform = DEFAULT_TRANSFORMS
    val_transform = [ToTensor()]
    
    train_set = PatchDataset("/data_isilon_main/isilon_images/10_MetaSystems/MetaSystemsData/MYCN_SpikeIn/SAMPLE_TIFS/H5/DATASET.h5", transform=train_transform, dataset="train", n=1024)
    val_set = PatchDataset("/data_isilon_main/isilon_images/10_MetaSystems/MetaSystemsData/MYCN_SpikeIn/SAMPLE_TIFS/H5/DATASET.h5", transform=val_transform, dataset="val", n=256)

    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, num_workers=4)
    validation_loader = DataLoader(val_set, batch_size=512, shuffle=True, num_workers=4)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.9999)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    ret_dict = train(1000, model, train_loader, validation_loader, loss_fn, optimizer, save_dir="/home/simon_g/MICCAI/test", use_gpu=True)