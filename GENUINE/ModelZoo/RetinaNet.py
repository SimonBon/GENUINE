from torch import nn
from torchvision.models.detection import retinanet_resnet50_fpn, RetinaNet_ResNet50_FPN_Weights
import torch 
import math

class RetinaNet(nn.Module):
    
    def __init__(self, weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT):
        super().__init__()

        self.num_classes = 4
        self.model = retinanet_resnet50_fpn(weights=weights)
                
        num_anchors = self.model.head.classification_head.num_anchors
        self.model.head.classification_head.num_classes = self.num_classes

        out_channels = 256

        cls_logits = torch.nn.Conv2d(out_channels, num_anchors * self.num_classes, kernel_size = 3, stride=1, padding=1)
        torch.nn.init.normal_(cls_logits.weight, std=0.01)
        torch.nn.init.constant_(cls_logits.bias, -math.log((1 - 0.01) / 0.01))

        self.model.head.classification_head.cls_logits = cls_logits

        self.model = self.model.float()
        
    def forward(self, X, y=None, debug=False):
        
        if self.model.training:
            return self.model(X, y)
        else:
            return self.model(X)
        

    def train_fn(self, train_loader, loss_fn, optimizer, scheduler, device):
        
        train_loss = 0
        for X, y in train_loader:
            
            X = [x.to(device) for x in X]
            y = [{k: v.to(device) for k,v in sub_y.items()} for sub_y in y]
            
            ŷ = self.forward(X, y)
            loss = loss_fn(val for val in ŷ.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not isinstance(scheduler, type(None)):
                scheduler.step()
            
            train_loss += loss.item()
            del ŷ, X, y, loss
            
        train_loss = train_loss/len(train_loader)
        ret = {"train_loss": train_loss}
            
        return ret


    def validation_fn(self, validation_loader, loss_fn, device):
        
        val_loss = 0
        with torch.no_grad():

            for X, y in validation_loader:
                
                X = [im.to(device) for im in X]
                y = [{k: v.to(device) for k,v in sub_y.items()} for sub_y in y]

                out = self.forward(X, y)
                
                val_loss += loss_fn(loss for loss in out.values()).item()
                del out, X, y

        val_loss = val_loss/len(validation_loader)
        ret_dict = {"val_loss": val_loss}
        return ret_dict