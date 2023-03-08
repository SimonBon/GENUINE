import torch
from tqdm import tqdm

def train_fn(self, train_loader, loss_fn, optimizer, scheduler, device):
    
    train_loss = 0
    correct = []
    with tqdm(train_loader) as train_loop:
        
        for n, (X, y) in enumerate(train_loop):
            
            ŷ = self.forward(X.float().to(device)).squeeze()
            loss = loss_fn(ŷ, y.float().to(device))
            correct.extend((ŷ.cpu() > 0)==y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if not isinstance(scheduler, type(None)):
                scheduler.step()
            
            train_loss += loss.item()
            
            train_loop.set_postfix({"train_accuracy": (torch.tensor(correct).sum()/len(correct)).item()*100,
                                    "tain_loss": train_loss/(n+1)})
            
        train_accuracy = (torch.tensor(correct).sum()/len(correct)).item()*100
        train_loss = train_loss/len(train_loader)
        ret = {"train_loss": train_loss,
            "train_accuracy": train_accuracy}
        
    return ret
    
    
def validation_fn(self, validation_loader, loss_fn, device):
        
    self.eval()
    
    val_loss = 0
    accuracy, n = 0, 0
    with torch.no_grad():

        for X, y in validation_loader:
            
            y = y.squeeze()
            n += len(y)
            ŷ = self.forward(X.float().to(device)).squeeze()
            val_loss += loss_fn(ŷ, y.float().to(device)).item()
            accuracy += sum((ŷ>0) == y.to(device))

    val_loss = val_loss/len(validation_loader)
    ret_dict = {"val_loss": val_loss,
                "accuracy": (accuracy/n).item()*100}
    
    self.train() 
    
    return ret_dict