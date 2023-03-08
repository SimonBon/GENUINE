import torch
import logging
import numpy as np
from datetime import datetime
import os  
from pathlib import Path
from typing import Union

class TrainingScheduler():
    
    def __init__(self, model, optimizer, save_dir, patients=10, top=5):
        
        self.model = model
        self.optimizer = optimizer
        self.save_dir = save_dir
        self.val_losses = np.array([])
        self.train_losses = np.array([])
        self.accuracies = np.array([])
        self.patients = patients
        self.stop_early = False
        self._top=top
        self._saved_dict = {"models": [], "losses": []}
        
        self._patients_counter = 0
        
    def add_loss(self, epoch, train_loss, val_loss, accuracy=None):
        
        self.saved = False
        self.val_loss = val_loss

        top = np.sort(self.val_losses)[:self._top]
        if len(self.val_losses) < self._top:
            if not isinstance(self.save_dir, type(None)):
                self.save_model(epoch, train_loss, accuracy)
            self.saved = True
            self._patients_counter = -1
            
        elif any(self.val_loss < top):
            if not isinstance(self.save_dir, type(None)):
                self.save_model(epoch, train_loss, accuracy)
                self.remove_worst()
            self.saved = True
            self._patients_counter = -1
           
        elif self._patients_counter == self.patients:
            self.stop_early = True
                
        self.val_losses = np.append(self.val_losses, val_loss) 
        self.train_losses = np.append(self.train_losses, train_loss) 
        self.accuracies = np.append(self.accuracies, accuracy) 
        
        self._patients_counter += 1
        
    def save_model(self, epoch, train_loss, accuracy): 
        
        
        time = datetime.now().strftime("%Y%m%d_%H%M%S%f")[:-3]
        
        save_dict = {
            "model": self.model,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "accuracy": accuracy,
            "train_loss": train_loss,
            "validation_loss": self.val_loss,
            "epoch": epoch,
            "val_losses": self.val_losses,
            "train_losses": self.train_losses,
            "accuracies": self.accuracies,
            "model_type": self.model.__class__.__name__
            }

        if hasattr(self.model, "kwargs"):
        
            save_dict["model_kwargs"] = self.model.kwargs
           
        if not isinstance(self.save_dir, type(None)):
            torch.save(
                save_dict,
                os.path.join(self.save_dir, f"{self.model.__class__.__name__}_{time}.pt")
            )

        self._saved_dict["losses"].append(self.val_loss)
        self._saved_dict["models"].append(os.path.join(self.save_dir, f"{self.model.__class__.__name__}_{time}.pt"))
    

    def remove_worst(self):
        
        remove_loss = [x for x, val in enumerate(self._saved_dict["losses"]) if val == max(self._saved_dict["losses"])][0]
        os.remove(self._saved_dict["models"][remove_loss])
        self._saved_dict["losses"].pop(remove_loss)
        self._saved_dict["models"].pop(remove_loss)
        