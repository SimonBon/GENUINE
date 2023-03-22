import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

def bbox_on_image(data, sz=(400,400), ret=False, threshold=None, title=None):
    
    if isinstance(data, dict):
    
        im = (np.array(data["image"].cpu().detach().numpy()*255).transpose(1,2,0)).astype(np.uint8).copy()
        boxes = np.array(data["boxes"].cpu().detach().numpy()).astype(int)
        labels = np.array(data["labels"].cpu().detach().numpy()).astype(int)
        scores = data["scores"].cpu().detach().numpy().astype(float)

        
    elif isinstance(data, tuple):
        
        im = (data[0]*255).astype(np.uint8).copy()
        fs = data[1].squeeze()
        boxes = fs[:, :4].astype(int).squeeze()
        labels = fs[:, 4].astype(int).squeeze()
        scores = fs[:, 5].astype(float).squeeze()
        
        
    if isinstance(threshold, float):
        idxs = (scores > threshold).squeeze()
        boxes = boxes[idxs]
        labels = labels[scores > threshold]
        
    for bbox, label in zip(boxes, labels):
    
        add_bbox(im, bbox, label)
        
    if ret:
        return im, title
            
    plt.imshow(im) 
    if not isinstance(title, type(None)):
         plt.title(title)
    plt.axis("off")
    plt.show()
        
def add_bbox(im, bbox, label):
    
    try:
        if label == 1:
            cv2.rectangle(im, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color=(255,0,0))
        elif label == 2:
            cv2.rectangle(im, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color=(0,255,0))
        elif label == 3:
            cv2.rectangle(im, (bbox[0],bbox[1]), (bbox[2],bbox[3]), color=(0,0,255))
    except:
        pass
        
        
def plot_results(results_dict, return_image=False, accuracy_ylim=[0,105], loss_ylim=None):
    
    results_df = pd.DataFrame({
        "accuracy": results_dict["accuracies"] if "accuracies" in results_dict else None,
        "val_losses": results_dict["val_losses"],
        "train_losses": results_dict["train_losses"],
    })
    
    fig, ax = plt.subplots()
    
    for val, items in results_df.items():
        
        if val == "accuracy":
            acc = ax.plot([], label="accuracy")
            ax2 = ax.twinx()
            ax2.plot(items, color=acc[0].get_color())
            ax2.set_ylim(accuracy_ylim[0], accuracy_ylim[1])
            ax2.set_ylabel("Accuracy /%")
            
        else:
            ax.plot(items)
        
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        
    ax.set_yscale("log")
    if isinstance(loss_ylim, list):
        ax.set_ylim([loss_ylim[0], loss_ylim[1]])
    ax.legend(results_df.columns, bbox_to_anchor=[0.5, 1.06], ncol=3, loc="center")
    
    if return_image:
        return fig
    else:
        plt.show()
        
def gridPlot(ims, labels=None, targets=None, sz=(10,10), vmin=0, vmax=1, save_path=None, plot=True, title=None):
    
    fig, axs = plt.subplots(sz[0], sz[1], figsize=(2*sz[1], 2*sz[0]))
    print(len(ims))
    for n, (ax, im) in enumerate(zip(axs.ravel(), ims[:sz[0]*sz[1]])):
        
        ax.imshow(im, vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])
        if isinstance(labels, (list, np.ndarray)) and isinstance(targets, (list, np.ndarray)):
            ax.set_title([labels[n], targets[n]])
        elif isinstance(labels, (list, np.ndarray)):
            ax.set_title(labels[n])
        else:
            ax.set_title(n)
      
    if isinstance(title, str):       
        fig.suptitle(title, fontsize=20)
    
    plt.tight_layout()
        
    if isinstance(save_path, str):
        plt.savefig(save_path)
        plt.close(fig)
    if plot: 
        plt.show()
        
        
def rand_col_seg(seg) -> np.ndarray:
    
    vals = np.unique(seg)
    colors = np.random.uniform(0.1, 1, (vals.max()+1, 3))
    colors[0] = [0, 0, 0]

    return colors[seg]
