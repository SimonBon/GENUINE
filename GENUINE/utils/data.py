import os
import numpy as np
from collections import Counter
from xml.etree import ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import cv2
import torch

CLASS_NAMES = ["NMI", "MYCN", "CLUSTER"]

class Range:
    """like builtin range, but when sliced gives a list"""
    __slots__ = "_range"
    
    def __init__(self, *args):
        self._range = range(*args) # takes no keyword arguments.
        
    def __getattr__(self, name):
        return getattr(self._range, name)
    
    def __getitem__(self, subscript):
        result = self._range.__getitem__(subscript)
        
        if isinstance(subscript, slice):
            return list(result)
        
        else:
            return result

def filelist(root, file_type):
    
    return  [Path(os.path.join(directory_path, f)) for directory_path, _, 
            files in os.walk(root) for f in files if f.endswith(file_type)]

def _match_lists(xml_list, im_list):
    
    xml_files, im_files = [], []
     
    im_samples = []
    for file in im_list:
        im_samples.append(file.stem)  
        
    for file in xml_list:
        xml_files.append(file)
        im_files.append(im_list[im_samples.index(file.stem)])
        
    return xml_files, im_files
  
  
def _bbox_from_xml(file):
    
    root = ET.parse(file).getroot()
    annotations = root.findall("./object/name")
    bboxes = root.findall("./object/bndbox")
    bbox_dict = []
    for anno, box in zip(annotations,bboxes):
        ymin = int(box.find("ymin").text)
        ymax = int(box.find("ymax").text)
        xmin = int(box.find("xmin").text)
        xmax = int(box.find("xmax").text)
        bbox_dict.append({"type": anno.text,
                            "box": [xmin, ymin, xmax, ymax]})

    return bbox_dict  


def _get_annotations_dataframe(xml_files, im_files):
    
    annotations = []
    for xml_file, png_file in zip(xml_files, im_files):
        
        boxes = _bbox_from_xml(xml_file)
        
        for i, inst in enumerate(boxes):
            anno_dict = {}
            anno_dict["filename"] = png_file.resolve()
            anno_dict["idx"] = i
            anno_dict["class"] = inst["type"]
            anno_dict["box"] = inst["box"]
            annotations.append(anno_dict)
    
    annotation_df = pd.DataFrame(annotations)
    return annotation_df

def _prepare_h5_dict(df):
    
    h5_dict = {}
    for filename in df.filename.unique():
        im = (cv2.cvtColor(cv2.imread(str(filename)), cv2.COLOR_BGR2RGB)/255).astype(float)
        tmp = df[df.filename == filename]
        
        labels, boxes = [], []
        for _, instance in tmp.iterrows():
            for n, class_name in enumerate(CLASS_NAMES):
                if instance["class"] == class_name:
                    labels.append(n+1)
                
            boxes.append(instance.box)
            
        h5_dict[str(filename.stem)] = {
            "labels": labels,
            "boxes": np.array(boxes),
            "image": np.array(im)
            }

    return h5_dict


def _create_h5_file(h5_dict, out_name, split):
    
    idxs = np.array(list(h5_dict.keys()))
    n_train = int(split[0]*len(idxs))
    np.random.shuffle(idxs)
    idxs_train = idxs[:n_train]
    idxs_val = idxs[n_train:]

    with h5py.File(out_name, "w") as fin:
    
        fin.create_group("train")
        fin.create_group("val")
    
        keys = np.array(list(h5_dict.keys()))
        np.random.shuffle(keys)
        
        for n, key in enumerate(keys):
    
            items = h5_dict[key]
            if key in idxs_train:
                group = fin.create_group(f"train/{n}")
                
            else:
                group = fin.create_group(f"val/{n}")
                
            group.create_dataset("image", data=items["image"])
            group.create_dataset("labels", data=items["labels"])
            group.create_dataset("boxes", data=items["boxes"])
    

def create_bbox_h5(root, out_path, split=[0.9, 0.1], im_file_ext = ".jpg"):
    
    xml_files = filelist(root, ".xml")
    png_files = filelist(root, im_file_ext)
    xml_files, png_files = _match_lists(xml_files, png_files)
    
    annotation_df = _get_annotations_dataframe(xml_files, png_files)
    h5_dict = _prepare_h5_dict(annotation_df)
    
    _create_h5_file(h5_dict, out_path, split)


def collate(batch):
    
    ims = []
    targets = []
    for b in batch:
        ims.append(b["image"].type(torch.FloatTensor))
        targets.append({"boxes": b["boxes"],
                        "labels": b["labels"]})
        
    return ims, targets