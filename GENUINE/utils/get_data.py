import requests
import wget
import os
import sys

def _check_existance(file_name, path):
    if os.path.isfile(os.path.join(path, file_name)):
        print(f"Found existing {file_name} in {path}")
        return True
    else:
        return False 
    
def bar_progress(current, total, width):
    
    width = 80
    bar_sz = int((current/total)*width)
    empty_sz = width - bar_sz
    
    progress_message = f"Downloading: [{'='*bar_sz}{' '*empty_sz}] {round(current/total,2):%} [{current} / {total}] Bytes"
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()
  
def _zenodo_get(doi, sandbox=False, data_str=None, out=os.path.join(os.path.expanduser('~'), ".GENUINE")):
    
    doi_num = doi.split(".")[-1]
    
    if sandbox:
        url = "https://sandbox.zenodo.org/api/records/" + doi_num
        
    else:
        url = "https://zenodo.org/api/records/" + doi_num
    
    r = requests.get(url)
    
    files = [f["links"]["self"] for f in r.json()["files"]]
    
    if not os.path.isdir(out):
        os.makedirs(out)
        
    if isinstance(data_str, type(None)):
        
        to_download = files
        
    elif isinstance(data_str, str):
        
        to_download = [x for x in files if data_str in x]

    else:
        raise ValueError("Enter None or str as data_str")
        
    for file in to_download:
        
        file_name = file.split("/")[-1]
        if _check_existance(file_name, out):
            continue
        else:
            print(f"Downloading {file} to {out}")
            s = wget.download(file, out=out, bar=bar_progress)
            print("\n")
    

        