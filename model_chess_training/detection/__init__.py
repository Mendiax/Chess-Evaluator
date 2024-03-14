import os
import requests
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# Function to download a file from a URL
def download_file(url, filepath):
    response = requests.get(url)
    with open(filepath, 'wb') as f:
        f.write(response.content)

# URLs and filenames for the files to download
files_to_download = [
    ('https://raw.githubusercontent.com/pytorch/vision/main/references/detection/engine.py', 'engine.py'),
    ('https://raw.githubusercontent.com/pytorch/vision/main/references/detection/utils.py', 'utils.py'),
    ('https://raw.githubusercontent.com/pytorch/vision/main/references/detection/transforms.py', 'transforms.py'),
    ('https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_utils.py', 'coco_utils.py'),
    ('https://raw.githubusercontent.com/pytorch/vision/main/references/detection/coco_eval.py', 'coco_eval.py')
]

# Directory to save the files
save_dir = os.path.dirname(__file__)

# Create the detection directory if it does not exist
detection_dir = os.path.join(save_dir, '.')
os.makedirs(detection_dir, exist_ok=True)

# Download the files
for url, filename in files_to_download:
    filepath = os.path.join(detection_dir, filename)
    # Check if the file already exists
    if not os.path.exists(filepath):
        print(f'Downloading {filename}...')
        download_file(url, filepath)
    else:
        print(f'{filename} already exists, skipping download.')

from .utils import *
from .engine import *
