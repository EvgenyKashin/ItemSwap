import subprocess
import yaml
from glob import glob
import shutil

# Before: remove all videos, faces, binary_mask
downloading_cmd = 'youtube-dl --recode-video mp4 -o data_faces/videos/{}_{}.mp4 {}'  # fold, num, url # -f 22
cropping_cmd = 'python scripts_faces/images_from_video.py {} {} data_faces/videos/{}_{}.mp4'
# fold, num, fold, num
mask_cmd = 'python scripts_faces/prep_binary_mask.py'

with open('urls_faces.yaml', 'r') as f:
    urls = yaml.load(f)

print(urls)

# Downloading videos
# Attention: there are some not html5 videos, which can't be downloaded. Should be manually!
print('Downloading videos..')
for folder in ['A', 'B']:
    for i, url in enumerate(urls[folder]):
        print(f'Folder {folder}, url: {url}')
        subprocess.call(downloading_cmd.format(folder, i, url), shell=True)

# Small fix of youtube-dl
for p in glob('data_faces/videos/*.mp4.mp4'):
    shutil.move(p, p.replace('.mp4.mp4', '.mp4'))

# Cropping face
print('Cropping faces..')
for folder in ['A', 'B']:
    for i, url in enumerate(urls[folder]):
        print(f'Folder {folder}, url: {url}')
        subprocess.call(cropping_cmd.format(folder, i, folder, i), shell=True)

# Mask
print('Preparing binary mask..')
subprocess.call(mask_cmd, shell=True)