import subprocess
import yaml

# Before: remove all videos, faces, binary_mask
downloading_cmd = 'youtube-dl -f 22 -o data/videos/{}_{}.mp4 {}'  # fold, num, url
cropping_cmd = 'python scripts/images_from_video.py {} {} data/videos/{}_{}.mp4'
# fold, num, fold, num
mask_cmd = 'python scripts/prep_binary_mask.py'

with open('urls.yaml', 'r') as f:
    urls = yaml.load(f)
#
# print(urls)
#
# # Downloading videos
# # Attention: there are some not html5 videos, which can't be downloaded. Should be manually!
# print('Downloading videos..')
# for folder in ['A', 'B']:
#     for i, url in enumerate(urls[folder]):
#         print(f'Folder {folder}, url: {url}')
#         subprocess.call(downloading_cmd.format(folder, i, url), shell=True)
#
# # Cropping face
# print('Cropping faces..')
# for folder in ['A', 'B']:
#     for i, url in enumerate(urls[folder]):
#         print(f'Folder {folder}, url: {url}')
#         subprocess.call(cropping_cmd.format(folder, i, folder, i), shell=True)

# Mask
print('Preparing binary mask..')
subprocess.call(mask_cmd, shell=True)