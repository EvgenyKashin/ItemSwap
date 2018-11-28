import subprocess
import yaml
from glob import glob
import shutil
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--urls_path', default='urls_faces.yaml')
    parser.add_argument('--images_path', default=None)
    parser.add_argument('--data_folder_path', default='data_faces')
    parser.add_argument('--cuda_device', default='0')
    
    args = parser.parse_args()
    urls_path = args.urls_path
    images_path = args.images_path
    data_folder_path = args.data_folder_path
    cuda_device = args.cuda_device

    print(f'Urls path: {urls_path}')
    print(f'Images path: {images_path}')
    print(f'Data folder path: {data_folder_path}')

    # Before: remove all videos, faces, binary_mask
    with open(urls_path, 'r') as f:
        urls = yaml.load(f)
    print(urls)

    # Downloading videos
    print('Downloading videos..')
    for folder in ['A', 'B']:
        if urls[folder] is None:
            continue
        for i, url in enumerate(urls[folder]):
            print(f'Folder {folder}, url: {url}')
            subprocess.call(
                f'youtube-dl --recode-video mp4 -o {data_folder_path}/videos/{folder}_{i}.mp4 {url}',
                shell=True)

    # Small fix of youtube-dl
    for p in glob(f'{data_folder_path}/videos/*.mp4.mp4'):
        shutil.move(p, p.replace('.mp4.mp4', '.mp4'))

    # Cropping face
    print('Cropping faces..')
    for folder in ['A', 'B']:
        if urls[folder] is None:
            continue
        for i, url in enumerate(urls[folder]):
            print(f'Folder {folder}, url: {url}')
            subprocess.call(f'python scripts_faces/images_from_video.py {folder} {i} '
                            f'{data_folder_path}/videos/{folder}_{i}.mp4 '
                            f'--data_folder {data_folder_path} '
                            f'--cuda_device {cuda_device}', shell=True)

    # Cropping face from images
    if images_path is not None:
        subprocess.call(f'python scripts/images_from_images.py {folder} {images_path} '
                        f'--data_folder {data_folder_path} '
                        f'--cuda_device {cuda_device}', shell=True)

    # Mask
    # TODO: for images too
    print('Preparing binary mask..')
    subprocess.call(f'python scripts_faces/prep_binary_mask.py '
                    f'--data_folder {data_folder_path} '
                    f'--cuda_device {cuda_device}', shell=True)
