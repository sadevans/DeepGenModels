import os
import zipfile

import gdown
from dotenv import load_dotenv

load_dotenv()


def unzip_file(zip_file_path, extract_to_folder):
    os.makedirs(extract_to_folder, exist_ok=True)

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to_folder)
        print(f'Extracted all files to {extract_to_folder}')


def main():
    PROJECT_DIR = os.getenv('PROJECT_DIR')
    HW_DIR = os.getenv('HW_DIR')
    DATA_DIR = os.path.join(PROJECT_DIR, HW_DIR, 'data')

    file_id = '1DHuQ3DBsgab6NtZIZfAKUHS2rW3-vmtb'
    url = f'https://drive.google.com/uc?id={file_id}'

    destination = f'{DATA_DIR}/dataset.zip'

    gdown.download(url, destination, quiet=False)

    unzip_file(destination, DATA_DIR)


if __name__ == '__main__':
    main()
