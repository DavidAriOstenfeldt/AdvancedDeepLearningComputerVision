import os
import gdown
import zipfile


path_to_download = 'data'
# DAVIS
# Google drive mirror: https://drive.google.com/drive/folders/1hEczGHw7qcMScbCJukZsoOW4Q9byx16A?usp=sharing
os.makedirs(f'{path_to_download}', exist_ok=True)

print('Downloading DAVIS 2017 trainval...')
gdown.download('https://drive.google.com/uc?id=1kiaxrX_4GuW6NmiVuKGSGVoKGWjOdp6d', output=f'{path_to_download}/DAVIS-2017-trainval-480p.zip', quiet=False)

print('Extracting DAVIS dataset...')

with zipfile.ZipFile(f'{path_to_download}/DAVIS-2017-trainval-480p.zip', 'r') as zip_file:
    zip_file.extractall(f'{path_to_download}')

print('Cleaning up ...')
os.remove(f'{path_to_download}/DAVIS-2017-trainval-480p.zip')
os.rename(os.path.join(path_to_download, 'DAVIS'), os.path.join(path_to_download, 'DAVIS17'))
print('Done.')