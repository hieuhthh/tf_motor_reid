import zipfile
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor
import time
import os

from_dir = 'download'
des = 'unzip'

os.mkdir(des)

def fast_unzip(zip_path, out_path):
    print(zip_path)
    try:
        start = time.time()
        with ZipFile(zip_path) as handle:
            with ThreadPoolExecutor(2) as exe:
                _ = [exe.submit(handle.extract, m, out_path) for m in handle.namelist()]
    except:
        pass
    finally:
        print('Unzip', zip_path, 'Time:', time.time() - start)

zip_path = 'download/MoRe_Dataset.zip'
fast_unzip(zip_path, des)

