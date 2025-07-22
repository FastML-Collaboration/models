import glob
import os.path
import tqdm
import numpy as np
cleanup_dir = '/ssd3/giorgian/hits-data-january-2024-yasser/trigger/'
#cleanup_dir = '/ssd3/giorgian/hits-data-january-2024-yasser/nontrigger/'
cleanup_events = os.path.join(cleanup_dir, 'events/*.npz')
empty_dir = os.path.join(cleanup_dir, 'empty/')

files = glob.glob(cleanup_events)
os.makedirs(empty_dir, exist_ok=True)
for file in tqdm.tqdm(files):
    with np.load(file) as data:
        if data['hit_cartesian'].shape[0] == 0:
            # move file to empty directory
            os.rename(file, os.path.join(empty_dir, os.path.basename(file)))
        
