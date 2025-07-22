import ujson
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from statistics import mode
import tqdm
from joblib import Parallel, delayed
import networkx as nx
import os

LAYER_ID_TO_LAYER = np.array([
        0,
        1,
        2,
        3,
        3,
        4,
        4
])

def process(filename, file_number, output_dir):
    trigger = 0
    with np.load(filename) as f:
        tracks = []
        particle_ids = []
        momentums = []
        track_origins = []
        trigger_nodes = []
        track_n_hits = []
        for pid in np.unique(f['particle_id']):
            hits = f['hit_cartesian'][f['particle_id'] == pid]
            layers = LAYER_ID_TO_LAYER[f['layer_id'][f['particle_id'] == pid].astype(int)]
            track = np.zeros((np.max(LAYER_ID_TO_LAYER) + 1, 3))
            n_hits = np.zeros(np.max(LAYER_ID_TO_LAYER) + 1)
            for h, l in zip(hits, layers):
                track[l] += h
                n_hits[l] += 1
            track /= np.maximum(n_hits, 1).reshape(-1, 1)
            tracks.append(track.reshape(-1))
            track_n_hits.append(n_hits)
            particle_ids.append(pid)
            trigger_nodes.append(f['trigger_node'][f['particle_id'] == pid][0])
            momentums.append(f['momentum'][f['particle_id'] == pid][0])
            track_origins.append(f['track_origin'][f['particle_id'] == pid][0])
            particle_ids.append(pid)


        track_hits = np.stack(tracks, axis=0)
        interaction_point = f['interaction_point']

        output_dir_event = os.path.join(output_dir, os.path.basename(filename))


        np.savez(output_dir_event,
               track_hits=track_hits,
               track_n_hits=np.stack(track_n_hits, axis=0),
               particle_id=particle_ids,
               track_origin=np.stack(track_origins, axis=0),
               momentum=np.stack(momentums, axis=0),
               interaction_point=interaction_point,
               trigger_node=np.array(trigger_nodes),
               trigger=False,
            )



def main():
    global MVTX_TYPES_LOCK
    """Main function"""
    #DATA_DIR = '/disks/disk2/yasser/FastML/output/pp2bbbar_b2OpenCharmX_241219_0-99/'
    #DATA_DIR = '/disks/disk2/yasser/FastML/output/pp2bbbar_b2OpenCharmX_250115_0-99/'
    DATA_DIR = '/home1/giorgian/real-data/parsed/nontrigger/events/'
    output_dir = '/home1/giorgian/real-data/parsed-tracks/nontrigger/'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'events'), exist_ok=True)
    with open(os.path.join(output_dir, 'info.txt'), 'w') as f:
        print(f'data_dir: {DATA_DIR}', file=f)
        print(f'output_dir: {output_dir}', file=f)

    n_workers = 16
    files = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)]
    event_output_dir = os.path.join(output_dir, 'events')
    Parallel(n_jobs=n_workers)(delayed(process)(file, i, event_output_dir) for i, file in enumerate(tqdm.tqdm(files)))

if __name__ == '__main__':
    main()
