import glob
import os.path
import tqdm
import numpy as np
from joblib import Parallel, delayed

def pair(f, groupa, groupb):
    MK = 0.493677
    MPi = 0.1395
    flag = False
    pairs = []
    for a in groupa:
        pa = f['momentum'][a]
        Ea = np.sqrt(sum(pa**2) + MK**2)
        for b in groupb:
            pb = f['momentum'][b]
            Eb = np.sqrt(sum(pb**2) + MPi**2)
            p = pa + pb
            E = Ea + Eb
            M = np.sqrt(E**2 - sum(p**2))
            if abs(M - 1.8648) < 0.0001:
                if (f['track_origin'][a] == f['track_origin'][b]).all() and (f['track_origin'][a] != f['interaction_point']).any():
                    pairs.append((a, b))
    return pairs


def process(filename):
    with np.load(filename) as f:
        pos321 = np.where(f['particle_types'] == 321)[0]
        neg321 = np.where(f['particle_types'] == -321)[0]
        pos211 = np.where(f['particle_types'] == 211)[0]
        neg211 = np.where(f['particle_types'] == -211)[0]
        trigger_node = np.zeros(f['particle_types'].shape[0], dtype=int)
        trigger_pairs_1 = pair(f, pos321, neg211)
        trigger_pairs_2 = pair(f, neg321, pos211)
        trigger_pairs = trigger_pairs_1 + trigger_pairs_2
        for a, b in trigger_pairs:
            trigger_node[a] = 1
            trigger_node[b] = 1
        data = {k:f[k] for k in f.keys()}
        data['trigger_node'] = trigger_node
        np.savez(filename, **data)



trigger_output_dir = '/ssd3/giorgian/tracks-data-august-2022/trigger/1/'
nontrigger_output_dir = '/ssd3/giorgian/tracks-data-august-2022/nontrigger/0/'
Parallel(n_jobs=8)(delayed(process)(filename) for filename in tqdm.tqdm(glob.glob(os.path.join(trigger_output_dir, '*.npz'))))
Parallel(n_jobs=8)(delayed(process)(filename) for filename in tqdm.tqdm(glob.glob(os.path.join(nontrigger_output_dir, '*.npz'))))
