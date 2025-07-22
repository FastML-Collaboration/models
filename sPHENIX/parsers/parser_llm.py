import rapidjson as ujson
import logging
import glob
import bz2
import os
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist 
from statistics import mode
import multiprocessing as mp
from functools import partial
from itertools import product
import itertools
from numpy.linalg import inv
from icecream import ic
import pickle
import random
from collections import defaultdict
import tqdm
from joblib import Parallel, delayed

SORT = False

# DATA_DIR = '/home/yuantian/Data/D0toPiKInAcceptanceSignal_Iteration6'
# output_dir = 'parsed_D0_Iteration6'
# DATA_DIR = '/home/zhaozhongshi/HFMLTrigger/NewDataUpdated/Signal'
# output_dir = 'parsed_trackvec_new/trigger'
# DATA_DIR = '/home/zhaozhongshi/HFMLTrigger/NewDataUpdated/Background'
# output_dir = 'parsed_trackvec_new/nontrigger'
TRIGGER_DATA_DIR = '/disks/disk1/tingtingxuan/HFMLNewFiles-old/Signal'

# In order to ensure the cluster types dictionary is shared in between them
NONTRIGGER_DATA_DIR = '/disks/disk1/tingtingxuan/HFMLNewFiles-old/Background'

data_dir = sorted(glob.glob(NONTRIGGER_DATA_DIR + '/*.json')) + sorted(glob.glob(TRIGGER_DATA_DIR + '/*.json'))
random.shuffle(data_dir)
filenumber = 0
total_number = 100000
raw_layer_to_layer = { 0:0, 1:1, 2:2, 3:3, 4:3, 5:4, 6:4}
filenames = data_dir[filenumber:(filenumber+total_number)]
output_dirs = ('d02pik/nontrigger/', 'd02pik/trigger/')
NOISE = 0
for output_dir in output_dirs:
    os.makedirs(output_dir, exist_ok=True)

def pair(groupa, groupb, ip):
    MK = 0.493677
    MPi = 0.1395
    flag = False
    paired_tracks = []
    for a in groupa:
        pa = np.array(a['TrackMomentum'])
        Ea = np.sqrt(sum(pa**2) + MK**2)
        for b in groupb:
            pb = np.array(b['TrackMomentum'])
            Eb = np.sqrt(sum(pb**2) + MPi**2)
            p = pa + pb
            E = Ea + Eb
            M = np.sqrt(E**2 - sum(p**2))
            if abs(M - 1.8648) < 0.0001:
                if a['OriginVertexPoint'] == b['OriginVertexPoint'] and (a['OriginVertexPoint'] != ip).any():
                    flag = True
                    paired_tracks.append((a['pair_tracking_id'], b['pair_tracking_id']))
                    # print(a['trackVtxPnt'], b['trackVtxPnt'], ip)
                # print('------------------')
    return flag, paired_tracks 



def process(filename, i):
    try:
        events = ujson.load(open(filename))
    except Exception as e:
        print(f'Could not load file: {e}')
        return

    for j, event in enumerate(events['Events']):
        trigger = int(event['TruthTriggerFlag']['Flags']['D0toPiKInAcceptance'])
        with open(os.path.join(output_dirs[trigger], f'event{trigger}_{i}_{j}.txt'), 'w') as fout:
            tracks = event['TruthHit']['TruthTracks']

            pos321 = []
            neg321 = []
            pos211 = []
            neg211 = []
            for i, track in enumerate(tracks):
                track['trigger_node'] = 0
                track['pair_tracking_id'] = i
                if track['ParticleTypeID'] == 321:
                    pos321.append(track)
                elif track['ParticleTypeID'] == -321:
                    neg321.append(track)
                elif track['ParticleTypeID'] == 211:
                    pos211.append(track)
                elif track['ParticleTypeID'] == -211:
                    neg211.append(track)

     
            collision_vertex = event['MetaData']['CollisionVertex']
            print(f'Here is a particle collision event with {len(tracks)} tracks.', file=fout)
            print(f'The collision vertex is {tuple(collision_vertex)}.', file=fout)
            ovs = []
            for track in tracks:
                ovs.append(track['OriginVertexPoint'])

            ovs = np.array(ovs)
            if len(ovs) > 0:
                if SORT:
                    order = np.lexsort(np.rot90(ovs))
                else:
                    order = np.arange(len(tracks))
            else:
                order = []
            compiled_tracks = []
            for j, i in enumerate(order):
                track = tracks[i] 
                ov = tuple(float(x) for x in (track['OriginVertexPoint'] + np.random.normal(size=3, scale=NOISE)))
                ptype = track['ParticleTypeID']
                momentum = track['TrackMomentum'] + np.random.normal(size=3, scale=NOISE)
                p_t = np.sqrt(momentum[0]**2 + momentum[1]**2)
                p_z = momentum[2]
                energy = track['TrackEnergy']
                #ht = track['HitInTruthTrack']
                #print(f'{ptype_id=} {ov=} {momentum=} {energy=}')
                #print(f'{ht=}')
                #print(f'{track["MVTXHitID"]=}')
                #print(f'{track["INTTHitID"]=}')
                layers = defaultdict(list)
                for mvtx_id in track["MVTXHitID"][0]:
                    hit = event['RawHit']['MVTXHits'][mvtx_id]
                    layer = raw_layer_to_layer[hit['ID']['Layer']]
                    layers[layer].append(hit['Coordinate'])

                for intt_id in track["INTTHitID"][0]:
                    hit = event['RawHit']['INTTHITS'][intt_id]
                    layer = raw_layer_to_layer[hit['ID']['Layer']]
                    layers[layer].append(hit['Coordinate'])

                track = []
                for layer, hits in sorted(layers.items()):
                    track.append(tuple(float(x) for x in (np.mean(hits, axis=0) + np.random.normal(size=3, scale=NOISE))))
                compiled_tracks.append(track)
                #print(f'Track number {i+1} has a particle type of {ptype}, an origin vertex of {ov}, a transverse momentum of {p_t}, a parallel momentum of {p_z}, and energy of {energy} and a trajectory of {track} as the particle flew through the detector.', file=fout)
                print(f'Track number {j+1} has an origin vertex of {ov}, a transverse momentum of {p_t}, a parallel momentum of {p_z} and a trajectory of {track} as the particle flew through the detector.', file=fout)
            collision_vertex = np.array(collision_vertex)
            has_trigger_1, trigger_pairs_1 = pair(pos321, neg211, collision_vertex)
            has_trigger_2, trigger_pairs_2 = pair(neg321, pos211, collision_vertex)
            has_trigger = has_trigger_1 or has_trigger_2
            if has_trigger:
                print('This event is a trigger event because a D0 decay was found.', file=fout)
                order = order.tolist()
                for (a, b) in trigger_pairs_1:
                    a_ordered = order.index(a)
                    b_ordered = order.index(b)
                    print(f'Tracks ({a_ordered+1}, {b_ordered+1}) decayed from the D0 particle.', file=fout)

                for (a, b) in trigger_pairs_2:
                    a_ordered = order.index(a)
                    b_ordered = order.index(b)
                    print(f'Tracks ({a_ordered+1}, {b_ordered+1}) decayed from the D0 particle.', file=fout)

            


Parallel(n_jobs=32)(delayed(process)(filename, i) for i, filename in enumerate(tqdm.tqdm(filenames)))
