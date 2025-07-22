import ujson
import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from statistics import mode
import tqdm
from joblib import Parallel, delayed
import networkx as nx
import os

PHI_SLOPE_MAX = 0.12
Z0_MAX = 800

MVTX_CLUSTER_DTYPE = np.dtype([
    ('x', 'f8'),
    ('y', 'f8'),
    ('z', 'f8'),
    ('chip', 'i4'),
    ('cluskey', 'i8'),
    ('col', 'i4'),
    ('layer', 'i4'),
    ('row', 'i4'),
    ('stave', 'i4'),
    ('track_id', 'i4'),
    ('trigger', 'i4')
])

INTT_CLUSTER_DTYPE = np.dtype([
    ('x', 'f8'),
    ('y', 'f8'),
    ('z', 'f8'),
    ('cluskey', 'i8'),
    ('col', 'i4'),
    ('ladder_phi_id', 'i4'),
    ('ladder_z_id', 'i4'),
    ('layer', 'i4'),
    ('row', 'i4'),
    ('track_id', 'i4'),
    ('trigger', 'i4')

])

CLUSTER_DTYPE = np.dtype([
    ('x', 'f8'),
    ('y', 'f8'),
    ('z', 'f8'),
    ('r', 'f8'),
    ('phi', 'f8'),
    ('track_id', 'i4'),
    ('trigger', 'i4'),
    ('layer', 'i4'),
])


TRACK_DTYPE = np.dtype([
    ('id', 'i4'),
    ('vx', 'f8'),
    ('vy', 'f8'),
    ('vz', 'f8'),
    ('px', 'f8'),
    ('py', 'f8'),
    ('pz', 'f8'),
    ('trigger', 'i4'),
])




def process_mvtx_clusters(event):
    mvtx_clusters = []
    for mvtx_hit in event['AllTracks']['MVTXClusters']:
        # TODO: be able to handle multiple tracks activating a pixel
        mvtx_clusters.append((
            *mvtx_hit['Coordinate'],
            mvtx_hit['ID']['Chip'],
            mvtx_hit['ID']['ClusKey'],
            mvtx_hit['ID']['Col'],
            mvtx_hit['ID']['Layer'],
            mvtx_hit['ID']['Row'],
            mvtx_hit['ID']['Stave'],
            -1,
            0,
        ))

    for mvtx_hit in event['TriggerTracks']['MVTXClusters']:
        # TODO: be able to handle multiple tracks activating a pixel
        mvtx_clusters.append((
            *mvtx_hit['Coordinate'],
            mvtx_hit['ID']['Chip'],
            mvtx_hit['ID']['ClusKey'],
            mvtx_hit['ID']['Col'],
            mvtx_hit['ID']['Layer'],
            mvtx_hit['ID']['Row'],
            mvtx_hit['ID']['Stave'],
            -1,
            0,
        ))

    mvtx_clusters = np.array(mvtx_clusters, dtype=MVTX_CLUSTER_DTYPE)

    return mvtx_clusters

def process_intt_clusters(event):
    intt_clusters = []
    for intt_hit in event['AllTracks']['INTTClusters']:
        # TODO: be able to handle multiple tracks activating a pixel
        intt_clusters.append((
            *intt_hit['Coordinate'],
            intt_hit['ID']['ClusKey'],
            intt_hit['ID']['Col'],
            intt_hit['ID']['LadderPhiId'],
            intt_hit['ID']['LadderZId'],
            intt_hit['ID']['Layer'],
            intt_hit['ID']['Row'],
            -1,
            0
        ))

    for intt_hit in event['TriggerTracks']['INTTClusters']:
        # TODO: be able to handle multiple tracks activating a pixel
        intt_clusters.append((
            *intt_hit['Coordinate'],
            intt_hit['ID']['ClusKey'],
            intt_hit['ID']['Col'],
            intt_hit['ID']['LadderPhiId'],
            intt_hit['ID']['LadderZId'],
            intt_hit['ID']['Layer'],
            intt_hit['ID']['Row'],
            -1,
            0
        ))

    intt_clusters = np.array(intt_clusters, dtype=INTT_CLUSTER_DTYPE)

    return intt_clusters

def process_tracks(event, mvtx_clusters, intt_clusters):
    tracks = []
    for track in event['AllTracks']['Tracks']:
        tracks.append((
            track['TrackSequenceInEvent'],
            *track['TrackPosition'],
            *track['TrackMomentum'],
            False,
        ))
        for mvtx_hit in track['MVTXclusterKeys']:
            mvtx_clusters['track_id'][mvtx_clusters['cluskey'] == mvtx_hit] = track['TrackSequenceInEvent']

        for intt_hit in track['INTTclusterKeys']:
            intt_clusters['track_id'][intt_clusters['cluskey'] == intt_hit] = track['TrackSequenceInEvent']

    for track in event['TriggerTracks']['Tracks']:
        tracks.append((
            track['TrackSequenceInEvent'],
            *track['TrackPosition'],
            *track['TrackMomentum'],
            True,
        ))
        for mvtx_hit in track['MVTXclusterKeys']:
            mvtx_clusters['track_id'][mvtx_clusters['cluskey'] == mvtx_hit] = track['TrackSequenceInEvent']
            mvtx_clusters['trigger'][mvtx_clusters['cluskey'] == mvtx_hit] = 1

        for intt_hit in track['INTTclusterKeys']:
            intt_clusters['track_id'][intt_clusters['cluskey'] == intt_hit] = track['TrackSequenceInEvent']
            intt_clusters['trigger'][intt_clusters['cluskey'] == intt_hit] = 1



    tracks = np.array(tracks, dtype=TRACK_DTYPE)
    return tracks

def process_clusters(mvtx_clusters, intt_clusters):
    clusters = []
    for mvtx_hit in mvtx_clusters:
        clusters.append((
            mvtx_hit['x'],
            mvtx_hit['y'],
            mvtx_hit['z'],
            np.sqrt(mvtx_hit['x']**2 + mvtx_hit['y']**2),
            np.arctan2(mvtx_hit['y'], mvtx_hit['x']),
            mvtx_hit['track_id'],
            mvtx_hit['trigger'],
            mvtx_hit['layer'],
        ))

    for intt_hit in intt_clusters:
        clusters.append((
            intt_hit['x'],
            intt_hit['y'],
            intt_hit['z'],
            np.sqrt(intt_hit['x']**2 + intt_hit['y']**2),
            np.arctan2(intt_hit['y'], intt_hit['x']),
            intt_hit['track_id'],
            intt_hit['trigger'],
            intt_hit['layer'],
        ))

    clusters = np.array(clusters, dtype=CLUSTER_DTYPE)
    return clusters

def select_segments(h1, h2, h1_indices, h2_indices, phi_slope_max, z0_max):
    dphi = calc_dphi(h1[:, 1], h2[:, 1])
    dz = h1[:, 2, None] - h2[None, :, 2]
    dr = h1[:, 0, None] - h2[None, :, 0]
    dr[dr == 0] = 1e-6
    phi_slope = dphi / dr
    z0 = h1[:, 2, None] - h1[:, 0, None] * dz / dr
    good = (np.abs(phi_slope) < phi_slope_max) & (np.abs(z0) < z0_max)
    h1i, h2i = np.meshgrid(h1_indices, h2_indices, indexing='ij')

    return np.stack([h1i[good], h2i[good]], axis=0), phi_slope[good], z0[good]

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2[None, :] - phi1[:, None]
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi


    



def process(filename, file_number, output_dir):
    with open(filename, 'rb') as f:
        try:
            data = ujson.load(f)
        except Exception as e:
            print(f'{filename=} {e}')

    layer_pairs = np.array([(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (1, 3), (1,4), (2,4), (2,5), (3,5), (3,6), (4,6)])
    # 3 and 4 are the same layer
    # 5 and 6 are the same layer
    layer_pairs_1 = np.array([
       (0, 6), (1, 6), (2, 6), (3, 6), (4, 6), 
       (0, 5), (1, 5), (2, 5), (3, 5), (4, 5), 
       (0, 4), (1, 4), (2, 4), 
       (0, 3), (1, 3), (2, 3), 
       (6, 4), (6, 3),
       (5, 4), (5, 3)
       ])


    for i, event in enumerate(data['Events']):
        mvtx_clusters = process_mvtx_clusters(event)
        intt_clusters = process_intt_clusters(event)

        tracks = process_tracks(event, mvtx_clusters, intt_clusters)
        clusters = process_clusters(mvtx_clusters, intt_clusters)
        track_dict = dict(zip(tracks['id'], range(len(tracks))))

        hit_cartesian = np.stack([clusters['x'], clusters['y'], clusters['z']], axis=1)
        hit_cylindrical = np.stack([clusters['r'], clusters['phi'], clusters['z']], axis=1)
        hit_indices = np.arange(hit_cylindrical.shape[0])

        edge_phi_slope = []
        edge_z0 = []
        edge_indices = []
        for (l1, l2) in layer_pairs:
            h1 = hit_cylindrical[clusters['layer'] == l1]
            h2 = hit_cylindrical[clusters['layer'] == l2]
            h1_indices = hit_indices[clusters['layer'] == l1]
            h2_indices = hit_indices[clusters['layer'] == l2]
            segment_indices, phi_slope, z0 = select_segments(h1, h2, h1_indices, h2_indices, PHI_SLOPE_MAX, Z0_MAX)
            edge_phi_slope.append(phi_slope)
            edge_z0.append(z0)
            edge_indices.append(segment_indices)

        edge_phi_slope = np.concatenate(edge_phi_slope)
        edge_z0 = np.concatenate(edge_z0)
        edge_indices = np.concatenate(edge_indices, axis=1)

        edge_phi_slope_1 = []
        edge_z0_1 = []
        edge_indices_1 = []
        for (l1, l2) in layer_pairs_1:
            h1 = hit_cylindrical[clusters['layer'] == l1]
            h2 = hit_cylindrical[clusters['layer'] == l2]
            h1_indices = hit_indices[clusters['layer'] == l1]
            h2_indices = hit_indices[clusters['layer'] == l2]
            segment_indices, phi_slope, z0 = select_segments(h1, h2, h1_indices, h2_indices, PHI_SLOPE_MAX, Z0_MAX)
            edge_phi_slope_1.append(phi_slope)
            edge_z0_1.append(z0)
            edge_indices_1.append(segment_indices)

        edge_phi_slope_1 = np.concatenate(edge_phi_slope_1)
        edge_z0_1 = np.concatenate(edge_z0_1)
        edge_indices_1 = np.concatenate(edge_indices_1, axis=1)


        p_x = np.array([tracks[track_dict[h['track_id']]]['px'] if h['track_id'] in track_dict else float('nan') for h in clusters])
        p_y = np.array([tracks[track_dict[h['track_id']]]['py'] if h['track_id'] in track_dict else float('nan') for h in clusters])
        p_z = np.array([tracks[track_dict[h['track_id']]]['pz'] if h['track_id'] in track_dict else float('nan') for h in clusters])
        momentum = np.stack([p_x, p_y, p_z], axis=-1)
        v_x = np.array([tracks[track_dict[h['track_id']]]['vx'] if h['track_id'] in track_dict else float('nan') for h in clusters])
        v_y = np.array([tracks[track_dict[h['track_id']]]['vy'] if h['track_id'] in track_dict else float('nan') for h in clusters])
        v_z = np.array([tracks[track_dict[h['track_id']]]['vz'] if h['track_id'] in track_dict else float('nan') for h in clusters])
        track_origin = np.stack([v_x, v_y, v_z], axis=-1)
        track_id = clusters['track_id']
        trigger = 0

        interaction_point = np.array(event["Metadata"]["CollisionVertex"])

        output_dir_event = os.path.join(output_dir, 'event%01i%05i%03i'%(trigger, file_number, i))
        trigger_node = clusters['trigger']
        particle_id = clusters['track_id']


        np.savez(output_dir_event,
               hit_cartesian=hit_cartesian,
               hit_cylindrical=hit_cylindrical,
               layer_id=clusters['layer'],
               momentum=momentum,
               interaction_point=interaction_point,
               trigger = False,
               trigger_node=trigger_node,
               #trigger=trigger,
               track_origin=track_origin,
               particle_id=particle_id,
               edge_index=edge_indices,
               edge_index_1=edge_indices_1,
               edge_phi_slope=edge_phi_slope,
               edge_phi_slope_1=edge_phi_slope_1,
               edge_z0=edge_z0,
               edge_z0_1=edge_z0_1,
               phi_slope_max=PHI_SLOPE_MAX,
               z0_max=Z0_MAX,
        )

def main():
    global MVTX_TYPES_LOCK
    """Main function"""
    #DATA_DIR = '/disks/disk2/yasser/FastML/output/pp2bbbar_b2OpenCharmX_241219_0-99/'
    #DATA_DIR = '/disks/disk2/yasser/FastML/output/pp2bbbar_b2OpenCharmX_250115_0-99/'
    DATA_DIR = '/home1/giorgian/real-data/nontrigger'
    output_dir = '/home1/giorgian/real-data/parsed/nontrigger/'
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
