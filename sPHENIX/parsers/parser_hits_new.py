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

MVTX_PIXEL_DTYPE = np.dtype([
    ('hit_assoc', 'i8'),
    ('track_id', 'i8'),
    ('hit_time', 'f8'),
    ('layer', 'i4'),
    ('strobe_id', 'i4'),
    ('stave', 'i4'),
    ('chip', 'i4'),
    ('pixel_x', 'i4'),
    ('pixel_z', 'i4'),
    ('x_l', 'f8'),
    ('y_l', 'f8'),
    ('z_l', 'f8'),
    ('x_g', 'f8'),
    ('y_g', 'f8'),
    ('z_g', 'f8')
])

INTT_PIXEL_DTYPE = np.dtype([
    ('hit_assoc', 'i8'),
    ('track_id', 'i8'),
    ('hit_time', 'f8'),
    ('layer', 'i4'),
    ('ladder_z_id', 'i4'),
    ('ladder_phi_id', 'i4'),
    ('row', 'i4'),
    ('col', 'i4'),
    ('ADC', 'f8'),
    ('x_g', 'f8'),
    ('y_g', 'f8'),
    ('z_g', 'f8')
])

CLUSTER_DTYPE = np.dtype([
    ('hit_assoc', 'i8'),
    ('track_id', 'i8'),
    ('hit_time', 'f8'),
    ('layer', 'i4'),
    ('n_pixels', 'i4'),
    ('x', 'f8'),
    ('y', 'f8'),
    ('z', 'f8'),
    ('r', 'f8'),
    ('phi', 'f8'),
    ('r_std', 'f8'),
    ('phi_std', 'f8'),
    ('z_std', 'f8')
])



TRACK_DTYPE = np.dtype([
    ('id', 'i4'),
    ('ptype', 'i4'),
    ('parent_id', 'i4'),
    ('parent_ptype', 'i4'),
    ('gparent_id', 'i4'),
    ('gparent_ptype', 'i4'),
    ('type', 'i4'),
    ('vx', 'f8'),
    ('vy', 'f8'),
    ('vz', 'f8'),
    ('px', 'f8'),
    ('py', 'f8'),
    ('pz', 'f8'),
    ('energy', 'f8')
])




def process_mvtx_pixels(event):
    mvtx_pixels = []
    for mvtx_hit in event['MvtxTrkrHits']:
        # TODO: be able to handle multiple tracks activating a pixel
        mvtx_pixels.append((
            mvtx_hit['ID']['MvtxHitAssoc'][0],
            mvtx_hit['ID']['MvtxTrkID'][0],
            mvtx_hit['ID']['MvtxHitTime'][0],
            mvtx_hit['ID']['Layer'],
            mvtx_hit['ID']['StrobeId'],
            mvtx_hit['ID']['Stave'],
            mvtx_hit['ID']['Chip'],
            mvtx_hit['ID']['Pixel_x'],
            mvtx_hit['ID']['Pixel_z'],
            *mvtx_hit['LocalCoordinate'],
            *mvtx_hit['GlobalCoordinate']
        ))
    mvtx_pixels = np.array(mvtx_pixels, dtype=MVTX_PIXEL_DTYPE)

    return mvtx_pixels

def get_mvtx_pixel_graph(mvtx_pixels):
    g = nx.Graph()
    d = {}

    edges = []
    for i, p in enumerate(mvtx_pixels):
        key = (p['layer'], p['pixel_x'], p['pixel_z'], p['stave'], p['chip'])
        d[key] = i
        edges.append((i, i))


    for i, p in enumerate(mvtx_pixels):
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                if dx == 0 and dz == 0:
                    continue
                new_key = (p['layer'], p['pixel_x'] + dx, p['pixel_z'] + dz, p['stave'], p['chip'])
                if new_key in d:
                    edges.append((i, d[new_key]))

    g.add_edges_from(edges)
    return g

def get_intt_pixel_graph(intt_pixels):
    g = nx.Graph()
    d = {}

    edges = []
    for i, p in enumerate(intt_pixels):
        key = (p['layer'], p['row'], p['col'], p['ladder_z_id'], p['ladder_phi_id'])
        d[key] = i
        edges.append((i, i))

    for i, p in enumerate(intt_pixels):
        for dr in range(-1, 2):
            if dr == 0:
                continue
            new_key = (p['layer'], p['row'] + dr, p['col'], p['ladder_z_id'], p['ladder_phi_id'])
            if new_key in d:
                edges.append((i, d[new_key]))

    g.add_edges_from(edges)
    return g


def process_intt_pixels(event):
    intt_pixels = []
    for intt_hit in event['InttTrkrHits']:
        # TODO: be able to handle multiple tracks activating a pixel
        intt_pixels.append((
            intt_hit['ID']['InttHitAssoc'][0],
            intt_hit['ID']['InttTrkId'][0],
            intt_hit['ID']['InttHitTime'][0],
            intt_hit['ID']['Layer'],
            intt_hit['ID']['LadderZId'],
            intt_hit['ID']['LadderPhiId'],
            intt_hit['ID']['Row'],
            intt_hit['ID']['Col'],
            intt_hit['ID']['ADC'],
            *intt_hit['GlobalCoordinate']
        ))
    intt_pixels = np.array(intt_pixels, dtype=INTT_PIXEL_DTYPE)

    return intt_pixels

def process_tracks(event):
    tracks = []
    for track in event['TruthPHG4Info']['TruthTracks']:
        tracks.append((
            track['TrackId'],
            track['TrackPDG'],
            track['TrackParentId'],
            track['TrackParentPDG'],
            track['TrackGrandParentId'],
            track['TrackGrandParentPDG'],
            track['TrackType'],
            *track['trackVtxPnt'],
            *track['TrackMomentum'],
            track['TrackEnergy']
        ))

    tracks = np.array(tracks, dtype=TRACK_DTYPE)
    return tracks

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


    

def cluster(pixels, graph):
    clusters = []
    for cluster in nx.connected_components(graph):
        cluster = list(cluster)
        p = pixels[cluster]
        x = p['x_g']
        y = p['y_g']
        z = p['z_g']

        r = np.sqrt(x**2 + y**2)

        r_std = np.std(r)
        z_std = np.std(z)

        x = np.mean(x)
        y = np.mean(y)
        z = np.mean(z)
        r = np.mean(r)

        phi = np.atan2(y, x)
        phi_std = np.sqrt(-np.log(min(x**2 + y**2, 1)))

        clusters.append((
            mode(p['hit_assoc']),
            mode(p['track_id']),
            np.mean(p['hit_time']),
            mode(p['layer']),
            len(cluster),
            x,
            y,
            z,
            r,
            phi,
            r_std,
            phi_std,
            z_std,
        ))

    clusters = np.array(clusters, dtype=CLUSTER_DTYPE)
    return clusters



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
        mvtx_pixels = process_mvtx_pixels(event)
        mvtx_graph = get_mvtx_pixel_graph(mvtx_pixels)
        mvtx_clusters = cluster(mvtx_pixels, mvtx_graph)

        intt_pixels = process_intt_pixels(event)
        intt_graph = get_intt_pixel_graph(intt_pixels)
        intt_clusters = cluster(intt_pixels, intt_graph)

        tracks = process_tracks(event)
        track_dict = dict(zip(tracks['id'], range(len(tracks))))

        clusters = np.concatenate([mvtx_clusters, intt_clusters])
        hit_cartesian = np.stack([clusters['x'], clusters['y'], clusters['z']], axis=1)
        hit_cylindrical = np.stack([clusters['r'], clusters['phi'], clusters['z']], axis=1)
        n_pixels = clusters['n_pixels']
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


        energy = np.array([tracks[track_dict[h['track_id']]]['energy'] if h['track_id'] in track_dict else float('nan') for h in clusters])
        p_x = np.array([tracks[track_dict[h['track_id']]]['px'] if h['track_id'] in track_dict else float('nan') for h in clusters])
        p_y = np.array([tracks[track_dict[h['track_id']]]['py'] if h['track_id'] in track_dict else float('nan') for h in clusters])
        p_z = np.array([tracks[track_dict[h['track_id']]]['pz'] if h['track_id'] in track_dict else float('nan') for h in clusters])
        momentum = np.stack([p_x, p_y, p_z], axis=-1)
        v_x = np.array([tracks[track_dict[h['track_id']]]['vx'] if h['track_id'] in track_dict else float('nan') for h in clusters])
        v_y = np.array([tracks[track_dict[h['track_id']]]['vy'] if h['track_id'] in track_dict else float('nan') for h in clusters])
        v_z = np.array([tracks[track_dict[h['track_id']]]['vz'] if h['track_id'] in track_dict else float('nan') for h in clusters])
        track_origin = np.stack([v_x, v_y, v_z], axis=-1)
        track_id = clusters['track_id']
        particle_type = np.array([tracks[track_dict[h['track_id']]]['ptype'] if h['track_id'] in track_dict else float('nan') for h in clusters])
        parent_id = np.array([tracks[track_dict[h['track_id']]]['parent_id'] if h['track_id'] in track_dict else float('nan') for h in clusters])
        parent_particle_type = np.array([tracks[track_dict[h['track_id']]]['parent_ptype'] if h['track_id'] in track_dict else float('nan') for h in clusters])
        gparent_id = np.array([tracks[track_dict[h['track_id']]]['gparent_id'] if h['track_id'] in track_dict else float('nan') for h in clusters])
        gparent_particle_type = np.array([tracks[track_dict[h['track_id']]]['gparent_ptype'] if h['track_id'] in track_dict else float('nan') for h in clusters])

        cylindrical_std = np.stack([clusters['r_std'], clusters['phi_std'], clusters['z_std']], axis=1)
        trigger = event["TruthTriggerFlag"]["Flags"]["HFMLTriggerHepMCTrigger"]

        interaction_point = np.array(event["TruthPHG4Info"]["TruthCollisionVertex"])

        output_dir_event = os.path.join(output_dir, 'event%01i%05i%03i'%(trigger, file_number, i))


        np.savez(output_dir_event,
               hit_cartesian=hit_cartesian,
               hit_cylindrical=hit_cylindrical,
               layer_id=clusters['layer'],
               n_pixels=n_pixels,
               energy=energy,
               momentum=momentum,
               interaction_point=interaction_point,
               trigger = False,
               #trigger=trigger,
               track_origin=track_origin,
               edge_index=edge_indices,
               edge_index_1=edge_indices_1,
               edge_phi_slope=edge_phi_slope,
               edge_phi_slope_1=edge_phi_slope_1,
               edge_z0=edge_z0,
               edge_z0_1=edge_z0_1,
               phi_slope_max=PHI_SLOPE_MAX,
               z0_max=Z0_MAX,
               particle_id=track_id,
               particle_type=particle_type,
               parent_id=parent_id,
               parent_particle_type=parent_particle_type,
               gparent_id=gparent_id,
               gparent_particle_type=gparent_particle_type,
               cylindrical_std=cylindrical_std
        )

def main():
    global MVTX_TYPES_LOCK
    """Main function"""
    #DATA_DIR = '/disks/disk2/yasser/FastML/output/pp2bbbar_b2OpenCharmX_241219_0-99/'
    #DATA_DIR = '/disks/disk2/yasser/FastML/output/pp2bbbar_b2OpenCharmX_250115_0-99/'
    DATA_DIR = '/disks/disk2/yasser/FastML/output/ppInQCD_MinBias_250114_0-99/'
    output_dir = '/ssd3/giorgian/hits-data-january-2024-yasser/nontrigger/'
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
