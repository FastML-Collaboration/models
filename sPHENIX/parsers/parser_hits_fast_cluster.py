from binascii import Incomplete
import ujson
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
from datetime import datetime
from icecream import ic
import itertools
from collections import defaultdict

# Geometric constraints
PHI_SLOPE_MAX = 0.12
Z0_MAX = 800

def find_medoid(pixels):
    # pixels: (N, 3)
    deltas = pixels[None] - pixels[:, None]
    # deltas: (N, N, 3)
    deltas = np.linalg.norm(deltas, axis=0)
    # deltas: (N, N)
    mean_distance = np.mean(deltas, axis=0)
    return pixels[np.argmin(mean_distance)]

def get_cluster_info(mvtx_cluster_types, one_cluster, type_hits):
    mvtx_dict = mvtx_cluster_types

    cluster_coords = np.array([(hit['x'],hit['y'],hit['z']) for hit in one_cluster])
    r = np.linalg.norm(cluster_coords[:, :2], axis=-1)
    phi = np.arctan2(cluster_coords[:, 1], cluster_coords[:, 0])
    x = np.mean(np.cos(phi))
    y = np.mean(np.sin(phi))

    phi_std = np.sqrt(-np.log(min(x**2 + y**2, 1)))
    r_std = np.std(r)
    z_std = np.std(cluster_coords[:, 2])

    if type_hits == 'MVTX':
        pixel_x = [hit['pixel_x'] for hit in one_cluster]
        pixel_z = [hit['pixel_z'] for hit in one_cluster]
        selected_pixels = np.array(sorted([tuple(p) for p in zip(pixel_x, pixel_z)]))
        medoid = find_medoid(selected_pixels)
        diffs = selected_pixels - medoid[None]
        medoid_diff = np.array(sorted([tuple(p) for p in diffs]))
        min_x, min_y = np.min(medoid_diff[:, 0]), np.min(medoid_diff[:, 1])
        medoid_diff[:, 0] -= min_x
        medoid_diff[:, 1] -= min_y
        cluster = tuple(sorted(tuple(p) for p in medoid_diff))
        with MVTX_TYPES_LOCK:
            if cluster in mvtx_dict:
                cluster_type = mvtx_dict[cluster]
            else:
                cluster_type = len(mvtx_dict)
                mvtx_dict[cluster] = cluster_type
    elif type_hits == 'INTT':
        # for INTT, the clustered hits are only straight lines
        # Thus, the type of cluster is simply the number of pixels activated
        cluster_type = len(one_cluster)

    return cluster_type, r_std, phi_std, z_std


def preprocessing_hits(event, type_hits):
    hits = []
    if type_hits == 'MVTX':
        hits = event['RawHit'][type_hits + 'Hits']
    elif type_hits == 'INTT':
        hits = event['RawHit'][type_hits + 'HITS']
    else:
        print('Invalid type.')

    if len(hits) == 0:
        print('no hits here!')
        return hits
    hits_list = []
    key_list = []
    n_MVTXHits = len(event['RawHit']['MVTXHits'])
    for hit in hits:
        hit_id = hit['ID']['HitSequenceInEvent']
        x, y, z = hit['Coordinate']
        if type_hits == 'MVTX':
            layer_id = hit["ID"]["Layer"] 
            chip_id = hit['ID']['Chip'] #Chip ID used
            stave_id = hit['ID']['Stave']
            pixel_x = hit['ID']['Pixel_x']
            pixel_z= hit['ID']['Pixel_z']
            one_hit = {'hit_id': hit_id, 'x': x, 'y': y, 'z': z, 
                        'pixel_x':pixel_x, 'pixel_z':pixel_z,
                        'stave_id': stave_id, 'layer_id': layer_id, 'chip_id': chip_id}
            hits_list.append(one_hit)
            one_key = (layer_id,pixel_x,pixel_z,chip_id,stave_id)
            key_list.append(one_key)
        elif type_hits == 'INTT':
            hit_id += n_MVTXHits
            layer_id = hit["ID"]["Layer"]
            row = hit['ID']['Row']
            col = hit['ID']['Col']
            ladder_z_id = hit['ID']['LadderZId']
            ladder_phi_id = hit['ID']['LadderPhiId']
            one_hit = {'hit_id': hit_id, 'x': x, 'y': y, 'z': z, 
                        'row': row, 'col': col, 'layer_id': layer_id, 
                        'ladder_z_id': ladder_z_id, 'ladder_phi_id': ladder_phi_id}
            hits_list.append(one_hit)
            one_key = (layer_id, row, col, ladder_z_id, ladder_phi_id)
            key_list.append(one_key)
    hits_dict=dict(zip(key_list, hits_list))
    return hits_dict

def clustering_hits(mvtx_cluster_types, type_hits, hits_dict, pid_dic, p_dic):
    hits_grouped = defaultdict(list)

    for k, v in hits_dict.items():
        if type_hit == 'MVTX':
            layer_id, pixel_x, pixel_z, chip_id, stave_id = k
            hits_grouped[(layer_id, stave_id, chip_id)].append(v)
        elif type_hit == 'INTT':
            layer_id, row, col, ladder_z_id, ladder_phi_id = k
            hits_grouped[(layer_id, ladder_z_id, ladder_phi_d]).append(v)


    hits_clustered = []
    for k, v in hits_grouped.items():
        if type_hit == 'MVTX':
            v = sorted(v, key=lambda x: (x['pixel_x'], x['pixel_z']))
        elif type_hit == 'INTT':
            v = sorted(v, key=lambda x: (x['row'], x['col']))

    for k, v in hits_grouped.item():
        if types_hits == 'MVTX':
            n_cols = max(map(lambda x: x['pixel_z'], v))
            cols = 


def clustering_hits(mvtx_cluster_types, type_hits, hits_dict, pid_dic, p_dic):
    hits_clustered = []
    while hits_dict:
        root_hit = hits_dict.pop(list(hits_dict.keys())[0])
        one_cluster = []
        queue = []
        one_cluster.append(root_hit)
        queue.append(root_hit)
        pid_list = [pid_dic[type_hits].get(root_hit['hit_id'], np.nan)]
        hid_list = [root_hit['hit_id']]
        while queue:
            one_hit = queue.pop(0)
            neighbouring_keys = []
            if type_hits == 'MVTX':
                layer_id, pixel_x, pixel_z, stave_id, chip_id = one_hit['layer_id'],one_hit['pixel_x'],one_hit['pixel_z'], one_hit['stave_id'], one_hit['chip_id']
                neighbouring_keys=[(x[0],x[1],x[2],x[3],x[4]) for x in np.array( \
                    np.meshgrid( \
                        [layer_id], \
                        [pixel_x-2,pixel_x-1,pixel_x,pixel_x+1,pixel_x+2], \
                        [pixel_z-2,pixel_z-1,pixel_z,pixel_z+1,pixel_z+2], \
                        [chip_id], \
                        [stave_id]
                        )).T.reshape(-1,5)]
            elif type_hits == 'INTT':
                layer_id, row, col, ladder_z_id, ladder_phi_id = one_hit['layer_id'], one_hit['row'], one_hit['col'], one_hit['ladder_z_id'],one_hit['ladder_phi_id']
                neighbouring_keys=[(x[0], x[1], x[2], x[3],x[4]) for x in np.array( \
                    np.meshgrid( \
                        [layer_id], [row - 1, row, row + 1], [col], [ladder_z_id], [ladder_phi_id])).T.reshape(-1,5)]
            else:
                print('Invalid type of hits.')
                return
                
            for key in neighbouring_keys:
                try:
                    neighbouring_hit = hits_dict.pop(key)
                    queue.append(neighbouring_hit)
                    one_cluster.append(neighbouring_hit)
                    pid_list.append(pid_dic[type_hits].get(neighbouring_hit['hit_id'], np.nan))
                    hid_list.append(neighbouring_hit['hit_id'])
                except KeyError:
                    pass
        cluster_coords = [(hit['x'],hit['y'],hit['z']) for hit in one_cluster]
        clustered_hit = one_cluster[cdist(cluster_coords,cluster_coords).sum(axis=1).argmin()]
        clustered_hit['n_pixels']=len(one_cluster)
        clustered_hit['pid'] = mode(pid_list)
        clustered_hit['psv'] = p_dic[clustered_hit['pid']]['OriginVertexPoint']
        clustered_hit['p_momentum'] = np.array(p_dic[clustered_hit['pid']]['TrackMomentum'])
        clustered_hit['layer_id'] = layer_id
        clustered_hit['energy'] = p_dic[clustered_hit['pid']]['TrackEnergy']
        clustered_hit['ParticleTypeID'] = p_dic[clustered_hit['pid']]['ParticleTypeID']
        clustered_hit['TriggerTrackFlag'] = p_dic[clustered_hit['pid']]['TriggerTrackFlag']
        # find a good hit_id
        cluster_type, r_std, phi_std, z_std = get_cluster_info(mvtx_cluster_types, one_cluster, type_hits)
        clustered_hit['cluster_type'] = cluster_type
        clustered_hit['r_std'] = r_std
        clustered_hit['phi_std'] = phi_std
        clustered_hit['z_std'] = z_std

        clustered_hit['hit_id'] = hid_list[pid_list.index(clustered_hit['pid'])]
        clustered_hit['x'] = np.mean([t['x'] for t in one_cluster])
        clustered_hit['y'] = np.mean([t['y'] for t in one_cluster])
        clustered_hit['z'] = np.mean([t['z'] for t in one_cluster])
        hits_clustered.append(clustered_hit)
    hits_df=pd.DataFrame(hits_clustered)
    return hits_df



def cluster_hits_by_event(mvtx_cluster_types, event, pid_dic, p_dic):
    hits_dict = {}
    hits_dict['MVTX'] = preprocessing_hits(event, 'MVTX')
    hits_dict['INTT'] = preprocessing_hits(event, 'INTT')
    len_mvtx = len(hits_dict['MVTX'])
    len_intt = len(hits_dict['INTT'])
#     if len_mvtx == 0 or len_intt == 0:
#         # TODO to confirm
#         return hits_dict
    list_pd = [clustering_hits(mvtx_cluster_types, type_hits, hits_dict[type_hits], pid_dic, p_dic) for type_hits in ['MVTX', 'INTT']]
    hits_df = pd.concat(list_pd)
    hits_df = hits_df.reset_index()
    return hits_df

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi

def select_segments(hits1, hits2, phi_slope_max, z0_max):
    """
    Construct a list of selected segments from the pairings
    between hits1 and hits2, filtered with the specified
    phi slope and z0 criteria.

    Returns: pd DataFrame of (index_1, index_2), corresponding to the
    DataFrame hit label-indices in hits1 and hits2, respectively.
    """
    # Start with all possible pairs of hits
    keys = ['evtid', 'r', 'phi', 'z']
    hit_pairs = hits1[keys].reset_index().merge(
        hits2[keys].reset_index(), on='evtid', suffixes=('_1', '_2'))
    # Compute line through the points
    dphi = calc_dphi(hit_pairs.phi_1, hit_pairs.phi_2)
    dz = hit_pairs.z_2 - hit_pairs.z_1
    dr = hit_pairs.r_2 - hit_pairs.r_1
    phi_slope = dphi / dr
    z0 = hit_pairs.z_1 - hit_pairs.r_1 * dz / dr
    # Filter segments according to criteria
    good_seg_mask = (phi_slope.abs() < phi_slope_max) & (z0.abs() < z0_max)
    return hit_pairs[['index_1', 'index_2']][good_seg_mask], phi_slope[good_seg_mask], z0[good_seg_mask]

def pair(groupa, groupb, ip):
    MK = 0.493677
    MPi = 0.1395
    flag = False
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
                    a['trigger_node'] = True
                    b['trigger_node'] = True
                    # print(a['OriginVertexPoint'], b['OriginVertexPoint'], ip)
                # print('------------------')
    return flag

def Parse_event(mvtx_cluster_types, filename, output_dir, file_number):
    now = datetime.now()
    # current_time = now.strftime("%H:%M:%S")
    # print("Current Time =", current_time)
    # print(filename, output_dir, file_number)
    with open(filename,'rb') as z:
        try:
            raw_data = ujson.loads(z.read())
        except:
            print(str(file_number) + ': an error occurred when loading the json file!!!! ###################################')
            return
    n_event = len(raw_data['Events'])
    total_event = 0
    valid_trigger_event = 0
    invalid_trigger_event = 0
    empty_event = 0
    no_edge_event = 0
    for event in raw_data['Events']:
        total_event += 1
        event_ID = event['MetaData']['EventID']
        trigger = int(event['TruthTriggerFlag']['Flags']['D0toPiKInAcceptance'])
        ip = np.array(event["MetaData"]['CollisionVertex'])
        # the output event filename is consist of three part: 
        # the first number represent whether this is a trgger event or not (1: trigger, 0: non-trigger)
        # the second part is the self-defined file_number (to make they different)
        # the thrid part is the eventID in json files. (0-999)
        evtid = 'event%01i%05i%03i'%(trigger, file_number, event_ID)
        
        # precessing MVTX hits
        n_MVTXHits = len(event['RawHit']['MVTXHits'])
        n_INTTHits = len(event['RawHit']['INTTHITS'])
        if n_MVTXHits == 0:
            empty_event += 1
            print('no MVTX hits here!')
            continue
        elif n_INTTHits == 0:
            empty_event += 1
            print('no INTT hits here!')
            continue
        

        pid_dic = {}
        p_dic = {np.nan: {'OriginVertexPoint': np.nan*np.ones(3), 'TrackMomentum': np.nan*np.ones(3), 'TrackEnergy':np.nan, 'ParticleTypeID':np.nan, 'TriggerTrackFlag':0}}
        pid_dic['MVTX'] = {}
        pid_dic['INTT'] = {}
        truth_tracks = event['TruthHit']['TruthTracks']

        pos321 = []
        neg321 = []
        pos211 = []
        neg211 = []
        for track in truth_tracks:
            track['trigger_node'] = False
            if track['ParticleTypeID'] == 321:
                pos321.append(track)
            elif track['ParticleTypeID'] == -321:
                neg321.append(track)
            elif track['ParticleTypeID'] == 211:
                pos211.append(track)
            elif track['ParticleTypeID'] == -211:
                neg211.append(track)
        valid_trigger_flag = pair(pos321, neg211, ip) or pair(neg321, pos211, ip)

        for truth_track in truth_tracks:
            pid = truth_track['TrackSequenceInEvent']
            p_dic[pid] = {'OriginVertexPoint': truth_track['OriginVertexPoint'], 'TrackMomentum': truth_track['TrackMomentum'], 'TrackEnergy': truth_track['TrackEnergy'], 'ParticleTypeID': truth_track['ParticleTypeID'], 'TriggerTrackFlag': truth_track['trigger_node']}
            for hit_id in truth_track['MVTXHitID'][0]:
                pid_dic['MVTX'][hit_id] = pid
            for hit_id in truth_track['INTTHitID'][0]:
                pid_dic['INTT'][hit_id + n_MVTXHits] = pid

        hits_df = cluster_hits_by_event(mvtx_cluster_types, event, pid_dic, p_dic)
        
        if len(hits_df) == 0:
            empty_event += 1
            print('length of hits_df is zero!')
            continue
            # raise KeyboardInterrupt
        # print(hits_df)
        if not 'x' in hits_df:
            empty_event += 1
            print('hits_df was not generated!')
            print(hits_df.keys)
            continue

        # generate edge_index
        # layer_pairs = np.array([(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (1, 3), (1,4), (2,4), (2,5), (3,5), (3,6), (4,6)])
        layer_pairs = np.array([(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (1, 3), (1,4), (2,4), (2,5), (3,5), (3,6), (4,6)])

        r = np.sqrt(hits_df.x**2 + hits_df.y**2)
        phi = np.arctan2(hits_df.y, hits_df.x)
        hits_df = hits_df.assign(r=r, phi=phi, evtid=evtid)
        
        layer_groups = hits_df.groupby('layer_id')
        segments = []
        edge_phi_slope = []
        edge_z0 = [] 
        for (layer1, layer2) in layer_pairs:
            # Find and join all hit pairs
            try:
                hits1 = layer_groups.get_group(layer1)
                hits2 = layer_groups.get_group(layer2)
            # If an event has no hits on a layer, we get a KeyError.
            # In that case we just skip to the next layer pair
            except KeyError as e:
                logging.info('skipping empty layer: %s' % e)
                continue
            # Construct the segments
            res = select_segments(hits1, hits2, phi_slope_max=PHI_SLOPE_MAX, z0_max=Z0_MAX)
            segments.append(res[0])
            edge_phi_slope.append(res[1])
            edge_z0.append(res[2])
        # Combine segments from all layer pairs
        try:
            segments = pd.concat(segments)
            edge_phi_slope = pd.concat(edge_phi_slope)
            edge_z0 = pd.concat(edge_z0)
        except:
            no_edge_event += 1
            print("no edge!", filename)
            # print(segments)
            continue
        edge_index = np.transpose(np.array(segments))
        edge_phi_slope = np.array(edge_phi_slope)
        edge_z0 = np.array(edge_z0)
        
        hits = hits_df[['r', 'phi', 'z']].values
        hits_cylindrical = ( hits ).astype(np.float32)
        hit_cartesian = hits_df[['x', 'y', 'z']]
        n_pixels = hits_df['n_pixels'].values
        n_hits = hits_df.shape[0]
        n_tracks = len(p_dic)-1
        hit_type = hits_df['cluster_type'].values
        cylindrical_std = hits_df[['r_std', 'phi_std', 'z_std']].values

        if valid_trigger_flag:
            valid_trigger_event += 1
        else:
            invalid_trigger_event += 1

        # save to npz file
        output_dir_event = os.path.join(output_dir, str(int(valid_trigger_flag)), 'event%01i%05i%03i'%(trigger, file_number, event_ID))
        np.savez(output_dir_event, 
                hit_cartesian=hits_df[['x', 'y', 'z']].values.astype(np.float64),
                hit_cylindrical=hits_df[['r', 'phi', 'z']].values.astype(np.float64),
                layer_id=hits_df['layer_id'].values.astype(int),
                n_pixels=hits_df['n_pixels'].values.astype(int),
                energy=hits_df['energy'].values.astype(np.float64),
                momentum=np.stack(hits_df['p_momentum'].values, axis=0).astype(np.float64),
                interaction_point=ip,
                trigger=trigger,
                has_trigger_pair=valid_trigger_flag,
                track_origin=hits_df['psv'],
                edge_index=edge_index,
                edge_z0=edge_z0,
                edge_phi_slope=edge_phi_slope,
                phi_slope_max=PHI_SLOPE_MAX,
                z0_max=Z0_MAX,
                trigger_node=hits_df['TriggerTrackFlag'].values.astype(int),
                particle_id=hits_df['pid'].values.astype(np.float64),
                particle_type=hits_df['ParticleTypeID'].values.astype(np.float64),
                parent_particle_type=np.nan*np.ones_like(hits_df['pid'].values).astype(np.float64),
                hit_type=hit_type.astype(np.float64),
                cylindrical_std=cylindrical_std.astype(np.float64)
        )
                

    ic(total_event, valid_trigger_event, invalid_trigger_event, empty_event, no_edge_event)
    #print(f'total_event: {total_event}, empty_event: {empty event}, no_edge_event: {no_edge_event}')
        # print(output_dir_event)
    # print(filename + '\t' + str(n_event) + '\t' + str(empty_event) + '\t' + str(no_edge_event))

MVTX_TYPES_LOCK = None
def main():
    global MVTX_TYPES_LOCK
    """Main function"""
    TRIGGER_DATA_DIR = '/disks/disk1/tingtingxuan/HFMLNewFiles-old/Signal'
    trigger_output_dir = '/ssd1/giorgian/hits-data-august-2022-ctypes/trigger'
    
    n_workers = 16
    os.makedirs(trigger_output_dir, exist_ok=True)
    os.makedirs(trigger_output_dir+'/0', exist_ok=True)
    os.makedirs(trigger_output_dir+'/1', exist_ok=True)
    os.makedirs(trigger_output_dir+'/empty', exist_ok=True)
    data_dir = sorted(glob.glob(TRIGGER_DATA_DIR + '/*.json'))
    filenumber = 0
    total_number = len(data_dir)
    filenames = data_dir[filenumber:(filenumber+total_number)]
    filenumber =  [i for i in range(filenumber, (filenumber+total_number))]
    print(filenumber)
    manager = mp.Manager()
    MVTX_CLUSTER_TYPES = manager.dict()
    MVTX_TYPES_LOCK = mp.Lock()
    with mp.Pool(processes=n_workers) as pool:
        pool.starmap(Parse_event, zip(itertools.repeat(MVTX_CLUSTER_TYPES), filenames, [trigger_output_dir]*total_number, filenumber))

    # In order to ensure the cluster types dictionary is shared in between them
    NONTRIGGER_DATA_DIR = '/disks/disk1/tingtingxuan/HFMLNewFiles-old/Background'
    nontrigger_output_dir = '/ssd1/giorgian/hits-data-august-2022-ctypes/nontrigger'
    
    n_workers = 16
    os.makedirs(nontrigger_output_dir, exist_ok=True)
    os.makedirs(nontrigger_output_dir+'/0', exist_ok=True)
    os.makedirs(nontrigger_output_dir+'/1', exist_ok=True)
    os.makedirs(nontrigger_output_dir+'/empty', exist_ok=True)
    data_dir = sorted(glob.glob(NONTRIGGER_DATA_DIR + '/*.json'))
    filenumber = 0
    total_number = len(data_dir)
    filenames = data_dir[filenumber:(filenumber+total_number)]
    filenumber =  [i for i in range(filenumber, (filenumber+total_number))]
    print(filenumber)
    with mp.Pool(processes=n_workers) as pool:
        pool.starmap(Parse_event, zip(itertools.repeat(MVTX_CLUSTER_TYPES), filenames, [nontrigger_output_dir]*total_number, filenumber))

    # Parse_event(filename=data_dir[0], output_dir=PARSED_DATA_DIR, file_number=0)
    print('Done!')
    with open(os.path.join(os.path.dirname(trigger_output_dir), 'mvtx_cluster_types.pickle'), 'wb') as f:
        pickle.dump(MVTX_CLUSTER_TYPES.copy(), f)


if __name__ == '__main__':
    main()
