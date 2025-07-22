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
import itertools
from numpy.linalg import inv
from icecream import ic
import pickle

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi

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
    hits_clustered = {}
    cluster_id = 0
    while hits_dict:
        root_hit = hits_dict.pop(list(hits_dict.keys())[0])
        one_cluster = []
        queue = []
        one_cluster.append(root_hit)
        queue.append(root_hit)
        pid_list = [pid_dic[type_hits].get(root_hit['hit_id'], -1)]
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
                    pid_list.append(pid_dic[type_hits].get(neighbouring_hit['hit_id'], -1))
                    hid_list.append(neighbouring_hit['hit_id'])
                except KeyError:
                    pass
        cluster_coords = [(hit['x'],hit['y'],hit['z']) for hit in one_cluster]
        clustered_hit = one_cluster[cdist(cluster_coords,cluster_coords).sum(axis=1).argmin()]
        clustered_hit['n_pixels']=len(one_cluster)
        clustered_hit['pid'] = mode(pid_list)
        clustered_hit['psv'] = p_dic[clustered_hit['pid']]['OriginVertexPoint']
        clustered_hit['p_momentum'] = p_dic[clustered_hit['pid']]['TrackMomentum']
        clustered_hit['layer_id'] = layer_id
        clustered_hit['energy'] = p_dic[clustered_hit['pid']]['TrackEnergy']
        clustered_hit['ParticleTypeID'] = p_dic[clustered_hit['pid']]['ParticleTypeID']
        clustered_hit['TriggerTrackFlag'] = p_dic[clustered_hit['pid']]['TriggerTrackFlag']
        cluster_type, r_std, phi_std, z_std = get_cluster_info(mvtx_cluster_types, one_cluster, type_hits)
        clustered_hit['cluster_type'] = cluster_type
        clustered_hit['r_std'] = r_std
        clustered_hit['phi_std'] = phi_std
        clustered_hit['z_std'] = z_std
        # find a goodhit_id
        clustered_hit['hit_id'] = hid_list[pid_list.index(clustered_hit['pid'])]
        clustered_hit['x'] = np.mean([t['x'] for t in one_cluster])
        clustered_hit['y'] = np.mean([t['y'] for t in one_cluster])
        clustered_hit['z'] = np.mean([t['z'] for t in one_cluster])
        clustered_hit['cluster_id'] = cluster_id
        cluster_id += 1
        for hit_id in hid_list:
            hits_clustered[hit_id] = clustered_hit

    return hits_clustered

def cluster_hits_by_event(mvtx_cluster_types, event, pid_dic, p_dic):
    hits_dict = {}
    hits_dict['MVTX'] = preprocessing_hits(event, 'MVTX')
    hits_dict['INTT'] = preprocessing_hits(event, 'INTT')
    len_mvtx = len(hits_dict['MVTX'])
    len_intt = len(hits_dict['INTT'])
#     if len_mvtx == 0 or len_intt == 0:
#         # TODO to confirm
#         return hits_dict
    mvtx_df, intt_df = [clustering_hits(mvtx_cluster_types, type_hits, hits_dict[type_hits], pid_dic, p_dic) for type_hits in ['MVTX', 'INTT']]
    return mvtx_df, intt_df


def matmul_3D(A, B):
    return np.einsum('lij,ljk->lik', A, B)

def get_approximate_radii(tracks_info, n_hits, good_hits):
    x_indices = [3*j for j in range(5)]
    y_indices = [3*j+1 for j in range(5)]
    r = np.zeros((tracks_info.shape[0], 1))
    centers = np.zeros((tracks_info.shape[0], 2))
    for n_hit in range(3, 5 + 1):
        complete_tracks = tracks_info[n_hits == n_hit]
        hit_indices = good_hits[n_hits == n_hit]
        if complete_tracks.shape[0] == 0:
            continue

        A = np.ones((complete_tracks.shape[0], n_hit, 3))
        x_values = complete_tracks[:, x_indices]
        x_values = x_values[hit_indices].reshape(complete_tracks.shape[0], n_hit)

        y_values = complete_tracks[:, y_indices]
        y_values = y_values[hit_indices].reshape(complete_tracks.shape[0], n_hit)
        A[:, :, 0] = x_values
        A[:, :, 1] = y_values

        y = - x_values**2 - y_values**2
        y = y.reshape((y.shape[0], y.shape[1], 1))
        AT = np.transpose(A, axes=(0, 2, 1))
        c = matmul_3D(matmul_3D(inv(matmul_3D(AT, A)), AT), y)
        r[n_hits == n_hit] = np.sqrt(c[:, 0]**2 + c[:, 1]**2 - 4*c[:, 2])/200
        centers[n_hits == n_hit] = np.concatenate([-c[:, 0]/2, -c[:, 1]/2], axis=-1)

    #test = get_approximate_radius(tracks_info, n_hits == 5)
    #assert np.allclose(test, r[n_hits == 5])

    return r, centers

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
                    a['trigger_track_flag'] = True
                    b['trigger_track_flag'] = True
                    # print(a['OriginVertexPoint'], b['OriginVertexPoint'], ip)
                # print('------------------')
    return flag

def Parse_event(mvtx_cluster_types, filename, output_dir, file_number):
    # print(filename, output_dir, file_number)
    with open(filename,'rb') as z:
        try:
            raw_data = ujson.loads(z.read())
        except Exception as e:
            print('an error occurred when loading the json file!!!! ###################################')
            print(e)
            return

    for event in raw_data['Events']:
        event_ID = event['MetaData']['EventID']
        trigger = int(event['TruthTriggerFlag']['Flags']['D0toPiKInAcceptance'])
        # trigger = 1
        ip = np.array(event["MetaData"]['CollisionVertex'])
        # the output event filename is consist of three part: 
        # the first number represent whether this is a trgger event or not (1: trigger, 0: non-trigger)
        # the second part is the self-defined file_number (to make they different)
        # the thrid part is the eventID in json files. (0-999)
        evtid = 'event%01i%05i%03i'%(trigger, file_number, event_ID)
        
        # preprocessing MVTX hits
        pid_dic = {}
        p_dic = {-1: {'OriginVertexPoint': None, 'TrackMomentum': None, 'TrackEnergy':None, 'ParticleTypeID':None, 'TriggerTrackFlag':None, 'is_complete_trk':None}}
        pid_dic['MVTX'] = {}
        pid_dic['INTT'] = {}

        pos321 = []
        neg321 = []
        pos211 = []
        neg211 = []

        n_MVTXHits = len(event['RawHit']['MVTXHits'])
        n_INTTHits = len(event['RawHit']['INTTHITS'])

        truth_tracks = event['TruthHit']['TruthTracks']
        for track in truth_tracks:
            track['trigger_track_flag'] = False
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
            p_dic[pid] = {'OriginVertexPoint': truth_track['OriginVertexPoint'], 'TrackMomentum': truth_track['TrackMomentum'], 'TrackEnergy': truth_track['TrackEnergy'], 'ParticleTypeID': truth_track['ParticleTypeID'], 'TriggerTrackFlag': truth_track['trigger_track_flag']}
            for hit_id in truth_track['MVTXHitID'][0]:
                pid_dic['MVTX'][hit_id] = pid
            for hit_id in truth_track['INTTHitID'][0]:
                pid_dic['INTT'][hit_id + n_MVTXHits] = pid



        MVTX_hit_dict, INTT_hit_dict = cluster_hits_by_event(mvtx_cluster_types, event, pid_dic, p_dic)
        
        
        p_dic = {}
        complete_flags = []
        track_origin = []
        pids = []
        momentums = []
        energy = []
        ptypes = []
        trigger_track_flags = []
        track_hits = np.zeros((len(truth_tracks), 15))
        n_pixels_per_layer = np.zeros((len(truth_tracks), 5))
        n_hits_per_layer = np.zeros((len(truth_tracks), 5))
        stds = np.zeros((len(truth_tracks), 5, 3))
        # We support up to ten hits, which is excessive 
        cluster_types = -1*np.ones((len(truth_tracks), 5, 10))

        i = 0

        for truth_track in truth_tracks:
            pid = truth_track['TrackSequenceInEvent']
            ptype = truth_track['ParticleTypeID']
            momentum = truth_track['TrackMomentum']
            if momentum[0] ** 2 + momentum[1] ** 2 <= 0.04:
                continue
            p_dic[pid] = {'OriginVertexPoint': truth_track['OriginVertexPoint'], 'TrackMomentum': truth_track['TrackMomentum'], 'TriggerTrackFlag': truth_track['trigger_track_flag']}
            p_dic[pid]['n_MVTX'] = len(truth_track['MVTXHitID'][0])
            p_dic[pid]['n_INTT'] = len(truth_track['INTTHitID'][0])
            
            # layer 0, 1, 2, (3, 4), (5, 6)
            layer_dict = {0:[], 1:[], 2:[], 3:[], 4:[]}
            mvtx_cluster_ids = set()
            intt_cluster_ids = set()
            for hit_id in truth_track['MVTXHitID'][0]:
                hit = MVTX_hit_dict[hit_id]
                if hit['cluster_id'] not in mvtx_cluster_ids:
                    layer_dict[hit['layer_id']].append((hit['x'], hit['y'], hit['z'], hit['n_pixels'], hit['r_std'], hit['phi_std'], hit['z_std'], hit['cluster_type']))
                    mvtx_cluster_ids.add(hit['cluster_id'])
            for hit_id in truth_track['INTTHitID'][0]:
                # The code was originally INTT_hit_dict[hit_id]
                # Adding the + n_MVTXHits was necessary to make it work due to the +n_MVTX_Hits in
                # the preprocessing events function
                hit = INTT_hit_dict[hit_id + n_MVTXHits]
                if hit['cluster_id'] not in intt_cluster_ids:
                    layer_dict[(hit['layer_id']-3)//2 + 3].append((hit['x'], hit['y'], hit['z'], hit['n_pixels'], hit['r_std'], hit['phi_std'], hit['z_std'], hit['cluster_type']))
                    intt_cluster_ids.add(hit['cluster_id'])

            
            # construct track vector
            complete_flag = True
            total_hits = 0
            loopy = False
            for l in range(5):
                if len(layer_dict.get(l)) != 0:
                    h = np.array(layer_dict.get(l))
                    total_pixels = np.sum(h[:, 3])
                    std = np.sum(h[:, 4:7]*h[:, 3:4], axis=0)/total_pixels
                    types, counts = np.unique(h[:, 7], return_counts=True)
                    types = types[np.argsort(counts)[::-1]]

                    n_pixels_per_layer[i, l] = total_pixels
                    track_hits[i, 3*l:(3*l+3)] = np.sum(h[:, 3:4]*h[:, :3], axis=0)/total_pixels
                    n_hits_per_layer[i, l] = len(layer_dict.get(l))
                    stds[i, l] = std
                    n = min(len(cluster_types[i, l]), len(types))
                    assert np.all(types != -1)
                    cluster_types[i, l, :n] = types[np.argsort(counts)[::-1]][:n]
                    total_hits += len(layer_dict.get(l))
                    loopy |= len(layer_dict.get(l)) > 3
                else:
                    complete_flag = False

            loopy = False
            if (total_hits >= 10 or loopy) and truth_track['trigger_track_flag']:
                file = 'event%01i%05i%03i'%(trigger, file_number, event_ID)
                track_momentum = np.sqrt(np.sum(np.array(truth_track['TrackMomentum'])**2))
                track_energy = truth_track['TrackEnergy']
                print(f'{total_hits=} {loopy=} {track_momentum=} {track_energy=} {file=}')

            if (total_hits < 10 and not loopy) or truth_track['trigger_track_flag']:
                complete_flags.append(complete_flag)
                track_origin.append(truth_track['OriginVertexPoint'])
                momentums.append(truth_track['TrackMomentum'])
                pids.append(pid)
                ptypes.append(ptype)
                energy.append(truth_track['TrackEnergy'])
                trigger_track_flags.append(truth_track['trigger_track_flag'])
                i += 1

        track_hits = track_hits[:i]
        n_hits_per_layer = n_hits_per_layer[:i]
        n_pixels_per_layer = n_pixels_per_layer[:i]
        track_origin = np.array(track_origin)
        momentums = np.array(momentums)
        pids = np.array(pids)
        energy = np.array(energy)
        trigger = np.array(trigger)
        valid_trigger_flag = np.array(valid_trigger_flag)
        ptypes = np.array(ptypes)
        stds = stds[:i]
        cluster_types = cluster_types[:i]

        output_dir_event = os.path.join(output_dir, str(int(valid_trigger_flag)), 'event%01i%05i%03i'%(trigger, file_number, event_ID))
        # save to npz file
        np.savez(output_dir_event, 
                track_hits=track_hits.astype(np.float64),
                track_origin=track_origin.astype(np.float64),
                momentum=momentums.astype(np.float64),
                energy=energy.astype(np.float64),
                interaction_point=ip.astype(np.float64),
                trigger=trigger.astype(int),
                n_pixels=n_pixels_per_layer.astype(np.float64),
                has_trigger_pair=valid_trigger_flag.astype(int),
                track_n_hits=n_hits_per_layer.astype(int),
                particle_id=pids.astype(np.float64),
                particle_types=ptypes.astype(np.float64),
                parent_particle_type=np.nan*np.ones_like(pids).astype(np.float64),
                track_hits_cylindrical_std=stds,
                track_hit_types=cluster_types,
            )
        # print(output_dir_event)

MVTX_TYPES_LOCK = None
def main():
    global MVTX_TYPES_LOCK
    """Main function"""
    # DATA_DIR = '/home/yuantian/Data/D0toPiKInAcceptanceSignal_Iteration6'
    # output_dir = 'parsed_D0_Iteration6'
    # DATA_DIR = '/home/zhaozhongshi/HFMLTrigger/NewDataUpdated/Signal'
    # output_dir = 'parsed_trackvec_new/trigger'
    # DATA_DIR = '/home/zhaozhongshi/HFMLTrigger/NewDataUpdated/Background'
    # output_dir = 'parsed_trackvec_new/nontrigger'

 
    TRIGGER_DATA_DIR = '/disks/disk1/tingtingxuan/HFMLNewFiles-old/Signal'
    trigger_output_dir = '/ssd3/giorgian/tracks-data-august-2022-1/trigger'
    # In order to ensure the cluster types dictionary is shared in between them
    NONTRIGGER_DATA_DIR = '/disks/disk1/tingtingxuan/HFMLNewFiles-old/Background'
    nontrigger_output_dir = '/ssd3/giorgian/tracks-data-august-2022/nontrigger'
 
  
    n_workers = 16
    os.makedirs(trigger_output_dir, exist_ok=True)
    os.makedirs(trigger_output_dir+'/0', exist_ok=True)
    os.makedirs(trigger_output_dir+'/1', exist_ok=True)
    os.makedirs(trigger_output_dir+'/empty', exist_ok=True)

    with open(os.path.join(os.path.dirname(trigger_output_dir), 'info.txt'), 'w') as f:
        print(f'trigger_data_dir: {TRIGGER_DATA_DIR}', file=f)
        print(f'trigger_output_dir: {trigger_output_dir}', file=f)
        print(f'nontrigger_data_dir: {NONTRIGGER_DATA_DIR}', file=f)
        print(f'nontrigger_output_dir: {nontrigger_output_dir}', file=f)
 
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
