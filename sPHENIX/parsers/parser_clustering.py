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

# Geometric constraints
PHI_SLOPE_MAX = 0.12
Z0_MAX = 800

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

def cluster_hits_by_event(event):
    hits_dict = {}
    hits_dict['MVTX'] = preprocessing_hits(event, 'MVTX')
    hits_dict['INTT'] = preprocessing_hits(event, 'INTT')
    return hits_dict

def Parse_event(filename, output_dir, file_number):
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
        
        # preprocessing MVTX hits
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
        


        hits_df = cluster_hits_by_event(event)
        if len(hits_df) == 0:
            empty_event += 1
            print('length of hits_df is zero!')
            continue
            # raise KeyboardInterrupt

        mvtx_cartesian = np.array(list(hits_df['MVTX'].keys()))
        intt_cartesian = np.array(list(hits_df['INTT'].keys()))
        

        # save to npz file
        output_dir_event = os.path.join(output_dir, str(int(trigger)), 'event%01i%05i%03i'%(trigger, file_number, event_ID))
        np.savez(output_dir_event, 
                mvtx_cartesian=mvtx_cartesian,
                intt_cartesian=intt_cartesian
        )
                


def main():
    """Main function"""
    # DATA_DIR = '/ssd2/tingting/HFMLNewFiles/Background'
    # output_dir = 'new_parsed_INTTclustered_hits2/nontrigger'
    # DATA_DIR = '/ssd2/tingting/HFMLNewFiles/Signal'
    # output_dir = 'new_parsed_INTTclustered_hits2/trigger'
    # output_dir = 'test/trigger'
    # DATA_DIR = 'json_compare_files'
    # output_dir = 'test/compare'
    DATA_DIR = '/disks/disk1/tingtingxuan/HFMLNewFiles-old/Background'
    output_dir = '/ssd2/giorgian/cluster-data-august-2022/nontrigger'
    
    n_workers = 16
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir+'/0', exist_ok=True)
    os.makedirs(output_dir+'/1', exist_ok=True)
    os.makedirs(output_dir+'/empty', exist_ok=True)
    data_dir = sorted(glob.glob(DATA_DIR + '/*.json'))
    filenumber = 0
    total_number = len(data_dir)
    filenames = data_dir[filenumber:(filenumber+total_number)]
    filenumber =  [i for i in range(filenumber, (filenumber+total_number))]
    # print(filenumber)
    with mp.Pool(processes=n_workers) as pool:
        pool.starmap(Parse_event, zip(filenames, [output_dir]*total_number, filenumber))
    print('Done!')

if __name__ == '__main__':
    main()
