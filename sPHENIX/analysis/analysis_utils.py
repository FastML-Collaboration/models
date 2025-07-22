from collections import namedtuple
import numpy as np
from icecream import ic
from numpy.linalg import inv

UnpackedTrack = namedtuple('UnpackedTrack', 
        'radius radius_per_layer_distances '
        'x_z_m_b x_z_per_layer_distances '
        'y_z_m_b y_z_per_layer_distances '
        'hits '
        'geometric_features'
)

UnpackedFixedTrack = namedtuple('UnpackedTrack', 
        'radius '
        'x_z_m_b '
        'y_z_m_b '
        'hits '
        'geometric_features'
)


RadiusFit = namedtuple('RadiusFit',
        'radius per_layer_distances'
)

LineFit = namedtuple('LineFit',
        'm_b per_layer_distances'
)

CircleFit = namedtuple('CircleFit',
        'radius center')

def noise_track(track_vectors, std_dev, hits_percentage):
    unpacked = unpack_track(track_vectors)
    thits = np.array(unpacked.hits.reshape((unpacked.hits.shape[0], 5, 3)))
    good_hits = np.logical_not(np.all(thits == 0, axis=-1))
    noise = np.random.normal(scale=std_dev, size=thits.shape)
    selected_hits = (np.random.uniform(size=good_hits.shape) > 1-hits_percentage)*good_hits
    thits[selected_hits] += noise[selected_hits]


    thits_center = np.mean(thits[good_hits], axis=0)

    thits = thits.reshape((thits.shape[0], 5*3))
    x_z_m_b, xzpld = get_x_z_m_b(thits)
    y_z_m_b, yzpld = get_x_z_m_b(thits)
    gf = calc_geometric_features(thits, thits_center)

    radius, rpld = get_radius(thits)

    if track_vectors.shape[-1] == 48:
        return pack_track(UnpackedTrack(
            radius=radius,
            radius_per_layer_distances=rpld,
            x_z_m_b=x_z_m_b,
            x_z_per_layer_distances=xzpld,
            y_z_m_b=y_z_m_b,
            y_z_per_layer_distances=yzpld,
            hits=thits,
            geometric_features=gf
        ))
    elif track_vectors.shape[-1] == 33:
         return simple_pack_track(UnpackedFixedTrack(
            radius=radius,
            x_z_m_b=x_z_m_b,
            y_z_m_b=y_z_m_b,
            hits=thits,
            geometric_features=gf
        ))



def rotate_track(track_vectors, theta):
    unpacked = unpack_track(track_vectors)
    R = np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
    ])
    rhits = unpacked.hits.reshape((unpacked.hits.shape[0], 5, 3))
    hits = np.einsum('ij,thj->thi', R, rhits).reshape((rhits.shape[0], 5*3))
    rhits_center = R @ unpacked.geometric_features[0, 10:13]

    x_z_m_b, xzpld = get_x_z_m_b(hits)
    y_z_m_b, yzpld = get_x_z_m_b(hits)
    gf = calc_geometric_features(hits, rhits_center)


    if track_vectors.shape[-1] == 48:
        return pack_track(UnpackedTrack(
            radius=unpacked.radius,
            radius_per_layer_distances=unpacked.radius_per_layer_distances,
            x_z_m_b=x_z_m_b,
            x_z_per_layer_distances=xzpld,
            y_z_m_b=y_z_m_b,
            y_z_per_layer_distances=yzpld,
            hits=hits,
            geometric_features=gf
        ))
    elif track_vectors.shape[-1] == 33:
         return simple_pack_track(UnpackedFixedTrack(
            radius=unpacked.radius,
            x_z_m_b=x_z_m_b,
            y_z_m_b=y_z_m_b,
            hits=hits,
            geometric_features=gf
        ))


def translate_track(track_vectors, z):
    unpacked = unpack_track(track_vectors)
    thits = np.array(unpacked.hits.reshape((unpacked.hits.shape[0], 5, 3)))
    good_hits = np.logical_not(np.all(thits == 0, axis=-1))
    thits[..., -1] += z
    thits[np.logical_not(good_hits)] = 0
    thits_center = np.array(unpacked.geometric_features[0, 10:13])
    thits_center[..., -1] += z

    thits = thits.reshape((thits.shape[0], 5*3))
    x_z_m_b, xzpld = get_x_z_m_b(thits)
    y_z_m_b, yzpld = get_x_z_m_b(thits)
    gf = calc_geometric_features(thits, thits_center)


    if track_vectors.shape[-1] == 48:
        return pack_track(UnpackedTrack(
            radius=unpacked.radius,
            radius_per_layer_distances=unpacked.radius_per_layer_distances,
            x_z_m_b=x_z_m_b,
            x_z_per_layer_distances=xzpld,
            y_z_m_b=y_z_m_b,
            y_z_per_layer_distances=yzpld,
            hits=thits,
            geometric_features=gf
        ))
    elif track_vectors.shape[-1] == 33:
         return simple_pack_track(UnpackedFixedTrack(
            radius=unpacked.radius,
            x_z_m_b=x_z_m_b,
            y_z_m_b=y_z_m_b,
            hits=thits,
            geometric_features=gf
        ))




def unpack_track(track_vectors):
    if track_vectors.shape[-1] == 48:
        cur = 0
        r = track_vectors[:, cur]
        cur += 1
        rpld = track_vectors[:, cur:cur+5]
        cur += 5
        xzmb = track_vectors[:, cur:cur+2]
        cur += 2
        xzpld = track_vectors[:, cur:cur+5]
        cur += 5
        yzmb = track_vectors[:, cur:cur+2]
        cur += 2
        yzpld = track_vectors[:, cur:cur+5]
        cur += 5
        hits = track_vectors[:, cur:cur+15]
        cur += 15
        gf = track_vectors[:, cur:cur+13]
        cur += 13

        return UnpackedTrack(
                radius=r,
                radius_per_layer_distances=rpld,
                x_z_m_b=xzmb,
                x_z_per_layer_distances=xzpld,
                y_z_m_b=yzmb,
                y_z_per_layer_distances=yzpld,
                hits=hits,
                geometric_features=gf
        )
    elif track_vectors.shape[-1] == 33:
        cur = 0
        r = track_vectors[:, cur]
        cur += 1
        xzmb = track_vectors[:, cur:cur+2]
        cur += 2
        yzmb = track_vectors[:, cur:cur+2]
        cur += 2
        hits = track_vectors[:, cur:cur+15]
        cur += 15
        gf = track_vectors[:, cur:cur+13]
        cur += 13

        return UnpackedFixedTrack(
                radius=r,
                x_z_m_b=xzmb,
                y_z_m_b=yzmb,
                hits=hits,
                geometric_features=gf
        )



def pack_track(unpacked_track):
    ut = unpacked_track
    return np.concatenate([
        np.expand_dims(ut.radius, axis=-1),
        ut.radius_per_layer_distances,
        ut.x_z_m_b,
        ut.x_z_per_layer_distances,
        ut.y_z_m_b,
        ut.y_z_per_layer_distances,
        ut.hits,
        ut.geometric_features
    ], axis=-1)
    


def simple_pack_track(unpacked_track):
    ut = unpacked_track
    return np.concatenate([
        np.expand_dims(ut.radius, axis=-1),
        ut.x_z_m_b,
        ut.y_z_m_b,
        ut.hits,
        ut.geometric_features
    ], axis=-1)



def get_radius(hits):
    tracks_info = hits
    hits = hits.reshape((hits.shape[0], 5, 3))
    good_hits = np.all(hits != 0, axis=-1)
    n_hits = np.sum(good_hits, axis=-1)

    x_indices = np.array([3*j for j in range(5)])
    y_indices = np.array([3*j+1 for j in range(5)])
    r = np.zeros(hits.shape[0])
    diffs = np.zeros((hits.shape[0], 5))
    for n_hit in range(3, 5 + 1):
        complete_tracks = tracks_info[n_hits == n_hit]
        complete_hits = hits[n_hits == n_hit]
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
        local_r = np.squeeze(np.sqrt(c[:, 0]**2 + c[:, 1]**2 - 4*c[:, 2])/200, axis=-1)
        r[n_hits == n_hit] = local_r
        center = -c[:, :2, 0]/2
        distances = np.linalg.norm(complete_hits[..., :2] - np.expand_dims(center, axis=1), axis=-1)
        errors = np.abs(distances - np.expand_dims(local_r, axis=-1))
        local_diff = np.zeros((complete_hits.shape[0], 5))
        local_diff[hit_indices] = errors[hit_indices]
        diffs[n_hits == n_hit] = local_diff

    return RadiusFit(radius=r, per_layer_distances=diffs)

def matmul_3D(A, B):
    return np.einsum('lij,ljk->lik', A, B)

def get_x_z_m_b(hits):
    tracks_info = hits
    hits = hits.reshape((hits.shape[0], 5, 3))
    good_hits = np.all(hits != 0, axis=-1)
    n_hits = np.sum(good_hits, axis=-1)

    x_indices = np.array([3*j for j in range(5)])
    y_indices = np.array([3*j+1 for j in range(5)])
    z_indices = np.array([3*j+2 for j in range(5)])
    mbs = np.zeros((hits.shape[0], 2))
    diffs = np.zeros((hits.shape[0], 5))
    for n_hit in range(3, 5 + 1):
        complete_tracks = tracks_info[n_hits == n_hit]
        complete_hits = hits[n_hits == n_hit]
        hit_indices = good_hits[n_hits == n_hit]

        if complete_tracks.shape[0] == 0:
            continue

        A = np.ones((complete_tracks.shape[0], n_hit, 2))
        x_values = complete_tracks[:, x_indices]
        x_values = x_values[hit_indices].reshape(complete_tracks.shape[0], n_hit)

        z_values = complete_tracks[:, z_indices]
        z_values = z_values[hit_indices].reshape(complete_tracks.shape[0], n_hit)
        A[:, :, 0] = x_values
        b = z_values
        xs = []
        for i in range(complete_hits.shape[0]):
            x = np.linalg.lstsq(A[i], b[i], rcond=None)[0]
            xs.append(x)
        local_x = np.stack(xs, axis=0)
        mbs[n_hits == n_hit] = local_x
        errors = b - np.einsum('lij,lj->li', A, local_x)
        local_diff = np.zeros((complete_hits.shape[0], 5))
        local_diff[hit_indices] = errors.flatten()

        diffs[n_hits == n_hit] = local_diff

    return LineFit(m_b=mbs, per_layer_distances=diffs)

def get_y_z_m_b(hits):
    tracks_info = hits
    hits = hits.reshape((hits.shape[0], 5, 3))
    good_hits = np.all(hits != 0, axis=-1)
    n_hits = np.sum(good_hits, axis=-1)

    x_indices = np.array([3*j for j in range(5)])
    y_indices = np.array([3*j+1 for j in range(5)])
    z_indices = np.array([3*j+2 for j in range(5)])
    mbs = np.zeros((hits.shape[0], 2))
    diffs = np.zeros((hits.shape[0], 5))
    for n_hit in range(3, 5 + 1):
        complete_tracks = tracks_info[n_hits == n_hit]
        complete_hits = hits[n_hits == n_hit]
        hit_indices = good_hits[n_hits == n_hit]

        if complete_tracks.shape[0] == 0:
            continue

        A = np.ones((complete_tracks.shape[0], n_hit, 2))
        y_values = complete_tracks[:, y_indices]
        y_values = y_values[hit_indices].reshape(complete_tracks.shape[0], n_hit)

        z_values = complete_tracks[:, z_indices]
        z_values = z_values[hit_indices].reshape(complete_tracks.shape[0], n_hit)
        A[:, :, 0] = y_values
        b = z_values
        xs = []
        for i in range(complete_hits.shape[0]):
            x = np.linalg.lstsq(A[i], b[i])[0]
            xs.append(x)
        local_x = np.stack(xs, axis=0)
        mbs[n_hits == n_hit] = local_x
        errors = b - np.einsum('lij,lj->li', A, local_x)
        local_diff = np.zeros((complete_hits.shape[0], 5))
        local_diff[hit_indices] = errors.flatten()

        diffs[n_hits == n_hit] = local_diff

    return LineFit(m_b=mbs, per_layer_distances=diffs)

def calc_geometric_features(hits, hits_center):
    geo_features = np.zeros((hits.shape[0], 13))
    phi  = np.zeros((hits.shape[0], 5))
    geo_features[:, 5] = np.arctan2(hits[:, 1], hits[:, 0])
    track_vectors = hits
    for i in range(4):
        geo_features[:, i] = get_length(track_vectors[:, (3*i+3):(3*i+6)], track_vectors[:, (3*i):(3*i+3)])
    for i in range(5):
        phi[:, i] = np.arctan2(track_vectors[:, (3*i)+1], track_vectors[:, (3*i)])

    geo_features[:, 5] = get_length(track_vectors[:, 12:15], track_vectors[:, 0:3])
    geo_features[:, 6:10] = np.diff(phi)
    geo_features[:, 10:13] = hits_center
    return geo_features

def get_length(start, end):
    return np.sqrt(np.sum((start - end)**2, axis=1))



def get_circle(hits):
    tracks_info = hits
    hits = hits.reshape((hits.shape[0], 5, 3))
    good_hits = np.all(hits != 0, axis=-1)
    n_hits = np.sum(good_hits, axis=-1)

    x_indices = np.array([3*j for j in range(5)])
    y_indices = np.array([3*j+1 for j in range(5)])
    r = np.zeros(hits.shape[0])
    centers = np.zeros((hits.shape[0], 2))
    diffs = np.zeros((hits.shape[0], 5))
    for n_hit in range(3, 5 + 1):
        complete_tracks = tracks_info[n_hits == n_hit]
        complete_hits = hits[n_hits == n_hit]
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
        local_r = np.squeeze(np.sqrt(c[:, 0]**2 + c[:, 1]**2 - 4*c[:, 2])/200, axis=-1)
        r[n_hits == n_hit] = local_r
        center = -c[:, :2, 0]/2
        centers[n_hits == n_hit] = center
        distances = np.linalg.norm(complete_hits[..., :2] - np.expand_dims(center, axis=1), axis=-1)
        errors = np.abs(distances - np.expand_dims(local_r, axis=-1))
        local_diff = np.zeros((complete_hits.shape[0], 5))
        local_diff[hit_indices] = errors[hit_indices]
        diffs[n_hits == n_hit] = local_diff

    return CircleFit(radius=r, center=centers)

def get_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

path_track_clustered = '/ssd2/tingting/HFMLNewFiles-old-parsed-hits'

# hit-file loader
def load_hit_graph(file):
    with np.load(file, allow_pickle=True) as f:
        # print(list(f.keys()))
        hits = f['hits']
        scaled_hits = f['scaled_hits']
        hits_xyz = f['hits_xyz']
        noise_label = f['noise_label']
        layer_id = f['layer_id']
        edge_index = f['edge_index']
        pid = f['pid']
        n_hits = f['n_hits']
        n_tracks =f ['n_tracks']
        trigger_flag = f['trigger']
        ip = f['ip']
        psv = f['psv'] # secondary vertex
        p_momentum = f['p_momentum']
        trigger_track_flag = f['trigger_track_flag']
    return hits, scaled_hits, hits_xyz, noise_label, layer_id, edge_index, pid, n_hits, n_tracks, trigger_flag, ip, psv, p_momentum, trigger_track_flag

def get_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))
    
def rule0(track, ip): # always True
    return track['TrackID'] != -1

def rule1(track, ip):
    return track['TrackID'] > 0

def rule2(track, ip):
    momentum = track['TrackMomentum']
    return track['TrackID'] != -1 and momentum[0] ** 2 + momentum[1] ** 2 >= 0.04

def rule3(track, ip):
    momentum = track['TrackMomentum']
    track_origin = track['OriginVertexPoint']
    d = get_distance(track_origin, ip)
    return track['TrackID'] != -1 and momentum[0] ** 2 + momentum[1] ** 2 >= 0.04 and d > 0.00001 and d < 1

def load_layerid_and_pid_with_rule(filename, rule_type):
    hits, scaled_hits, hits_xyz, noise_label, layer_id, edge_index, pid, n_hits, n_tracks, trigger_flag, ip, psv, p_momentum, trigger_track_flag = load_hit_graph(filename)
    selected = (pid != -1)
    if rule_type == 0:
        pass
    elif rule_type == 1:
        momentum_mask = [m is not None and m[0] ** 2 + m[1] ** 2 > 0.04 for m in p_momentum]
        selected = np.logical_and(selected, momentum_mask)
    # # elif rule_type == 2:
    #     print(trigger_track_flag, len(trigger_track_flag))
    #     selected = np.logical_and(selected, trigger_track_flag)
    #     selected = np.array(selected, np.bool)
    #     print(selected, len(selected))
    #     print(len(layer_id))
    else:
        print('wrong type!')
    
    return layer_id[selected], pid[selected]

# Percentage of tracks of 1 hit per layer
def percentage_tracks_one_hit_per_layer(num_of_hits_per_layer_by_track):
    return  np.sum(np.all(num_of_hits_per_layer_by_track == 1, axis = 1)) / len(num_of_hits_per_layer_by_track)

# Percentage of tracks of less than or equal to 1 hit per layer
def percentage_tracks_le_one_hit_per_layer(num_of_hits_per_layer_by_track):
    return  np.sum(np.all(num_of_hits_per_layer_by_track <= 1, axis = 1)) / len(num_of_hits_per_layer_by_track)

def getLayer(layer_id):
    if layer_id < 0 or layer_id > 6:
        print('Invalid Layer ID')
    elif layer_id < 3:
        return layer_id
    else:
        return 3 + (layer_id - 3) // 2 

'''
@Output: numpy 2D array of size [n_files, n_rules] where n_files is input and n_rules = 2
'''
def get_percentages_tracks_hit_le_per_layer(n_files = 100, type = '/trigger'):
    n_rules = 2
    n_layers = 5
    percentages_tracks_one_hit_per_layer = np.zeros([n_files, n_rules])
    percentages_tracks_le_one_hit_per_layer = np.zeros([n_files, n_rules])
    i_event = 0

    hits_input_dir = path_track_clustered + type
    filenames = sorted([f for f in os.listdir(hits_input_dir) if f.startswith('event')])[:n_files]

    layer_distribution = np.zeros([n_files, n_rules, n_layers, 4]) # n events, 4 rules, 4 layers, 3 types: == 0, == 1, > 1

    for filename in filenames:
        full_filename = os.path.join(hits_input_dir, filename)
        for rule_number in range(2):
            layer_id, pid = load_layerid_and_pid_with_rule(full_filename, rule_number)
            pid_set = set(set(pid)) # sorted
            dict_track_to_index = {} # key: track; value: index
            dict_track_to_index = dict(zip(set(pid_set), [i for i in range(len(pid_set))]))

            num_of_hits_per_layer_by_track = np.zeros([len(pid_set), n_layers])

            for l, t in list(zip(layer_id, pid)):
                num_of_hits_per_layer_by_track[dict_track_to_index[t]][getLayer(l)] += 1
            
            for i_l in range(n_layers):
                layer_distribution[i_event][rule_number][i_l][3] = len(pid_set)

            for i_t in range(len(pid_set)):
                for i_l in range(n_layers):
                    if num_of_hits_per_layer_by_track[i_t][i_l] == 0:
                        layer_distribution[i_event][rule_number][i_l][0] += 1
                    elif num_of_hits_per_layer_by_track[i_t][i_l] == 1:
                        layer_distribution[i_event][rule_number][i_l][1] += 1
                    else:
                        layer_distribution[i_event][rule_number][i_l][2] += 1

            percentages_tracks_one_hit_per_layer[i_event][rule_number] = percentage_tracks_one_hit_per_layer(num_of_hits_per_layer_by_track)
            percentages_tracks_le_one_hit_per_layer[i_event][rule_number] = percentage_tracks_le_one_hit_per_layer(num_of_hits_per_layer_by_track)
            
        i_event += 1
    return percentages_tracks_le_one_hit_per_layer

'''
Modify distribution matrix for given event at matrix[i_event][i_rule].
It should 2 extra dimension [layer][track_type] to save the counts of tracks of differents types
'''
def get_track_dristribution_by_layers_for_event(full_filename, i_event, layer_distribution):
    n_layers = 5
    n_rule = 2
    for rule_number in range(n_rule):
        layer_id, pid = load_layerid_and_pid_with_rule(full_filename, rule_number)
        pid_set = set(set(pid)) # sorted
        dict_track_to_index = {} # key: track; value: index
        dict_track_to_index = dict(zip(set(pid_set), [i for i in range(len(pid_set))]))

        num_of_hits_per_layer_by_track = np.zeros([len(pid_set), n_layers])

        for l, t in list(zip(layer_id, pid)):
            num_of_hits_per_layer_by_track[dict_track_to_index[t]][getLayer(l)] += 1
        
        for i_l in range(n_layers):
            layer_distribution[i_event][rule_number][i_l][3] = len(pid_set)

        for i_t in range(len(pid_set)):
            for i_l in range(n_layers):
                if num_of_hits_per_layer_by_track[i_t][i_l] == 0:
                    layer_distribution[i_event][rule_number][i_l][0] += 1
                elif num_of_hits_per_layer_by_track[i_t][i_l] == 1:
                    layer_distribution[i_event][rule_number][i_l][1] += 1
                else:
                    layer_distribution[i_event][rule_number][i_l][2] += 1
    return layer_distribution

'''
A convenient method to get num of hit distribution for given files.
'''
def get_track_dristribution_by_layers_for_event_for_files(hits_input_dir, filenames, layer_distribution):
    n_rules = 2
    n_layers = 5
    i_event = 0
    layer_distribution = np.zeros([len(filenames), n_rules, n_layers, 4]) # n events, 4 rules, 4 layers, 3 types: == 0, == 1, > 1

    for filename in filenames:
        full_filename = os.path.join(hits_input_dir, filename)
        get_track_dristribution_by_layers_for_event(full_filename, i_event, layer_distribution)
        i_event += 1
    return layer_distribution

'''
A convenient method to get num of hit distribution for first n files.
'''
def get_track_dristribution_for_layers_for_n_files(n_files = 100, type = '/trigger', save=False):
    n_rules = 2
    n_layers = 5

    hits_input_dir = path_track_clustered + type
    filenames = sorted([f for f in os.listdir(hits_input_dir) if f.startswith('event')])[:n_files]

    layer_distribution = np.zeros([n_files, n_rules, n_layers, 4]) # n events, 4 rules, 4 layers, 3 types: == 0, == 1, > 1

    layer_distribution = get_track_dristribution_by_layers_for_event_for_files(hits_input_dir, filenames, layer_distribution)

    return layer_distribution

def plot_track_dristribution_for_layers(track_dristribution_for_layers, i_rule):
    n_layers = 5
    for i_l in range(n_layers):
        xs = range(len(track_dristribution_for_layers))

        zeros = track_dristribution_for_layers[:, i_rule, i_l, 0] / track_dristribution_for_layers[:, i_rule, i_l, 3]
        ones = track_dristribution_for_layers[:, i_rule, i_l, 1] / track_dristribution_for_layers[:, i_rule, i_l, 3]
        other = track_dristribution_for_layers[:, i_rule, i_l, 2] / track_dristribution_for_layers[:, i_rule, i_l, 3]

        plt.plot([],[],color='salmon', label='zero', linewidth=3)
        plt.plot([],[],color='whitesmoke', label='one', linewidth=3)
        plt.plot([],[],color='deepskyblue', label='other', linewidth=3)
        plt.stackplot(xs, zeros, ones, other, colors=['salmon','whitesmoke','deepskyblue'])

        plt.xlabel('Event')
        plt.ylabel('Percentage')
        plt.title('Distribution of # Hits per Track - Layer ' + str(i_l+1))
        plt.legend(loc=7)
        plt.savefig('plots/layer'+str(i_l))
        plt.show()