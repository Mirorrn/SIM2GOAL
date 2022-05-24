import logging
import os
import math
# from IPython import embed
import pickle
from collections import defaultdict
import itertools
import json
import random

import numpy as np

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)
from config import *
config = Config()
from data.augmenter import Transformer

from collections import namedtuple


TrackRow = namedtuple('Row', ['frame', 'pedestrian', 'x', 'y', 'prediction_number', 'scene_id'])
TrackRow.__new__.__defaults__ = (None, None, None, None, None, None)
SceneRow = namedtuple('Row', ['scene', 'pedestrian', 'start', 'end', 'fps', 'tag'])
SceneRow.__new__.__defaults__ = (None, None, None, None, None, None)

class Reader(object):
    """Read trajnet files.
    :param scene_type: None -> numpy.array, 'rows' -> TrackRow and SceneRow, 'paths': grouped rows (primary pedestrian first), 'tags': numpy.array and scene tag
    :param image_file: Associated image file of the scene
    """
    def __init__(self, input_file, scene_type=None, image_file=None):
        if scene_type is not None and scene_type not in {'rows', 'paths', 'tags'}:
            raise Exception('scene_type not supported')
        self.scene_type = scene_type

        self.tracks_by_frame = defaultdict(list)
        self.scenes_by_id = dict()

        self.read_file(input_file)

    def read_file(self, input_file):
        with open(input_file, 'r') as f:
            for line in f:
                line = json.loads(line)

                track = line.get('track')
                if track is not None:
                    row = TrackRow(track['f'], track['p'], track['x'], track['y'], \
                                   track.get('prediction_number'), track.get('scene_id'))
                    self.tracks_by_frame[row.frame].append(row)
                    continue

                scene = line.get('scene')
                if scene is not None:
                    row = SceneRow(scene['id'], scene['p'], scene['s'], scene['e'], \
                                   scene.get('fps'), scene.get('tag'))
                    self.scenes_by_id[row.scene] = row

    def scenes(self, randomize=False, limit=0, ids=None, sample=None):
        scene_ids = self.scenes_by_id.keys()
        if ids is not None:
            scene_ids = ids
        if randomize:
            scene_ids = list(scene_ids)
            random.shuffle(scene_ids)
        if limit:
            scene_ids = itertools.islice(scene_ids, limit)
        if sample is not None:
            scene_ids = random.sample(scene_ids, int(len(scene_ids) * sample))
        for scene_id in scene_ids:
            yield self.scene(scene_id)

    @staticmethod
    def track_rows_to_paths(primary_pedestrian, track_rows):
        primary_path = []
        other_paths = defaultdict(list)
        for row in track_rows:
            if row.pedestrian == primary_pedestrian:
                primary_path.append(row)
                continue
            other_paths[row.pedestrian].append(row)

        return [primary_path] + list(other_paths.values())

    @staticmethod
    def paths_to_xy(paths):
        """Convert paths to numpy array with nan as blanks."""
        frames = set(r.frame for r in paths[0])
        pedestrians = set(row.pedestrian
                          for path in paths
                          for row in path if row.frame in frames)
        paths = [path for path in paths if path[0].pedestrian in pedestrians]
        frames = sorted(frames)
        pedestrians = list(pedestrians)

        frame_to_index = {frame: i for i, frame in enumerate(frames)}
        xy = np.full((len(frames), len(pedestrians), 2), np.nan)

        for ped_index, path in enumerate(paths):
            for row in path:
                if row.frame not in frame_to_index:
                    continue
                entry = xy[frame_to_index[row.frame]][ped_index]
                entry[0] = row.x
                entry[1] = row.y

        return xy

    def scene(self, scene_id):
        scene = self.scenes_by_id.get(scene_id)
        if scene is None:
            raise Exception('scene with that id not found')

        frames = range(scene.start, scene.end + 1)
        track_rows = [r
                      for frame in frames
                      for r in self.tracks_by_frame.get(frame, [])]

        # return as rows
        if self.scene_type == 'rows':
            return scene_id, scene.pedestrian, track_rows

        # return as paths
        paths = self.track_rows_to_paths(scene.pedestrian, track_rows)
        if self.scene_type == 'paths':
            return scene_id, paths

        ## return with scene tag
        if self.scene_type == 'tags':
            return scene_id, scene.tag, self.paths_to_xy(paths)

        # return a numpy array
        return scene_id, self.paths_to_xy(paths)

def prepare_data(path, subset='/train/', sample=1.0, goals=True):
    """ Prepares the train/val scenes and corresponding goals

    Parameters
    ----------
    subset: String ['/train/', '/val/']
        Determines the subset of data to be processed
    sample: Float (0.0, 1.0]
        Determines the ratio of data to be sampled
    goals: Bool
        If true, the goals of each track are extracted
        The corresponding goal file must be present in the 'goal_files' folder
        The name of the goal file must be the same as the name of the training file

    Returns
    -------
    all_scenes: List
        List of all processed scenes
    all_goals: Dictionary
        Dictionary of goals corresponding to each dataset file.
        None if 'goals' argument is False.
    Flag: Bool
        True if the corresponding folder exists else False.
    """

    ## Check if folder exists
    if not os.path.isdir(path + subset):
        if 'train' in subset:
            print("Train folder does NOT exist")
            exit()
        if 'val' in subset:
            print("Validation folder does NOT exist")
            return None, None, False

    ## read goal files
    all_goals = {}
    all_scenes = []

    ## List file names
    files = [f.split('.')[-2] for f in os.listdir(path + subset) if f.endswith('.ndjson')]
    ## Iterate over file names
    for file in files:
        reader = Reader(path + subset + file + '.ndjson', scene_type='paths')
        ## Necessary modification of train scene to add filename
        scene = [(file, s_id, s) for s_id, s in reader.scenes(sample=sample)]
        if goals:
            goal_dict = pickle.load(open('goal_files/' + subset + file + '.pkl', "rb"))
            ## Get goals corresponding to train scene
            all_goals[file] = {s_id: [goal_dict[path[0].pedestrian] for path in s] for _, s_id, s in scene}
        all_scenes += scene

    if goals:
        return all_scenes, all_goals, True
    return all_scenes, None, True


def read_file(_path, delim="\t"):
    data = []
    if delim == "tab":
        delim = "\t"
    elif delim == "space":
        delim = " "
    with open(_path, "r") as f:
        for line in f:
            line = line.strip().split(delim)
            line = [float(i) for i in line]
            data.append(line)
    return np.asarray(data)


def drop_distant(xy, r=6.0):
    """
    Drops pedestrians more than r meters away from primary ped
    """
    distance_2 = np.sum(np.square(xy - xy[:, 0:1]), axis=2)
    mask = np.nanmin(distance_2, axis=0) < r**2
    return xy[:, mask], mask


class TrajectoryDatasetTrajnet(Dataset):
    """Dataloder for the Trajectory datasets"""

    def __init__(
            self,
            data_dir,
            save_dir,
            obs_len=9,
            pred_len=12,
            skip=1,
            threshold=0.002,
            min_ped=1,
            delim="\t",
    ):
        """
        Args:
        - data_dir: Directory containing dataset files in the format
        <frame_id> <ped_id> <x> <y>
        - obs_len: Number of time-steps in input trajectories
        - pred_len: Number of time-steps in output trajectories
        - skip: Number of frames to skip while making the dataset
        - threshold: Minimum error to be considered for non linear traj
        when using a linear predictor
        - min_ped: Minimum number of pedestrians that should be in a seqeunce
        - delim: Delimiter in the dataset files
        """
        super(TrajectoryDatasetTrajnet, self).__init__()

        self.data_dir = data_dir
        self.save_dir = save_dir
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.skip = skip
        self.seq_len = self.obs_len + self.pred_len
        self.delim = delim
        # Split Test=3200, Train=12800   , val=3200
        data_scenes, _, _ = prepare_data(data_dir, subset='/', sample=1., goals=None)
        Test_scenes = data_scenes[0:3200]
        Train_scenes = data_scenes[3200:3200 + 12800]
        Val_scenes = data_scenes[3200 + 12800:12800 + 2*3200]
        self.create_pickl_file(Test_scenes, 'synth', self.save_dir + 'test')
        self.create_pickl_file(Train_scenes, 'synth', self.save_dir + 'train')
        self.create_pickl_file(Val_scenes, 'synth', self.save_dir + 'val')

    def create_pickl_file(self, data_scenes, dataset_name, save_dir_split):
        # all_files = os.listdir(self.data_dir)
        # all_files = [os.path.join(self.data_dir, _path) for _path in all_files]
        num_peds_in_seq = []
        seq_list = []
        loss_mask_list = []
        loss_mask_rel = []
        val_loss_mask = []
        # if not os.path.isdir(data_dir + subset):
        # dataset_name = data_dir.split('/')[-2]

        for scene_i, (filename, scene_id, paths) in enumerate(data_scenes):
            scene = Reader.paths_to_xy(paths)
            ## Drop Distant
            scene, mask = drop_distant(scene)
            ## Drop partial observed, ugly but works!
            scene[np.isnan(scene)] = np.inf
            mask = np.sum(scene, axis=0)
            scene = scene[:,mask[:, 0] != np.inf]
            assert mask[0,0] != np.inf
            num_peds_in_seq.append(scene.shape[1])
            scene = scene.transpose((1,2,0))
            loss_msk = np.ones_like(scene)
            rel_mask = np.ones_like(scene)
            rel_mask[:,:,0] = 0
            loss_mask_rel.append(rel_mask)
            loss_mask_list.append(loss_msk)
            seq_list.append(scene)
            val_loss_mask.append(loss_msk)

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        loss_mask = np.concatenate(loss_mask_list, axis=0)
        self.loss_maks_rel = np.concatenate(loss_mask_rel, axis=0)
        val_loss_mask = np.concatenate(val_loss_mask, axis=0)
        # non_linear_ped = np.asarray(non_linear_ped)
        self.obs_traj = seq_list[:, :, : self.obs_len]
        self.pred_traj = seq_list[:, :, self.obs_len :]
        self.loss_mask = loss_mask
        self.val_mask = val_loss_mask
        cum_start_idx = [0] + np.cumsum(num_peds_in_seq).tolist()
        self.seq_start_end = [
            (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]
        with open(save_dir_split + '/' + dataset_name + '.pkl', 'wb') as handle:
            pickle.dump([self.num_seq, self.loss_maks_rel, self.obs_traj, self.pred_traj, self.loss_mask,
                         self.val_mask, self.seq_start_end], handle)
        print('Data saved to:' + save_dir_split + '/' + dataset_name + '.pkl')


    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):
        start, end = self.seq_start_end[index]
        out = [
            self.obs_traj[start:end, :],
            self.pred_traj[start:end, :],
            self.val_mask[start:end, :],
            self.loss_mask[start:end, :],
            self.loss_maks_rel[start:end, :],
        ]
        return out

_dir = os.path.dirname(__file__)
_dir = _dir.split("/")[:-2]
_dir = "/".join(_dir) + "/Sim2Goal"
dir = os.path.join(_dir, 'datasets/trajnet', 'orca', 'test')
save_dir_pre = os.path.join(_dir, 'datasets/trajnet', 'synth/')
# save_dir_pre = '/home/WIN-UNI-DUE/adi205v/NFTraj/trajnetdataset/synth/'
TrajectoryDatasetTrajnet(data_dir=dir,save_dir=save_dir_pre)