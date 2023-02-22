import os
import json
import numpy as np
import torch
import platform

from random import randint
from copy import deepcopy
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from pytorchvideo import transforms as T

from torch.utils.data import DataLoader
from data_transform import pad_frames, RandomVerticalFlip, RandomHorizontalFlip


anomalies = [
    'start_stop_or_stationary',  # 1 ST
    'moving_ahead_or_waiting',  # 2 AH
    'lateral',  # 3 LA
    'oncoming',  # 4 OC
    'turning',  # 5 TC
    'pedestrian',  # 6 VP
    'obstacle',  # 7 VO
    'leave_to_right',  # 8 OO-r
    'leave_to_left',  # 9 OO-l
    'unknown',  # 10 UK
]


def read_file(path):
    return np.asarray(Image.open(path))


def has_objects(ann):
    return sum([len(labels['objects']) for labels in ann['labels']]) != 0


def gt_cls_target(curtime_batch, toa_batch, tea_batch):
    return (
        (toa_batch >= 0) &
        (curtime_batch >= toa_batch) & (
            (curtime_batch < tea_batch) |
            # case when sub batch end with a positive frame
            (toa_batch == tea_batch)
        )
    )


class AnomalySubBatch(object):
    def __init__(self, dota, index):
        key = dota.keys[index]
        num_frames = dota.metadata[key]['num_frames']
        self.begin, self.end = dota._get_random_subbatch(num_frames)
        # negative case
        if self.end >= dota.metadata[key]['anomaly_start'] and \
                self.begin <= dota.metadata[key]['anomaly_end']:
            self.label = 1
            self.a_start = max(
                0, dota.metadata[key]['anomaly_start'] - self.begin
            )
            self.a_end = min(
                dota.metadata[key]['anomaly_end'] - self.begin,
                self.end - self.begin
            )
        else:
            self.label = -1
            self.a_start = -1
            self.a_end = -1


class Dota(Dataset):
    def __init__(
            self, root_path, phase,
            transforms={'image': None},
            VCL=None,
            vertical_flip_prob=0., horizontal_flip_prob=0.):
        self.root_path = root_path
        self.phase = phase  # 'train', 'test', 'play'
        self.transforms = transforms
        self.fps = 10
        self.VCL = VCL

        if vertical_flip_prob > 0.:
            self.vflipper = RandomVerticalFlip(vertical_flip_prob)

        if horizontal_flip_prob > 0.:
            self.hflipper = RandomHorizontalFlip(horizontal_flip_prob)

        self.get_data_list()

    def get_data_list(self):
        list_file = os.path.join(
            self.root_path, 'metadata', 'metadata_{}.json'.format(self.phase))
        assert os.path.exists(list_file), \
            "File does not exist! %s" % (list_file)
        with open(list_file, 'r') as f:
            self.metadata = json.load(f)

        # load annotations
        self._load_anns()
        # clean wrong metadata and reload
        self._filter_wrong_metadata()
        self._load_anns()

    def _load_anns(self):
        self.keys = list(self.metadata.keys())
        self.annotations = []
        for key in self.keys:
            self.annotations.append(self._load_ann(key))

    def _filter_wrong_metadata(self):
        # check if #ann != #images
        metadata = deepcopy(self.metadata)
        for index in range(len(self.metadata)):
            ann = self.annotations[index]
            video_file = self.keys[index]
            frames_dir = os.path.join(
                self.root_path, 'frames', video_file, 'images')
            count_files = len(os.listdir(frames_dir))
            count_ann = len(ann['labels'])
            count_meta = self.metadata[video_file]['num_frames']
            if count_ann != count_files or count_files != count_meta \
                    or count_ann != count_meta:
                # remove wrong!
                del metadata[video_file]
                #  del self.annotations[index]
        print('removed {} videos'.format(len(self.metadata) - len(metadata)))
        self.metadata = metadata

    def _load_ann(self, key):
        ann_file = os.path.join(
            self.root_path, 'annotations', '{}.json'.format(key))
        with open(ann_file, 'r') as f:
            ann = json.load(f)
        return ann

    def __len__(self):
        return len(self.metadata)

    def _get_random_subbatch(self, count):
        if self.VCL is None:
            return 0, count
        else:
            # if video is small then VCL, return full video
            if count <= self.VCL:
                return 0, count
            max_ = count - self.VCL
            begin = randint(0, max_)
            end = begin + self.VCL
            return begin, end

    def _add_video_filler(self, frames):
        try:
            filler_count = self.VCL - len(frames)
        except TypeError:
            return frames
        if filler_count > 0:
            filler = np.full((filler_count,) + frames.shape[1:], 0)
            frames = np.concatenate((frames, filler), axis=0)
        return frames

    def read_video(self, index, sub_batch):
        """Read video frames."""
        video_file = self.keys[index]
        label = sub_batch.label

        frames_dir = os.path.join(
            self.root_path, 'frames', video_file, 'images')
        frames = []
        names = []
        frames = sorted(os.listdir(frames_dir))
        for index, name in enumerate(frames[sub_batch.begin:sub_batch.end]):
            # filter accident frames if it is a negative example
            if label == 1 or index < sub_batch.a_start or \
                    index > sub_batch.a_end:
                path = os.path.join(frames_dir, name)
                names.append(path)
        frames = np.array(list(map(read_file, names)))
        video_len_orig = len(frames)
        frames = self._add_video_filler(frames)
        return frames.astype('float32'), video_len_orig

    def gather_info(self, index, sub_batch, video_len_orig):
        ann = self.annotations[index]
        label = sub_batch.label
        return np.array([
            video_len_orig,
            self.keys.index(ann['video_name']),
            sub_batch.a_start,
            sub_batch.a_end,
            label,
            # min value because of reward formula
            sub_batch.a_start,
            sub_batch.a_end,
            ann['accident_id'],
            int(ann['ego_involve']),
            int(ann['night']),
            int(has_objects(ann)),
        ]).astype('float')

    def __getitem__(self, index):
        # init sub_batch
        sub_batch = AnomalySubBatch(self, index)
        # read RGB video (trimmed)
        video_data, video_len_orig = self.read_video(index, sub_batch)
        # gather info
        data_info = self.gather_info(index, sub_batch, video_len_orig)

        # pre-process
        if self.transforms['image'] is not None:
            video_data = self.transforms['image'](video_data)  # (T, C, H, W)

        if hasattr(self, 'hflipper'):
            _, video_data = self.hflipper(video_data)

        if hasattr(self, 'vflipper'):
            _, video_data = self.vflipper(video_data)

        return video_data, data_info


def setup_dota(Dota, cfg, num_workers=-1,
               VCL=None, phase=None):
    mean = cfg.get('data_mean', [0.218, 0.220, 0.209])
    std = cfg.get('data_std', [0.277, 0.280, 0.277])
    params = {
        'input_shape': cfg.input_shape,
        'mean': mean,
        'std': std,
    }

    vertical_flip_prob = cfg.get('vertical_flip_prob', 0.)
    horizontal_flip_prob = cfg.get('horizontal_flip_prob', 0.)

    #  def transf_train(x):
    transform_dict = {
        'image': transforms.Compose([
            pad_frames(cfg.input_shape),
            transforms.Lambda(lambda x: torch.tensor(x)),
            # [T, H, W, C] -> [T, C, H, W]
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Normalize(params['mean'], params['std']),
            # [T, C, H, W]
        ]),
    }

    transform_dict_train = {
        'image': transforms.Compose([
            pad_frames(cfg.input_shape),
            transforms.Lambda(lambda x: torch.tensor(x)),
            # [T, H, W, C] -> [T, C, H, W]
            transforms.Lambda(lambda x: x.permute(0, 3, 1, 2)),
            T.AugMix(),
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Normalize(params['mean'], params['std']),
            # [T, C, H, W]
        ]),
    }

    transform_dict_to_play = {
        'image': None,
    }

    traindata_loader, testdata_loader = None, None

    # training dataset
    pin_memory_val = True
    if platform.system() == 'Windows':
        pin_memory_val = False

    # testing dataset
    if phase == 'train':
        train_data = Dota(
            cfg.data_path, 'train',
            transforms=transform_dict_train,
            VCL=VCL,
            vertical_flip_prob=vertical_flip_prob,
            horizontal_flip_prob=horizontal_flip_prob)
        traindata_loader = DataLoader(
            dataset=train_data, batch_size=cfg.batch_size,
            shuffle=True, drop_last=True, num_workers=num_workers,
            pin_memory=pin_memory_val)
        print("# train set: {}".format(len(train_data)))
        cfg.update(FPS=train_data.fps)
    else:
        if phase == 'test':
            # validation dataset
            test_data = Dota(cfg.data_path, 'val', transforms=transform_dict)
            testdata_loader = DataLoader(
                dataset=test_data, batch_size=1, shuffle=False,
                drop_last=True, num_workers=num_workers,
                pin_memory=pin_memory_val)
            print("# test set: %d" % (len(test_data)))
        elif phase == 'play':
            # validation dataset
            test_data = Dota(
                cfg.data_path, 'val',
                transforms=transform_dict_to_play)
            testdata_loader = DataLoader(
                dataset=test_data, batch_size=1, shuffle=False,
                drop_last=True, num_workers=num_workers,
                pin_memory=pin_memory_val)
            print("# test set: %d" % (len(test_data)))
        cfg.update(FPS=test_data.fps)

    return traindata_loader, testdata_loader
