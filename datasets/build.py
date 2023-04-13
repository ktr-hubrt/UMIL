from logging import Logger
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch
import numpy as np
from functools import partial
import random

import io
import os
import os.path as osp
import shutil
import warnings
from collections.abc import Mapping, Sequence
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import Dataset
import copy
import os.path as osp
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict
import os.path as osp
import mmcv
import numpy as np
import torch
import math
import tarfile
from .pipeline import *
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from mmcv.parallel import collate
import pandas as pd

PIPELINES = Registry('pipeline')
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)


class RawFramesTestRecord(object):
    def __init__(self, row, temp_label=None):
        self._data = row
        self.temp_label = temp_label

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        if int(self._data[1])==0:
            return int(self._data[2])
        else:
            return int(self._data[1])

    @property
    def label(self):
        if len(self._data) == 15:
            label = np.zeros((32, 1))
            class_onehot = []
            for i in range(2, len(self._data)):
                class_onehot.append(int(self._data[i]))
            if self.temp_label is None:
                return label
            else:
                for j in range(len(self.temp_label)):
                    start = round(self.temp_label[j][0] * 32.0 / self.num_frames)
                    end = round(self.temp_label[j][1] * 32.0 / self.num_frames)
                    label[start:end] += 1

                    if start > 0:
                        label[start - 1] = 2
                    if end < 32:
                        label[end] = 2
                    if end < 31:
                        label[end - 1] = 2
                    label[start] = 2

            return label
        else:
            if len(self._data) > 2 and int(self._data[1])>0:
                return int(self._data[2])
            elif len(self._data) > 2 and int(self._data[1])==0:
                return int(self._data[3])
            else:
                return 0


class BaseDataset(Dataset, metaclass=ABCMeta):
    def __init__(self,
                 ann_file,
                 pipeline,
                 repeat = 1,
                 pipeline_=None,
                 data_prefix=None,
                 test_mode=False,
                 multi_class=False,
                 num_classes=None,
                 start_index=1,
                 modality='RGB',
                 sample_by_class=False,
                 filename_tmpl='img_{:08}.jpg',
                 seg_interval=30,
                 power=0,
                 dynamic_length=False,):
        super().__init__()
        self.use_tar_format = True if ".tar" in data_prefix else False
        data_prefix = data_prefix.replace(".tar", "")
        self.ann_file = ann_file
        self.repeat = repeat
        self.data_prefix = osp.realpath(
            data_prefix) if data_prefix is not None and osp.isdir(
                data_prefix) else data_prefix
        self.filename_tmpl = filename_tmpl
        self.test_mode = test_mode
        self.multi_class = multi_class
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        self.sample_by_class = sample_by_class
        self.power = power
        self.seg_interval = seg_interval
        self.dynamic_length = dynamic_length

        assert not (self.multi_class and self.sample_by_class)

        self.pipeline = Compose(pipeline)
        if pipeline_ is not None:
            self.pipeline_ = Compose(pipeline_)
            self.repeat += 1
        self.video_infos = self.load_annotations()
        if self.sample_by_class:
            self.video_infos_by_class = self.parse_by_class()

            class_prob = []
            for _, samples in self.video_infos_by_class.items():
                class_prob.append(len(samples) / len(self.video_infos))
            class_prob = [x**self.power for x in class_prob]

            summ = sum(class_prob)
            class_prob = [x / summ for x in class_prob]

            self.class_prob = dict(zip(self.video_infos_by_class, class_prob))

    @abstractmethod
    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""

    # json annotations already looks like video_infos, so for each dataset,
    # this func should be the same
    def load_json_annotations(self):
        """Load json annotation file to get video information."""
        video_infos = mmcv.load(self.ann_file)
        num_videos = len(video_infos)
        path_key = 'frame_dir' if 'frame_dir' in video_infos[0] else 'filename'
        for i in range(num_videos):
            path_value = video_infos[i][path_key]
            if self.data_prefix is not None:
                path_value = osp.join(self.data_prefix, path_value)
            video_infos[i][path_key] = path_value
            if self.multi_class:
                assert self.num_classes is not None
            else:
                assert len(video_infos[i]['label']) == 1
                video_infos[i]['label'] = video_infos[i]['label'][0]
        return video_infos

    def parse_by_class(self):
        video_infos_by_class = defaultdict(list)
        for item in self.video_infos:
            label = item['label']
            video_infos_by_class[label].append(item)
        return video_infos_by_class

    @staticmethod
    def label2array(num, label):
        arr = np.zeros(num, dtype=np.float32)
        arr[label] = 1.
        return arr

    @staticmethod
    def dump_results(results, out):
        """Dump data to json/yaml/pickle strings or files."""
        return mmcv.dump(results, out)

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['filename_tmpl'] = self.filename_tmpl
        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        aug1 = self.pipeline(results)
        # import pdb;pdb.set_trace()
        if self.repeat > 1:
            aug2 = self.pipeline_(results)
            ret = {"imgs": torch.stack((aug1['imgs'], aug2['imgs']), 0),
                   "label": aug1['label'].repeat(2),
                   "vid": aug1['vid'],
                   'frame_inds': aug1['frame_inds'],
                   'total_frames': aug1['total_frames'],
                       }
            return ret
        else:
            return aug1

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.video_infos[idx])
        results['modality'] = self.modality
        results['start_index'] = self.start_index

        # prepare tensor in getitem
        # If HVU, type(results['label']) is dict
        if self.multi_class and isinstance(results['label'], list):
            onehot = torch.zeros(self.num_classes)
            onehot[results['label']] = 1.
            results['label'] = onehot

        return self.pipeline(results)

    def __len__(self):
        """Get the size of the dataset."""
        return len(self.video_infos)

    def __getitem__(self, idx):
        """Get the sample for either training or testing given index."""
        if self.test_mode:
            return self.prepare_test_frames(idx)

        return self.prepare_train_frames(idx)


class FrameDataset(BaseDataset):
    def __init__(self, ann_file, pipeline, labels_file, start_index=0, **kwargs):
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)
        self.labels_file = labels_file

    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        vid = 0
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                if self.multi_class:
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(int, label))
                else:
                    if len(line_split) == 4:
                        filename, start, end, label = line_split
                    elif len(line_split) == 5:
                        filename, end, label, _, _ = line_split
                        start = 0
                    else:
                        filename, end, label = line_split[:3]
                        start = 0
                    label = int(label)
                if self.data_prefix is not None and self.data_prefix not in filename:
                    filename = osp.join(self.data_prefix, filename)
                video_infos.append(dict(frame_dir=filename, label=label, total_frames=int(end)-int(start), tar=self.use_tar_format, vid=vid))
                vid += 1
        return video_infos


class RawFramesTestDataset(BaseDataset):
    def __init__(self, ann_file, pipeline, labels_file, start_index=0, **kwargs):
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)
        self.labels_file = labels_file

    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()

    def load_annotations(self):
        segs = []
        path_list = []
        vid = 0
        for x in open(self.ann_file):
            video_info = RawFramesTestRecord(x.strip().split(' '))
            path_list.append(video_info.path)
            num_segs = math.ceil(video_info.num_frames / self.seg_interval)
            for i in range(num_segs):
                start = self.seg_interval * i
                end = min(self.seg_interval * (i + 1), video_info.num_frames)
                if end - start < 5:
                    continue
                filename = video_info.path
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                seg = dict(frame_dir=filename, label=video_info.label, start=int(start), total_frames=int(end)-int(start), tar=self.use_tar_format, vid=vid)
                segs.append(seg)
            vid += 1
        return segs


class VideoDataset(BaseDataset):
    def __init__(self, ann_file, pipeline, labels_file, start_index=0, **kwargs):
        super().__init__(ann_file, pipeline, start_index=start_index, **kwargs)
        self.labels_file = labels_file

    @property
    def classes(self):
        classes_all = pd.read_csv(self.labels_file)
        return classes_all.values.tolist()

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()

        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                import pdb;
                pdb.set_trace()
                if self.multi_class:
                    assert self.num_classes is not None
                    filename, label = line_split[0], line_split[1:]
                    label = list(map(int, label))
                else:
                    filename, label = line_split
                    label = int(label)
                if self.data_prefix is not None:
                    filename = osp.join(self.data_prefix, filename)
                video_infos.append(dict(filename=filename, label=label, tar=self.use_tar_format))

        return video_infos


class SubsetRandomSampler(torch.utils.data.Sampler):
    r"""Samples elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.epoch = 0
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in torch.randperm(len(self.indices)))

    def __len__(self):
        return len(self.indices)

    def set_epoch(self, epoch):
        self.epoch = epoch


def mmcv_collate(batch, samples_per_gpu=1): 
    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')
    if isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        return {
            key: mmcv_collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)


def build_dataloader(logger, config):
    scale_resize = int(256 / 224 * config.DATA.INPUT_SIZE)

    train_pipeline = [
        dict(type='SampleFrames', clip_len=config.DATA.NUM_FRAMES, frame_interval=config.DATA.FRAME_INTERVAL, num_clips=config.DATA.NUM_CLIPS),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(type='CenterCrop', crop_size=config.DATA.INPUT_SIZE),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs', 'label', 'vid', 'frame_inds', 'total_frames'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label']),
    ]

    train_pipeline_S = [
        dict(type='SampleFrames', clip_len=config.DATA.NUM_FRAMES, frame_interval=config.DATA.FRAME_INTERVAL,
             num_clips=config.DATA.NUM_CLIPS),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(
            type='MultiScaleCrop',
            input_size=config.DATA.INPUT_SIZE,
            scales=(1, 0.875, 0.75, 0.66),
            random_crop=False,
            max_wh_scale_gap=1),
        dict(type='Resize', scale=(config.DATA.INPUT_SIZE, config.DATA.INPUT_SIZE), keep_ratio=False),
        dict(type='ColorJitter', p=config.AUG.COLOR_JITTER),
        dict(type='GrayScale', p=config.AUG.GRAY_SCALE),
        dict(type='RandAugment', auto_augment='rand-n{}-m{}-mstd0.5'.format(2, 10)),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs', 'label', 'vid', 'frame_inds', 'total_frames'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs', 'label']),
    ]
        

    train_data = FrameDataset(ann_file=config.DATA.TRAIN_FILE, data_prefix=config.DATA.ROOT,
                              filename_tmpl=config.DATA.FILENAME_TMPL, labels_file=config.DATA.LABEL_LIST,
                              pipeline=train_pipeline, pipeline_=train_pipeline_S)
    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        train_data, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    train_loader = DataLoader(
        train_data, sampler=sampler_train,
        batch_size=config.TRAIN.BATCH_SIZE,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(mmcv_collate, samples_per_gpu=config.TRAIN.BATCH_SIZE),
    )
    train_loader_umil = DataLoader(
        train_data, sampler=sampler_train,
        batch_size=config.TRAIN.BATCH_SIZE_UMIL,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=partial(mmcv_collate, samples_per_gpu=config.TRAIN.BATCH_SIZE_UMIL),
    )
    
    val_pipeline = [
        dict(type='SampleFrames', clip_len=config.DATA.NUM_FRAMES, frame_interval=config.DATA.FRAME_INTERVAL, num_clips=config.DATA.NUM_CLIPS, test_mode=True),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(type='CenterCrop', crop_size=config.DATA.INPUT_SIZE),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs', 'label', 'vid'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]

    if config.TEST.NUM_CROP == 3:
        val_pipeline[3] = dict(type='Resize', scale=(-1, config.DATA.INPUT_SIZE))
        val_pipeline[4] = dict(type='ThreeCrop', crop_size=config.DATA.INPUT_SIZE)
    if config.TEST.NUM_CLIP > 1:
        val_pipeline[0] = dict(type='SampleFrames', clip_len=config.DATA.NUM_FRAMES, frame_interval=config.DATA.FRAME_INTERVAL, num_clips=config.DATA.NUM_CLIPS, multiview=config.TEST.NUM_CLIP)

    val_data = FrameDataset(ann_file=config.DATA.VAL_FILE, data_prefix=config.DATA.ROOT, labels_file=config.DATA.LABEL_LIST, filename_tmpl=config.DATA.FILENAME_TMPL, pipeline=val_pipeline)

    sampler_val = torch.utils.data.SequentialSampler(val_data)
    val_loader = DataLoader(
        val_data, sampler=sampler_val,
        batch_size=2,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(mmcv_collate, samples_per_gpu=2),
    )

    test_pipeline = [
        dict(type='SampleFrames', clip_len=config.DATA.NUM_FRAMES, frame_interval=config.DATA.FRAME_INTERVAL,
             num_clips=1, test_mode=True),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(type='CenterCrop', crop_size=config.DATA.INPUT_SIZE),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs', 'label', 'vid'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    test_data = RawFramesTestDataset(ann_file=config.DATA.VAL_FILE, data_prefix=config.DATA.ROOT,
                            labels_file=config.DATA.LABEL_LIST, filename_tmpl=config.DATA.FILENAME_TMPL,
                            pipeline=test_pipeline, seg_interval=config.DATA.NUM_FRAMES*config.DATA.FRAME_INTERVAL)

    sampler_test = torch.utils.data.SequentialSampler(test_data)
    test_loader = DataLoader(
        test_data, sampler=sampler_test,
        batch_size=64,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(mmcv_collate, samples_per_gpu=2),
    )

    train_pipeline_test = [
        dict(type='SampleFrames', clip_len=config.DATA.NUM_FRAMES, frame_interval=config.DATA.FRAME_INTERVAL,
             num_clips=1, test_mode=True),
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(-1, scale_resize)),
        dict(type='CenterCrop', crop_size=config.DATA.INPUT_SIZE),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCTHW'),
        dict(type='Collect', keys=['imgs', 'label', 'vid'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    train_data_test = RawFramesTestDataset(ann_file=config.DATA.TRAIN_FILE, data_prefix=config.DATA.ROOT,
                                     labels_file=config.DATA.LABEL_LIST, filename_tmpl=config.DATA.FILENAME_TMPL,
                                     pipeline=train_pipeline_test,
                                     seg_interval=config.DATA.NUM_FRAMES * config.DATA.FRAME_INTERVAL)

    train_sampler_test = torch.utils.data.SequentialSampler(train_data_test)
    train_loader_test = DataLoader(
        train_data_test, sampler=train_sampler_test,
        batch_size=64,
        num_workers=16,
        pin_memory=True,
        drop_last=False,
        collate_fn=partial(mmcv_collate, samples_per_gpu=2),
    )

    return train_data, val_data, test_data, train_loader, val_loader, test_loader, train_loader_test, train_loader_umil