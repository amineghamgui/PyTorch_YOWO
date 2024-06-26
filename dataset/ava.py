#!/usr/bin/python
# encoding: utf-8

import os
import random
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image

try:
    import ava_helper
except:
    from . import ava_helper


# Dataset for AVA
class AVA_Dataset(Dataset):
    def __init__(self,
                 cfg,
                 is_train=False,
                 img_size=224,
                 transform=None,
                 len_clip=16,
                 sampling_rate=1):
        self._downsample = 4
        self.num_classes = 1             
        # self.num_classes = 80
        self.data_root = cfg['data_root']
        self.frames_dir = os.path.join(cfg['data_root'], cfg['frames_dir'])
        self.frame_list = os.path.join(cfg['data_root'], cfg['frame_list'])
        self.annotation_dir = os.path.join(cfg['data_root'], cfg['annotation_dir'])
        self.labelmap_file = os.path.join(cfg['data_root'], cfg['annotation_dir'], cfg['labelmap_file'])
        if is_train:
            self.gt_box_list = os.path.join(self.annotation_dir, cfg['train_gt_box_list'])
            self.exclusion_file = os.path.join(self.annotation_dir, cfg['train_exclusion_file'])
        else:
            self.gt_box_list = os.path.join(self.annotation_dir, cfg['val_gt_box_list'])
            self.exclusion_file = os.path.join(self.annotation_dir, cfg['val_exclusion_file'])

        self.transform = transform
        self.is_train = is_train
        
        self.img_size = img_size
        self.len_clip = len_clip
        self.sampling_rate = sampling_rate
        self.seq_len = self.len_clip * self.sampling_rate

        # load ava data
        self._load_data()


    def _load_data(self):
        # Loading frame paths.
        (
            self._image_paths,
            self._video_idx_to_name,
        ) = ava_helper.load_image_lists(
            self.frames_dir,
            self.frame_list,
            self.is_train
            )

        # Loading annotations for boxes and labels.
        # boxes_and_labels: {'<video_name>': {<frame_num>: a list of [box_i, box_i_labels]} }
        boxes_and_labels = ava_helper.load_boxes_and_labels(
            self.gt_box_list,
            self.exclusion_file,
            self.is_train,
            full_test_on_val=False
            )

        assert len(boxes_and_labels) == len(self._image_paths)

        # boxes_and_labels: a list of {<frame_num>: a list of [box_i, box_i_labels]}
        boxes_and_labels = [
            boxes_and_labels[self._video_idx_to_name[i]]
            for i in range(len(self._image_paths))
        ]

        # Get indices of keyframes and corresponding boxes and labels.
        # _keyframe_indices: [video_idx, sec_idx, sec, frame_index]
        # _keyframe_boxes_and_labels: list[list[list]], outer is video_idx, middle is sec_idx,
        # inner is a list of [box_i, box_i_labels]
        (
            self._keyframe_indices,
            self._keyframe_boxes_and_labels,
        ) = ava_helper.get_keyframe_data(boxes_and_labels)

        # Calculate the number of used boxes.
        self._num_boxes_used = ava_helper.get_num_boxes_used(
            self._keyframe_indices, self._keyframe_boxes_and_labels
        )

        self._max_objs = ava_helper.get_max_objs(
            self._keyframe_indices, self._keyframe_boxes_and_labels
        )

        print("=== AVA dataset summary ===")
        print("Train: {}".format(self.is_train))
        print("Number of videos: {}".format(len(self._image_paths)))
        total_frames = sum(
            len(video_img_paths) for video_img_paths in self._image_paths
        )
        print("Number of frames: {}".format(total_frames))
        print("Number of key frames: {}".format(len(self)))
        print("Number of boxes: {}.".format(self._num_boxes_used))
        #print("self._keyframe_boxes_and_labels",self._keyframe_boxes_and_labels[0])
        #print("boxes_and_labels ",boxes_and_labels )
        #print("\n self._keyframe_indices    ",self._keyframe_indices)


    def __len__(self):
        return len(self._keyframe_indices)


    def get_sequence(self, center_idx, half_len, sample_rate, num_frames):
        """
        Sample frames among the corresponding clip.

        Args:
            center_idx (int): center frame idx for current clip
            half_len (int): half of the clip length
            sample_rate (int): sampling rate for sampling frames inside of the clip
            num_frames (int): number of expected sampled frames

        Returns:
            seq (list): list of indexes of sampled frames in this clip.
        """
        # seq = list(range(center_idx - half_len, center_idx + half_len, sample_rate))
        seq = list(range(center_idx - half_len*2 + 1*sample_rate, center_idx+1*sample_rate, sample_rate))
        
        for seq_idx in range(len(seq)):
            if seq[seq_idx] < 0:
                seq[seq_idx] = 0
            elif seq[seq_idx] >= num_frames:
                seq[seq_idx] = num_frames - 1
        return seq


    def get_frame_idx(self, latest_idx, sample_length, sample_rate, num_frames):
        """
        Sample frames among the corresponding clip. But see keyframe as the latest frame,
        instead of viewing it in center
        """
        # seq = list(range(latest_idx - sample_length + 1, latest_idx + 1, sample_rate))
        seq = list(range(latest_idx, latest_idx - sample_length, -sample_rate))
        seq.reverse()
        for seq_idx in range(len(seq)):
            if seq[seq_idx] < 0:
                seq[seq_idx] = 0
            elif seq[seq_idx] >= num_frames:
                seq[seq_idx] = num_frames - 1

        return seq


    def __getitem__(self, idx):
        # load a data
        frame_idx, video_clip, target = self.pull_item(idx)

        return frame_idx, video_clip, target


    def pull_item(self, idx):
        # Get the frame idxs for current clip. We can use it as center or latest
        video_idx, sec_idx, sec, frame_idx = self._keyframe_indices[idx]
        clip_label_list = self._keyframe_boxes_and_labels[video_idx][sec_idx]

        # check label list
        assert len(clip_label_list) > 0
        assert len(clip_label_list) <= self._max_objs

        # get a sequence
        seq = self.get_sequence(
            frame_idx,
            self.seq_len // 2,
            self.sampling_rate,
            num_frames=len(self._image_paths[video_idx]),
        )
        image_paths = [self._image_paths[video_idx][frame - 1] for frame in seq]
        keyframe_info = self._image_paths[video_idx][frame_idx - 1]

        # load a video clip
        video_clip = []
        for img_path in image_paths:
            frame = Image.open(img_path).convert('RGB')
            video_clip.append(frame)
        ow, oh = frame.width, frame.height

        # Get boxes and labels for current clip.
        boxes = []
        labels = []
        for box_labels in clip_label_list:
            bbox = box_labels[0]
            label = box_labels[1]
            multi_hot_label = np.zeros(1 + self.num_classes)
            multi_hot_label[..., label] = 1.0

            boxes.append(bbox)
            labels.append(multi_hot_label[..., 1:].tolist())

        boxes = np.array(boxes).reshape(-1, 4)
        # renormalize bbox
        boxes[..., [0, 2]] *= ow
        boxes[..., [1, 3]] *= oh
        labels = np.array(labels).reshape(-1, self.num_classes)

        # target: [N, 4 + C]
        target = np.concatenate([boxes, labels], axis=-1)

        # transform
        video_clip, target = self.transform(video_clip, target)
        # List [T, 3, H, W] -> [3, T, H, W]
        video_clip = torch.stack(video_clip, dim=1)

        # reformat target
        target = {
            'boxes': target[:, :4].float(),  # [N, 4]
            'labels': target[:, 4:].long(),  # [N, C]
            'orig_size': [ow, oh],
            'video_idx': video_idx,
            'sec': sec,
            # 'video_idx': "aaaaaaaaaaaaaaaaa",
            # 'sec': sec,
        }

        return keyframe_info, video_clip, target



if __name__ == '__main__':
    import cv2
    from transforms import Augmentation, BaseTransform

    is_train = False
    img_size = 224
    len_clip = 16
    dataset_config = {
        #'data_root': '/kaggle/input/data-ava/ava',
        'data_root':'/kaggle/input/ava-version2-oneclass/ava-20240312T085221Z-001/ava',
        #'data_root': '/kaggle/input/data-ava/ava',
        'frames_dir': 'frames/',
        'frame_list': 'frame_lists/',
        'annotation_dir': 'annotations/',
        'train_gt_box_list': 'ava_v2.2/ava_train_v2.2.csv',
        'val_gt_box_list': 'ava_v2.2/ava_val_v2.2.csv',
        'train_exclusion_file': 'ava_v2.2/ava_train_excluded_timestamps_v2.2.csv',
        'val_exclusion_file': 'ava_v2.2/ava_val_excluded_timestamps_v2.2.csv',
        'labelmap_file': 'ava_v2.2/ava_action_list_v2.2.pbtxt',
    }
    
    trans_config = {
        'pixel_mean': [0.45, 0.45, 0.45],
        'pixel_std': [0.225, 0.225, 0.225],
        'jitter': 0.2,
        'hue': 0.1,
        'saturation': 1.5,
        'exposure': 1.5
    }
    transform = Augmentation(
        img_size=img_size,
        pixel_mean=trans_config['pixel_mean'],
        pixel_std=trans_config['pixel_std'],
        jitter=trans_config['jitter'],
        saturation=trans_config['saturation'],
        exposure=trans_config['exposure']
        )
    # transform = BaseTransform(
    #     img_size=img_size,
    #     pixel_mean=trans_config['pixel_mean'],
    #     pixel_std=trans_config['pixel_std']
    #     )

    train_dataset = AVA_Dataset(
        cfg=dataset_config,
        is_train=is_train,
        img_size=img_size,
        transform=transform,
        len_clip=len_clip,
        sampling_rate=1
    )

    print("*************************************************************************************amine")
    print(train_dataset)
    print(len(train_dataset))
    std = trans_config['pixel_std']
    mean = trans_config['pixel_mean']
    for i in range(len(train_dataset)):
        frame_id, video_clip, target = train_dataset[i]
        key_frame = video_clip[:, -1, :, :]

        key_frame = key_frame.permute(1, 2, 0).numpy()
        key_frame = ((key_frame * std + mean) * 255).astype(np.uint8)
        H, W, C = key_frame.shape

        key_frame = key_frame.copy()
        bboxes = target['boxes']
        labels = target['labels']

        for box, cls_id in zip(bboxes, labels):
            x1, y1, x2, y2 = box
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            x2 = int(x2 * W)
            y2 = int(y2 * H)
            key_frame = cv2.rectangle(key_frame, (x1, y1), (x2, y2), (255, 0, 0))

        # cv2 show
        import matplotlib.pyplot as plt
        plt.imshow( key_frame[..., (2, 1, 0)])
        # cv2.imshow('key frame', key_frame[..., (2, 1, 0)])
        # cv2.waitKey(0)
        
