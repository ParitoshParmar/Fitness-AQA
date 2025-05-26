# Author: Paritosh Parmar


import json
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageDraw
from opts_exercise_qa import *
from augmentations import image_augmentations

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True


def filenames2trajnames(filenames_list):
    for i in range(len(filenames_list)):
        filenames_list[i] += '.json'
    return filenames_list


def exclude_list(full_list, list_exclude):
    # print('excluding files...')
    # print('full list: ', full_list)
    # print('samples 2 exclude: ', list_exclude)
    # print('no of sample b4 exclusion: ', len(full_list))
    final_list = [x for x in full_list if x not in list_exclude]
    # print('no of sample after exclusion: ', len(final_list))
    return final_list


def traj2phase(trajectory):
    phase = []
    for i in range(len(trajectory)):
        phase.append(trajectory[i]*360)
    return phase


def load_image(image_path, transform=None, augmentations=None):
    image = Image.open(image_path)
    size = input_resize_2dcnn
    interpolator_idx = 1#random.randint(0,3)
    interpolators = [Image.NEAREST, Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]
    interpolator = Image.BILINEAR#interpolators[interpolator_idx]
    image = image.resize(size, interpolator)

    #------------applying augmentations-------------#
    image = image_augmentations.apply_augmentations(image, augmentations=augmentations)
    #-----------------------------------------------#
    # image.show()
    # input('hi')

    if transform is not None:
        image = transform(image)# .unsqueeze(0) commented because not video model
    return image


class VideoDataset(Dataset):
    def __init__(self, mode, ssl_contrastive_phase_gap):
        super(VideoDataset, self).__init__()
        self.mode = mode # train, val, or test
        # loading annotations
        self.trajectories_files = sorted(os.listdir(ssl_trajectories_dir))
        self.traj_nan = json.load(open(train_val_test_sets_dir + 'traj_nan.json'))
        self.trajectories_files = exclude_list(self.trajectories_files, self.traj_nan)
        # excluding val samples
        val_samples2exclude = filenames2trajnames(json.load(open(train_val_test_sets_dir + 'val_keys.json')))
        self.trajectories_files = exclude_list(self.trajectories_files, val_samples2exclude)
        # excluding test samples
        test_samples2exclude = filenames2trajnames(json.load(open(train_val_test_sets_dir + 'test_keys.json')))
        self.trajectories_files = exclude_list(self.trajectories_files, test_samples2exclude)
        # print('traj files: ', self.trajectories_files)
        # print('len of traj files b4: ', len(self.trajectories_files))
        if self.mode == 'train':
            self.trajectories_files = self.trajectories_files[:-500]
        elif self.mode == 'val':
            self.trajectories_files = self.trajectories_files[-500:]
        elif self.mode == 'test':
            self.trajectories_files = self.trajectories_files[-500:]
        else:
            input('Wrong type of mode selected. What do you want to do?')
        # print('len of ssl_traj_files: ', len(self.trajectories_files))
        # print(trajectories_files)
        # specify phase difference in degrees
        self.ssl_contrastive_phase_gap =  ssl_contrastive_phase_gap # random.randint(45,50)



    def __getitem__(self, ix_0):
        # print('from dataloader, phase_diff_ll: ', self.phase_diff_ll)
        transform = transforms.Compose([transforms.CenterCrop(H_2dcnn),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


        # [todo: might want to select second sample of length as the first one]
        second_sample_found = False
        while second_sample_found == False:
            ix_1 = random.randint(0, len(self.trajectories_files)-1)
            if ix_1 != ix_0:
                second_sample_found = True

        # load bar trajectories
        trajectory_0 = json.load(open(ssl_trajectories_dir + self.trajectories_files[ix_0]))
        # print('ix_1: ', ix_1)
        trajectory_1 = json.load(open(ssl_trajectories_dir + self.trajectories_files[ix_1]))

        normed_trajectory_0 = []
        for i in range(len(trajectory_0)):
            temp = (trajectory_0[i] - min(trajectory_0)) / (max(trajectory_0) - min(trajectory_0))
            normed_trajectory_0.append(temp)
        # print(normed_trajectory_0)
        # print('len of traj 0 b4 smoothing: ', len(normed_trajectory_0))
        phase_0 = traj2phase(normed_trajectory_0)

        normed_trajectory_1 = []
        for i in range(len(trajectory_1)):
            temp = (trajectory_1[i] - min(trajectory_1)) / (max(trajectory_1) - min(trajectory_1))
            normed_trajectory_1.append(temp)
        # print(normed_trajectory_1)
        phase_1 = traj2phase(normed_trajectory_1)

        # list all the frames in both videos
        video_id_0 = self.trajectories_files[ix_0].split('.')[0]
        video_id_1 = self.trajectories_files[ix_1].split('.')[0]
        # print('video id: ', video_id_0)
        frames_list_0 = sorted(os.listdir(ssl_frames_dir + video_id_0 + '/'))
        frames_list_1 = sorted(os.listdir(ssl_frames_dir + video_id_1 + '/'))
        # print('frame list 0: ', frames_list_0)
        # print('frame list 1: ', frames_list_1)


        # ---------phase code----------------#
        # choose a random phase from video_0/trajectory_0
        anchor_phase_temp = random.randint(int(max(min(phase_0), min(phase_1))),
                                           int(min(max(phase_0), max(phase_1))))

        # calculating phase to exclude
        phase_0_available = []
        for i in range(len(phase_0)):
            if not ((abs(anchor_phase_temp) - abs(self.ssl_contrastive_phase_gap)) < abs(phase_0[i]) < (
                    abs(anchor_phase_temp) + abs(self.ssl_contrastive_phase_gap))):
                phase_0_available.append(phase_0[i])
        # following is for visualizing purpose
        #     else:
        #         phase_0_available.append(0)
        # phase_0_available_abs = phase_0_available.copy()
        # for i in range(len(phase_0_available_abs)):
        #     phase_0_available_abs[i] = abs(phase_0_available_abs[i])
        # plt.stem(phase_0_available_abs)
        # plt.show()
        #
        phase_1_available = []
        for i in range(len(phase_1)):
            if not ((abs(anchor_phase_temp) - abs(self.ssl_contrastive_phase_gap)) < abs(phase_1[i]) < (
                    abs(anchor_phase_temp) + abs(self.ssl_contrastive_phase_gap))):
                phase_1_available.append(phase_1[i])
            # following is for visualizing purpose
            # else:
            #     phase_1_available.append(0)
        # -----------------------------------#

        video_4_neg_ip = 1#random.randint(0, 1)
        # print('video_4_neg_ip: ', video_4_neg_ip)

        anchor_phase = anchor_phase_temp
        positive_phase = anchor_phase_temp

        # --------choosing negative input from phase_1_available----------#
        negative_phase = phase_1_available[random.randint(0, len(phase_1_available) - 2)]
        # print('-ve phase: ', negative_phase)
        # ----------------------------------------------------------------#

        # -------------phase code---------------#
        # find the frame with the phase closest to anchor_phase
        anchor_phase_diffs = []
        for i in range(len(phase_0)):
            anchor_phase_diffs.append(abs(anchor_phase - phase_0[i]))
        min_anchor_phase_diff = anchor_phase_diffs.index(min(anchor_phase_diffs))
        # print('min anchor phase diff: ', min_anchor_phase_diff)
        # print('len of anchor diffs: ', len(anchor_phase_diffs))
        anchor_idx = min_anchor_phase_diff
        # repeating for positive phase
        positive_phase_diffs = []
        for i in range(len(phase_1)):
            positive_phase_diffs.append(abs(positive_phase - phase_1[i]))
        min_positive_phase_diff = positive_phase_diffs.index(min(positive_phase_diffs))
        # print('min positive phase diff: ', min_positive_phase_diff)
        positive_ip_idx = min_positive_phase_diff
        # repeating for negative phase
        negative_phase_diffs = []
        for i in range(len(phase_1)):
            negative_phase_diffs.append(abs(negative_phase - phase_1[i]))
        min_negative_phase_diff = negative_phase_diffs.index(min(negative_phase_diffs))
        # print('min negative phase diff: ', min_negative_phase_diff)
        negative_ip_idx = min_negative_phase_diff
        # --------------------------------------#




##################################################################################################
        images_anchor = torch.zeros(C_2dcnn, H_2dcnn, W_2dcnn)
        images_positive = torch.zeros(C_2dcnn, H_2dcnn, W_2dcnn)
        images_negative = torch.zeros(C_2dcnn, H_2dcnn, W_2dcnn)

        # hori_flip = 0
        #
        # apply_input_mask_toss = random.randint(0, 1)
        # apply_input_mask = 0
        # if apply_input_mask_toss > 0:
        #     apply_input_mask = 1

        # ----------- preparing augmentations----------------#
        # different augmentations applied to anchor, +ve, -ve
        augmentations = [{}, {}, {}]  # order: anchor, +ve, -ve
        for triplet_i in range(3):
            augmentations[triplet_i] = {
                # 'hori_flip': {'apply': random.choice([0, 1])},
                'masking': {'apply': random.choice([0, 1]), 'type': 'fixed', 'mask_amt': random.uniform(0.4, 0.5)},
                # 'masking_checker_ol': {'apply': random.choice([0, 1])},
                # 'masking_checker_nool': {'apply': random.choice([0, 1])},
                # 'translation': {'apply': random.choice([0, 1]), 'type': 'fixed', 'x': random.uniform(-10, 10), 'y': random.uniform(-10, 10)},
                # 'rotation': {'apply': random.choice([0, 1]), 'type': 'fixed', 'rotation': random.uniform(-20, 20)},
                # 'blurring': {'apply': random.choice([0, 1]), 'type': 'fixed', 'blur_amt': random.uniform(0, 1.2)},
                # 'zooming': {'apply': random.choice([0, 1]), 'type': 'fixed', 'zoom_amt': random.uniform(0.8, 1.2)},
                # 'color_jittering': {'apply': random.choice([0, 1]), 'desired_order': random.sample([0, 1, 2], 3)}
            }
        # ---------------------------------------------------#

        images_anchor = load_image(ssl_frames_dir + video_id_0 + '/' + frames_list_0[anchor_idx], transform, augmentations[0])
        images_positive = load_image(ssl_frames_dir + video_id_1 + '/' + frames_list_1[positive_ip_idx], transform, augmentations[1])
        if video_4_neg_ip == 0:
            images_negative = load_image(ssl_frames_dir + video_id_0 + '/' + frames_list_0[negative_ip_idx], transform, augmentations[2])
        else:
            images_negative = load_image(ssl_frames_dir + video_id_1 + '/' + frames_list_1[negative_ip_idx], transform, augmentations[2])


        data = {}
        data['anchor_im'] = images_anchor
        data['positive_im'] = images_positive
        data['negative_im'] = images_negative
        # input('want to continue?')
        return data


    def __len__(self):
        return len(self.trajectories_files)
