# Author: Paritosh Parmar (https://github.com/ParitoshParmar)


import os
import torch
from torch.utils.data import DataLoader
from ssl_contrastive_image_cleaned.dataloader import VideoDataset
from ssl_contrastive_image_cleaned.dataloader_eval import VideoDataset_Eval
import random
import torch.optim as optim
import torch.nn.functional as F
from opts_exercise_qa import *
import numpy as np
from ssl_contrastive_image_cleaned.models import my_resnet, linear_layers

torch.manual_seed(randomseed); torch.cuda.manual_seed_all(randomseed); random.seed(randomseed); np.random.seed(randomseed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False


def save_model(model, model_name, epoch, path):
    model_path = os.path.join(path, '%s_%d.pth' % (model_name, epoch))
    torch.save(model.state_dict(), model_path)


def log_printer(line2print, saving_dir):
    f = open(saving_dir + "log.txt", "a")
    f.write(line2print)
    f.close()
    return


def custom_act_scaling(x):
    y = x / (1 + x**2)
    return y


def train_phase(train_dataloader, optimizer, epoch, saving_dir):

    model_CNN.train()
    model_linear_layers.train()
    # model_classifier.train()

    iteration = 0
    for data in train_dataloader:

        anchor_im = data['anchor_im'].cuda()
        positive_im = data['positive_im'].cuda()
        negative_im = data['negative_im'].cuda()

        anchor_im_feats = F.normalize(model_linear_layers(model_CNN(anchor_im)), dim=-1, p=2)
        positive_im_feats = F.normalize(model_linear_layers(model_CNN(positive_im)), dim=-1, p=2)
        negative_im_feats = F.normalize(model_linear_layers(model_CNN(negative_im)), dim=-1, p=2)

        ##############
        # anchor_im_feats = custom_act_scaling(model_CNN(anchor_im))
        # positive_im_feats = custom_act_scaling(model_CNN(positive_im))
        # negative_im_feats = custom_act_scaling(model_CNN(negative_im))
        #################
        # print('shape of anchor feats: ', anchor_im_feats.shape)
        loss = 0
        current_batch_size = anchor_im_feats.shape[0] # to handle partial batches
        for sample in range(current_batch_size):
            temp_loss_anc_pos = torch.exp(-1*sum(((anchor_im_feats[sample,:] - positive_im_feats[sample,:])**2)))
            temp_loss_anc_neg = torch.exp(-1*sum(((anchor_im_feats[sample,:] - negative_im_feats[sample,:])**2)))
            temp_loss_pos_neg = torch.exp(-1*sum(((positive_im_feats[sample,:] - negative_im_feats[sample,:])**2)))
            # temp_loss = -1*torch.log(temp_loss_anc_pos / (temp_loss_anc_pos + temp_loss_anc_neg))
            temp_loss = -1*torch.log(temp_loss_anc_pos / (temp_loss_anc_pos + temp_loss_anc_neg + temp_loss_pos_neg))
            temp_loss = temp_loss / current_batch_size
            # print('temp num shape:', temp_loss)
            loss += temp_loss



        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if iteration % 20 == 0:
            line2print = 'Epoch:' + str(epoch) + '    Iter:' + str(iteration) + '    Loss:' + str(loss.data.cpu().numpy())
            print(line2print)
            line2print += '\n'
            log_printer(line2print, saving_dir)

        iteration += 1
    line2print = '--------------------------------------------------------------\n'
    log_printer(line2print, saving_dir)


def test_phase(mode, test_dataloader, saving_dir):
    print('In Test Phase...')
    with torch.no_grad():

        model_CNN.eval()
        model_linear_layers.eval()

        dist_ap = []; dist_an = []
        iteration = 0
        for data in test_dataloader:
            anchor_im = data['anchor_im'].cuda()
            positive_im = data['positive_im'].cuda()
            negative_im = data['negative_im'].cuda()
            # print('shape of anchor clip: ', anchor_clip.shape)

            anchor_im_feats = F.normalize(model_linear_layers(model_CNN(anchor_im)), dim=-1, p=2)
            positive_im_feats = F.normalize(model_linear_layers(model_CNN(positive_im)), dim=-1, p=2)
            negative_im_feats = F.normalize(model_linear_layers(model_CNN(negative_im)), dim=-1, p=2)

            batch_size = anchor_im_feats.shape[0]

            for i in range(batch_size):
                # L2 distance based
                temp_dist_ap = torch.sqrt(torch.sum((anchor_im_feats[i] - positive_im_feats[i]) ** 2))
                # print('temp dist ap: ', dist_ap)
                temp_dist_an = torch.sqrt(torch.sum((anchor_im_feats[i] - negative_im_feats[i]) ** 2))
                # print('temp dist an: ', temp_dist_an)

                dist_ap.append(temp_dist_ap.item())
                dist_an.append(temp_dist_an.item())

            iteration += 1
            # print('iter: ', iteration)
            samples_tested = iteration * batch_size
            if samples_tested > 499:
                # print('breaking!!!')
                break

        # print('dist_ap: ', dist_ap)
        # print('dist_an: ', dist_an)
        correct = 0
        for i in range(len(dist_ap)):
            if dist_ap[i] < dist_an[i]:
                correct += 1
        accuracy = 100 * correct / len(dist_ap)
        line2print = mode + ' Accuracy: ' + str(accuracy) + '\n'
        line2print += mode + ' Average dist AP: ' + str(sum(dist_ap) / len(dist_ap)) + '\n'
        line2print += mode + ' Average dist AN: ' + str(sum(dist_an) / len(dist_an)) + '\n'
        line2print += mode + ' Average dist Ratio: ' + str((sum(dist_ap) / len(dist_ap))/(sum(dist_an) / len(dist_an))) + '\n'
        log_printer(line2print, saving_dir)
        print(line2print)
        # print('exiting')




def main():
    parameters_2_optimize = list(model_CNN.parameters())
    parameters_2_optimize_named = list(model_CNN.named_parameters())

    parameters_2_optimize += list(model_linear_layers.parameters())
    parameters_2_optimize_named += list(model_linear_layers.named_parameters())

    # parameters_2_optimize += list(model_classifier.parameters())
    # parameters_2_optimize_named += list(model_classifier.parameters())

    learning_rate = base_learning_rate
    optimizer = optim.Adam(parameters_2_optimize, lr=learning_rate)
    # print('Parameters that will be learnt: ', parameters_2_optimize_named)

    # train_dataset = VideoDataset('train')
    # val_dataset = VideoDataset('val')
    # test_dataset = VideoDataset('test')
    # train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    # val_dataloader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False)
    # test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
    # print('Length of train loader: ', len(train_dataloader))
    # print('Length of val loader: ', len(val_dataloader))
    # print('Length of test loader: ', len(test_dataloader))
    # print('Training set size: ', len(train_dataset.keys), ';    Val set size: ', len(val_dataset.keys), ';    Test set size: ', len(test_dataset.keys))

    # actual training, testing loops
    ssl_contrastive_phase_gap = 30
    for epoch in range(max_epochs):
        print('-------------------------------------------------------------------------------------------------------')
        #---------------------------------------#
        if (epoch+1)%4 == 0:
            if ssl_contrastive_phase_gap > 30:
                ssl_contrastive_phase_gap -= 1
                print('SSL_contrastive_phase_gap: ', ssl_contrastive_phase_gap)
        train_dataset = VideoDataset(mode='train', ssl_contrastive_phase_gap=ssl_contrastive_phase_gap)
        # val_dataset = VideoDataset(mode='val', ssl_contrastive_phase_gap=ssl_contrastive_phase_gap)
        train_dataset_min = VideoDataset_Eval(mode='train', ssl_contrastive_phase_gap=30)
        train_dataset_med = VideoDataset_Eval(mode='train', ssl_contrastive_phase_gap=45)
        train_dataset_max = VideoDataset_Eval(mode='train', ssl_contrastive_phase_gap=60)
        val_dataset_min = VideoDataset_Eval(mode='val', ssl_contrastive_phase_gap=30)
        val_dataset_med = VideoDataset_Eval(mode='val', ssl_contrastive_phase_gap=45)
        val_dataset_max = VideoDataset_Eval(mode='val', ssl_contrastive_phase_gap=60)
        # test_dataset = VideoDataset(mode='test', ssl_contrastive_phase_gap=ssl_contrastive_phase_gap)
        #
        train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size_2dcnn, shuffle=True)
        #
        train_dataloader_min = DataLoader(train_dataset_min, batch_size=test_batch_size_2dcnn, shuffle=False)
        train_dataloader_med = DataLoader(train_dataset_med, batch_size=test_batch_size_2dcnn, shuffle=False)
        train_dataloader_max = DataLoader(train_dataset_max, batch_size=test_batch_size_2dcnn, shuffle=False)
        val_dataloader_min = DataLoader(val_dataset_min, batch_size=test_batch_size_2dcnn, shuffle=False)
        val_dataloader_med = DataLoader(val_dataset_med, batch_size=test_batch_size_2dcnn, shuffle=False)
        val_dataloader_max = DataLoader(val_dataset_max, batch_size=test_batch_size_2dcnn, shuffle=False)
        # test_dataloader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)
        if epoch == 0:
            print('Length of train loader: ', len(train_dataloader))
            print('Length of val loader: ', len(val_dataloader_max))
            # print('Length of test loader: ', len(test_dataloader))
            # print('Training set size: ', len(train_dataset.keys), ';    Val set size: ', len(val_dataset.keys), ';    Test set size: ', len(test_dataset.keys))
        #---------------------------------------#

        saving_dir = '/data/paritosh_trained_wts/image_model/ssl/cvcrl_fc1_noaug_noolmaskgray/'

        for param_group in optimizer.param_groups:
            print('Current learning rate: ', param_group['lr'])
        # validation phase
        test_phase('Valid_Min', val_dataloader_min, saving_dir)
        test_phase('Valid_Med', val_dataloader_med, saving_dir)
        test_phase('Valid_Max', val_dataloader_max, saving_dir)
        # calculating train accuracy
        test_phase('Train_Min', train_dataloader_min, saving_dir)
        test_phase('Train_Med', train_dataloader_med, saving_dir)
        test_phase('Train_Max', train_dataloader_max, saving_dir)
        # training phase
        train_phase(train_dataloader, optimizer, epoch, saving_dir)


        if (epoch+1) % model_ckpt_interval == 0: # save models every 5 epochs
            save_model(model_CNN, 'model_CNN', epoch, saving_dir)
            save_model(model_linear_layers, 'model_linear_layers', epoch, saving_dir)

    # testing phase
    test_phase(test_dataloader)





if __name__ == '__main__':
    torch.cuda.set_device(0)
    model_CNN = my_resnet.resnet18(pretrained=False)
    #
    model_CNN_pretrained_dict = torch.load('/home/dockeruser/.cache/torch/hub/checkpoints/resnet18-5c106cde.pth')
    model_CNN_dict = model_CNN.state_dict()
    model_CNN_pretrained_dict = {k: v for k, v in model_CNN_pretrained_dict.items() if k in model_CNN_dict}
    model_CNN_dict.update(model_CNN_pretrained_dict)
    model_CNN.load_state_dict(model_CNN_dict)
    #
    model_CNN = model_CNN.cuda()
    # print('model cnn: ', model_CNN)

    # # loading our error classifier
    # model_classifier = C3D_dilated_head_classifier()
    # model_classifier = model_classifier.cuda()

    model_linear_layers = linear_layers.linear_layers()
    # model_linear_layers = nn.DataParallel(model_linear_layers)
    model_linear_layers = model_linear_layers.cuda()

    os.getcwd()

    main()
