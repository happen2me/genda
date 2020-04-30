import os
import argparse

import torch
import torch.optim as optim
from torch import nn

from models.lenet import LeNet5Encoder, LeNet5Classifier
from models.lenet_half import LeNet5HalfEncoder
from models.critic import Critic
from datasets.genimg import get_genimg
from datasets.usps import get_usps
from utils import eval_encoder_and_classifier, partial_load

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

teacher_path = 'cache/models/teacher.pt'
student_path = 'cache/models/student.pt'

parser = argparse.ArgumentParser()
parser.add_argument('--model_root', type=str, default='cache/models/', help='interval for testinh the model')
parser.add_argument('--num_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--c_learning_rate', type=float, default=1e-4, help='c learning rate')
parser.add_argument('--d_learning_rate', type=float, default=1e-4, help='d learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')
parser.add_argument('--log_step', type=int, default=150, help='interval for logging')
parser.add_argument('--save_step', type=int, default=150, help='interval for saving the model')
parser.add_argument('--eval_step', type=int, default=50, help='interval for testinh the model')
opt = parser.parse_args()


def train_tgt(src_encoder, tgt_encoder, critic,
              src_data_loader, tgt_data_loader, classifier):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    tgt_encoder.train()
    critic.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=opt.c_learning_rate,
                               betas=(opt.beta1, opt.beta2))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=opt.d_learning_rate,
                                  betas=(opt.beta1, opt.beta2))
    len_data_loader = min(len(src_data_loader), len(tgt_data_loader))

    ####################
    # 2. train network #
    ####################

    for epoch in range(opt.num_epochs):
        # zip source and target data pair

        data_zip = enumerate(zip(src_data_loader, tgt_data_loader))
        for step, ((images_src, _), (images_tgt, _)) in data_zip:
            ###########################
            # 2.1 train discriminator #
            ###########################

            # make images variable
            images_src = images_src.to(device)
            images_tgt = images_tgt.to(device)


            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(images_src)
            feat_tgt = tgt_encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0).to(device)

            # predict on discriminator
            pred_concat = critic(feat_concat.detach())

            # prepare real and fake label
            label_src = torch.ones(feat_src.size(0)).long().to(device)
            label_tgt = torch.zeros(feat_tgt.size(0)).long().to(device)
            label_concat = torch.cat((label_src, label_tgt), 0)

            # compute loss for critic
            loss_critic = criterion(pred_concat, label_concat)
            loss_critic.backward()

            # optimize critic
            optimizer_critic.step()

            pred_cls = torch.squeeze(pred_concat.max(1)[1])
            acc = (pred_cls == label_concat).float().mean()

            ############################
            # 2.2 train target encoder #
            ############################

            # zero gradients for optimizer
            optimizer_critic.zero_grad()
            optimizer_tgt.zero_grad()

            # extract and target features
            feat_tgt = tgt_encoder(images_tgt)

            # predict on discriminator
            pred_tgt = critic(feat_tgt)

            # prepare fake labels
            label_tgt = torch.ones(feat_tgt.size(0)).long().to(device)

            # compute loss for target encoder
            loss_tgt = criterion(pred_tgt, label_tgt)
            loss_tgt.backward()

            # optimize target encoder
            optimizer_tgt.step()

            #######################
            # 2.3 print step info #
            #######################
            if ((step + 1) % opt.log_step == 0):
                print("Epoch [{}/{}] Step [{}/{}]:"
                      "d_loss={:.5f} g_loss={:.5f} acc={:.5f}"
                      .format(epoch + 1,
                              opt.num_epochs,
                              step + 1,
                              len_data_loader,
                              loss_critic.data.item(),
                              loss_tgt.data.item(),
                              acc.data.item()))

        #############################
        # 2.4 save model parameters #
        #############################
        if ((epoch + 1) % opt.save_step == 0):
            torch.save(critic.state_dict(), os.path.join(
                opt.model_root,
                "critic.pt"))
            torch.save(tgt_encoder.state_dict(), os.path.join(
                opt.model_root,
                "tgt_encoder.pt"))

        if epoch % 10 == 0:
            eval_encoder_and_classifier(src_encoder, classifier, tgt_data_loader)
            eval_encoder_and_classifier(tgt_encoder, classifier, tgt_data_loader)

    return tgt_encoder


def run():
    src_encoder = partial_load(LeNet5Encoder, teacher_path)
    tgt_encoder = partial_load(LeNet5HalfEncoder, student_path)
    classifier = partial_load(LeNet5Classifier, teacher_path)
    critic = Critic(64, 84, 2)
    src_data_loader = get_genimg(True, opt.batch_size)
    tgt_data_loader = get_usps(False, opt.batch_size)

    train_tgt(src_encoder, tgt_encoder, critic,
              src_data_loader, tgt_data_loader, classifier)

    eval_encoder_and_classifier(tgt_encoder, classifier, tgt_data_loader)


if __name__ == '__main__':
    run()
