import os
import argparse

import torch
import torch.optim as optim
from torch import nn

from models.lenet import LeNet5
from models.lenet_half import LeNet5HalfEncoder, LeNet5HalfClassifier
from models.critic import Critic
from datasets.genimg import get_genimg
from datasets.usps import get_usps
from datasets.mnist import get_mnist
from utils import eval_encoder_and_classifier, partial_load, kd_loss_fn

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','cifar10','cifar100', 'USPS'])
parser.add_argument('--target', type=str, default='USPS', choices=['MNIST','cifar10','cifar100', 'USPS'])
parser.add_argument('--model_root', type=str, default='cache/models/', help='interval for testinh the model')
parser.add_argument('--num_epochs', type=int, default=2000, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--c_learning_rate', type=float, default=1e-4, help='c learning rate')
parser.add_argument('--d_learning_rate', type=float, default=1e-3, help='d learning rate')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam')
parser.add_argument('--beta2', type=float, default=0.9, help='beta2 for adam')
parser.add_argument('--log_step', type=int, default=20, help='interval for logging')
parser.add_argument('--save_step', type=int, default=100, help='interval for saving the model')
parser.add_argument('--eval_step', type=int, default=1, help='interval for testinh the model')
parser.add_argument('--img_opt_step', type=int, default=200, help='img optimization steps')
parser.add_argument('--lr_O', type=float, default=1e-2, help='img optimization steps')
parser.add_argument('--oh', type=float, default=1, help='img optimization steps')
parser.add_argument('--a', type=float, default=0.03, help='img optimization steps')
parser.add_argument('--ie', type=float, default=1, help='img optimization steps')
opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
teacher_path = 'cache/models/teacher_{}.pt'.format(opt.dataset)
student_path = 'cache/models/student_{}.pt'.format(opt.dataset)
encoder_path = 'cache/models/tgt_encoder_{}2{}.pt'.format(opt.dataset, opt.target)

teacher = partial_load(LeNet5, teacher_path)


def pre_train_critic(src_encoder, critic, src_dataloader, tgt_dataloader):
    critic = critic.to(device)
    num_epoch = 10
    criterion = nn.CrossEntropyLoss()
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=opt.d_learning_rate,
                                  betas=(opt.beta1, opt.beta2))

    for epoch in range(num_epoch):
        data_zip = enumerate(zip(src_dataloader, tgt_dataloader))
        for step, ((images_src, _), (images_tgt, _)) in data_zip:
            # make images variable
            images_src = images_src.to(device)
            images_tgt = images_tgt.to(device)

            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(images_src)
            feat_tgt = src_encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0).detach().to(device)

            # predict on discriminator
            pred_concat = critic(feat_concat)

            # prepare real and fake label, src is 1 and tgt is 0
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
            print('[pre-train critic]epoch {}:'.format(epoch), 'step {}:'.format(step), 'acc: {}'.format(acc))
    return critic


def train_tgt(src_encoder, tgt_encoder, critic, tgt_data_loader, classifier):
    """Train encoder for target domain."""
    ####################
    # 1. setup network #
    ####################

    # set train state for Dropout and BN layers
    tgt_encoder.to(device)
    tgt_encoder.train()
    critic.to(device)
    critic.train()

    # setup criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_tgt = optim.Adam(tgt_encoder.parameters(),
                               lr=opt.c_learning_rate,
                               betas=(opt.beta1, opt.beta2))
    optimizer_critic = optim.Adam(critic.parameters(),
                                  lr=opt.d_learning_rate,
                                  betas=(opt.beta1, opt.beta2))

    ####################
    # 2. train network #
    ####################
    best_acc = 0
    for epoch in range(opt.num_epochs):
        for step, (images_tgt, _) in enumerate(tgt_data_loader):
            ###########################
            # 2.1 train discriminator #
            ###########################

            # make images variable
            images_tgt = images_tgt.to(device)

            # initiate img with target data
            opt_imgs = images_tgt.clone().to(device)
            opt_imgs.requires_grad = True
            optimizer_img = torch.optim.Adam([opt_imgs], opt.lr_O)

            # optimize img
            for step in range(opt.img_opt_step):
                output, feature = teacher(opt_imgs, out_feature=True)
                loss_oh = criterion(output, output.data.max(1)[1])
                loss_act = -feature.abs().mean()
                softmax_o = torch.nn.functional.softmax(output, dim=1).mean(dim=0)
                loss_ie = (softmax_o * torch.log(softmax_o)).sum()
                loss = loss_oh * opt.oh + loss_act * opt.a + loss_ie * opt.ie
                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()

            # zero gradients for optimizer
            optimizer_critic.zero_grad()

            # extract and concat features
            feat_src = src_encoder(opt_imgs)
            feat_tgt = tgt_encoder(images_tgt)
            feat_concat = torch.cat((feat_src, feat_tgt), 0).detach().to(device)

            # predict on discriminator
            pred_concat = critic(feat_concat)

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
            domain_acc = (pred_cls == label_concat).float().mean()

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
            if (step + 1) % opt.log_step == 0:
                print("Epoch [{}/{}] Step [{}]:"
                      "d_loss={:.5f} g_loss={:.5f} d_acc={:.5f}"
                      .format(epoch + 1,
                              opt.num_epochs,
                              step + 1,
                              loss_critic.data.item(),
                              loss_tgt.data.item(),
                              domain_acc.data.item()))

        #############################
        # 2.4 save model parameters #
        #############################
        # if ((epoch + 1) % opt.save_step == 0):

        if epoch % opt.eval_step == 0:
            print('epoch ', epoch, ':')
            acc = eval_encoder_and_classifier(tgt_encoder, classifier, tgt_data_loader)
            if acc > best_acc:
                best_acc = acc
                torch.save(critic.state_dict(), os.path.join(
                    opt.model_root,
                    "critic.pt"))
                torch.save(tgt_encoder.state_dict(), os.path.join(
                    opt.model_root,
                    "tgt_encoder.pt"))

    return tgt_encoder


def run():
    src_encoder = partial_load(LeNet5HalfEncoder, student_path)
    tgt_encoder = partial_load(LeNet5HalfEncoder, student_path)
    classifier = partial_load(LeNet5HalfClassifier, student_path)
    critic = Critic(42, 84, 2)
    if opt.target == 'USPS':
        tgt_data_loader = get_usps(True, opt.batch_size)
    elif opt.target == 'MNIST':
        tgt_data_loader = get_mnist(True, opt.batch_size)
    else:
        print("adapting to {} is not yet implemented, abort")
        return

    for p in critic.parameters():
        p.requires_grad = True
    for p in src_encoder.parameters():
        p.requires_grad = False
    for p in tgt_encoder.parameters():
        p.requires_grad = True
    for p in classifier.parameters():
        p.requires_grad = False

    train_tgt(src_encoder, tgt_encoder, critic,
              tgt_data_loader, classifier)
    eval_encoder_and_classifier(tgt_encoder, classifier, tgt_data_loader)


if __name__ == '__main__':
    run()