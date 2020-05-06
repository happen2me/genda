import argparse
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from torchvision.datasets.mnist import MNIST

from models.lenet_half import LeNet5Half
from models.generator import Generator
from models.lenet import LeNet5
from datasets.mnist import get_mnist
from datasets.usps import get_usps
from utils import partial_load, eval_model

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','cifar10','cifar100', 'USPS'])
parser.add_argument('--target', type=str, default='USPS', choices=['MNIST','cifar10','cifar100', 'USPS'])
parser.add_argument('--data', type=str, default='cache/data/')
parser.add_argument('--teacher_dir', type=str, default='cache/models/')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1024, help='size of the batches')
parser.add_argument('--lr_G', type=float, default=0.2, help='learning rate')
parser.add_argument('--lr_S', type=float, default=2e-3, help='learning rate')
parser.add_argument('--lr_O', type=float, default=1e-2, help='optimize target img learning rate')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--oh', type=float, default=1, help='one hot loss')
parser.add_argument('--ie', type=float, default=15, help='information entropy loss')
parser.add_argument('--a', type=float, default=0.1, help='activation loss')
parser.add_argument('--kd', type=float, default=1, help='knowledge distillation loss')
parser.add_argument('--output_dir', type=str, default='cache/models/')
parser.add_argument('--num_classes', type=int, help='num of classes in the dataset', default=10)
parser.add_argument('--img_opt_step', type=int, default=200, help='img optimization steps')
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)
teacher_path = opt.output_dir + 'teacher_{}.pt'.format(opt.dataset)
student_path = opt.output_dir + 'student_{}.pt'.format(opt.dataset)

cuda = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

accr = 0
accr_best = 0


def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, reduction='sum') / y.shape[0]
    return l_kl


def run():
    teacher = partial_load(LeNet5, teacher_path)
    teacher.eval()
    # freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False
    criterion = torch.nn.CrossEntropyLoss().to(device)

    teacher = nn.DataParallel(teacher)

    # Configure data loader
    net = LeNet5Half().to(device)
    net = nn.DataParallel(net)

    if opt.dataset == 'MNIST':
        data_test_loader = get_mnist(True, batch_size=opt.batch_size)
    else:
        data_test_loader = get_usps(True,  batch_size=opt.batch_size)
    if opt.target == 'USPS':
        tgt_loader = get_usps(True, batch_size=opt.batch_size)
    else:
        tgt_loader = get_mnist(True, batch_size=opt.batch_size)

    # Optimizers
    optimizer_student = torch.optim.Adam(net.parameters(), lr=opt.lr_S)

    # ----------
    #  Training
    # ----------

    accr_best = 0
    for epoch in range(opt.n_epochs):
        for step, (tgt_imgs, _) in enumerate(tgt_loader):
            net.train()
            optimizer_student.zero_grad()

            # initiate img with target data
            opt_imgs = tgt_imgs.clone().to(device)
            opt_imgs.requires_grad = True
            optimizer_img = torch.optim.Adam([opt_imgs], opt.lr_O)

            # optimize img
            for img_opt_step in range(opt.img_opt_step):
                output, feature = teacher(opt_imgs, out_feature=True)
                loss_oh = criterion(output, output.data.max(1)[1])
                loss_act = -feature.abs().mean()
                softmax_o = torch.nn.functional.softmax(output, dim=1).mean(dim=0)
                loss_ie = (softmax_o * torch.log(softmax_o)).sum()
                loss = loss_oh * opt.oh + loss_act * opt.a + loss_ie * opt.ie
                optimizer_img.zero_grad()
                loss.backward()
                optimizer_img.step()

            output = teacher(opt_imgs)
            loss_kd = kdloss(net(opt_imgs), output.detach())
            loss_kd.backward()
            optimizer_student.step()
            if step == 0:
                print("[Epoch %d/%d] [loss_kd: %f] " % (epoch, opt.n_epochs, loss_kd.item()))

        accr = eval_model(net, data_test_loader)
        if accr > accr_best:
            torch.save(net.state_dict(), student_path)
            accr_best = accr
    print('best accuracy is {}'.format(accr_best))


if __name__ == "__main__":
    run()
