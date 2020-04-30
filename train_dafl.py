import argparse
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.transforms as transforms
from torchvision.datasets.mnist import MNIST

from models.lenet_half import LeNet5Half, LeNet5HalfEncoder
from models.generator import Generator
from models.lenet import LeNet5
from datasets.mnist import get_mnist
from datasets.usps import get_usps
from utils import partial_load, eval_model, alter_dict_key

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','cifar10','cifar100'])
parser.add_argument('--data', type=str, default='cache/data/')
parser.add_argument('--teacher_dir', type=str, default='cache/models/')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--lr_G', type=float, default=0.2, help='learning rate')
parser.add_argument('--lr_S', type=float, default=2e-3, help='learning rate')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--oh', type=float, default=1, help='one hot loss')
parser.add_argument('--ie', type=float, default=20, help='information entropy loss')
parser.add_argument('--a', type=float, default=0.1, help='activation loss')
parser.add_argument('--kd', type=float, default=1, help='knowledge distillation loss')
parser.add_argument('--output_dir', type=str, default='cache/models/')
parser.add_argument('--num_classes', type=int, help='num of classes in the dataset', default=10)
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)
teacher_path = opt.output_dir + 'teacher.pt'

cuda = True
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

accr = 0
accr_best = 0


def kdloss(y, teacher_scores):
    p = F.log_softmax(y, dim=1)
    q = F.softmax(teacher_scores, dim=1)
    l_kl = F.kl_div(p, q, reduction='sum') / y.shape[0]
    return l_kl


def load_classifier(student):
    teacher_dict = torch.load(teacher_path)
    # remove leading 'module.' in state dict if needed
    alter = False
    for key, val in teacher_dict.items():
        if key[:7] == 'module.':
            alter = True
        break
    if alter:
        print("keys in state dict starts with 'module.', trimming it.")
        teacher_dict = alter_dict_key(teacher_dict)
    classifier_dict = {k: v for k, v in teacher_dict.items() if k.startwith('f5')}
    student_state_dict = student.state_dict()
    student_state_dict.update(classifier_dict)
    student.load_state_dict(student_state_dict)
    for i, p in student.parameters():
        if i > 9:
            p.requires_grad = False
    return student


def run():
    generator = Generator().to(device)
    teacher = partial_load(LeNet5, teacher_path)
    teacher.eval()
    # freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False
    criterion = torch.nn.CrossEntropyLoss().to(device)

    teacher = nn.DataParallel(teacher)
    generator = nn.DataParallel(generator)

    # Configure data loader
    net = LeNet5Half().to(device)
    net = load_classifier(net)
    net = nn.DataParallel(net)



    data_test_loader = get_mnist(True, batch_size=opt.batch_size)

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_G)
    optimizer_S = torch.optim.Adam(net.parameters(), lr=opt.lr_S)

    def adjust_learning_rate(optimizer, epoch, learing_rate):
        if epoch < 800:
            lr = learing_rate
        elif epoch < 1600:
            lr = 0.1*learing_rate
        else:
            lr = 0.01*learing_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # ----------
    #  Training
    # ----------

    accr_best = 0
    for epoch in range(opt.n_epochs):

        total_correct = 0
        avg_loss = 0.0

        for i in range(120):
            net.train()
            z = torch.randn(opt.batch_size, opt.latent_dim).to(device)

            # generate random labels
            labels = torch.LongTensor(opt.batch_size, 1).random_() % opt.num_classes
            labels_onehot = torch.FloatTensor(opt.batch_size, opt.num_classes)
            labels_onehot.zero_()
            labels_onehot.scatter_(1, labels, 1)
            labels = labels.to(device)
            labels_onehot = labels_onehot.to(device)
            z = torch.cat((z, labels_onehot), dim=1)

            optimizer_G.zero_grad()
            optimizer_S.zero_grad()
            gen_imgs = generator(z)
            outputs_T, features_T = teacher(gen_imgs, out_feature=True)
            pred = outputs_T.data.max(1)[1]
            loss_activation = -features_T.abs().mean()
            loss_one_hot = criterion(outputs_T,pred)
            # loss_condition = criterion(outputs_T, labels.view(opt.batch_size))
            softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
            loss_information_entropy = (softmax_o_T * torch.log(softmax_o_T)).sum()
            loss_kd = kdloss(net(gen_imgs.detach()), outputs_T.detach())
            loss = loss_one_hot * opt.oh + loss_activation * opt.a + loss_kd * opt.kd + loss_information_entropy * opt.ie
            loss.sum().backward()
            optimizer_G.step()
            optimizer_S.step()
            if i == 1:
                print ("[Epoch %d/%d] [loss_one_hot: %f] [loss_a: %f] [loss_kd: %f]" % (epoch, opt.n_epochs,loss_one_hot.item(), loss_activation.item(), loss_kd.item()))

        accr = eval_model(net, data_test_loader)

        if accr > accr_best:
            torch.save(net.state_dict(), opt.output_dir + 'student.pt')
            torch.save(generator.state_dict(), opt.output_dir + "generator.pt")
            accr_best = accr
    print('best accuracy is {}'.format(accr_best))


if __name__ == "__main__":
    run()
