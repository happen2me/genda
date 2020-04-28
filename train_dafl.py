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
from utils import partial_load

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','cifar10','cifar100'])
parser.add_argument('--data', type=str, default='cache/data/')
parser.add_argument('--teacher_dir', type=str, default='cache/models/')
parser.add_argument('--n_epochs', type=int, default=600, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
parser.add_argument('--lr_G', type=float, default=0.2, help='learning rate')
parser.add_argument('--lr_S', type=float, default=2e-3, help='learning rate')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--oh', type=float, default=1, help='one hot loss')
parser.add_argument('--ie', type=float, default=10, help='information entropy loss')
parser.add_argument('--a', type=float, default=0.1, help='activation loss')
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
    l_kl = F.kl_div(p, q, size_average=False) / y.shape[0]
    return l_kl


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
    net = nn.DataParallel(net)
    data_test = MNIST(opt.data,
                      train=False,
                      transform=transforms.Compose([
                          transforms.Resize((32, 32)),
                          transforms.ToTensor(),
                          transforms.Normalize((0.1307,), (0.3081,))
                          ]))
    data_test_loader = DataLoader(data_test, batch_size=64, num_workers=1, shuffle=False)

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

    batches_done = 0
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
            # pred = outputs_T.data.max(1)[1]
            loss_activation = -features_T.abs().mean()
            # loss_one_hot = criterion(outputs_T,pred)
            loss_condition = criterion(outputs_T, labels.view(opt.batch_size))
            softmax_o_T = torch.nn.functional.softmax(outputs_T, dim = 1).mean(dim = 0)
            # loss_information_entropy = (softmax_o_T * torch.log(softmax_o_T)).sum()
            loss = loss_condition * opt.oh + loss_activation * opt.a
            loss_kd = kdloss(net(gen_imgs.detach()), outputs_T.detach())
            loss += loss_kd
            loss.backward()
            optimizer_G.step()
            optimizer_S.step()
            if i == 1:
                print ("[Epoch %d/%d] [loss_condition: %f] [loss_ie: %f] [loss_a: %f] [loss_kd: %f]" % (epoch, opt.n_epochs,loss_condition.item(), loss_information_entropy.item(), loss_activation.item(), loss_kd.item()))

        with torch.no_grad():
            for i, (images, labels) in enumerate(data_test_loader):
                images = images.to(device)
                labels = labels.to(device)
                net.eval()
                output = net(images)
                avg_loss += criterion(output, labels).sum()
                pred = output.data.max(1)[1]
                total_correct += pred.eq(labels.data.view_as(pred)).sum()

        avg_loss /= len(data_test)
        print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), float(total_correct) / len(data_test)))
        accr = round(float(total_correct) / len(data_test), 4)
        if accr > accr_best:
            torch.save(net.state_dict(), opt.output_dir + 'student.pt')
            torch.save(generator.state_dict(), opt.output_dir + "generator.pt")
            accr_best = accr


if __name__ == "__main__":
    run()
