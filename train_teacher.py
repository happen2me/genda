import os
from models.lenet import LeNet5
import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse
from datasets.mnist import get_mnist
from datasets.usps import get_usps
from datasets.mnist_m import get_mnist_m
from datasets.svhn import get_svhn
from utils import eval_model

parser = argparse.ArgumentParser(description='train-teacher-network')

# Basic model parameters.
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'MNIST-M', 'USPS', 'SVHN'])
parser.add_argument('--data', type=str, default='cache/data/')
parser.add_argument('--output_dir', type=str, default='cache/models/')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.CrossEntropyLoss().to(device)

acc = 0
acc_best = 0


def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if epoch < 80:
        lr = 0.1
    elif epoch < 120:
        lr = 0.01
    else:
        lr = 0.001
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(net, data_train_loader, optimizer, epoch):
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for step, (images, labels) in enumerate(data_train_loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        output = net(images)

        loss = criterion(output, labels)

        loss_list.append(loss.data.item())
        batch_list.append(step + 1)

        if step == 1:
            print('Train - Epoch %d, Batch: %d, Loss: %f' % (epoch, step, loss.data.item()))

        loss.backward()
        optimizer.step()


def test(net, data_test_loader):
    global acc, acc_best
    net.eval()
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = images.to(device), labels.to(device)
            output = net(images)
            avg_loss += criterion(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test_loader.dataset)
    acc = float(total_correct) / len(data_test_loader.dataset)
    if acc_best < acc:
        acc_best = acc
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), acc))


def train_and_test(net, data_train_loader, data_test_loader, optimizer, epoch):
    train(net, data_train_loader, optimizer, epoch)
    test(net, data_test_loader)


def main():
    if args.dataset == "MNIST":
        data_train_loader = get_mnist(True, args.batch_size)
        data_test_loader = get_mnist(False, args.batch_size)
    elif args.dataset == "USPS":
        data_train_loader = get_usps(True, args.batch_size)
        data_test_loader = get_usps(False, args.batch_size)
    elif args.dataset == "MNIST-M":
        data_train_loader = get_mnist_m(True, args.batch_size)
        data_test_loader = get_mnist_m(False, args.batch_size)
    elif args.dataset == 'SVHN':
        data_train_loader = get_svhn(True, args.batch_size)
        data_test_loader = get_svhn(False, args.batch_size)
    else:
        print('Dataset {} not find. Program terminated'.format(args.dataset))
        return

    if args.dataset == "MNIST-M" or args.dataset == 'SVHN':
        net = LeNet5(channel=3).to(device)
    else:
        net = LeNet5().to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    if args.dataset == 'MNIST':
        epoch = 10  # if dataset is MNIST or USPS, 10 for epoch is enough
    elif args.dataset == 'USPS':
        epoch = 20
    else:
        epoch = 100
    for e in range(1, epoch):
        train_and_test(net, data_train_loader, data_test_loader, optimizer, e)
    torch.save(net.state_dict(), args.output_dir + 'teacher_{}.pt'.format(args.dataset))
    print("teacher model saved at ", args.output_dir, 'teacher_{}.pt'.format(args.dataset))


if __name__ == '__main__':
    main()
