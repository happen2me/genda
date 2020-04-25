import argparse

from torch.autograd import Function
import torch.nn as nn
import torch.optim as optim
import torch
import numpy as np

from utils import partial_load, eval_model, eval_encoder_and_classifier, kd_loss_fn
from models.lenet import LeNet5, LeNet5Encoder, LeNet5Classifier
from models.lenet_half import LeNet5Half, LeNet5HalfEncoder, LeNet5HalfClassifier
from models.critic import Critic
from datasets.usps import get_usps
from datasets.genimg import get_genimg


encoder_path = 'cache/models/student.pt'

parser = argparse.ArgumentParser(description='adapt student model')
# Basic model parameters.
parser.add_argument('--lr', type=float, default='1e-3')
parser.add_argument('--n_epoch', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--temperature', type=int, default=8)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DannFullModel(nn.Module):
    def __init__(self):
        super(DannFullModel, self).__init__()
        self.encoder = LeNet5HalfEncoder()
        self.classifier = LeNet5HalfClassifier()
        self.discriminator = Critic(84, 84, 2)

    def forward(self, img, alpha):
        feature = self.encoder(img)
        class_label = self.classifier(feature)
        reversed_feature = ReverseLayerF.apply(feature, alpha)
        domain_label = self.discriminator(reversed_feature)
        return class_label, domain_label, feature

    def load(self, encoder_path=None, classifier_path=None):
        if encoder_path is not None:
            self.encoder = partial_load(LeNet5HalfEncoder, encoder_path)
        if classifier_path is not None:
            self.classifier = partial_load(LeNet5HalfClassifier, classifier_path)
        if self.classifier is not None:
            for p in self.classifier.parameters():
                p.requires_grad = False
        if self.encoder is not None:
            for p in self.encoder.parameters():
                p.requires_grad = True
        if self.discriminator is not None:
            for p in self.discriminator.parameters():
                p.requires_grad = True

    def get_encoder(self):
        return self.encoder

    def get_classifier(self):
        return self.classifier




def adapt(model, dataloader_source, dataloader_target, params, dataloader_target_eval=None):
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=params.lr)
    loss_class = torch.nn.NLLLoss().to(device)
    loss_domain = torch.nn.NLLLoss().to(device)
    loss_feature = kd_loss_fn

    ############
    # Training #
    ############

    len_dataloader = min(len(dataloader_source), len(dataloader_target))
    print("dataloader_source length is: {}, target length is: {}".format(len(dataloader_source), len(dataloader_target)))
    for epoch in range(params.n_epoch):
        for step, ((image_src, label_src), (image_tgt, _)) in enumerate(zip(dataloader_source, dataloader_target)):
            image_src = image_src.to(device)
            label_src = label_src.to(device)
            image_tgt = image_tgt.to(device)

            p = float(step + epoch * len_dataloader) / params.n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1
            model.zero_grad()
            # train with source data
            assert params.batch_size == len(label_src)  # make sure they have same batch size

            domain_src = torch.zeros(params.batch_size).long().to(device)

            label_src_pred, domain_src_pred, feature_src = model(image_src, alpha=alpha)

            # TODO: check compatibility
            err_src_label = loss_class(label_src_pred, torch.max(label_src.view(params.batch_size, 10), 1)[1])
            err_src_domain = loss_domain(domain_src_pred, domain_src)

            # train with target data
            assert params.batch_size == len(image_tgt)
            domain_tgt = torch.ones(params.batch_size).long().to(device)
            label_tgt_pred, domain_tgt_pred, feature_tgt = model(image_tgt, alpha=alpha)
            err_tgt_domain = loss_domain(domain_tgt_pred, domain_tgt)

            err = err_src_label + err_src_domain + err_tgt_domain

            try:
                err.backward()
                optimizer.step()
            except RuntimeError as e:
                print("backward failed: ", e)
                continue

            if dataloader_target_eval is not None:
                eval_encoder_and_classifier(model.get_encoder(), model.get_classifier(), dataloader_target_eval)


def run():
    src_data_loader = get_genimg(True, batch_size=args.batch_size)
    tgt_data_loader = get_usps(True, batch_size=args.batch_size)
    tgt_data_loader_eval = get_usps(False, batch_size=1024)

    model = DannFullModel()
    model.load(encoder_path, encoder_path)

    adapt(model, src_data_loader, tgt_data_loader, args, dataloader_target_eval=tgt_data_loader_eval)


if __name__ == '__main__':
    run()
