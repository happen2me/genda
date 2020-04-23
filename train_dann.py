from torch.autograd import Function
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch

from utils import partial_load, eval_model

from models.lenet import LeNet5, LeNet5Encoder, LeNet5Classifier
from models.lenet_half import LeNet5Half, LeNet5HalfEncoder
from models.discriminator import Discriminator

from datasets.usps import get_usps


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
        self.lenet5half_encoder = LeNet5HalfEncoder()
        self.classifier = LeNet5Classifier()
        self.discriminator = Discriminator(84, 84, 2)
        self.src_encoder = LeNet5Encoder()

    def forward(self, img, alpha):
        feature = self.lenet5half_encoder(img)
        src_feature = self.src_encoder(img)
        class_label = self.classifier(feature)
        reversed_feature = ReverseLayerF.apply(feature, alpha)
        domain_label = self.discriminator(reversed_feature)
        return class_label, domain_label, feature, src_feature

    def load(self, encoder_path=None, classifier_path=None, src_encoder_path=None):
        if encoder_path is not None:
            self.lenet5half_encoder = partial_load(LeNet5HalfEncoder, encoder_path)
        if classifier_path is not None:
            self.classifier = partial_load(LeNet5Classifier, classifier_path)
        if src_encoder_path is not None:
            self.src_encoder = partial_load(LeNet5Encoder, src_encoder_path)
        if self.classifier is not None:
            for p in self.classifier.parameters():
                p.requires_grad = False
        if self.lenet5half_encoder is not None:
            for p in self.lenet5half_encoder.parameters():
                p.requires_grad = True
        if self.discriminator is not None:
            for p in self.discriminator.parameters():
                p.requires_grad = True
        if self.src_encoder is not None:
            for p in self.src_encoder.parameters():
                p.requires_grad = False

    def get_encoder(self):
        return self.lenet5half_encoder

    def get_classifier(self):
        return self.classifier


def kd_loss_fn(s_output, t_output, temperature, labels=None, alpha=0.4, weights=None):
    s_output = F.log_softmax(s_output/temperature, dim=1)
    t_output = F.softmax(t_output/temperature, dim=1)
    kd_loss = F.kl_div(s_output, t_output, reduction='batchmean')
    entropy_loss = kd_loss if labels is None else F.cross_entropy(s_output, labels)
    loss = (1-alpha)*entropy_loss + alpha*kd_loss*temperature*temperature
    return loss


def adapt(model, dataloader_source, dataloader_target, params, dataloader_target_eval = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

            label_src_pred, domain_src_pred, feature_src, feature_src_org = model(image_src, alpha=alpha)
            # don't optimize encoder with src image? 相当于这里只训练域判别器?

            # print("==========================================")
            # print("label src pred shape ", label_src_pred.shape)
            # print("label src shape ", label_src.shape)
            # print("label src shape reshape", label_src.view(50, 10).shape)
            # print("==========================================")

            err_src_label = loss_class(label_src_pred, torch.max(label_src.view(50, 10), 1)[1])

            err_src_domain = loss_domain(domain_src_pred, domain_src)


            # train with target data
            assert params.batch_size == len(image_tgt)
            domain_tgt = torch.ones(params.batch_size).long().to(device)
            label_tgt_pred, domain_tgt_pred, feature_tgt, feature_tgt_org = model(image_tgt, alpha=0)
            err_tgt_domain = loss_domain(domain_tgt_pred, domain_tgt)

            err_kd_loss = loss_feature(feature_src_org, feature_src, params.temperature) \
                          + loss_feature(feature_tgt_org, feature_tgt, params.temperature)

            err = err_src_label + err_src_domain + err_tgt_domain + err_kd_loss

            try:
                err.backward()
                optimizer.step()
            except RuntimeError as e:
                print("backward failed: ", e)
                continue

            if dataloader_target_eval is not None:
                eval_model(model.get_encoder(), model.get_classifier(), dataloader_target_eval)


def run():
    tgt_data_loader = get_usps(True, batch_size=256)
    tgt_data_loader_eval = get_usps(False, batch_size=1024)