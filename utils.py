import torch.nn as nn
import torch.nn.functional as F
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval_model(model, data_loader):
    """Evaluation for target encoder by source classifier on target dataset."""
    # set eval state for Dropout and BN layers
    model.eval()

    # init loss and accuracy
    loss = 0
    acc = 0

    # set loss function
    criterion = nn.CrossEntropyLoss()

    # evaluate network
    for (images, labels) in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images)
        loss += criterion(preds, labels).data.item()

        pred_cls = preds.data.max(1)[1]
        acc += pred_cls.eq(labels.data).sum().item()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))

    return acc

def eval_encoder_and_classifier(encoder, classifier, data_loader):
    class Full(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.classifier = classifier

        def forward(self, img):
            feature = self.encoder(img)
            output = self.classifier(feature)
            return output

    full = Full()
    eval_model(full, data_loader)


def alter_dict_key(state_dict):
    new_dict = {}
    for key, val in state_dict.items():
        new_dict[key[7:]] = val
    return new_dict


def partial_load(model_cls, model_path):
    model = model_cls().to(device)
    model.eval()
    print("loading ", type(model).__name__, " from ", model_path)
    saved_state_dict = torch.load(model_path, map_location=device)

    # remove leading 'module.' in state dict if needed
    alter = False
    for key, val in saved_state_dict.items():
        if key[:7] == 'module.':
            alter = True
        break
    if alter:
        print("keys in state dict starts with 'module.', trimming it.")
        saved_state_dict = alter_dict_key(saved_state_dict)

    model_state_dict = model.state_dict()
    # filter state dict
    filtered_dict = {k: v for k, v in saved_state_dict.items() if k in model_state_dict}
    if len(filtered_dict) == len(saved_state_dict):
        print("model fully fits saved weights, performing complete load")
    else:
        print("model and saved weights doesn't fully match, performing partial load. common states: ",
              len(filtered_dict), ", saved states: ", len(saved_state_dict))
        print("an item in saved dict is: ")
        for key, val in saved_state_dict.items():
            print(key)
            break
    model_state_dict.update(filtered_dict)
    model.load_state_dict(model_state_dict)
    return model


def kd_loss_fn(s_output, t_output, temperature, labels=None, alpha=0.4, weights=None):
    s_output = F.log_softmax(s_output/temperature, dim=1)
    t_output = F.softmax(t_output/temperature, dim=1)
    kd_loss = F.kl_div(s_output, t_output, reduction='batchmean')
    entropy_loss = kd_loss if labels is None else F.cross_entropy(s_output, labels)
    loss = (1-alpha)*entropy_loss + alpha*kd_loss*temperature*temperature
    return loss
