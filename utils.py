import torch.nn as nn
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
        acc += pred_cls.eq(labels.data).cpu().sum().item()

    loss /= len(data_loader)
    acc /= len(data_loader.dataset)

    print("Avg Loss = {}, Avg Accuracy = {:2%}".format(loss, acc))


def partial_load(model_cls, model_path):
    model = model_cls().to(device)
    model.eval()
    print("loading ", type(model).__name__, " from ", model_path)
    saved_state_dict = torch.load(model_path, map_location=device)
    model_state_dict = model.state_dict()
    # filter state dict
    filtered_dict = {k: v for k, v in saved_state_dict.items() if k in model_state_dict}
    if len(filtered_dict) == len(saved_state_dict):
        print("model fully fits saved weights, performing complete load")
    else:
        print("model and saved weights doesn't fully match, performing partial load. common states: ",
              len(filtered_dict), ", saved states: ", len(saved_state_dict))
    model_state_dict.update(filtered_dict)
    model.load_state_dict(model_state_dict)
    return model
