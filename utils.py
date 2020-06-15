import torch.nn as nn
import torch.nn.functional as F
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def eval_model(model, data_loader):
    """在data_loader上衡量model的精确度"""
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
    """在data_loader上评估把encoder和classifier合在一起的模型的准确度"""
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
    return eval_model(full, data_loader)


def alter_dict_key(state_dict):
    """删除掉state_dict字典里的头7个字符"""
    new_dict = {}
    for key, val in state_dict.items():
        new_dict[key[7:]] = val
    return new_dict


def partial_load_model(model, model_path):
    """从model_path加载权重到model。仅加载key值一致的权重，忽略其它。"""
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


def partial_load(model_cls, model_path):
    """部分加载权重

    参数:
       model_cls: 模型的类定义
       model_path: 权重路径
    """
    model = model_cls().to(device)
    partial_load_model(model, model_path)


def kd_loss_fn(s_output, t_output, temperature, labels=None, alpha=0.4, weights=None):
    """蒸馏损失函数定义"""
    s_output = F.log_softmax(s_output/temperature, dim=1)
    t_output = F.softmax(t_output/temperature, dim=1)
    kd_loss = F.kl_div(s_output, t_output, reduction='batchmean')
    entropy_loss = kd_loss if labels is None else F.cross_entropy(s_output, labels)
    loss = (1-alpha)*entropy_loss + alpha*kd_loss*temperature*temperature
    return loss
