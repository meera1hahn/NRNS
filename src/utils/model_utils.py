import numpy as np
import scipy.sparse as sp
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn


def convert_model_to_state(model, args):
    state = {"args": vars(args), "state_dict": {}}
    # use copies instead of references
    for k, v in model.state_dict().items():
        state["state_dict"][k] = v.clone().to(torch.device("cpu"))
    return state


def load_resnet():
    resnet = models.resnet18(pretrained=True)
    resnet = nn.Sequential(*(list(resnet.children())[:-1]))
    resnet.eval()
    resnet.to("cuda")
    return resnet


def load_places_resnet():
    # load the pre-trained weights
    model = models.__dict__["resnet18"](num_classes=365)
    model_file = "../../models/resnet18_places365.pth.tar"
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {
        str.replace(k, "module.", ""): v for k, v in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model = nn.Sequential(*(list(model.children())[:-1]))
    model.eval()
    model.to("cuda")
    return model


def get_res_feats_batch(filename, images, resnet):
    batch_size = 54
    mean = (
        torch.tensor([0.485, 0.456, 0.406])
        .view(1, 3, 1, 1)
        .to(device=next(resnet.parameters()).device)
    )
    std = (
        torch.tensor([0.229, 0.224, 0.225])
        .view(1, 3, 1, 1)
        .to(device=next(resnet.parameters()).device)
    )

    with torch.no_grad():
        feats = []
        for i in range(0, len(images), batch_size):
            imgs = (
                torch.stack(
                    [torch.tensor(img) for img in images[i : i + batch_size]], 0
                )
                .to(device=next(resnet.parameters()).device)
                .permute(0, 3, 1, 2)
                .float()
                / 255
                - mean
            ) / std
            feats.append(resnet(imgs).detach().cpu())
        torch.save(torch.cat(feats, dim=0).squeeze(), filename)
        feats = []


def get_res_feats(img, resnet):
    mean = (
        torch.tensor([0.485, 0.456, 0.406])
        .view(1, 3, 1, 1)
        .to(device=next(resnet.parameters()).device)
    )
    std = (
        torch.tensor([0.229, 0.224, 0.225])
        .view(1, 3, 1, 1)
        .to(device=next(resnet.parameters()).device)
    )

    img = (
        torch.tensor(img)
        .unsqueeze(0)
        .to(device=next(resnet.parameters()).device)
        .permute(0, 3, 1, 2)
        .float()
        / 255
        - mean
    ) / std

    feats = resnet(img).detach().cpu().view(-1).unsqueeze(0)
    return feats


def calculate_spl(success, length_shortest, length_taken):
    spl = length_shortest / max(length_shortest, length_taken)
    if not success:
        spl = 0
    return spl
