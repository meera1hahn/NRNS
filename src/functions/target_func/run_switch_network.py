from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import tqdm
import os
import copy

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader

from src.functions.target_func.switch_data_loader import Loader
from src.functions.target_func.switch_model import XRN
from src.utils.cfg import input_paths

parser = argparse.ArgumentParser()
parser = input_paths(parser)
parser.add_argument("--run_name", type=str, default="switch_mlp")
parser.add_argument("--train", action="store_true", default=True)
parser.add_argument("--node_feat_size", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--dist_max", type=float, default=3.0)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="Weight decay (L2 loss on parameters).",
)
parser.add_argument("--early_stopping", type=int, default=10)
args = parser.parse_args()
args.base_dir += f"{args.dataset}/"
args.data_splits += f"{args.dataset}/"
# args.run_name += f"_{args.dataset}"
args.trajectory_data_dir = f"{args.base_dir}{args.trajectory_data_dir}"

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def load_data():
    loader = Loader(args)
    loader.build_dataset(
        split="train",
    )
    loader.build_dataset(
        split="valUnseen",
    )
    return loader


def train(model, train_iterator):
    train_loss = 0.0
    train_acc = []
    train_ratio = []
    pred_ratio = []
    dist_loss, switch_loss, rot_loss = [], [], []
    rot_err, dist_err = [], []
    for batch_data in tqdm.tqdm(train_iterator):
        losses, switch_acc, tr, pr, loc_err = model.train_emb(*batch_data)
        train_loss += sum(losses)
        train_acc.append(switch_acc)
        train_ratio.append(tr)
        pred_ratio.append(pr)
        rot_err.append(loc_err[0])
        dist_err.append(loc_err[1])

        dist_loss.append(losses[0])
        switch_loss.append(losses[1])
        rot_loss.append(losses[2])

    train_loss = train_loss / len(train_iterator)

    print(
        "Epoch: {:02d}".format(epoch + 1),
        "[Train- Switch]",
        "true_ratio: {:.4f}".format(np.mean(train_ratio)),
        "pred_ratio: {:.4f}".format(np.mean(pred_ratio)),
        "loss_train: {:.4f}".format(train_loss),
        "acc_train: {:.4f}".format(np.mean(train_acc) * 100),
    )
    print(
        "[Train - Dist]",
        "Error Dist: {:.4f}".format(np.mean(dist_err)),
    )
    print(
        "[Train - Angle]",
        "Error Angle: {:.4f}".format(np.mean(rot_err)),
    )
    print(
        "[Train - Losses]",
        "dist_loss: {:.4f}".format(np.mean(dist_loss)),
        "switch_loss: {:.4f}".format(np.mean(switch_loss)),
        "rot_loss: {:.4f}".format(np.mean(rot_loss)),
    )


def evaluate(model, data_iterator, split):
    with torch.no_grad():
        eval_loss = []
        eval_acc = []
        eval_ratio = []
        pred_ratio = []
        rot_err, dist_err = [], []

        for batch_data in tqdm.tqdm(data_iterator):
            if 1.0 not in batch_data[0].numpy():
                continue
            loss, switch_acc, tr, pr, loc_err = model.eval_emb(*batch_data)

            eval_loss.append(loss)
            eval_acc.append(switch_acc)
            eval_ratio.append(tr)
            pred_ratio.append(pr)
            rot_err.append(loc_err[0])
            dist_err.append(loc_err[1])

    print(
        "Epoch: {:02d}".format(epoch + 1),
        "[Val]",
        "true_ratio: {:.4f}".format(np.mean(eval_ratio)),
        "pred_ratio: {:.4f}".format(np.mean(pred_ratio)),
        "loss_" + split + ": {:.4f}".format(np.mean(eval_loss)),
        "acc_" + split + ": {:.4f}".format(np.mean(eval_acc) * 100),
    )
    print(
        "[Val - Dist]",
        "Error Dist:  {:.4f}".format(np.mean(dist_err)),
    )
    print(
        "[Val - Angle]",
        "Error Angle:  {:.4f}".format(np.mean(rot_err)),
    )
    return np.mean(eval_acc)


if __name__ == "__main__":
    # Load data
    loader = load_data()

    # Model and optimizer
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = XRN(args)

    train_iterator = DataLoader(
        loader.datasets["train"], batch_size=args.batch_size, shuffle=True
    )
    val_iterator = DataLoader(
        loader.datasets["valUnseen"], batch_size=args.batch_size, shuffle=False
    )

    if args.train:
        print("Starting Training...")
        best_val_acc = float("-inf")
        best_model = None
        patience = 0

        start_time = time.time()
        for epoch in range(args.epochs):
            model.train_start()
            train(model, train_iterator)
            model.eval_start()
            val_acc = evaluate(model, val_iterator, "validation")

            if val_acc > best_val_acc:
                patience = 0
                best_val_acc = val_acc
                """save model"""
                best_model = copy.deepcopy(model.get_model())
                save_path = os.path.join(
                    args.saved_model_dir,
                    "switch_func/",
                    args.run_name + "_distAcc{:.2f}_epoch{}.pt".format(val_acc, epoch),
                )
                torch.save(best_model.state_dict(), save_path)
                print("Saved model at:", str(save_path))
            else:
                patience += 1
                if patience >= args.early_stopping:
                    print("Patience reached... ended training")
                    break
            print("Patience:", patience)

        print("Training Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - start_time))
        model.set_model(best_model)
        test_acc = evaluate(model, val_iterator, "test")
        print("best model is saved at:", save_path)
