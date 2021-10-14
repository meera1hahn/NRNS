from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import tqdm
import os
import copy

import torch
from torch.utils.data import DataLoader
from src.functions.target_func.goal_data_loader import Loader
from src.functions.target_func.goal_model import XRN
from src.utils.cfg import input_paths

# Training settings
parser = argparse.ArgumentParser()
parser = input_paths(parser)
parser.add_argument("--run_name", type=str, default="goal_mlp")
parser.add_argument("--train", action="store_true", default=True)
parser.add_argument("--node_feat_size", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--dist_max", type=float, default=3.0)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--early_stopping", type=int, default=20)
args = parser.parse_args()
args.base_dir += f"{args.dataset}/"
args.data_splits += f"{args.dataset}/"
args.run_name += f"_{args.dataset}"
args.trajectory_data_dir = f"{args.base_dir}{args.trajectory_data_dir}"

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


def load_data():
    loader = Loader(args)
    loader.build_dataset(split="train")
    loader.build_dataset(split="valUnseen")
    return loader


def train(model, train_iterator):
    train_loss = []
    train_acc = []
    dist_loss, rot_loss = [], []
    dist_err = []
    for batch_data in tqdm.tqdm(train_iterator):
        loss, losses, location_err, dist_output, rot_output = model.train_emb(
            *batch_data
        )
        train_loss.append(loss)
        dist_err.append(location_err)
        dist_loss.append(losses[0])
        rot_loss.append(losses[1])

    train_acc = (np.asarray(dist_err) <= 0.1).sum() * 1.0 / len(dist_err)

    print(
        "Epoch: {:02d}".format(epoch + 1),
        "[Train]",
        "dist_loss:  {:.4f}".format(np.mean(dist_loss)),
        "rot_loss:  {:.4f}".format(np.mean(rot_loss)),
        "loss_train: {:.4f}".format(np.mean(train_loss)),
        "acc_train  @.1m:  {:.4f}".format(train_acc),
    )
    print(
        "[Train - Dist]",
        "Error Dist:  {:.4f}".format(np.mean(dist_err)),
    )


def evaluate(model, data_iterator, split):
    with torch.no_grad():
        eval_loss = []
        eval_acc = []
        dist_err = []
        outputs = [[], []]

        for batch_data in tqdm.tqdm(data_iterator):
            loss, _, location_err, dist_output, rot_output = model.eval_emb(*batch_data)
            eval_loss.append(loss)
            dist_err.append(location_err)
            outputs[0].extend(dist_output)
            outputs[1].extend(rot_output)

    eval_acc = (np.asarray(dist_err) <= 0.1).sum() * 1.0 / len(dist_err)
    print(
        "Epoch: {:02d}".format(epoch + 1),
        "[Val]",
        "loss_" + split + ": {:.4f}".format(np.mean(eval_loss)),
        "acc_" + split + " @.1m:  {:.4f}".format(eval_acc),
    )
    print(
        "[Val - Dist2Goal]",
        "Error: {:.4f}".format(np.mean(dist_err)),
        "Distance: {:.4f}".format(np.mean(outputs[0])),
        "Rot Mean: {:.4f}".format(np.mean(outputs[1])),
        "Rot Std: {:.4f}".format(np.std(outputs[1])),
        "Rot min: {:.4f}".format(np.min(outputs[1])),
        "Rot max: {:.4f}".format(np.max(outputs[1])),
    )
    return eval_acc, np.mean(dist_err)


if __name__ == "__main__":
    print("Runnning", args.run_name)
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
        print("Starting Training...", args.run_name)
        best_val_dist = float("inf")
        best_model = None
        patience = 0

        start_time = time.time()
        for epoch in range(args.epochs):
            model.train_start()
            train(model, train_iterator)
            model.eval_start()
            val_acc, dist_err = evaluate(model, val_iterator, "validation")

            if dist_err < best_val_dist:
                patience = 0
                best_val_dist = dist_err
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
        print("best model is saved at:", save_path)

        print("Running Test Data")
        model.set_model(best_model)
        test_acc = evaluate(model, val_iterator, "test")
