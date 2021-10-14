from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import tqdm

import torch
import os
import copy
from torch.utils.data import DataLoader
from src.utils.cfg import input_paths

# Training settings
parser = argparse.ArgumentParser()
parser = input_paths(parser)
parser.add_argument("--bc_model_type", type=str, default="map", help="gru or map")
parser.add_argument("--run_name", type=str, default="action_network")
parser.add_argument("--train", action="store_true", default=True)
parser.add_argument("--node_feat_size", type=int, default=512)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument(
    "--weight_decay",
    type=float,
    default=5e-4,
    help="Weight decay (L2 loss on parameters).",
)
parser.add_argument("--early_stopping", type=int, default=20)
args = parser.parse_args()
args.base_dir += f"{args.dataset}/"
args.data_splits += f"{args.dataset}/"
args.run_name += f"_{args.bc_model_type}_{args.dataset}"

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
    train_loss = []
    train_acc = []
    ratios = []
    for batch_data in tqdm.tqdm(train_iterator):
        loss, acc, pred_ratio = model.train_emb(*batch_data)
        train_loss.append(loss)
        train_acc.append(acc)
        ratios.append(pred_ratio)
    print(
        "Epoch: {:02d}".format(epoch + 1),
        "[Train - Action]",
        "loss_train: {:.4f}".format(np.mean(train_loss)),
        "acc_train:  {:.4f}".format(np.mean(train_acc) * 100),
        "ratios: {:s}".format(str(np.mean(np.asarray(ratios), axis=0))),
    )


def evaluate(model, data_iterator, split, vis):
    with torch.no_grad():
        eval_loss = []
        eval_acc = []
        for batch_data in tqdm.tqdm(data_iterator):
            loss, acc = model.eval_emb(*batch_data)
            eval_loss.append(loss)
            eval_acc.append(acc)
    print(
        "Epoch: {:02d}".format(epoch + 1),
        "[Validation - Action]",
        "loss_" + split + ": {:.4f}".format(np.mean(eval_loss)),
        "acc_" + split + ":  {:.4f}".format(np.mean(eval_acc) * 100),
    )
    return np.mean(eval_loss)


if __name__ == "__main__":
    # Metric Map or GRU
    if args.bc_model_type == "map":
        from src.functions.bc_func.bc_map_network import XRN
        from src.functions.bc_func.bc_map_loader import Loader
    else:  # bc_model_type == "gru"
        from src.functions.bc_func.bc_gru_network import XRN
        from src.functions.bc_func.bc_gru_loader import Loader

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
        best_val_loss = float("inf")
        best_model = None
        patience = 0

        start_time = time.time()
        for epoch in range(args.epochs):
            model.train_start()
            train(model, train_iterator)
            model.eval_start()
            eval_loss = evaluate(model, val_iterator, "validation", False)

            if eval_loss < best_val_loss:
                patience = 0
                best_val_loss = eval_loss
                """save model"""
                best_model = copy.deepcopy(model.get_model())
                save_path = os.path.join(
                    args.saved_model_dir,
                    "action_func/",
                    args.run_name + "_Acc{:.2f}_epoch{}.pt".format(eval_loss, epoch),
                )
                torch.save(best_model.state_dict(), save_path)
                print("Saved model at: ", str(save_path))
            else:
                patience += 1
                if patience >= args.early_stopping:
                    print("Patience reached... ended training")
                    break
            print("Patience:", patience)

        print("Training Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - start_time))
