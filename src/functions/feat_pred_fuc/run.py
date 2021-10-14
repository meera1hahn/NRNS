from __future__ import division
from __future__ import print_function

import time
import argparse
import numpy as np
import tqdm
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import DataListLoader
from src.functions.feat_pred_fuc.deepgcn import XRN
from src.functions.feat_pred_fuc.batch_traj_loader import Loader
from src.utils.cfg import input_paths

# Training settings
parser = argparse.ArgumentParser()
parser = input_paths(parser)
parser.add_argument("--run_name", type=str, default="CGConv")
parser.add_argument("--train", action="store_true", default=True)
parser.add_argument("--test", action="store_true", default=True)
parser.add_argument("--dropout", type=float, default=0.5)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--seed", type=int, default=42, help="Random seed.")
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--weight_decay", type=float, default=5e-4)
parser.add_argument("--early_stopping", type=int, default=15)
args = parser.parse_args()
args.base_dir += f"{args.dataset}/"
args.data_splits += f"{args.dataset}/"
args.run_name += f"_{args.dataset}"
args.trajectory_data_dir = f"{args.base_dir}{args.trajectory_data_dir}"
args.clustered_graph_dir = f"{args.base_dir}{args.clustered_graph_dir}"
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
writer = SummaryWriter(args.summary_dir + args.run_name)


def load_data():
    loader = Loader(args)
    loader.build_dataset(split="train")
    loader.build_dataset(split="valUnseen")
    return loader


def evaluate(model, train_iterator):
    model.eval_start()
    eval_loss, eval_err, eval_acc, eval_acctopk = [], [], [], []
    for data_list in tqdm.tqdm(train_iterator):
        loss, dist_err, dist_acc, topk_acc = model.eval_emb(data_list)
        eval_loss.append(loss)
        eval_err.append(dist_err)
        eval_acc.append(dist_acc)
        eval_acctopk.append(topk_acc)
    mode = "Val"
    writer.add_scalar("Loss/" + mode, np.mean(eval_loss), epoch)
    writer.add_scalar("Error/" + mode, np.mean(eval_err), epoch)
    writer.add_scalar("DistAcc/" + mode, np.mean(eval_acc), epoch)
    writer.add_scalar("TopKAcc/" + mode, np.mean(eval_acctopk), epoch)
    print(
        "Epoch: {:02d}".format(epoch + 1),
        "--Eval:",
        "eval_loss: {:.4f}".format(np.mean(eval_loss)),
        "eval_dist_err: {:.4f}".format(np.mean(eval_err)),
        "eval_dist_acc: {:.4f}".format(np.mean(eval_acc)),
        "eval_topk_acc: {:.4f}".format(np.mean(eval_acctopk)),
    )

    return np.mean(eval_acctopk)


def train(model, train_iterator):
    model.train_start()
    train_loss, train_err, train_acc, train_acctopk = [], [], [], []
    for data_list in tqdm.tqdm(train_iterator):
        loss, dist_err, dist_acc, topk_acc = model.train_emb(data_list)
        train_loss.append(loss)
        train_err.append(dist_err)
        train_acc.append(dist_acc)
        train_acctopk.append(topk_acc)
    mode = "Train"
    writer.add_scalar("Loss/" + mode, np.mean(loss), epoch)
    writer.add_scalar("Error/" + mode, np.mean(train_err), epoch)
    writer.add_scalar("DistAcc/" + mode, np.mean(train_acc), epoch)
    writer.add_scalar("TopKAcc/" + mode, np.mean(train_acctopk), epoch)

    print(
        "Epoch: {:02d}".format(epoch + 1),
        "--Train:",
        "train_loss: {:.4f}".format(np.mean(train_loss)),
        "train_dist_err: {:.4f}".format(np.mean(train_err)),
        "train_dist_acc: {:.4f}".format(np.mean(train_acc)),
        "train_topk_acc: {:.4f}".format(np.mean(train_acctopk)),
    )


if __name__ == "__main__":

    # Load data
    loader = load_data()
    train_iterator = DataListLoader(
        loader.datasets["train"], batch_size=args.batch_size, shuffle=True
    )
    val_iterator = DataListLoader(
        loader.datasets["valUnseen"], batch_size=args.batch_size, shuffle=True
    )

    # Create Model
    model = XRN(args)

    # Train Model
    print("Starting Training...")
    save_path = ""
    best_val_acc = float("-inf")
    best_model = None
    patience = 0

    start_time = time.time()

    for epoch in range(args.epochs):
        train(model, train_iterator)
        val_acc = evaluate(model, val_iterator)
        model.my_lr_scheduler.step()

        best_model = model.get_model()
        save_path = os.path.join(
            args.saved_model_dir + "feat_pred/",
            args.run_name + "_unseenAcc{:.2f}_epoch{}.pt".format(val_acc, epoch),
        )
        torch.save(best_model, save_path)
        print("Saved model at:", str(save_path))

        if val_acc > best_val_acc:
            patience = 0
            best_val_acc = val_acc
        else:
            patience += 1
            if patience >= args.early_stopping:
                print("Patience reached... ended training")
                break

        print("Patience:", patience)

    print("Training Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - start_time))

    test_iterator = DataListLoader(
        loader.datasets["valUnseen"], batch_size=args.batch_size, shuffle=True
    )

    # Evaluate Best Model
    model.set_model(best_model)
    model.eval_start()
    test_acc = evaluate(model, test_iterator)
    print("Testing Finished!")
