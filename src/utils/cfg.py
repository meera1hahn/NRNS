import argparse
import datetime

# Data/Input Paths
def input_paths(parser):
    parser.add_argument(
        "--noise",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        # default="gibson",
        default="mp3d",
    )
    parser.add_argument(
        "--sim_dir",
        type=str,
        default="../../data/scene_datasets/",
    )
    parser.add_argument(
        "--data_splits",
        type=str,
        default="../../data/data_splits/",
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        default="../../data/topo_nav/",
    )
    # generated data
    parser.add_argument(
        "--trajectory_data_dir",
        type=str,
        default="no_noise/trajectory_data/",
    )
    parser.add_argument(
        "--clustered_graph_dir",
        type=str,
        default="no_noise/clustered_graph/",
    )
    parser.add_argument(
        "--distance_data_dir",
        type=str,
        default="no_noise/distance_data_straight/",
    )
    parser.add_argument(
        "--action_dir",
        type=str,
        default="behavioral_cloning/",
    )
    # save folders
    parser.add_argument(
        "--saved_model_dir",
        type=str,
        default="../../models/",
    )
    parser.add_argument(
        "--submitit_log_dir",
        type=str,
        default="../../logs/submitit/log_test",
    )
    parser.add_argument(
        "--summary_dir",
        type=str,
        default="../../logs/tensorboard/",
    )

    return parser
