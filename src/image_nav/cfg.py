import argparse

parser = argparse.ArgumentParser(description="Image Nav Task")
parser.add_argument("--dataset", type=str, default="gibson")
parser.add_argument(
    "--path_type", type=str, default="straight", help="options: [straight, curved]"
)
parser.add_argument(
    "--difficulty", type=str, default="hard", help="options: [easy, medium, hard]"
)

parser.add_argument("--visualize", default=False, action="store_true")
parser.add_argument("--visualization_dir", type=str, default="visualizations/")

# Baselines
parser.add_argument("--behavioral_cloning", default=False, action="store_true")
parser.add_argument(
    "--bc_type", type=str, default="gru", help="options: [gru, map, random]"
)

# Abalations/GT
parser.add_argument("--use_gt_distances", default=False, action="store_true")
parser.add_argument("--use_gt_exporable_area", default=False, action="store_true")

# NOISE
parser.add_argument("--pose_noise", default=False, action="store_true")
parser.add_argument("--actuation_noise", default=False, action="store_true")

# RANDOM
parser.add_argument("--sample_used", type=float, default=1.0)
parser.add_argument("--max_steps", type=int, default=500)

# Data/Input Paths
parser.add_argument("--base_dir", type=str, default="../../data/topo_nav/")
parser.add_argument("--sim_dir", type=str, default="../../data/scene_datasets/")
parser.add_argument("--floorplan_dir", type=str, default="../../data/mp3d_floorplans/")

# Models
parser.add_argument("--model_dir", type=str, default="../../models/")
parser.add_argument(
    "--distance_model_path",
    type=str,
    default="distance_gcn.pt",
    help="options: [distance_gcn.pt, distance_gcn_noise.pt]",  # NO NOISE or NOISY
)

parser.add_argument(
    "--goal_model_path",
    type=str,
    default="goal_mlp.pt",
    help="options: [goal_mlp.pt, goal_mlp_noise.pt]",  # NO NOISE or NOISY
)

parser.add_argument(
    "--switch_model_path",
    type=str,
    default="switch_mlp.pt",
    help="options: [switch_mlp.pt, switch_mlp_noise.pt]",  # NO NOISE or NOISY
)

parser.add_argument(
    "--bc_model_path",
    type=str,
    default="bc_gru.pt",
    help="options: [bc_gru.pt, bc_metric_map.pt]",  # (Metric Map + GRU + weighting) or (ResNet + Prev action + GRU + weighting)
)


def parse_args():
    args = parser.parse_args()
    args.base_dir += f"{args.dataset}/"
    args.test_dir = f"{args.base_dir}image_nav_episodes/"

    if args.dataset == "mp3d":
        args.sim_dir += "mp3d/"
        args.bc_model_path = f"mp3d/mp3d_{args.bc_model_path}"
        args.switch_model_path = f"mp3d/mp3d_{args.switch_model_path}"
        args.goal_model_path = f"mp3d/mp3d_{args.goal_model_path}"
        args.distance_model_path = f"mp3d/mp3d_{args.distance_model_path}"
    else:
        args.sim_dir += "gibson_train_val/"
        args.bc_model_path = f"gibson/gibson_{args.bc_model_path}"
        args.switch_model_path = f"gibson/gibson_{args.switch_model_path}"
        args.goal_model_path = f"gibson/gibson_{args.goal_model_path}"
        args.distance_model_path = f"gibson/gibson_{args.distance_model_path}"
    return args
