import torch
import numpy as np
import tqdm, json, gzip, quaternion
from habitat.utils.geometry_utils import quaternion_from_coeff

from src.utils.model_utils import load_places_resnet
from src.image_nav.visualize import Visualizer
from src.image_nav.cfg import parse_args
from src.image_nav.agent import Agent
from src.image_nav.navigate import single_image_nav
from src.image_nav.bcloning_navigate import single_image_nav_BC
from src.image_nav.utils import load_models, evaluate_episode
from src.utils.sim_utils import (
    set_up_habitat_noise,
    add_noise_actions_habitat,
)


turn_angle = 15


def create_habitat(args, sim, current_scan):
    if sim is not None:
        sim.close()
    if args.dataset == "mp3d":
        scene = "{}{}/{}.glb".format(args.sim_dir, current_scan, current_scan)
    else:
        scene = "{}{}.glb".format(args.sim_dir, current_scan)

    return set_up_habitat_noise(scene, turn_angle)


def main(args):
    print(f"starting run: {args.path_type} data")
    print("Data Type:", args.difficulty)
    print(f"Pose Noise: {args.pose_noise}; Actuation Noise: {args.actuation_noise}")
    """Evaluation Metrics"""
    maxed_out = 0
    rates = {
        "success": [],
        "spl": [],
        "dist2goal": [],
        "taken_path_total": [],
        "taken_path_success": [],
        "gt_path_total": [],
        "gt_path_success": [],
    }
    visCounter = {}
    visualizer = Visualizer(args)

    """Load Models"""
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    resnet = load_places_resnet()
    model_switch, model_goal, model_feat_pred, model_action = load_models(args)
    print("finished loading models.")

    """Create Habitat Sim"""
    add_noise_actions_habitat()
    sim, pathfinder, current_scan = None, None, None

    """Load test episodes"""
    jsonfilename = f"{args.test_dir}/{args.path_type}_{args.difficulty}.json.gz"
    with gzip.open(jsonfilename, "r") as fin:
        data = json.loads(fin.read().decode("utf-8"))["episodes"]
    """Loop over test episodes"""
    for instance in tqdm.tqdm(data):
        scan_name = instance["scene_id"].split("/")[-1].split(".")[0]
        episode_id = instance["episode_id"]
        length_shortest = instance["length_shortest"]

        """Load habitat scene"""
        if current_scan != scan_name:
            current_scan = scan_name
            print(current_scan)
            sim, pathfinder = create_habitat(args, sim, current_scan)
            visCounter[current_scan] = 0

        """ Image nav agent per episode"""
        agent = Agent(
            args,
            sim,
            pathfinder,
            resnet,
            current_scan,
            model_goal,
            model_switch,
            model_feat_pred,
            model_action,
        )
        start_position = instance["start_position"]
        start_rotation = quaternion.as_float_array(
            quaternion_from_coeff(instance["start_rotation"])
        )
        goal_position = instance["goals"][0]["position"]
        goal_rotation = quaternion.as_float_array(
            quaternion_from_coeff(instance["goals"][0]["rotation"])
        )
        agent.reset_agent(start_position, start_rotation, goal_position, goal_rotation)

        """ Run a image nav episode"""
        if args.behavioral_cloning:
            single_image_nav_BC(agent, args)
        else:
            single_image_nav(agent, visualizer, args)

        """ Evaluate result of episode"""
        dist_to_goal, episode_spl, success = evaluate_episode(
            agent, args, length_shortest
        )
        if agent.steps >= args.max_steps:
            maxed_out += 1
        rates["success"].append(success)
        rates["spl"].append(episode_spl)
        rates["dist2goal"].append(dist_to_goal)
        rates["taken_path_total"].append(agent.length_taken)
        rates["gt_path_total"].append(length_shortest)

        if success:
            rates["taken_path_success"].append(agent.length_taken)
            rates["gt_path_success"].append(length_shortest)

        """Visualize Episode"""
        if args.visualize:
            print("Creating visualization of episode...")
            visCounter[current_scan] += 1
            if args.dataset == "mp3d":
                visualizer.create_layout_mp3d(episode_id)
            else:
                visualizer.create_layout(agent, episode_id)

    """Print Stats"""
    print("\nType of Run: ")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Data: {args.path_type.upper()}")
    print(f"Data Type: {args.difficulty}")
    print(f"Pose Noise: {args.pose_noise}; Actuation Noise: {args.actuation_noise}")

    print("\nStats of Runs: ")
    print(f"Success Rate: {np.mean(rates['success']):.4f}")
    print(f"SPL: {np.mean(rates['spl']):.4f}")
    print(f"Avg dist to goal: {np.mean(rates['dist2goal']):.4f}")
    print(f"Avg taken path len - total: {np.mean(rates['taken_path_total']):.4f}")
    print(f"Avg taken path len - success: {np.mean(rates['taken_path_success']):.4f}")
    print(f"Avg gt path len - total: {np.mean(rates['gt_path_total']):.4f}")
    print(f"Avg gt path len - success: {np.mean(rates['gt_path_success']):.4f}")

    print("\nFor excel in above order: ")
    print(f"{np.mean(rates['success']):.4f}")
    print(f"{np.mean(rates['spl']):.4f}")
    print(f"{np.mean(rates['dist2goal']):.4f}")
    print(f"{np.mean(rates['taken_path_total']):.4f}")
    print(f"{np.mean(rates['taken_path_success']):.4f}")
    print(f"{np.mean(rates['gt_path_total']):.4f}")
    print(f"{np.mean(rates['gt_path_success']):.4f}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
