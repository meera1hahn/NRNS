from src.utils.model_utils import get_res_feats
from src.functions.validity_func.local_nav import LocalAgent
from src.functions.validity_func.validity_utils import get_sim_location
import torch
import numpy as np
import quaternion
import random


def predict_action(args, agent, current_feature, agent_map=None):
    action = 0
    with torch.no_grad():
        if args.bc_type == "gru":
            output = agent.action_model(
                torch.tensor(agent.actions[-1])
                .long()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(args.device),
                current_feature.clone().detach().unsqueeze(0).to(args.device),
                agent.goal_feat.clone().detach().unsqueeze(0).to(args.device),
            )
            action = torch.max(output.detach().cpu()[0][0], 0)[1].item()
        elif args.bc_type == "map":
            # maps = np.asarray(agent.map)
            maps = torch.from_numpy(np.repeat(agent_map[..., np.newaxis], 3, -1))
            output = agent.action_model(
                torch.tensor(agent.actions[-1])
                .long()
                .unsqueeze(0)
                .unsqueeze(0)
                .to(args.device),
                maps.long().unsqueeze(0).unsqueeze(0).to(args.device),
                current_feature.clone().detach().unsqueeze(0).to(args.device),
                agent.goal_feat.clone().detach().unsqueeze(0).to(args.device),
            )
            action = torch.max(output.detach().cpu()[0][0], 0)[1].item()
        elif args.bc_type == "random":
            action = random.randint(0, 3)
    return action


def single_image_nav_BC(agent, args):
    # set vars
    prev_pos = agent.current_pos
    agent.sim.set_agent_state(
        agent.current_pos.numpy().tolist(),
        quaternion.from_float_array(agent.current_rot.numpy()),
    )
    obs = agent.sim.get_observations_at(
        agent.current_pos.numpy().tolist(),
        quaternion.from_float_array(agent.current_rot.numpy()),
    )
    agent.rgb_img = obs["rgb"][:, :, :3]
    agent.depth_img = obs["depth"]

    agent_map = None
    if args.bc_type == "map":
        local_agent = LocalAgent(
            actuation_noise=False,
            pose_noise=False,
            curr_pos=agent.current_pos.numpy(),
            curr_rot=agent.current_rot.numpy(),
            map_size_cm=1200,
            map_resolution=5,
        )
        local_agent.update_local_map(agent.depth_img)
        agent_map = local_agent.get_map()
    """While not excuted terminate action OR exceed max steps:"""
    while agent.steps < args.max_steps:

        """
        1) Get current image feats
        """
        current_feature = get_res_feats(agent.rgb_img, agent.resnet)

        """ 
        2) Get next navigation step
        """
        next_action = predict_action(args, agent, current_feature, agent_map)

        """
        3) If step is 0 break
        """
        if next_action == 0:
            break

        """
        5) Take step, update feats
        """
        obs = agent.sim.step(next_action)
        agent.actions.append(next_action)
        agent.rgb_img = obs["rgb"][:, :, :3]
        agent.depth_img = obs["depth"]
        state = agent.sim.get_agent_state()
        agent.current_pos, agent.current_rot = (
            torch.tensor(state.position),
            torch.tensor(quaternion.as_float_array(state.rotation)),
        )
        if args.bc_type == "map":
            local_agent.new_sim_origin = get_sim_location(
                state.position, state.rotation
            )
            local_agent.update_local_map(agent.depth_img)
            agent_map = local_agent.get_map()

        agent.length_taken += np.linalg.norm(
            agent.current_pos.numpy() - prev_pos.numpy()
        )
        prev_pos = agent.current_pos

        """
        6) Check if exceeding step length
        """
        # Add Steps to Counter
        agent.steps += 1
        if agent.steps >= args.max_steps:
            print("max_steps")
            break
