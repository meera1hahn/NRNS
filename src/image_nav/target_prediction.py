import torch
import numpy as np
import quaternion

from src.functions.validity_func.local_nav import LocalAgent, loop_nav


"""
predict if you want to switch to local navigation
"""

def predict_end_exploration(args, agent, visualizer):
    switchThreshold = 0.55
    switch = False
    with torch.no_grad():
        batch_goal = agent.goal_feat.clone().detach()
        batch_nodes = (
            agent.node_feats[agent.current_node, :].clone().detach().unsqueeze(0)
        )

        switch_pred, _, _ = agent.switch_model(
            batch_nodes.to(args.device), batch_goal.to(args.device)
        )
        switch_pred = switch_pred.detach().cpu()[0].item()
        if switch_pred >= switchThreshold:
            switch = True
            rho, phi = agent.goal_model(
                batch_nodes.to(args.device), batch_goal.to(args.device)
            )
            rho = rho.cpu().detach().item()
            phi = phi.cpu().detach().item()

            localnav(agent, rho, phi, visualizer)

        else:
            switch = False
    return switch

def run_vis(agent, visualizer, prev_poses):
    for p in prev_poses:
        img = agent.sim.get_observations_at(p[0], quaternion.from_float_array(p[1]),)[
            "rgb"
        ][:, :, :3]
        agent.current_pos, agent.current_rot = torch.tensor(p[0]), torch.tensor(p[1])
        visualizer.seen_images.append(img)
        visualizer.current_graph(agent, switch=True)


def localnav(agent, rho, phi, visualizer):
    agent.sim.set_agent_state(
        agent.current_pos.numpy(),
        quaternion.from_float_array(agent.current_rot.numpy()),
    )
    agent.switch_index = len(agent.prev_poses)
    agent.prev_poses.append([agent.current_pos.numpy(), agent.current_rot.numpy()])
    try:
        agent.sim.set_agent_state(
            agent.current_pos.numpy(),
            quaternion.from_float_array(agent.current_rot.numpy()),
        )
        local_agent = LocalAgent(
            agent.actuation_noise,
            agent.pose_noise,
            agent.current_pos.numpy(),
            agent.current_rot.numpy(),
            map_size_cm=1200,
            map_resolution=5,
        )
        final_pos, final_rot, nav_length, prev_poses = loop_nav(
            agent.sim,
            local_agent,
            agent.current_pos.numpy(),
            agent.current_rot.numpy(),
            rho,
            phi,
            min(100, 499 - agent.steps),
        )
        if agent.visualize:
            run_vis(agent, visualizer, prev_poses)
        agent.prev_poses.extend(prev_poses)
        agent.current_pos = torch.tensor(final_pos)
        agent.current_rot = torch.tensor(final_rot)
        agent.length_taken += nav_length
        return np.linalg.norm(agent.goal_pos - final_pos)
    except:
        print("ERROR: local navigation through error")

    return np.linalg.norm(agent.goal_pos - agent.current_pos.numpy())
