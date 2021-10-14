# Data Generation: Trajectories & Graph 

This directory contains the code for creating the passive dataset for training our image nav modules


## Trajectories

The folder `trajectories/` contains code for:
1. randomly generating trajectories through the scenes
    1.1 run `generate_trajectories.py`
2. simulating trajectories with habitat, grabing scene actions (points, rotations), features, images, videos
    1.1 run `simulate_trajectories.py`

## Graphs

The folder `graphs/` contains code for:

1. Taking the info from the simulated trajectories, storing it in a graph form and clustering the step-wise graph to a topologolgical graph.
    For each trajectories it saves a clustered graph with the following info:
    * nodes
    * edges
    * edge_attrs - (pose difference)
    * invalid_points - (per each node -list of invalid points)
    * valid_points - (per each node -list of valid points)
    * floor
    * scene
    * scan_name

2. Taking the info from the simulated trajectories, storing it in a graph form and clustering the step-wise graph to a topologolgical graph. and pulling distance data out of it. 
    Per each trajectory, the code tries to get 20 pairs of nodes and stores the following
    * n1
    * n2
    * geodesic
    * euclidean
    * num_steps
    * rotation_diff
    * traj
    * floor
    * scan_name


