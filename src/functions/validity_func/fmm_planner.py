import cv2, ctypes, logging, os, numpy as np, pickle
from numpy import ma
from collections import OrderedDict
from skimage.morphology import binary_closing, disk
import scipy, skfmm, skimage
import matplotlib.pyplot as plt
import math

step_size = 5
num_rots = 36


def get_mask(sx, sy, scale):
    size = int(5 // scale) * 2 + 1
    mask = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if ((i + 0.5) - (size // 2 + sx)) ** 2 + (
                (j + 0.5) - (size // 2 + sy)
            ) ** 2 <= 25:
                mask[i, j] = 1
    return mask


class FMMPlanner:
    def __init__(self, traversible, num_rots, scale):
        self.scale = scale
        if scale != 1.0:
            self.traversible = cv2.resize(
                traversible,
                (traversible.shape[1] // scale, traversible.shape[0] // scale),
                interpolation=cv2.INTER_NEAREST,
            )
            self.traversible = np.rint(self.traversible)
        else:
            self.traversible = traversible

        self.angle_value = [0, 2.0 * np.pi / num_rots, -2.0 * np.pi / num_rots, 0]
        self.du = int(step_size / (self.scale * 1.0))
        self.num_rots = num_rots

    def set_goal(self, goal, auto_improve=False):
        traversible_ma = ma.masked_values(self.traversible * 1, 0)
        goal_x, goal_y = int(goal[0] / (self.scale * 1.0)), int(
            goal[1] / (self.scale * 1.0)
        )

        goal_x = min(goal_x, self.traversible.shape[0])
        goal_x = max(goal_x, 0)
        goal_y = min(goal_y, self.traversible.shape[1])
        goal_y = max(goal_y, 0)

        if self.traversible[goal_x, goal_y] == 0.0 and auto_improve:
            goal_x, goal_y = self._find_nearest_goal([goal_x, goal_y])

        traversible_ma[goal_x, goal_y] = 0
        dd = skfmm.distance(traversible_ma, dx=1)
        dd_mask = np.invert(np.isnan(ma.filled(dd, np.nan)))
        dd = ma.filled(dd, np.max(dd) + 1)
        self.fmm_dist = dd
        self._goal = (goal_x, goal_y)
        return dd_mask

    def get_goal(self):
        return self._goal

    def get_short_term_goal2(self, state):
        scale = self.scale * 1.0
        state = [x / scale for x in state]
        dx, dy = state[0] - int(state[0]), state[1] - int(state[1])
        mask = get_mask(dx, dy, scale)
        state = [int(x) for x in state]

        dist = np.pad(
            self.fmm_dist,
            self.du,
            "constant",
            constant_values=self.fmm_dist.shape[0] ** 2,
        )
        subset = dist[
            state[0] : state[0] + 2 * self.du + 1, state[1] : state[1] + 2 * self.du + 1
        ]
        subset *= mask
        subset += (1 - mask) * self.fmm_dist.shape[0] ** 2
        subset -= subset[self.du, self.du]
        subset[subset < -(self.du)] = 1
        (stg_x, stg_y) = np.unravel_index(np.argmin(subset), subset.shape)
        if subset[stg_x, stg_y] > -0.0001:
            replan = True
        else:
            replan = False
        return (
            (stg_x + state[0] - self.du) * scale + 0.5,
            (stg_y + state[1] - self.du) * scale + 0.5,
            replan,
        )

    def get_next_action(self, state, stg, agent_orientation):
        stg_x, stg_y = stg[0], stg[1]
        relative_dist = get_l2_distance(stg_x, state[0], stg_y, state[1])

        angle_st_goal = math.degrees(math.atan2(stg_x - state[0], stg_y - state[1]))
        angle_agent = (agent_orientation) % 360.0
        if angle_agent > 180:
            angle_agent -= 360

        relative_angle = (angle_agent - angle_st_goal) % 360.0
        if relative_angle > 180:
            relative_angle -= 360

        if relative_angle > 10.0:
            action = 3
        elif relative_angle < -10.0:
            action = 2
        else:
            action = 1
        return action

    def _find_nearest_goal(self, goal):
        traversible4 = (
            skimage.morphology.binary_dilation(
                np.zeros(self.traversible.shape), skimage.morphology.disk(2)
            )
            != True
        )
        rots = 360 // 10.0
        planner4 = FMMPlanner(traversible4, rots, 1)
        planner4.set_goal(goal)

        mask = self.traversible

        dist_map = planner4.fmm_dist * mask
        dist_map[dist_map == 0] = dist_map.max()

        goal = np.unravel_index(dist_map.argmin(), dist_map.shape)

        return goal


def get_l2_distance(x1, x2, y1, y2):
    """
    Computes the L2 distance between two points.
    """
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
