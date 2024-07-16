import numpy as np

import torch
from torch_geometric import data as pyg_data

from fmp.datasets.fingerspelling5 import transforms
from fmp.datasets.fingerspelling5 import utils as fs5_utils


def test_rotation():
    palm_normal = (0, 0, -1)
    knuckle_direction = (1, 0, 0)

    hand = np.zeros((21, 3))
    hand[fs5_utils.mediapipe_hand_landmarks.nodes.wrist] = (0, 0, 1)
    hand[fs5_utils.mediapipe_hand_landmarks.nodes.index_mcp] = (2, 0, 4)
    hand[fs5_utils.mediapipe_hand_landmarks.nodes.pinky_mcp] = (-2, 0, 4)

    expected_hand = np.zeros((21, 3))
    expected_hand[fs5_utils.mediapipe_hand_landmarks.nodes.wrist] = (0, -1, 0)
    expected_hand[fs5_utils.mediapipe_hand_landmarks.nodes.index_mcp] = (2, -4, 0)
    expected_hand[fs5_utils.mediapipe_hand_landmarks.nodes.pinky_mcp] = (-2, -4, 0)

    hand = torch.from_numpy(hand)
    expected_hand = torch.from_numpy(expected_hand)

    hand_data = pyg_data.Data(pos=hand)

    rotator = transforms.pyg.RotateHand(
        palm_normal=palm_normal, knuckle_direction=knuckle_direction
    )
    output = rotator(hand_data)

    torch.testing.assert_close(expected_hand, output.pos)
