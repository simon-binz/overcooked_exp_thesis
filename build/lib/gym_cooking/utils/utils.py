import numpy as np

MAP_NUMBER_TO_OBJECT = {
    0: 'Floor',
    1: 'Counter',
    2: 'CutBoard',
    3: 'DeliverSquare',
    4: 'Tomato',
    5: 'ChoppedTomato',
    6: 'Lettuce',
    7: 'ChoppedLettuce',
    8: 'Onion',
    9: 'ChoppedOnion',
    10: 'Plate',
    11: 'Blender',
    12: 'Carrot',
    13: 'ChoppedCarrot',
    14: 'MashedCarrot',  # (or switch with ChoppedCarrot?)
    15: 'Agent',
    16: 'OwnAgent',
    17: 'OtherAgent',
    18: 'AgentOriented1 (West)',
    19: 'AgentOriented2 (East)',
    20: 'AgentOriented3 (South)',
    21: 'AgentOriented4 (North)'
}

MAP_OBJECT_TO_NUMBER = {v: k for k, v in MAP_NUMBER_TO_OBJECT.items()}

ACTION_MAP = {
    'nop': 0,
    'move_left': 1,
    'move_right': 2,
    'move_down': 3,
    'move_up': 4,
    'interact': 5
}
LOW_LEVEL_ACTION_IMAGE_MAP = {
    0: 'DoNothing',
    1: 'Left',
    2: 'Right',
    3: 'Down',
    4: 'Up',
    5: 'F'
}
HIGH_LEVEL_ACTION_IMAGE_MAP = {
    0 : 'SliceLettuce',
    1 : 'SliceTomato',
    2 : 'SliceOnion',
    3 : 'SlicePeanut',

    4 : 'ChoppedTomatoPlate',
    5 : 'ChoppedLettucePlate',
    6 : 'ChoppedOnionPlate',
    7 : 'ChoppedPeanutPlate',
    8 : 'ChoppedLettuceTomatoPlate',

    9 : 'GiveTomato',
    10 : 'GiveLettuce',
    11 : 'GiveOnion',
    #12 : 'GivePeanut',

    13 : 'GiveChoppedTomato',
    14 : 'GiveChoppedLettuce',
    15 : 'GiveChoppedOnion',
    #16 : 'GiveChoppedPeanut',
    17 : 'GivePlate',

    18 : 'delivery',

    19 : 'DoNothing'
}

HIGH_LEVEL_IMAGE_ACTION_MAP = {v: k for k, v in HIGH_LEVEL_ACTION_IMAGE_MAP.items()}
LOW_LEVEL_IMAGE_ACTION_MAP = {v: k for k, v in LOW_LEVEL_ACTION_IMAGE_MAP.items()}

def correct_fm_tensor(fm):
    """Corrects the given tensor to be aligned with the game window (transposed)."""
    return np.transpose(fm, axis=2)
