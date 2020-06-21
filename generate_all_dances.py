"""Script to generate all dances for a song."""
from visualize import save_matrices, save_dance
from multiprocessing import Pool
from PIL import Image
import numpy as np
import itertools
import argparse
import librosa
import madmom
import random
import torch
import scipy

random.seed(123)

# parse arguments
parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('-songpath', '--songpath', type=str, default='./audio_files/fluetesong.mp3',
                    help='Path to .mp3 song')
parser.add_argument('-songname', '--songname', type=str, default='flutesong',
                    help='Name of song')
parser.add_argument('-steps', '--steps', type=int, default=100,
                    help='Number of equidistant LRS steps the agent should take')
parser.add_argument('-type', '--type', type=str, default='action',
                    help='Type of dance -- state, action, stateplusaction')
parser.add_argument('-baseline', '--baseline', type=str, default='none',
                    help='Generate baseline -- none, unsync_random, unsync_sequential, sync_sequential, sync_random')
parser.add_argument('-visfolder', '--visfolder', type=str, default='./vis_num_steps_20/dancing_person_20',
                    help='path to folder containing agent visualizations')
args = parser.parse_args()

# global variables
GRID_SIZE = 20
REWARD_INTERVAL = 5
ALL_ACTION_COMBS = list(set(itertools.permutations([-1 for _ in range(REWARD_INTERVAL)] + [1 for _ in range(REWARD_INTERVAL)] + [0 for _ in range(REWARD_INTERVAL)], REWARD_INTERVAL)))
START_POSITION = int(GRID_SIZE / 2)
ACTION_MAPPING = {-1: 'L', 1: 'R', 0: 'S'}    # L: left, R: right, S; stay
MUSIC_MODE = 'affinity'
MUSIC_METRIC = 'euclidean'
HOP_LENGTH = 512

# **************************************************************************************************************** #
# BASELINES


def unsync_random(num_steps):
    """Baseline 1 : unsynced - random."""
    states, actions = [], []
    curr = START_POSITION

    # get state and action sequences
    for i in range(num_steps):
        act = random.choice([-1, 0, 1])
        newcurr = curr + act
        if newcurr < 0:
            curr = 0
        elif newcurr == GRID_SIZE:
            curr = GRID_SIZE - 1
        else:
            curr = newcurr
        states.append(curr)
        actions.append(act)
    return [states, actions]


def unsync_sequential(num_steps):
    """Baseline 2 : unsynced - left2right."""
    states, actions = [], []
    curr = START_POSITION
    curr, prev = START_POSITION, START_POSITION - 1

    # get state and action sequences
    for i in range(num_steps):
        act = (curr - prev)
        newcurr = curr + act
        if newcurr < 0:
            prev = 0
            curr = 1
        elif newcurr == GRID_SIZE:
            prev = GRID_SIZE - 1
            curr = GRID_SIZE - 2
        else:
            prev = curr
            curr = newcurr
        states.append(curr)
        actions.append(act)
    return [states, actions]


def sync_sequential(num_steps, filename, duration):
    """Baseline 3 : synced - left2right."""
    # get beat information
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(filename)
    beat_times = np.around(proc(act) * num_steps / duration)

    states, actions = [], []
    curr = START_POSITION
    curr, prev = START_POSITION, START_POSITION - 1

    for i in range(num_steps):
        if i in beat_times:
            act = (curr - prev)
            newcurr = curr + act
            if newcurr < 0:
                prev = 0
                curr = 1
            elif newcurr == GRID_SIZE:
                prev = GRID_SIZE - 1
                curr = GRID_SIZE - 2
            else:
                prev = curr
                curr = newcurr
        else:
            act = 0
        states.append(curr)
        actions.append(act)
    return [states, actions]


def sync_random(num_steps, filename, duration):
    """Baseline 4 : synced - random."""
    # get beat information
    proc = madmom.features.beats.DBNBeatTrackingProcessor(fps=100)
    act = madmom.features.beats.RNNBeatProcessor()(filename)
    beat_times = np.around(proc(act) * num_steps / duration)

    states, actions = [], []
    curr = START_POSITION

    for i in range(num_steps):
        if i in beat_times:
            act = random.choice([-1, 0, 1])
            newcurr = curr + act
            if newcurr < 0:
                curr = 0
            elif newcurr == GRID_SIZE:
                curr = GRID_SIZE - 1
            else:
                curr = newcurr
        else:
            act = 0
        states.append(curr)
        actions.append(act)
    return [states, actions]

# **************************************************************************************************************** #
# DANCE MATRIX CREATION


def fill_dance_aff_matrix_diststate(states):
    """Fill state action affinity matrix - relative distance based states."""
    s = len(states)
    rowtile = np.tile(states, (s, 1))
    coltile = rowtile.T
    sa_aff = 1. - np.abs(rowtile-coltile) / (GRID_SIZE-1)
    # sa_aff = (sa_aff - np.min(sa_aff)) / (np.max(sa_aff) - np.min(sa_aff))    # normalize
    return sa_aff


def fill_dance_aff_matrix_action(actions):
    """Fill state action affinity matrix - action based."""
    s = len(actions)
    rowtile = np.tile(actions, (s, 1))
    coltile = rowtile.T
    sa_aff = np.maximum(1. - np.abs(rowtile-coltile), 0.)
    # sa_aff = (sa_aff - np.min(sa_aff)) / (np.max(sa_aff) - np.min(sa_aff))    # normalize
    return sa_aff


def fill_dance_aff_matrix_diststateplusaction(states, actions):
    """Fill state action affinity matrix - action based."""
    state_matrix = fill_dance_aff_matrix_diststate(states)
    action_matrix = fill_dance_aff_matrix_action(actions)
    sa_aff = (state_matrix + action_matrix) / 2.
    # sa_aff = (sa_aff - np.min(sa_aff)) / (np.max(sa_aff) - np.min(sa_aff))    # normalize
    return sa_aff


def get_dance_matrix(states, actions, dance_matrix_type, music_matrix_full):
    """Pass to appropriate dance matrix generation function based on dance_matrix_type."""
    if dance_matrix_type == 'state':
        dance_matrix = fill_dance_aff_matrix_diststate(states)
    elif dance_matrix_type == 'action':
        dance_matrix = fill_dance_aff_matrix_action(actions)
    elif dance_matrix_type == 'stateplusaction':
        dance_matrix = fill_dance_aff_matrix_diststateplusaction(states, actions)
    else:
        print("err")
    dance_matrix = np.array(Image.fromarray(np.uint8(dance_matrix * 255)).resize(music_matrix_full.shape, Image.NEAREST)) / 255.
    return dance_matrix

# **************************************************************************************************************** #
# MUSIC MATRIX COMPUTATION


def compute_music_matrix(y, sr, mode, metric):
    """Return music affinity matrix based on mode."""
    lifter = 0
    n_mfcc = 20
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, lifter=lifter, hop_length=HOP_LENGTH)
    R = torch.from_numpy(librosa.segment.recurrence_matrix(mfcc, metric=metric, mode=mode, sym=True).T)    # already normalized in 0-1
    R = R + torch.zeros(R.shape)    # change (True, False) to (1, 0)
    R.fill_diagonal_(1)    # make diagonal entries 1
    return R

# **************************************************************************************************************** #
# REWARD COMPUTATION


def music_reward(music_matrix, dance_matrix, mtype):
    """Return the reward given music matrix and dance matrix."""
    # compute distance based on mtype
    if mtype == 'pearson':
        if np.array(music_matrix).std() == 0 or np.array(dance_matrix).std() == 0:
            reward = 0
        else:
            reward, p_val = scipy.stats.pearsonr(music_matrix.flatten(), dance_matrix.flatten())
    elif mtype == 'spearman':
        reward, p_val = scipy.stats.spearmanr(music_matrix.flatten(), dance_matrix.flatten())
    else:
        print("err")
    return reward


# **************************************************************************************************************** #
# BRUTE FORCE METHODS


def get_rsa_for_actionset(args):
    """Return reward, state, action set."""
    actionset, music_matrix, loc, num_actions, prev_states, prev_actions, dance_matrix_type = args
    curr_states = []
    curr_actions = []
    start_loc = loc
    for action in list(actionset):
        newpos = start_loc + action
        if newpos == 0 or newpos == GRID_SIZE:
            # hit wall
            break
        curr_states.append(newpos)
        curr_actions.append(action)
        start_loc = newpos

    # if not completed, ignore
    if len(curr_actions) != num_actions:
        return False, [], []

    # get dance up till now
    states = prev_states + curr_states
    actions = prev_actions + curr_actions

    # get dance matrix
    if dance_matrix_type == 'state':
        dance_matrix = fill_dance_aff_matrix_diststate(states)
    elif dance_matrix_type == 'action':
        dance_matrix = fill_dance_aff_matrix_action(actions)
    else:
        dance_matrix = fill_dance_aff_matrix_diststateplusaction(states, actions)
    dance_matrix = np.array(Image.fromarray(np.uint8(dance_matrix * 255)).resize(music_matrix.shape, Image.NEAREST)) / 255.
    # check how good dance up till now is by computing reward
    curr_reward = music_reward(music_matrix, dance_matrix, 'pearson')
    return curr_reward, states, actions


def getbest(loc, num_actions, prev_states, prev_actions, music_matrix_full, num_steps, dance_matrix_type):
    """Return best combination of size num_actions.

    Start from `loc` in grid of size `GRID_SIZE`.
    """
    scale = int(music_matrix_full.shape[0] * (len(prev_states)+num_actions) / num_steps)
    music_matrix = np.array([music_matrix_full.numpy()[i][:scale] for i in range(scale)])
    # get best dance for this music matrix
    bestreward = 0
    p = Pool()
    args = ((actionset, music_matrix, loc, num_actions, prev_states, prev_actions, dance_matrix_type) for actionset in ALL_ACTION_COMBS)
    res = p.map(get_rsa_for_actionset, args)
    p.close()
    for curr_reward, states, actions in res:
        if curr_reward is not False and curr_reward > bestreward:
            bestreward = curr_reward
            beststates = states
            bestactions = actions
    return beststates, bestactions, bestreward

# **************************************************************************************************************** #
# MAIN


if __name__ == "__main__":

    # get args
    songname = args.songname
    baseline = args.baseline
    dance_matrix_type = args.type
    num_steps = args.steps
    visfolder = args.visfolder
    songpath = args.songpath

    # load song
    y, sr = librosa.load(songpath)    # default sampling rate 22050
    duration = librosa.get_duration(y=y, sr=sr)

    # get music matrix
    music_matrix_full = compute_music_matrix(y, sr, MUSIC_MODE, MUSIC_METRIC)

    # main code
    if baseline in ['unsync_random', 'unsync_sequential', 'sync_sequential', 'sync_random']:
        # check baselines
        if baseline == 'unsync_random':
            states, actions = unsync_random(num_steps=num_steps)
        if baseline == 'unsync_sequential':
            states, actions = unsync_sequential(num_steps=num_steps)
        if baseline == 'sync_sequential':
            states, actions = sync_sequential(num_steps=num_steps, filename=songpath, duration=duration)
        else:
            states, actions = sync_random(num_steps=num_steps, filename=songpath, duration=duration)

        # compute dance matrix
        dance_matrix = get_dance_matrix(states, actions, dance_matrix_type, music_matrix_full)

        # compute correlation
        reward = music_reward(music_matrix_full, dance_matrix, 'pearson')
    else:
        # our approach
        # try out all combinations of `REWARD_INTERVAL` actions and compute reward
        prev_states = []
        prev_actions = []

        for i in range(num_steps):
            # apply greedy algo to get dance matrix with best reward
            if len(prev_actions) == 0:
                prev_states, prev_actions, reward = getbest(loc=START_POSITION,
                                                            num_actions=REWARD_INTERVAL,
                                                            prev_states=prev_states,
                                                            prev_actions=prev_actions,
                                                            music_matrix_full=music_matrix_full,
                                                            num_steps=num_steps,
                                                            dance_matrix_type=dance_matrix_type)
            elif not i % REWARD_INTERVAL:
                prev_states, prev_actions, reward = getbest(loc=prev_states[-1],
                                                            num_actions=REWARD_INTERVAL,
                                                            prev_states=prev_states,
                                                            prev_actions=prev_actions,
                                                            music_matrix_full=music_matrix_full,
                                                            num_steps=num_steps,
                                                            dance_matrix_type=dance_matrix_type)
            elif num_steps - len(prev_actions) != 0 and num_steps - len(prev_actions) < REWARD_INTERVAL:
                prev_states, prev_actions, reward = getbest(loc=prev_states[-1],
                                                            num_actions=num_steps-len(prev_states),
                                                            prev_states=prev_states,
                                                            prev_actions=prev_actions,
                                                            music_matrix_full=music_matrix_full,
                                                            num_steps=num_steps,
                                                            dance_matrix_type=dance_matrix_type)
            else:
                continue

        # get best dance matrix
        dance_matrix = get_dance_matrix(prev_states, prev_actions, dance_matrix_type, music_matrix_full)

        # assign states and actions correctly correctly
        states = prev_states
        actions = prev_actions

    # map actions correctly
    actions = [ACTION_MAPPING[a] for a in actions]

    # print results
    print("Correlation = ", reward)
    print("State sequence = ", states)
    print("Action sequence = ", actions)

    # visualize results
    save_matrices(music_matrix=music_matrix_full, dance_matrix=dance_matrix, duration=duration)
    save_dance(states=states, visfolder=args.visfolder, songname=songname, duration=duration, num_steps=num_steps)

    print(songname, " :: DONE!")
