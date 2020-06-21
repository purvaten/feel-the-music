"""Read pickle file generated from `generate_all_dances`.

Create video for any approach. Visualize matrices.
"""
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import subprocess
import argparse
import librosa
import shutil
import pickle
import os


# parse arguments
parser = argparse.ArgumentParser(description='Process arguments.')
parser.add_argument('-songpath', '--songpath', type=str, default='./audio_files/fluetesong.mp3',
                    help='path to .mp3 song -- e.g., ./audio_files/fluetesong.mp3')
parser.add_argument('-songname', '--songname', type=str, default='flutesong',
                    help='name of song -- e.g., flutesong')
parser.add_argument('-visfolder', '--visfolder', type=str, default='./vis_num_steps_20/dancing_person_20',
                    help='path to folder containing agent visualizations -- e.g., ./vis_num_steps_20/dancing_person_20')
parser.add_argument('-num', '--num', type=int, default=15,
                    help='integer from 0 to 20 indicating which visualization you want to see -- e.g., 15 indicates action-based dance with 100 steps. Check visualize.py for more details.')
args = parser.parse_args()


def save_video(foldername, songname, songlen, num_steps, output):
    """Make video from given frames. Add audio appropriately."""
    num_steps_by_len = num_steps / songlen
    p = subprocess.Popen(['ffmpeg', '-f', 'image2', '-r', str(num_steps_by_len), '-i', '%d.png', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', 'movie.mp4'], cwd=foldername)
    p.wait()

    p = subprocess.Popen(['ffmpeg', '-i', 'movie.mp4', '-i', '../audio_files/' + songname + '.mp3', '-map', '0:v', '-map', '1:a', '-c', 'copy', output], cwd=foldername)
    p.wait()


if __name__ == "__main__":

    # decide dance type to load.
    # -----------------------------------------------------------------------------------------
    # state | action | stateplusaction | baseline1 | baseline2 | baseline3 | baseline4 | steps
    # -----------------------------------------------------------------------------------------
    # 0		| 	1	 | 		  2 	   |     3     |     4     |     5     |     6     |   25
    # -----------------------------------------------------------------------------------------
    # 7     |   8    |        9        |     10    |     11    |     12    |     13    |   50
    # -----------------------------------------------------------------------------------------
    # 14    |   15   |       16        |     17    |     18    |     19    |     20    |   100
    # -----------------------------------------------------------------------------------------

    songname = args.songname
    filename = args.songpath
    num = args.num
    mp4_filename = "dance_" + songname + "_" + str(num)
    with open('pickles/' + songname + '.pickle', 'rb') as handle:
        songdict = pickle.load(handle)

    music_matrix = songdict['music_matrix']
    correlations = songdict['correlations'][num]
    dance_matrix = songdict['dance_matrices'][num]
    num_steps = songdict['nums_steps'][num]
    print("Number of steps = ", num_steps)

    # load song
    y, sr = librosa.load(filename)    # default sampling rate 22050
    duration = librosa.get_duration(y=y, sr=sr)
    d = int(duration) + 1

    # save music matrix
    plt.tick_params(labelsize=22)
    plt.xticks(np.arange(0, music_matrix.shape[0], music_matrix.shape[0] / duration), np.arange(0, d, 1))
    plt.yticks(np.arange(0, music_matrix.shape[0], music_matrix.shape[0] / duration), np.arange(0, d, 1))
    plt.imshow(music_matrix, cmap='gray')
    plt.savefig('music.png', bbox_inches='tight')
    plt.close()

    plt.tick_params(labelsize=22)
    plt.xticks(np.arange(0, music_matrix.shape[0], music_matrix.shape[0] / duration), np.arange(0, d, 1))
    plt.yticks(np.arange(0, music_matrix.shape[0], music_matrix.shape[0] / duration), np.arange(0, d, 1))
    plt.imshow(dance_matrix, cmap='gray')
    plt.savefig('dance.png', bbox_inches='tight')
    plt.close()

    # ******************************************************************************************************************* #
    states = songdict['state_sequences'][num]
    actions = songdict['action_sequences'][num]
    # ******************************************************************************************************************* #

    # Make folder if not already exists
    if not os.path.exists('./plots/'):
        os.makedirs('plots/')

    # Delete old items
    print("Starting file deletions")
    for item in os.listdir('./plots/'):
        delfile = os.path.join('./plots/', item)
        os.remove(delfile)
    print("File deletions complete")

    # Create dance video
    print("Creating dance video frames")
    a = ""
    c = 0
    for i, state in enumerate(states):
        # ****************** stick figure agent ******************
        shutil.copy(args.visfolder + "/" + str(state+1) + '.png', 'plots/' + str(i+1) + '.png')
        c += 1

    # Save video
    save_video('./plots', songname, duration, num_steps, mp4_filename + '.mp4')

    print("Done :: ", songname)
