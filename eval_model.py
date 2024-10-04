import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.distributions import Categorical

import fixed_env as env
import load_trace
import a3c

S_INFO = 6  
S_LEN = 8  
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [300,750,1200,1850,2850,4300] 
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3  
SMOOTH_PENALTY = 1
DEFAULT_QUALITY = 1  
RANDOM_SEED = 42
RAND_RANGE = 1000
LOG_FILE = './test_results/log'
TEST_TRACES = './cooked_test_traces/'

def main(epoch):

    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    all_cooked_time, all_cooked_bw, all_file_names = load_trace.load_trace(TEST_TRACES)

    net_env = env.Environment(all_cooked_time=all_cooked_time,
                              all_cooked_bw=all_cooked_bw, 
                              random_seed=RANDOM_SEED,  )

    log_path = LOG_FILE + '_' + str(net_env.trace_idx)
    log_file = open(log_path, 'w')

    net = a3c.A3C(is_central=False, s_dim=[S_INFO, S_LEN], a_dim=A_DIM, actor_lr=ACTOR_LR_RATE, critic_lr=CRITIC_LR_RATE)

    net.actor.load_state_dict(torch.load('./models/actor_'+str(epoch)+'.pkl'))
    print("Testing model restored.")

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    video_count = 0
    state=torch.zeros((S_INFO,S_LEN))

    while True:
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)

        time_stamp += delay 
        time_stamp += sleep_time  

        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] -
                                           VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K

        last_bit_rate = bit_rate

        log_file.write(str(reward) + '\n')
        log_file.flush()

        state = torch.roll(state,-1,dims=-1)

        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE)) 
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  
        state[4, :A_DIM] = torch.tensor(next_video_chunk_sizes) / M_IN_K / M_IN_K  
        state[5, -1] = min(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        with torch.no_grad():
            logits = net.actor.forward(state.unsqueeze(0))
            probs = F.softmax(logits, dim=-1)
            m = Categorical(probs)
            bit_rate = m.sample().item()

        if end_of_video:
            log_file.close()

            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY 

            state=torch.zeros((S_INFO,S_LEN))

            video_count += 1

            if video_count >= len(all_file_names):
                break

            log_path = LOG_FILE + '_' + str(net_env.trace_idx)
            log_file = open(log_path, 'w')


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=0)
    args = parser.parse_args()
    main(args.epoch)