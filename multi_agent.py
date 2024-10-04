import os
import logging
import argparse
import torch.multiprocessing as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datetime import datetime
import logging
from tensorboardX import SummaryWriter

from a3c import A3C
import load_trace
import env

S_INFO = 6
S_LEN = 8
A_DIM = 6
ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
# need to change
NUM_AGENTS = 8
TRAIN_SEQ_LEN = 100
MODEL_SAVE_INTERVAL = 100
VIDEO_BIT_RATE = [300, 750, 1200, 1850, 2850, 4300]
BUFFER_NORM_FACTOR = 10.0
CHUNK_TIL_VIDEO_END_CAP = 48.0
M_IN_K = 1000.0
REBUF_PENALTY = 4.3
SMOOTH_PENALTY = 1.0
DEFAULT_QUALITY = 1  
RANDOM_SEED = 42
RAND_RANGE = 1000

EPOCH = 100000

MODEL_FOLDER = './models'
TEST_RESULTS = './test_results'
TRAIN_TRACES = './cooked_traces/'
TEST_TRACES = './cooked_test_traces/'

def eval_model(epoch):
    os.system("python eval_model.py --epoch %d" % epoch)
    files = os.listdir(TEST_RESULTS)
    reward_list = []
    for file in files:
        with open(os.path.join(TEST_RESULTS, file), 'r') as f:
            # print(file)
            next(f)
            for line in f:
                reward_list.append(float(line.strip()))
                # print(float(line.strip()[-1]))
    return np.mean(reward_list)

def central_agent(net_param_queues, exp_queues):

    assert len(net_param_queues) == NUM_AGENTS
    assert len(exp_queues) == NUM_AGENTS

    writer = SummaryWriter()
    logging.basicConfig(level=logging.INFO,
                        filename='./results/train_log', 
                        filemode='w')

    a3c = A3C(is_central=True, s_dim=[S_INFO, S_LEN], a_dim=A_DIM, actor_lr=ACTOR_LR_RATE, critic_lr=CRITIC_LR_RATE)

    for epoch in tqdm(range(EPOCH)):
        actor_param, critci_param = a3c.actor.state_dict(), a3c.critic.state_dict()
        for i in range(NUM_AGENTS):
            net_param_queues[i].put([actor_param, critci_param])
        
        total_reward = 0.0
        total_batch_len = 0.0
        # total_td_loss = 0.0
        # total_entropy = 0.0
        for i in range(NUM_AGENTS):
            s_batch, a_batch, r_batch, end_of_video = exp_queues[i].get()

            s_batch = torch.from_numpy(np.stack(s_batch)).float()
            a_batch = torch.from_numpy(np.stack(a_batch)).float()
            r_batch = torch.from_numpy(np.stack(r_batch)).float()

            a3c.train(s_batch, a_batch, r_batch, end_of_video, epoch)
        
            total_reward += torch.sum(r_batch)
            total_batch_len += len(r_batch)
            # total_td_loss += a3c.td_loss

        a3c.update()

        avg_reward = total_reward / total_batch_len

        if (epoch + 1) % MODEL_SAVE_INTERVAL == 0:
            torch.save(a3c.actor.state_dict(), os.path.join(MODEL_FOLDER, 'actor_%d.pkl' % (epoch + 1)))
            torch.save(a3c.critic.state_dict(), os.path.join(MODEL_FOLDER, 'critic_%d.pkl' % (epoch + 1)))

            reward_mean = eval_model(epoch + 1)

            print('Epoch %d, train reward: %f, test reward: %f' % (epoch + 1, avg_reward, reward_mean))
            logging.info('Epoch %d, train reward: %f, test reward: %f' % (epoch + 1, avg_reward, reward_mean))

            writer.add_scalar('train/avg_reward', avg_reward, epoch + 1)
            writer.add_scalar('test/avg_reward', reward_mean, epoch + 1)

            writer.flush()

def agent(agent_id, all_cooked_time, all_cooked_bw, net_param_queue, exp_queue):

    net_env = env.Environment(all_cooked_time, all_cooked_bw, random_seed=agent_id)

    actor = A3C(is_central=False, s_dim=[S_INFO, S_LEN], a_dim=A_DIM, actor_lr=ACTOR_LR_RATE, critic_lr=CRITIC_LR_RATE)
    actor_net_params, _ = net_param_queue.get()
    actor.actor.load_state_dict(actor_net_params)

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY
    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []

    time_stamp = 0
    while True:
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video, video_chunk_remain = \
            net_env.get_video_chunk(bit_rate)
        
        time_stamp += delay 
        time_stamp += sleep_time

        reward = VIDEO_BIT_RATE[bit_rate] / M_IN_K \
                - REBUF_PENALTY * rebuf \
                - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[bit_rate] - VIDEO_BIT_RATE[last_bit_rate]) / M_IN_K
        
        last_bit_rate = bit_rate
        r_batch.append(reward)

        state = np.array(s_batch[-1], copy=True)
        state = np.roll(state, -1, axis=1)  

        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR  
        state[2, -1] = float(video_chunk_size) / float(delay) / M_IN_K  
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  
        state[4, :A_DIM] = np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K  
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)

        # need to change
        # print(torch.from_numpy(state).float().unsqueeze(0).shape)
        bit_rate = actor.select_action(torch.from_numpy(state).float().unsqueeze(0))

        if len(r_batch) >= TRAIN_SEQ_LEN or end_of_video:
            exp_queue.put([s_batch[1:], 
                           a_batch[1:], 
                           r_batch[1:], 
                           end_of_video])

            actor_net_params, _ = net_param_queue.get()
            actor.actor.load_state_dict(actor_net_params)

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

        if end_of_video:
            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY

            s_batch = [np.zeros((S_INFO, S_LEN))]
            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1
            a_batch = [action_vec]
        else:
            s_batch.append(state)
            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1
            a_batch.append(action_vec)

def main():

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    net_param_queue = []
    exp_queue = []
    for i in range(NUM_AGENTS):
        net_param_queue.append(mp.Queue(1))
        exp_queue.append(mp.Queue(1))

    coordinator = mp.Process(
        target=central_agent, 
        args=(net_param_queue, exp_queue)
    )
    coordinator.start()

    all_cooked_time, all_cooked_bw, _ = load_trace.load_trace(TRAIN_TRACES)

    agents = []
    for i in range(NUM_AGENTS):
        agents.append(mp.Process(
            target=agent, 
            args=(i, all_cooked_time, all_cooked_bw, 
                  net_param_queue[i], exp_queue[i])
        ))
        agents[i].start()

    coordinator.join()
    for single_agent in agents:
        single_agent.join()    

if __name__ == '__main__':
    torch.set_num_threads(1)
    main()
