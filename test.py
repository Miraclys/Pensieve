import os
import numpy as np

TEST_RESULTS = './test_results'

files = os.listdir(TEST_RESULTS)
reward_list = []
for file in files:
    with open(os.path.join(TEST_RESULTS, file), 'r') as f:
        next(f)
        for line in f:
            reward_list.append(float(line.strip()[-1]))
print(np.mean(reward_list))
print(len(reward_list))
