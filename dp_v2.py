# given a result of a metric, calculate DP
'''
{
    'nq': {
        'newbing': [...],
        'gpt0125': [...], ...
    },
    'hotpotqa': {...}
}
'''
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import json, tqdm, math, itertools, random, argparse
import numpy as np
arguments = argparse.ArgumentParser()
arguments.add_argument('--metric', type=str, default='')
arguments.add_argument('--dataset', type=str, default='')
args = arguments.parse_args()


# ===== load metric result =====
file_path = f'/home/u2022000150/ufo/results/{args.metric}.json'
with open(file_path, 'r', encoding='utf-8') as fp:
    data = json.load(fp)
x = list()
system2score_dict = data[args.dataset]
for system, score_list in system2score_dict.items():
    x.append(score_list)
x = np.array(x)
'''
- x.shape: [systems, scores]
[
    [0.5, 0.2, 0.3],    # Bing Chat (samples count Q=3)
    [0.2, 0.1, 0.2],    # ChatGPT
    ...
]
'''



def M(s, random_indexes):
    '''
    calculate the average under the random_indexes
    '''
    return np.mean(s[random_indexes])

def calculate_metrics(x, fuzziness, bootstrap_times=1000):
    systems, Q = x.shape
    C_sum = math.comb(systems, 2)
    # [1, 2, 4] --> [(1, 2), (1, 4), (2, 4)]
    combinations_list = list(itertools.combinations(x, 2))
    # for each system pair (X, Y) \in C
    EQs, GTs = list(), list()
    for X, Y in tqdm.tqdm(combinations_list, desc='sampling', ncols=100):
        EQ, GT_xy, GT_yx = 0, 0, 0
        for b in range(bootstrap_times):
            # 每一轮获取bootstrap采样下标
            # (0, 1, 2, 3) --> [0, 0, 3, 2]
            random_indexes = random.choices(range(Q), k=Q)
            metric_output_x, metric_output_y = M(X, random_indexes), M(Y, random_indexes)
            margin = fuzziness * max(metric_output_x, metric_output_y)
            if abs(metric_output_x - metric_output_y) < margin:
                EQ += 1
            elif metric_output_x > metric_output_y:
                GT_xy += 1
            else:
                GT_yx += 1
        EQs.append(EQ)
        GTs.append(min(GT_xy, GT_yx))
    MR = np.sum(GTs) / (bootstrap_times * C_sum)
    PT = np.sum(EQs) / (bootstrap_times * C_sum)
    DP = 1-MR
    print(f'pt: {PT:.5f}\tdp: {DP:.5f}')
    return DP, PT

def binary_search_fuzziness(x, target_pt=0.05, tolerance=0.0001, max_iter=1000):
    low, high = 0.0, 1.0
    best_fuzziness, best_dp = None, None
    
    for i in range(max_iter):
        mid = (low + high) / 2
        dp, pt = calculate_metrics(x, mid)
        
        if abs(pt - target_pt) < tolerance:
            best_fuzziness = mid
            best_dp = dp
            break
        
        if pt < target_pt:
            low = mid
        else:
            high = mid
    
    return best_fuzziness, best_dp, pt
    

if __name__ == '__main__':
    best_fuzziness, best_dp, pt = binary_search_fuzziness(x, target_pt=0.05)    
    print(f'dataset: {args.dataset}\tmetric: {args.metric}\tthreshold: {best_fuzziness:.4f}\talpha: {pt:.4f}\tDP: {best_dp:.4f}')