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

# ===== initialize relevant functions =====
def t_test(X):
    '''
    In order to conduct a Hypothesis Test, 
    we need a test statistic t and a null hypothesis distribution.
    '''
    m, s = np.mean(X), np.std(X, ddof=1)
    t = m / (s / np.sqrt(len(X)))
    return t
def create_bootstrap(w: list):
    '''
    Figure 1 from Sakai
    randomly sampling with replacement size n=|Q|
    returns: w^{*b} = w_{1}^{*b}, w_{2}^{*b}, w_{3}^{*b}, ...
    '''
    random_indexes = random.choices(range(len(w)), k=len(w))
    return w[random_indexes]
def get_ASL(X: list, Y: list, bootstrap_times: int):
    '''
    Figure 2 from Sakai
    estimating the Achieved Significance Level (ASL) based on the **Paired Test**.
    '''
    # z_i = x_i - y_i
    Z = X - Y
    # t(z)
    t_Z = t_test(Z)
    # w_i = z_i - \overline{z}
    W = Z - np.mean(Z)
    # get the t(w_{X, Y}^{*i})
    t_w_XY_list = list()
    count = 0
    for b in range(bootstrap_times):
        # get w^{*b}
        W_bootstrap = create_bootstrap(W)
        # get t(w^{*b})
        t_W = t_test(W_bootstrap)
        t_w_XY_list.append((abs(t_W), abs(np.mean(W_bootstrap))))
        if abs(t_W) >= abs(t_Z):
            count += 1
    ASL = count / bootstrap_times
    return ASL, t_w_XY_list

def calculate_dp(x: list, bootstrap_times:int=1000, alpha:float=0.05):
    '''
    Figure 5 from Sakai
    get the satisfied_num, combination_total, estimated_diff for a metric
    '''
    estimated_diff = list()
    satisfied_num = 0
    systems, Q = x.shape
    # get the total pairs of samples
    combination_total = math.comb(systems, 2)   
    # get pairs of combinations
    # e.g., [1, 2, 4] --> [(1, 2), (1, 4), (2, 4)]
    combinations_list = list(itertools.combinations(x, 2))
    # The idea is simple: Perform a Bootstrap Hypothesis Test for every system pair (X, Y) \in C, and count how many of the pairs (satisfied_num) satisfy ASL < \alpha.
    for X, Y in tqdm.tqdm(combinations_list, desc='sampling', ncols=80):
        ASL, t_w_XY_list = get_ASL(X, Y, bootstrap_times)
        if ASL < alpha: 
            satisfied_num += 1
        # sort t(w_{X,Y})
        sorted_t_w_XY_list = sorted(t_w_XY_list, key=lambda x: x[0])
        index = int(bootstrap_times * alpha)
        estimated_diff.append(sorted_t_w_XY_list[-index][1])
    estimated_diff = max(estimated_diff)
    return satisfied_num, combination_total, estimated_diff


# ===== running =====
if __name__ == '__main__':
    satisfied_num, combination_total, estimated_diff = calculate_dp(x, bootstrap_times=1000, alpha=0.05)
    ASL_satisfied = satisfied_num / combination_total
    print(f"{args.metric}; {args.dataset};\nASL < alpha: {ASL_satisfied:.3f}\nestimated_diff: {estimated_diff:.3f}")