# given a result of a metric, calculate DP
'''
nq:
{
    'newbing': [...],
    'gpt0125': [...],
}
'''
import json, tqdm, math, itertools, random, argparse, os
import numpy as np
arguments = argparse.ArgumentParser()
arguments.add_argument('--datasets', nargs='+')
arguments.add_argument('--prefix', type=str, default='')
arguments.add_argument('--metric', type=str, default='')
arguments.add_argument('--bisearch', action='store_true')
arguments.add_argument('--ratio', type=float)
arguments.add_argument('--bootstrap_times', type=int)
args = arguments.parse_args()


# ===== load metric result =====
def get_dataset_results(dataset: str) -> np.array:
    file_path = f'{args.prefix}/{args.metric}/{dataset}.json'    
    with open(file_path, 'r', encoding='utf-8') as fp:
        system2score_dict = json.load(fp)
    x = list()
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
    return x


def M(s, random_indexes):
    '''
    calculate the average under the random_indexes
    '''
    return s[random_indexes], np.mean(s[random_indexes])

def calculate_metrics(x, fuzziness, bootstrap_times=200, ratio=1):
    systems, Q = x.shape
    C_sum = math.comb(systems, 2)
    # [1, 2, 4] --> [(1, 2), (1, 4), (2, 4)]
    combinations_list = list(itertools.combinations(x, 2))
    # for each system pair (X, Y) \in C
    EQs, GTs = list(), list()
    for X, Y in combinations_list:
        EQ, GT_xy, GT_yx = 0, 0, 0
        for b in range(bootstrap_times):
            # get bootstrap index
            # (0, 1, 2, 3) --> [0, 0, 3, 2]
            random_indexes = random.choices(range(Q), k=int(Q*ratio))
            X_selected, metric_output_x = M(X, random_indexes)
            Y_selected, metric_output_y = M(Y, random_indexes)
            
            if metric_output_x < 0 and metric_output_y < 0:
                crit = max(max(X_selected)-min(X_selected), max(Y_selected)-min(Y_selected))
                # crit = abs(min(metric_output_x, metric_output_y))
            else:
                crit = max(max(X_selected)-min(X_selected), max(Y_selected)-min(Y_selected))
                # crit = max(metric_output_x, metric_output_y)
            margin = fuzziness * crit
            # print(f'x: {metric_output_x:.4f}\ty: {metric_output_y:.4f}\tmargin: {margin:.4f}')
            if abs(metric_output_x - metric_output_y) < margin:
                EQ += 1
            elif metric_output_x > metric_output_y:
                GT_xy += 1
            else:
                GT_yx += 1
        # print(f'EQ: {EQ}\tGT: {min(GT_xy, GT_yx)}\txy: {GT_xy}\tyx: {GT_yx}')
        EQs.append(EQ)
        GTs.append(min(GT_xy, GT_yx))
    MR = np.sum(GTs) / (bootstrap_times * C_sum)    # MR estimates the chance of reaching a wrong conclusion about a system pair
    PT = np.sum(EQs) / (bootstrap_times * C_sum)    # while P T reflects lack of discriminative power. Thus, for a good performance metric, both of these values should be small.
    DP = 1-MR
    print(f'f: {fuzziness:.3f}\tmr: {MR:.3f}\tpt: {PT:.3f}\tdp: {DP:.3f}')
    return fuzziness, MR, PT, DP

def binary_search_fuzziness(x, target_pt=0.05, tolerance=0.0001, bootstrap_times=1000, max_iter=1000, ratio=0.2):
    low, high = 0.0, 1.0
    best_fuzziness, best_dp = None, None
    
    for i in range(max_iter):
        mid = (low + high) / 2
        dp, pt = calculate_metrics(x, mid, ratio=ratio, bootstrap_times=bootstrap_times)
        
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
    os.makedirs(f'dp/{args.metric}', exist_ok=True)
    for dataset in args.datasets:
        output_fn = f'dp/{args.metric}/{dataset}_ratio_{args.ratio}_bs_{args.bootstrap_times}.json'
        print(f'dataset: {dataset}')
        x = get_dataset_results(dataset)
        if args.bisearch:
            print('binary search!')
            best_fuzziness, best_dp, pt = binary_search_fuzziness(x, target_pt=0.05, ratio=0.01)
            print(f'dataset: {dataset}\tmetric: {args.metric}\tthreshold: {best_fuzziness:.4f}\talpha: {pt:.4f}\tDP: {best_dp:.4f}')
        else:
            print("given fuzziness!")
            with open(output_fn, 'w', encoding='utf-8') as fw:
                # for fuzziness in np.arange(0.002, 0.1, 0.002):
                _result = dict()
                for fuzziness in list(np.arange(0.005, 0.205, 0.005)):
                    _fuzziness, MR, PT, DP = calculate_metrics(x, fuzziness, bootstrap_times=args.bootstrap_times, ratio=args.ratio)
                    _result[_fuzziness] = {
                        "MR": f'{MR:.3f}', 
                        "PT": f'{PT:.3f}',
                        "DP": f'{DP:.3f}'
                    }
                json.dump(_result, fw, ensure_ascii=False, indent=2)