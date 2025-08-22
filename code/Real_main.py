import numpy as np
import heapq
import matplotlib.pyplot as plt
import random
import os
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, cpu_count
from numpy.random import f

def linear_map(stars_series,a,b):
    
    current_min = stars_series.min()
    current_max = stars_series.max()

    # Linear mapping formula: 
    # new_value = a + (value - current_min) * (b - a) / (current_max - current_min)
    mapped_stars = a + (stars_series - current_min) * (b - a) / (current_max - current_min)
    return mapped_stars

def oracle_select(ucb_values, k):
    """Selects top-k arms using UCB values."""
    return heapq.nlargest(k, range(len(ucb_values)), key=lambda i: ucb_values[i])

def expect_reward(mu,super_arm):
    product = 1
    for i in super_arm:
        product *= (1 - mu[i]) # must be mu_on
    return 1 - product

def trigger(super_arm,mu_on):
    mu_update = np.zeros(len(super_arm))# X_i,t
    observed_arm = []
    for i in super_arm:
        observed_arm.append(i)
        if random.random() < mu_on[i]: # X_i,t == 1
            mu_update[len(observed_arm) - 1] = 1
            return mu_update,observed_arm
    return mu_update,observed_arm # if 0, it is no arm searched out

def offline(m,mu_off,num_each_arm):
    delta = 0.1
    sample = np.zeros(m)
    log_CLCB = np.log(np.divide(2*m*num_each_arm ,delta))# n in paper: 一个臂拉取多少次，fix
    mu_hat_off = np.zeros(m)
    LCB = np.zeros(m)
    if num_each_arm == 10:
        choose = [np.random.randint(1, num_each_arm + 1) for i in range(m)]
    elif num_each_arm == 50:
        choose = [np.random.randint(30, num_each_arm + 1) for i in range(m)]
    else:
        choose = [np.random.randint(150, num_each_arm + 1) for i in range(m)]
    for i in range(m):
        sample[i] = np.random.binomial(choose[i],mu_off[i])
        mu_hat_off[i] = np.divide(sample[i],num_each_arm)
        LCB[i] = mu_hat_off[i]- np.sqrt(np.divide(log_CLCB,2* num_each_arm))
        
    return choose,mu_hat_off, LCB

def MeanRewardOff(m, mu_off, k, mu_on,num_each_arm,off_turn):
    all_rewards = []
    all_N = []         
    all_mu_hat_off = []
    
    for t in range(off_turn):
        N, mu_hat_off ,LCB= offline(m, mu_off,num_each_arm)
        super_arm = oracle_select(mu_hat_off, k)

        reward = expect_reward(mu_on, super_arm)
        # print(f"reward:{reward}")
        
        all_rewards.append(reward)
        all_N.append(N)
        all_mu_hat_off.append(mu_hat_off)
    
    # 计算均值
    mean_reward = np.mean(all_rewards)
    mean_N = np.mean(all_N, axis=0)  # axis = 0: do summation to row vector
    mean_mu_hat_off = np.mean(all_mu_hat_off, axis=0)
    
    return mean_reward, mean_N, mean_mu_hat_off

def hybrid(m,k,t,N_online,mu_on,mu_hat_on,N_off,mu_hat_off,V):
    ucb_online = np.zeros(m)
    ucb_hybrid = np.zeros(m)
    
    for i in range(m):
        # Pure online UCB
        # log_term = np.log(4 * m * t)
        log_term = np.log(100)

        # N_online
        if N_online[i] == 0:
            ucb_online[i] = 1
        else:
            ucb_online[i] = mu_hat_on[i] + np.sqrt(2 * log_term / (N_online[i]))

        # N[i] + N_online[i] == 0
        if N_off[i] + N_online[i] == 0:
            ucb_hybrid[i] = 1
        else:
            ucb_hybrid[i] = np.divide(N_off[i]*mu_hat_off[i] + N_online[i]* mu_hat_on[i], N_off[i] + N_online[i]) + np.sqrt(2 * log_term / (N_online[i]+N_off[i])) + np.divide(N_off[i]*V[i],N_off[i] + N_online[i])
            
    # ucb = np.minimum(ucb_online, ucb_hybrid)
    ucb = np.minimum(np.minimum(ucb_online, ucb_hybrid), np.ones(m))
    # print(f"ucb={ucb}")

    ## Trigger part
    # super_arm = oracle_select(ucb,k)
    super_arm = oracle_select(ucb,k)
    # print(f"hybrid super_arm:{super_arm}")

    reward_online = expect_reward(mu_on,super_arm)

    mu_update, observed_arm = trigger(super_arm,mu_on)


    # Reward part
    for i in observed_arm: 
        N_online[i] += 1

    if mu_update.sum() != 0:
        for idx, arm in enumerate(observed_arm):
            if idx == len(observed_arm) - 1: # chosen one
                mu_hat_on[arm] += np.divide(1 - mu_hat_on[arm],N_online[arm])
                continue
            mu_hat_on[arm] += np.divide(0 - mu_hat_on[arm],N_online[arm]) # triggered but not chosen

    else:
        for arm in observed_arm: # all did not chosen by observed
            mu_hat_on[arm] += np.divide(0 - mu_hat_on[arm],N_online[arm])
    return reward_online,N_online,mu_hat_on

def single_run_hybrid(m, k, T, reward_star, mu_on,N_off, mu_hat_off, V1):
    
    print('doing now')
    
    # 1.initial
    N_online,mu_hat_on = np.zeros(m),np.zeros(m)
    N_biased,mu_hat_biased  = np.zeros(m),np.zeros(m)
    gap_online = np.zeros(T)
    gap_hybrid_biased = np.zeros(T)
    cumulative_online = 0.0
    cumulative_hybrid = 0.0

    # 2. running 
    for t in range(1, T+1):
        # print("--------------------------------------------------")
        # print(f"t={t}")
        reward_online,N_online,mu_hat_on= hybrid(m,k,t,N_online,mu_on,mu_hat_on,np.zeros(m),mu_hat_off=np.zeros(m),V = np.zeros(m))
        cumulative_online += (reward_star - reward_online)
        gap_online[t - 1] = cumulative_online

        reward_bias,N_biased,mu_hat_biased= hybrid(m,k,t,N_biased,mu_on,mu_hat_biased,N_off,mu_hat_off,V1)
        cumulative_hybrid += (reward_star - reward_bias)
        gap_hybrid_biased[t - 1] = cumulative_hybrid        
    return gap_online , gap_hybrid_biased


def process_json():
    
    file_path = r"D:\Context_Bandit\Yelp-JSON\Yelp JSON\yelp_academic_dataset_business.json"

    # Initialize an empty list to store the business data
    business_data = []

    # Read the file line by line
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            # Parse each line as a separate JSON object
            business = json.loads(line)
            business_data.append(business)

    print(f"Successfully loaded {len(business_data)} business records")


    # Extract only business_id and stars from your loaded data
    business_stars = pd.DataFrame({
        'business_id': [b['business_id'] for b in business_data],
        'stars': [b['stars'] for b in business_data]
    })


    business_stars_unique = business_stars.drop_duplicates(subset=['business_id'], keep='first')

    business_stars_unique['numeric_id'] = pd.factorize(business_stars_unique['business_id'])[0] + 1  # +1 to start from 1 instead of 0



    # Add noise to stars
    business_stars_unique['stars_muon'] = business_stars_unique['stars'] + np.random.normal(0, 0.1)

    # Create a list of all indices and shuffle them randomly
    all_indices = list(range(len(business_stars_unique)))
    np.random.shuffle(all_indices)

    unique_stars_rows = pd.DataFrame()
    seen_stars = set()
    index_pos = 0  # Position in the shuffled indices list
    count = 0

    # First, collect 10 unique rows
    while count < 10 and index_pos < len(all_indices):
        # Get a random index from the shuffled list
        random_index = all_indices[index_pos]
        current_row = business_stars_unique.iloc[random_index]
        current_star = current_row['stars_muon']
        
        if current_star not in seen_stars:
            # Add the row to our result DataFrame
            unique_stars_rows = pd.concat([unique_stars_rows, current_row.to_frame().T])
            seen_stars.add(current_star)
            count += 1
        
        index_pos += 1

    # Reset index for the result
    unique_stars_rows = unique_stars_rows.reset_index(drop=True)

    # Create artificial 10th category if we only found 9 unique values
    if len(unique_stars_rows) == 9:
        print("Creating artificial 10th category...")
        # Take one value and transform it significantly
        modified_row = unique_stars_rows.iloc[0].copy()
        modified_row['stars_muon'] = 0.5  # Create a new category
        unique_stars_rows = pd.concat([unique_stars_rows, modified_row.to_frame().T])
        unique_stars_rows = unique_stars_rows.reset_index(drop=True)
    return unique_stars_rows

if __name__ == "__main__":

    unique_stars_rows = process_json()

    unique_stars_rows['stars_muon'] = linear_map(unique_stars_rows['stars_muon'],a=0.4,b=0.6)




    # clear_variables()

    print("Clear up")
    m = 10
    k = 5
    num_trials = 20
    # mu_on = np.linspace(0.4,0.6,m)
    mu_on = unique_stars_rows['stars_muon'].to_numpy()

    save_dir = r'D:\Github\ContextualMAB.github.io\graph_results'
    # mu_on = np.random.uniform(0.3,0.7,m) # Adjust : 0 can be changed to 0.5,0.6
    # mu_on[0] = 0.699

    # for bias in {0.2, 0.3, 0.4}:
    # for bias in {0.3}:
    # for num_each_arm in {10,50,200}:
    for num_each_arm in {200}:
        bias = 0
        # bias generation
        # while True:
        #     V = [bias] * m # Bias bound
        #     count = 0
        #     for i in range(m):
        #         if random.random() < 0.5:
        #             V[i] = -V[i]
        #             count += 1
        #     if m / 2 - 5 <= count <= m / 2 + 5:
        #         break


        V = [bias] * m # Bias bound
        # for i in range(m):
        #     if random.random() < 0.5:
        #         V[i] = -V[i]


        print(f"V={V}")


        mu_off =  mu_on + V 
        print(f"mu_off:{mu_off}")

        V1 = [bias] * m

        # 2.Find the oracle 
        super_arm_oracle = oracle_select(mu_on,k)
        reward_star = expect_reward(mu_on,super_arm_oracle)

        # 3. do offline, online, hybrid-unbiased, hybrid-biased
        ## 3.1 offline
        reward_off = 0
        reward = []
        # num_each_arm = 200
        reward_off,N_off,mu_hat_off = MeanRewardOff(m,mu_off,k,mu_on,num_each_arm,off_turn= 20)# 5000 is not good, maybe it's because of the seed
        T = 8000
        os.makedirs(save_dir, exist_ok=True)

        gap_offline = (reward_star - reward_off ) * range(1,T+1)
        gap_off_array = np.array(gap_offline)
        np.save(os.path.join(save_dir, f'gap_off_{num_each_arm}.npy'), gap_off_array)
        print("Off is done!")

        gap_online = np.zeros(T)
        gap_hybrid_biased = np.zeros(T)
        gap_on = []
        gap_hb = []

        # 3.2 online, hybrid-unbiased, hybrid-biased

        # Run the simulation 5 times

        N_off = [num_each_arm] * m

        # gap_online , gap_hybrid_biased = single_run_hybrid(m,k,T,reward_star,mu_on,N_off, mu_hat_off ,V1)
        print(f"gap_online:{gap_online}")
        print(f"gap_hybrid_biased:{gap_hybrid_biased}")

        args = (m,k,T,reward_star,mu_on,N_off,mu_hat_off,V1)
        tasks = [args] * num_trials
        num_processes = cpu_count() - 1



        with Pool(processes=num_processes) as pool:
            results = pool.starmap(single_run_hybrid, tasks)

        
        gap_online_all, gap_hybrid_all = zip(*results) 
        gap_online_all = np.vstack(gap_online_all)         # shape (num_trials, T)
        gap_hybrid_all = np.vstack(gap_hybrid_all)         # shape (num_trials, T)
        

        mean_gap_online = gap_online_all.mean(axis=0)      # shape (T,)
        mean_gap_hybrid = gap_hybrid_all.mean(axis=0)      # shape (T,)

        np.save(os.path.join(save_dir, f'gap_online_{num_each_arm}.npy'), gap_online_all)
        np.save(os.path.join(save_dir, f'gap_hb_{num_each_arm}.npy'), gap_hybrid_all)



        std_gap_online = gap_online_all.std(axis=0)
        std_gap_hybrid = gap_hybrid_all.std(axis=0)


        # 4.Plot
        plt.plot(gap_offline, color='green', label='CLCB')
        print("There is offline part0")
        plt.plot(mean_gap_online, color='blue', label='CUCB')
        plt.plot(mean_gap_hybrid, color='red', label='H-CUCB')

        plt.fill_between(range(1,T+1),
                    mean_gap_online - std_gap_online/np.sqrt(num_trials),
                    mean_gap_online + std_gap_online/np.sqrt(num_trials),
                    color='blue',
                    alpha=0.3,
                    label='±1 Std Dev')
        
        plt.fill_between(range(1,T+1),
                    mean_gap_hybrid - std_gap_hybrid/np.sqrt(num_trials),
                    mean_gap_hybrid + std_gap_hybrid/np.sqrt(num_trials),
                    color='red',
                    alpha=0.3,
                    label='±1 Std Dev')


        plt.xlim(1,T)
        plt.ylim(0,300)
        plt.xlabel("Time Steps (t)")
        plt.ylabel("Cumulative Regret")
        plt.title(f"V={bias},N={num_each_arm}")

        # Customize the legend location and appearance
        plt.legend(loc='upper left', frameon=True, shadow=True)

        plt.grid(True, linestyle='--', alpha=0.7)
        plt.savefig(rf"D:\Github\ContextualMAB.github.io\graph_results\{num_each_arm}.pdf", bbox_inches='tight', dpi=300)

        plt.figure()

        # plt.show()



    
