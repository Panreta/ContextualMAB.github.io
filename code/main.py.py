import numpy as np
import heapq
import matplotlib.pyplot as plt
import random

def clear_variables():
    """Clears only variables while keeping functions and imports."""
    # List of protected identifiers (functions, imports, etc.)
    protected = {
        name for name, obj in globals().items() 
        if name.startswith('__') or 
           callable(obj) or 
           isinstance(obj, type) or 
           name in ['np', 'plt', 'heapq', 'random','clear_variables']  # Add your imports here
    }
    
    # Get all non-protected variables
    to_delete = [name for name in globals() 
                if not name.startswith('_') and name not in protected]
    
    # Delete them
    for name in to_delete:
        del globals()[name]

def oracle_select(ucb_values, k):
    """Selects top-k arms using UCB values."""
    return heapq.nlargest(k, range(len(ucb_values)), key=lambda i: ucb_values[i])

def expect_reward(mu,super_arm):
    product = 1
    for i in super_arm:
        product *= (1-mu[i])
    return 1 - product


def trigger(super_arm,mu_on):
    mu_update = np.zeros(len(super_arm))
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
    choose = [np.random.randint(10, num_each_arm + 1) for i in range(m)]
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
        print(f"mu_hat_off:{mu_hat_off}")
        super_arm = oracle_select(mu_hat_off, k)
        print(f"super_arm_off:{super_arm}")
        print(f"super_arm:{super_arm}")
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

def hybrid(m,k,t,N_online,mu_on,mu_hat_on,N,mu_hat_off,V):
    ucb_online = np.zeros(m)
    ucb_hybrid = np.zeros(m)
    
    for i in range(m):
        # log_term = np.log(4 * m * t )
        log_term = np.log(100)

        # N_online
        if N_online[i] == 0:
            ucb_online[i] = 1
        else:
            ucb_online[i] = mu_hat_on[i] + np.sqrt(2 * log_term / (N_online[i]))

        # N[i] + N_online[i] == 0
        if N[i] + N_online[i] == 0:
            ucb_hybrid[i] = 1
        else:
            ucb_hybrid[i] = np.divide(N[i]*mu_hat_off[i] + N_online[i]* mu_hat_on[i], N[i] + N_online[i]) + np.sqrt(2 * log_term / (N_online[i]+N[i])) + np.divide(N[i]*V[i],N[i] + N_online[i])
            
    # ucb = np.minimum(ucb_online, ucb_hybrid)
    ucb = np.minimum(np.minimum(ucb_online, ucb_hybrid), np.ones(m))
    print(f"ucb={ucb}")

    ## Trigger part
    # super_arm = oracle_select(ucb,k)
    super_arm = oracle_select(ucb,k)
    print(f"hybrid super_arm:{super_arm}")

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

def single_run_hybrid(m, k, T, reward_star, mu_on,gap_offline, N, mu_hat_off, V1):
    import os
    from scipy.interpolate import make_interp_spline
    from scipy.signal import savgol_filter

    # 1.initial
    N_online,mu_hat_on = np.zeros(m),np.zeros(m)
    N_biased,mu_hat_biased  = np.zeros(m),np.zeros(m)
    gap_online = np.zeros(T)
    gap_hybrid_biased = np.zeros(T)
    cumulative_online = 0.0
    cumulative_hybrid = 0.0

    # 2. running 
    for t in range(1, T+1):
        print("--------------------------------------------------")
       # save_path as you like
        print(f"t={t}")
        print("online part:")
        reward_online,N_online,mu_hat_on= hybrid(m,k,t,N_online,mu_on,mu_hat_on,N = np.zeros(m),mu_hat_off=np.zeros(m),V = np.zeros(m))
        cumulative_online += (reward_star - reward_online)
        print(f"gap_on:{reward_star - reward_online}")
        gap_online[t - 1] = cumulative_online
        
        print("##########################--------------------------")
        print("biased part:")
        reward_bias,N_biased,mu_hat_biased= hybrid(m,k,t,N_biased,mu_on,mu_hat_biased,N,mu_hat_off,V1)
        print(f"reward_bias:{reward_bias}")
        cumulative_hybrid += (reward_star - reward_bias)
        gap_hybrid_biased[t - 1] = cumulative_hybrid
        print(f"gap_hybrid:{reward_star - reward_bias}")
      

        # if t % 200 == 0:
        #     plt.figure()


        #     plt.xlim(1, t)
        #     plt.ylim(0, 50)
        #     plt.xlabel("Time Steps (t)")
        #     plt.ylabel("Cumulative Regret")
        #     plt.title(f"V={V1[0]}, N={200}")
        #     plt.legend(loc='upper left', frameon=True, shadow=True)
        #     plt.grid(True, linestyle='--', alpha=0.7)

        #     file_name = f"graph_t{t}.pdf"
        #     plt.savefig(os.path.join(save_path, file_name), format='pdf')
        #     plt.close()


    return gap_online , gap_hybrid_biased


    # 1. setting
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Pool, cpu_count
if __name__ == "__main__":
    clear_variables()
    print("Clear up")
    m = 10
    k = 5
    num_trials = 20
    bias = 0
    mu_on = np.linspace(0,0.5,m)

    print(f"mu_on:{mu_on}")

    # # bias generation, if bias choose this part
    # while True:
    #     V = [bias] * m # Bias bound
    #     count = 0
    #     for i in range(m):
    #         if random.random() < 0.5:
    #             V[i] = -V[i]
    #             count += 1
    #     if m / 2 - 5 <= count <= m / 2 + 5:
    #         break


    # mu_off =  mu_on + V # Adjust : may fix
    mu_off =  mu_on # Adjust : may fix

    V1 = [bias] * m


    # 2.Find the oracle 
    super_arm_oracle = oracle_select(mu_on,k)
    print(f"super_arm_oracle:{super_arm_oracle}")
    reward_star = expect_reward(mu_on,super_arm_oracle)
    print(f"reward_star:{reward_star}")


    # 3. do offline, online, hybrid-unbiased, hybrid-biased
    ## 3.1 offline
    reward_off = 0
    reward = []
    num_each_arm = 200
    reward_off,N,mu_hat_off = MeanRewardOff(m,mu_off,k,mu_on,num_each_arm,off_turn= 1)
    print(f"reward_off:{reward_off}")

    print(f"gap_off:{reward_star - reward_off }")
    T = int(input("T:"))

    gap_offline = (reward_star - reward_off ) * range(1,T+1)
    gap_online = np.zeros(T)
    gap_hybrid_biased = np.zeros(T)
    gap_on = []
    gap_hb = []

    # 3.2 online, hybrid-unbiased, hybrid-biased
    N = [num_each_arm] * m

    args = (m,k,T,reward_star,mu_on,gap_offline, N,mu_hat_off,V1)

    tasks = [args] * num_trials
    num_processes = cpu_count() - 1 
    # run trials in parallel
    print("Starting parallel processing...")

    with Pool(processes=num_processes) as pool:
        results = pool.starmap(single_run_hybrid, tasks)

    print("Parallel processing complete.")
    print(f"gap_offline:{gap_offline}")


    gap_online_all, gap_hybrid_all = zip(*results) 
    gap_online_all = np.vstack(gap_online_all)         # shape (num_trials, T)
    gap_hybrid_all = np.vstack(gap_hybrid_all)         # shape (num_trials, T)

    mean_gap_online = gap_online_all.mean(axis=0)      # shape (T,)
    mean_gap_hybrid = gap_hybrid_all.mean(axis=0)      # shape (T,)

    std_gap_online = gap_online_all.std(axis=0)
    std_gap_hybrid = gap_hybrid_all.std(axis=0)


    # 4.Plot
    plt.plot(gap_offline, color='green', label='Offline Algorithm')
    print("There is offline part0")
    plt.plot(mean_gap_online, color='blue', label='Online Algorithm')
    plt.plot(mean_gap_hybrid, color='red', label='Hybrid biased Algorithm')

    plt.fill_between(range(1,T+1),
                 mean_gap_online - std_gap_online/np.sqrt(num_trials),
                 mean_gap_online + std_gap_online/np.sqrt(num_trials),
                 color='blue',
                 alpha=0.3,
                 label='±1 Std Dev')
    
    plt.fill_between(range(1,T+1),
                 mean_gap_online - std_gap_online/np.sqrt(num_trials),
                 mean_gap_online + std_gap_online/np.sqrt(num_trials),
                 color='blue',
                 alpha=0.3,
                 label='±1 Std Dev')


    plt.xlim(1,T)
    plt.ylim(0,50)
    plt.xlabel("Time Steps (t)")
    plt.ylabel("Cumulative Regret")
    plt.title(f"V={bias},N={num_each_arm}")

    # Customize the legend location and appearance
    plt.legend(loc='upper left', frameon=True, shadow=True)

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()