# Import essential modules
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools; import joblib; import time
from scipy import linalg
import warnings

# Set pandas options
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Import DataFrame to fill out
df = pd.read_excel('empty.xlsx')

# Import reservoir modules
import module.predict_by_reservoir_single as lrt_1
import module.predict_by_reservoir_uncoupled as lrt_2
import module.predict_by_reservoir_coupled_low as lrt_3
import module.predict_by_reservoir_coupled_high as lrt_4

# Training parameters 
test_fraction = 0.2 
target_var = 'z'
ridge_lambda = 1.5

# Paratemers of the model
epsilon0 = 0.0667
q0 = 0.0008
gamma0 = 0.0000889
f0 = 0.65
alfa0 = 270
beta0 = 430
betap0 = 412.8
kappa0_low = 0.00002475
kappa0_high = 0.000099 

# Initial conditions 
in_phase = [0, 3.287, 0.006225, 0.4, 0.4]
out_of_phase = [0, 24.26, 0.04515, 0.4, 0.404]
anti_phase = [0.1411, 0, 0.06109, 0.4, 0.404]
in_in = [0, 3.287, 0.006225, 0.4, 0.4, 0, 3.287, 0.006225, 0.4, 0.4]
in_out = [0, 3.287, 0.006225, 0.4, 0.4, 0, 24.26, 0.04515, 0.4, 0.404]
in_anti = [0, 3.287, 0.006225, 0.4, 0.4, 0.1411, 0, 0.06109, 0.4, 0.404]
out_in = [0, 24.26, 0.04515, 0.4, 0.404, 0, 3.287, 0.006225, 0.4, 0.4]
out_out = [0, 24.26, 0.04515, 0.4, 0.404, 0, 24.26, 0.04515, 0.4, 0.404]
out_anti = [0, 24.26, 0.04515, 0.4, 0.404, 0.1411, 0, 0.06109, 0.4, 0.404]
anti_in = [0.1411, 0, 0.06109, 0.4, 0.404, 0, 3.287, 0.006225, 0.4, 0.4]
anti_out = [0.1411, 0, 0.06109, 0.4, 0.404, 0, 24.26, 0.04515, 0.4, 0.404]
anti_anti = [0.1411, 0, 0.06109, 0.4, 0.404, 0.1411, 0, 0.06109, 0.4, 0.404]

IC_single = [in_phase, out_of_phase, anti_phase]
IC_dual = [in_in, in_out, in_anti, out_in, out_out, out_anti, anti_in, anti_out, anti_anti]

def main(): 
    # We iterate over the four reservoir cases (single, uncoupled, coupled low, coupled high)
    
    for i in tqdm([1, 2, 3, 4]): # 1, 2, 3, 4
        # If i==1 we are going to use IC_single set of initial conditions (single oscillator)
        if i == 1:
            iterator = 0
            # We iterate over the different datasets
            for j in [100, 90, 80, 70, 60, 50, 40, 30, 20]:
                lorenz = pd.read_csv(f'./data/lorenz_{j}.csv')
                target_ts = lorenz
                # We iterate over the different initial conditions
                for m, init in enumerate(IC_single):
                    def single_reservoir_computing_parallel(init = init, target_ts = target_ts, w_in_sparsity = 0.1, w_in_strength = 0.1,
                                                            epsilon = epsilon0, q = q0, gamma = gamma0, f = f0, alfa = alfa0, beta = beta0):
                        par_lrc = lrt_1.Reservoir(network_name = "BZ Reaction")
                        par_lrc.prepare_target_data(target_ts, target_var, test_fraction)
                        par_lrc.initialize_reservoir(num_reservoir_nodes = 5, w_in_sparsity = w_in_sparsity, w_in_strength = w_in_strength)
                        par_lrc.compute_reservoir_state(epsilon = epsilon, q = q, gamma = gamma, f = f, alfa = alfa, beta = beta,
                                   initial_state = init)
                        par_lrc.learn_model(ridge_lambda = ridge_lambda, washout_fraction = 0.05)
                        par_lrc.predict()
                        par_lrc.summarize_stat()
                        return par_lrc.result_summary_df
                    warnings.filterwarnings("ignore")
                    # We do parameter search  for sparsity and strength 
                    WinSp_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                    WinSt_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
                    par_rc1 = joblib.Parallel(n_jobs=-3, verbose = 10)([joblib.delayed(single_reservoir_computing_parallel)
                                    (w_in_sparsity = x, w_in_strength = y) for x, y in itertools.product(WinSp_range, WinSt_range)])
                    output_all_df1 = par_rc1[0]
                    for k in range(1,len(par_rc1)): output_all_df1 = output_all_df1.append(par_rc1[k])
                    output_all_df1 = output_all_df1.reset_index(drop = True)
                    wspst_min = output_all_df1.loc[output_all_df1['NMSE_test'].min() == output_all_df1['NMSE_test']]
                    w_in_sp_best = float(np.array(wspst_min["Win_sparsity"])[0])
                    w_in_st_best = float(np.array(wspst_min["Win_strength"])[0])
                    # We perform RC using the best results
                    lrc = lrt_1.Reservoir(network_name = "BZ Reaction") 
                    lrc.prepare_target_data(target_ts, target_var, test_fraction) 
                    lrc.initialize_reservoir(num_reservoir_nodes=5, w_in_sparsity = w_in_sp_best, w_in_strength = w_in_st_best)
                    lrc.compute_reservoir_state(epsilon = epsilon0, q = q0, gamma = gamma0, f = f0, alfa = alfa0, 
                                                beta = beta0, initial_state = init)
                    lrc.learn_model(ridge_lambda = ridge_lambda)
                    lrc.predict()
                    lrc.summarize_stat() 
                    # We perform simple ridge regression under the same condition 
                    pred_wo_reservoir = lrt_1.Reservoir(network_name = "BZ Reaction")
                    pred_wo_reservoir.prepare_target_data(target_ts, target_var, test_fraction)
                    pred_wo_reservoir.learn_model_wo_reservoir(ridge_lambda = ridge_lambda) 
                    pred_wo_reservoir.predict_wo_reservoir() 
                    # We fill out the empty spaces in the DataFrame 
                    df['RMSE (test)'][m+iterator] = pred_wo_reservoir.test_rmse
                    df['RMSE (test)'][m+iterator+27] = float(lrc.result_summary_df["RMSE_test"][0])
                iterator += 3
        # If i==2 we are going to use IC_dual set of initial conditions (uncoupled)
        elif i == 2:
            iterator = 0
            # We iterate over the different datasets
            for j in [100, 90, 80, 70, 60, 50, 40, 30, 20]:
                lorenz = pd.read_csv(f'./data/lorenz_{j}.csv')
                target_ts = lorenz
                # We iterate over the different initial conditions
                for m, init in enumerate(IC_dual):
                    def uncoupled_reservoir_computing_parallel(init = init, target_ts = target_ts, w_in_sparsity = 0.1, w_in_strength = 0.1,
                                                               epsilon = epsilon0, q = q0, gamma = gamma0, f = f0, alfa = alfa0, 
                                                               beta = beta0, betap = betap0):
                        par_lrc = lrt_2.Reservoir(network_name = "BZ Reaction")
                        par_lrc.prepare_target_data(target_ts, target_var, test_fraction)
                        par_lrc.initialize_reservoir(num_reservoir_nodes = 10, w_in_sparsity = w_in_sparsity, w_in_strength = w_in_strength)
                        par_lrc.compute_reservoir_state(epsilon = epsilon, q = q, gamma = gamma, f = f, alfa = alfa, beta = beta, 
                                                        betap = betap, initial_state = init)
                        par_lrc.learn_model(ridge_lambda = ridge_lambda, washout_fraction = 0.05)
                        par_lrc.predict()
                        par_lrc.summarize_stat()
                        return par_lrc.result_summary_df 
                    warnings.filterwarnings("ignore")
                    # We do parameter search  for sparsity and strength 
                    WinSp_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
                    WinSt_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
                    par_rc1 = joblib.Parallel(n_jobs=-3, verbose = 10)([joblib.delayed(uncoupled_reservoir_computing_parallel)
                                    (w_in_sparsity = x, w_in_strength = y) for x, y in itertools.product(WinSp_range, WinSt_range)])
                    output_all_df1 = par_rc1[0]
                    for k in range(1,len(par_rc1)): output_all_df1 = output_all_df1.append(par_rc1[k])
                    output_all_df1 = output_all_df1.reset_index(drop = True)
                    wspst_min = output_all_df1.loc[output_all_df1['NMSE_test'].min() == output_all_df1['NMSE_test']]
                    w_in_sp_best = float(np.array(wspst_min["Win_sparsity"])[0])
                    w_in_st_best = float(np.array(wspst_min["Win_strength"])[0])
                    # We perform RC using the best results 
                    lrc = lrt_2.Reservoir(network_name = "BZ Reaction")
                    lrc.prepare_target_data(target_ts, target_var, test_fraction) 
                    lrc.initialize_reservoir(num_reservoir_nodes=10, w_in_sparsity = w_in_sp_best, w_in_strength = w_in_st_best)
                    lrc.compute_reservoir_state(epsilon = epsilon0, q = q0, gamma = gamma0, f = f0, alfa = alfa0, 
                                                beta = beta0, betap = betap0, initial_state = init)
                    lrc.learn_model(ridge_lambda = ridge_lambda)
                    lrc.predict()
                    lrc.summarize_stat() 
                    # We fill out the empty spaces in the DataFrame 
                    df['RMSE (test)'][m+iterator+54] = float(lrc.result_summary_df["RMSE_test"][0])
                iterator += 9
        # If i==3 we are going to use IC_dual set of initial conditions (coupled low)
        elif i == 3:
            iterator = 0
            # We iterate over the different datasets
            for j in [100, 90, 80, 70, 60, 50, 40, 30, 20]:
                lorenz = pd.read_csv(f'./data/lorenz_{j}.csv')
                target_ts = lorenz
                # We iterate over the different initial conditions
                for m, init in enumerate(IC_dual):
                    def coupled_low_reservoir_computing_parallel(init = init, target_ts = target_ts, w_in_sparsity = 0.1, 
                                                                 w_in_strength = 0.1, epsilon = epsilon0, q = q0, gamma = gamma0, 
                                                                 f = f0, alfa = alfa0, beta = beta0, betap = betap0, kappa = kappa0_low):
                        par_lrc = lrt_3.Reservoir(network_name = "BZ Reaction")
                        par_lrc.prepare_target_data(target_ts, target_var, test_fraction)
                        par_lrc.initialize_reservoir(num_reservoir_nodes = 10, w_in_sparsity = w_in_sparsity, w_in_strength = w_in_strength)
                        par_lrc.compute_reservoir_state(epsilon = epsilon, q = q, gamma = gamma, f = f, alfa = alfa, beta = beta, 
                                                        betap = betap, kappa = kappa, initial_state = init)
                        par_lrc.learn_model(ridge_lambda = ridge_lambda, washout_fraction = 0.05)
                        par_lrc.predict() 
                        par_lrc.summarize_stat() 
                        return par_lrc.result_summary_df 
                    warnings.filterwarnings("ignore")
                    # We do parameter search  for sparsity and strength
                    WinSp_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
                    WinSt_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
                    par_rc1 = joblib.Parallel(n_jobs=-3, verbose = 10)([joblib.delayed(coupled_low_reservoir_computing_parallel)
                                    (w_in_sparsity = x, w_in_strength = y) for x, y in itertools.product(WinSp_range, WinSt_range)])
                    output_all_df1 = par_rc1[0]
                    for k in range(1,len(par_rc1)): output_all_df1 = output_all_df1.append(par_rc1[k])
                    output_all_df1 = output_all_df1.reset_index(drop = True)
                    wspst_min = output_all_df1.loc[output_all_df1['NMSE_test'].min() == output_all_df1['NMSE_test']]
                    w_in_sp_best = float(np.array(wspst_min["Win_sparsity"])[0])
                    w_in_st_best = float(np.array(wspst_min["Win_strength"])[0])
                    # We perform RC using the best results 
                    lrc = lrt_3.Reservoir(network_name = "BZ Reaction")
                    lrc.prepare_target_data(target_ts, target_var, test_fraction)
                    lrc.initialize_reservoir(num_reservoir_nodes=10, w_in_sparsity = w_in_sp_best, w_in_strength = w_in_st_best)
                    lrc.compute_reservoir_state(epsilon = epsilon0, q = q0, gamma = gamma0, f = f0, alfa = alfa0, 
                                                beta = beta0, betap = betap0, kappa = kappa0_low, initial_state = init)
                    lrc.learn_model(ridge_lambda = ridge_lambda)
                    lrc.predict()
                    lrc.summarize_stat()
                    # We fill out the empty spaces in the DataFrame 
                    df['RMSE (test)'][m+iterator+135] = float(lrc.result_summary_df["RMSE_test"][0])
                iterator += 9
        # If i==4 we are going to use IC_dual set of initial conditions (coupled high)
        else:
            iterator = 0
            # We iterate over the different datasets
            for j in [100, 90, 80, 70, 60, 50, 40, 30, 20]:
                lorenz = pd.read_csv(f'./data/lorenz_{j}.csv')
                target_ts = lorenz
                # We iterate over the different initial conditions
                for m, init in enumerate(IC_dual):
                    def coupled_high_reservoir_computing_parallel(init = init, target_ts = target_ts, w_in_sparsity = 0.1, 
                                                                  w_in_strength = 0.1, epsilon = epsilon0, q = q0, gamma = gamma0, 
                                                                  f = f0, alfa = alfa0, beta = beta0, betap = betap0, kappa = kappa0_high):
                        par_lrc = lrt_4.Reservoir(network_name = "BZ Reaction")
                        par_lrc.prepare_target_data(target_ts, target_var, test_fraction)
                        par_lrc.initialize_reservoir(num_reservoir_nodes = 10, w_in_sparsity = w_in_sparsity, w_in_strength = w_in_strength)
                        par_lrc.compute_reservoir_state(epsilon = epsilon, q = q, gamma = gamma, f = f, alfa = alfa, beta = beta, 
                                                        betap = betap, kappa = kappa, initial_state = init)
                        par_lrc.learn_model(ridge_lambda = ridge_lambda, washout_fraction = 0.05)
                        par_lrc.predict() 
                        par_lrc.summarize_stat()
                        return par_lrc.result_summary_df
                    warnings.filterwarnings("ignore")
                    # We do parameter search  for sparsity and strength
                    WinSp_range = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] 
                    WinSt_range = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
                    par_rc1 = joblib.Parallel(n_jobs=-3, verbose = 10)([joblib.delayed(coupled_high_reservoir_computing_parallel)
                                    (w_in_sparsity = x, w_in_strength = y) for x, y in itertools.product(WinSp_range, WinSt_range)])
                    output_all_df1 = par_rc1[0]
                    for k in range(1,len(par_rc1)): output_all_df1 = output_all_df1.append(par_rc1[k])
                    output_all_df1 = output_all_df1.reset_index(drop = True)
                    wspst_min = output_all_df1.loc[output_all_df1['NMSE_test'].min() == output_all_df1['NMSE_test']]
                    w_in_sp_best = float(np.array(wspst_min["Win_sparsity"])[0])
                    w_in_st_best = float(np.array(wspst_min["Win_strength"])[0])
                    # We perform RC using the best results 
                    lrc = lrt_4.Reservoir(network_name = "BZ Reaction")
                    lrc.prepare_target_data(target_ts, target_var, test_fraction)
                    lrc.initialize_reservoir(num_reservoir_nodes=10, w_in_sparsity = w_in_sp_best, w_in_strength = w_in_st_best)
                    lrc.compute_reservoir_state(epsilon = epsilon0, q = q0, gamma = gamma0, f = f0, alfa = alfa0, 
                                                beta = beta0, betap = betap0, kappa = kappa0_high, initial_state = init)
                    lrc.learn_model(ridge_lambda = ridge_lambda)
                    lrc.predict()
                    lrc.summarize_stat()
                    # We fill out the empty spaces in the DataFrame 
                    df['RMSE (test)'][m+iterator+216] = float(lrc.result_summary_df["RMSE_test"][0])
                iterator += 9
                
    df.to_excel('results_lambda=1.5_violet.xlsx')
                
if __name__=="__main__":
    main()