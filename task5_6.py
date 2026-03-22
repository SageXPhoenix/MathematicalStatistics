import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.ndimage as ndimage
import scipy.special as special
import mplcursors
import math

def generate_pareto_sample(thetta, n):
    sample = stats.pareto.rvs(thetta-1, size=n)
    return sample

def find_accurate_mean_trusted_interval_pareto():
    num_of_obs = 100
    new_sample = generate_pareto_sample(3, num_of_obs)
    quantiles = np.zeros((2,))
    xmax = np.amax(new_sample)
    quantiles[0] = -np.log(1 - 0.025**(1/num_of_obs))
    quantiles[1] = -np.log(1 - 0.975**(1/num_of_obs))
    print(f"Accurate trusted interval for mean: [{2**(np.log(xmax)/quantiles[1])}, {2**(np.log(xmax)/quantiles[0])}]")

def find_asymptotic_trusted_interval_pareto():
    num_of_obs = 100
    new_sample = generate_pareto_sample(50, num_of_obs)
    log_summ = np.sum(np.log(new_sample))
    quantiles = stats.norm.ppf([0.025, 0.975])
    trusted_interval_boundaries = np.zeros((2,))
    trusted_interval_boundaries[0] = 1 + num_of_obs/log_summ - (1/np.sqrt(num_of_obs))*quantiles[1]*(num_of_obs/log_summ)
    trusted_interval_boundaries[1] = 1 + num_of_obs/log_summ - (1/np.sqrt(num_of_obs))*quantiles[0]*(num_of_obs/log_summ)
    print(f"Asymptotic trusted interval: [{trusted_interval_boundaries[0]}, {trusted_interval_boundaries[1]}]")

def find_parametric_bootstrap_trusted_interval_pareto():
    beta = 0.95
    num_of_obs = 100
    num_of_bootstrap = 50000
    new_sample = generate_pareto_sample(50, num_of_obs)

    #Взял OMP оценку т.к она более эффективная
    estimation_OMP = 1 + num_of_obs/np.sum(np.log(new_sample))
    diff_estimations = np.zeros((num_of_bootstrap,))

    for i in range(0, num_of_bootstrap, 1):
        new_sample = generate_pareto_sample(estimation_OMP, num_of_obs)
        diff_estimations[i] = estimation_OMP - 1 - num_of_obs/np.sum(np.log(new_sample))
    
    diff_estimations = np.sort(diff_estimations)
    k1 = int(((1 - beta)/2)*num_of_bootstrap)
    k2 = int(((1 + beta)/2)*num_of_bootstrap)

    print(f"Parametric bootstrap trusted interval: [{estimation_OMP - diff_estimations[k2]}, {estimation_OMP - diff_estimations[k1]}]")

def find_nonparametric_bootstrap_trusted_interval_pareto():
    num_of_bootstrap = 1000
    num_of_obs = 100
    beta = 0.95

    new_sample = generate_pareto_sample(50, num_of_obs)
    new_sample = new_sample.reshape(num_of_obs, )

    estimation_OMP = 1 + num_of_obs/np.sum(np.log(new_sample))
    diff_estimations = np.zeros((num_of_bootstrap,))

    for i in range(num_of_bootstrap):
        sample = np.random.choice(new_sample, size=num_of_obs, replace=True)
        diff_estimations[i] = 1 + num_of_obs/np.sum(np.log(sample)) - estimation_OMP
    
    diff_estimations = np.sort(diff_estimations)
    k1 = int(((1 - beta)/2)*num_of_bootstrap)
    k2 = int(((1 + beta)/2)*num_of_bootstrap)

    print(f"Nonparametric bootstrap trusted interval: [{estimation_OMP - diff_estimations[k2]}, {estimation_OMP - diff_estimations[k1]}]")

def generate_uniform_sample(thetta, n):
    sample = np.random.uniform(thetta, thetta*2, (1,n))
    return sample

def find_accurate_trusted_interval_uniform():
    num_of_obs = 100
    new_sample = generate_uniform_sample(20, num_of_obs)
    quantiles = np.zeros((2,))
    xmax = np.amax(new_sample)
    quantiles[0] = 0.025**(1/num_of_obs) + 1
    quantiles[1] = 0.975**(1/num_of_obs) + 1
    print(f"Accurate trusted interval: [{xmax/quantiles[1]}, {xmax/quantiles[0]}]")

def find_asymptotic_trusted_interval_uniform():
    num_of_obs = 100
    new_sample = generate_uniform_sample(20, num_of_obs)
    first_moment = np.mean(new_sample)
    second_moment = np.mean(new_sample**2)
    quantiles = stats.norm.ppf([0.025, 0.975])
    trusted_interval_boundaries = np.zeros((2,))
    trusted_interval_boundaries[0] = 2*(first_moment - quantiles[1]*np.sqrt(second_moment - first_moment**2)/np.sqrt(num_of_obs))/3
    trusted_interval_boundaries[1] = 2*(first_moment - quantiles[0]*np.sqrt(second_moment - first_moment**2)/np.sqrt(num_of_obs))/3
    print(f"Asymptotic trusted interval: [{trusted_interval_boundaries[0]}, {trusted_interval_boundaries[1]}]")

def find_parametric_bootstrap_trusted_interval_uniform():
    beta = 0.95
    num_of_obs = 100
    num_of_bootstrap = 50000
    new_sample = generate_uniform_sample(20, num_of_obs)

    #Взял OMP оценку т.к она более эффективная
    estimation_OMP = np.amax(new_sample)*(num_of_obs + 1)/(2*num_of_obs + 1) 
    diff_estimations = np.zeros((num_of_bootstrap,))

    for i in range(0, num_of_bootstrap, 1):
        new_sample = generate_uniform_sample(estimation_OMP, num_of_obs)
        diff_estimations[i] = np.amax(new_sample)*(num_of_obs + 1)/(2*num_of_obs + 1) - estimation_OMP
    
    diff_estimations = np.sort(diff_estimations)
    k1 = int(((1 - beta)/2)*num_of_bootstrap)
    k2 = int(((1 + beta)/2)*num_of_bootstrap)

    print(f"Parametric bootstrap trusted interval: [{estimation_OMP - diff_estimations[k2]}, {estimation_OMP - diff_estimations[k1]}]")

def find_nonparametric_bootstrap_trusted_interval_uniform():
    num_of_bootstrap = 1000
    num_of_obs = 100
    beta = 0.95

    new_sample = generate_uniform_sample(20, num_of_obs)
    new_sample = new_sample.reshape(num_of_obs, )

    estimation_OMP = np.amax(new_sample)*(num_of_obs + 1)/(2*num_of_obs + 1) 
    diff_estimations = np.zeros((num_of_bootstrap,))

    for i in range(num_of_bootstrap):
        sample = np.random.choice(new_sample, size=num_of_obs, replace=True)
        diff_estimations[i] = np.amax(sample)*(num_of_obs + 1)/(2*num_of_obs + 1) - estimation_OMP
    
    diff_estimations = np.sort(diff_estimations)
    k1 = int(((1 - beta)/2)*num_of_bootstrap)
    k2 = int(((1 + beta)/2)*num_of_bootstrap)

    print(f"Nonparametric bootstrap trusted interval: [{estimation_OMP - diff_estimations[k2]}, {estimation_OMP - diff_estimations[k1]}]")


def main():
    print("For task 5:")
    find_accurate_trusted_interval_uniform()
    find_asymptotic_trusted_interval_uniform()
    find_parametric_bootstrap_trusted_interval_uniform()
    find_nonparametric_bootstrap_trusted_interval_uniform()
    print("For task 6:")
    find_accurate_mean_trusted_interval_pareto()
    find_asymptotic_trusted_interval_pareto()
    find_parametric_bootstrap_trusted_interval_pareto()
    find_nonparametric_bootstrap_trusted_interval_pareto()


if __name__ == "__main__":
    main()