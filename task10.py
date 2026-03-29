import numpy as np
from scipy.stats import norm

def empirical_cdf(data):
    sorted_data = np.sort(data)
    total_count = len(data)
    
    def cdf_func(value):
        return np.searchsorted(sorted_data, value, side='left') / total_count
    
    return cdf_func

raw_sample = np.repeat(np.arange(10), [5,8,6,12,14,18,11,6,13,7])

theoretical_cdf = norm.cdf(raw_sample, loc=4.77, scale=np.sqrt(6.28))
cdf_estimator = empirical_cdf(raw_sample)
cdf_extended = np.concatenate((np.array([cdf_estimator(raw_sample[j]) for j in range(100)]), [1]))

max_diff = -1
for j in range(100):
    current_diff = max(abs(theoretical_cdf[j] - cdf_extended[j]), abs(theoretical_cdf[j] - cdf_extended[j+1]))
    max_diff = max(max_diff, current_diff)

max_diff = max_diff * 10
print(max_diff)

def resample_test(mu_param: float, var_param: float, obs_stat: float, iterations: int) -> float:
    test_stats = np.zeros(50000)
    for k in range(iterations):
        generated_data = np.random.normal(loc=mu_param, scale=np.sqrt(var_param), size=100)
        mu_updated = np.mean(generated_data)
        var_updated = np.mean((generated_data - mu_updated)**2)
        generated_data.sort()
        theoretical_cdf = norm.cdf(generated_data, loc=mu_updated, scale=np.sqrt(var_updated))
        cdf_estimator = empirical_cdf(generated_data)
        cdf_extended = np.concatenate((np.array([cdf_estimator(generated_data[j]) for j in range(100)]), [1]))
        test_diff = -1
        for j in range(100):
            current_diff = max(abs(theoretical_cdf[j] - cdf_extended[j]), abs(theoretical_cdf[j] - cdf_extended[j+1]))
            test_diff = max(test_diff, current_diff)
        test_stats[k] = test_diff
    result = 0
    test_stats *= 10
    for j in range(iterations):
        result = result + 1 if test_stats[j] >= obs_stat else result
    return result/iterations

print(resample_test(4.77, 6.28, max_diff, 50000))