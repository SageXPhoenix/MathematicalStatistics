import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import scipy.ndimage as ndimage
import scipy.special as special
import mplcursors
import math

#Получение значений оценки плотности распределения медианы
def median_density(arrange):
    median_estimation = np.zeros(200)

    for i in range(13, 26):
        median_estimation += special.comb(25, i)*np.exp((-1)*arrange)*((1 - np.exp((-1)*arrange))**(i-1))*((1-(1 - np.exp((-1)*arrange)))**(24-i))*(i - 25*(1 - np.exp((-1)*arrange)))
        
    return median_estimation

#Бутстраповская оценка медиана
def bootstrap_ar_mean(arrange):
    list_of_ar_mean = np.zeros(1000)

    for i in range(1, 1001, 1):
        sample = np.random.choice(arrange, size=25, replace=True)
        ar_mean = np.mean(sample)
        list_of_ar_mean[i-1] = ar_mean
    
    return list_of_ar_mean

#Бутстраповская оценка среднего арифметического
def bootstrap_median(example):
    list_of_ar_median = np.zeros(1000)

    for i in range(1, 1001, 1):
        sample = np.random.choice(example, size=25, replace=True)
        ar_median = ndimage.median(sample)
        list_of_ar_median[i-1] = ar_median
    
    return list_of_ar_median

#Бутстраповская оценка распределения коэффицента асимметрии
def bootstrap_coeff_asimm(arrange):
    list_of_coeffs_asimm = np.zeros(1000)
    for i in range(1, 1001, 1):
        sample = np.random.choice(arrange, size=25, replace=True)
        coeff_asimm = stats.skew(sample)
        list_of_coeffs_asimm[i-1] = coeff_asimm
    
    return list_of_coeffs_asimm

#Генерация выборки и создание вариационного ряда
sample = np.random.standard_exponential(25)
sample_sorted = np.sort(sample)

#Рассчет априорных величин
sample_aprior = np.arange(0, 5, 0.01)
func_distrib = 1 - np.exp(-1*sample_aprior)
y_density = np.exp(-1*sample_aprior)

#Расчет для нормального распределения плотности
range_dist = np.arange(0, 2, 0.01)
normal_dist = stats.norm.pdf(range_dist, loc=1, scale=0.2)

#Размах выборки
sample_size = sample_sorted[24] - sample_sorted[0]

#Медиана
median = np.median(sample_sorted)

#Мода(не имеет смысла)
moda = stats.mode(sample_sorted).mode 

#Коэффицент асимметрии
asimm_coeff = stats.skew(sample_sorted)

#Бутстраповская оценка среднего арифметического
bootstrap_estimation_ar_mean = bootstrap_ar_mean(sample_sorted)

#Бутстраповская оценка коэффицента асимметрии
bootstrap_estimation_coeff_asimm = bootstrap_coeff_asimm(sample_sorted)

#Бутстраповская оценка плотности медианы
bootstrap_estimation_median = bootstrap_median(sample_sorted)

#Получение оценки плотности медианы
arrange = np.arange(0, 2, 0.01)
median_estimation_distribution = median_density(arrange)

#Оценка вероятности коэффицента асимметрии
num_of_appropriate_coeff_asimm = 0
for i in range(1, 1001, 1):
    if(bootstrap_estimation_coeff_asimm[i - 1] < 1):
        num_of_appropriate_coeff_asimm += 1

estimation_coeff_asimm_prob = num_of_appropriate_coeff_asimm/1000.

#Построение графиков
fig, axs = plt.subplots(2, 3, num='Графики', figsize=(15,5), sharex=True)

bp = axs[0, 0].boxplot(sample_sorted, orientation='horizontal', autorange=True)
hist = axs[0, 1].hist(sample_sorted, bins=6, color='skyblue', density=True, edgecolor='red', alpha=0.7, label='Histogram')
density = axs[0, 1].plot(sample_aprior, y_density, color='red', label='Density')
ecdf = axs[1, 0].ecdf(sample_sorted, label='Estimation of distribution function')
distrib_func = axs[1, 0].plot(sample_aprior, func_distrib, color='red', label='Distribution function')
bootstrap_hist_ar_mean = axs[1, 1].hist(bootstrap_estimation_ar_mean, bins=10, color='green', density=True, edgecolor='purple', alpha=0.7, label='Bootstrap arithmetic mean')
normal_distrib = axs[1, 1].plot(range_dist, normal_dist, color='red', label='Normal distribution')
bootstrap_hist_coeff_asimm = axs[0, 2].hist(bootstrap_estimation_coeff_asimm, bins=10, color='yellow', density=True, edgecolor='green', alpha=0.7, label='Bootstrap coefficient of asymmetry')
bootstrap_hist_median = axs[1, 2].hist(bootstrap_estimation_median, bins=10, color='pink', density=True, edgecolor='purple', alpha=0.7, label='Bootstrap median')
median_density_estimation = axs[1, 2].plot(arrange, median_estimation_distribution, color='red', label='Density estimation')

axs[0, 0].set_title("Boxplot")
axs[0, 1].set_title("Histogram")
axs[1, 0].set_title("Empirical distribution function")
axs[1, 1].set_title("Bootstrap estimation of arithmetic mean")
axs[0, 0].set_xlabel('Значения')
axs[0, 1].set_xlabel('Значения')
axs[0, 1].set_ylabel('Плотность распределения')
axs[1, 0].set_xlabel('Значения')
axs[1, 0].set_ylabel('Плотность распределения')
axs[1, 1].set_ylabel('Плотность распределения')
axs[1, 1].set_xlabel('Значения')
axs[0, 2].set_ylabel('Плотность распределения')
axs[0, 2].set_xlabel('Значения')
axs[0, 2].set_title('Bootstrap estimation of coefficient of assymetry')
axs[1, 2].set_ylabel('Плотность распределения')
axs[1, 2].set_xlabel('Значения')
axs[1, 2].set_title('Bootstrap estimation of median')

axs[0, 1].legend(fontsize='small')
axs[1, 0].legend(fontsize='small')
axs[1, 1].legend(fontsize='small')
axs[0, 2].legend(fontsize='small')
axs[1, 2].legend(fontsize='small')

elements = (bp['boxes'] + bp['medians'] + bp['whiskers'] + 
            bp['caps'] + bp['fliers'] + bp['means'])
cursor = mplcursors.cursor(elements, hover=True)

print("Мода: не имеет смысл на вещественной выборке")
print("Медиана:", median)
print("Размах:", sample_size)
print("Оценка коэффицента асимметрии:", asimm_coeff)
print("Оценка вероятности бутстраповского коэффицента асимметрии:", estimation_coeff_asimm_prob)

fig.tight_layout()
plt.savefig("task2Plots.png")
