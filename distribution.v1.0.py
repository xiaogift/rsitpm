#!/usr/bin/env python3
# ============================================================================== #
# Distribution
# Powered by xiaolis@outlook.com 202307
# ============================================================================== #
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import shapiro
from pandas import read_csv

employee = read_csv( './data/Final_Employees_Data.csv', 
                     sep = ',', header = None,
                     names = [ 'Eid','Ename','Experience','Total_projects','Rating','Area_of_Interest_1',\
                               'Area_of_Interest_2','Area_of_Interest_3','Language1','Language2','Language3',\
                               'AI_project_count','ML_project_count','JS_project_count','Java_project_count',\
                               'DotNet_project_count','Mobile_project_count'] ).drop(0)
column_to_test = employee['Rating'].values.astype(float)


skewness = employee['Rating'].skew()
mean_rating = np.mean(column_to_test)
median_rating = np.median(column_to_test)
percentile_25 = np.percentile(column_to_test, 25)
percentile_75 = np.percentile(column_to_test, 75)

print(f"Skewness: {skewness}")
print(f"Mean Rating: {mean_rating}")
print(f"Median Rating: {median_rating}")
print(f"25th Percentile Rating: {percentile_25}")
print(f"75th Percentile Rating: {percentile_75}")

alpha = 0.05
plt.figure(figsize=(8, 6)) 
plt.hist(column_to_test, bins=50, density=True, color='skyblue', edgecolor='black', alpha=0.7)
mu, sigma = np.mean(column_to_test), np.std(column_to_test)
x = np.linspace(min(column_to_test), max(column_to_test), 100)
y = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
plt.axvline(2.5, color='skyblue', linestyle='dashed', linewidth=2, label='Median')
plt.plot(x, y, color='red', linewidth=2)
plt.title('Employee Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Density')
plt.grid(True)
plt.savefig('rating_distribution.png', dpi=1000)
plt.show()

statistic, p_value = shapiro(column_to_test)
print("Shapiro-Wilk statistic：", statistic)
print("p value：", p_value)
if p_value > alpha: print("Normal Distribution")
else: print("Not Normal Distribution")


# ============================================================================== #
