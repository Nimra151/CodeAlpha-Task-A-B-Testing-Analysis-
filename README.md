# CodeAlpha-Task-A-B-Testing-Analysis-
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest

# Load the dataset
file_path = 'marketing_AB.csv'
data = pd.read_csv(file_path)

# Display basic information and first few rows of the dataset
print(data.info())
print(data.head())

# Drop unnecessary columns (if any)
data_cleaned = data.drop(columns=['Unnamed: 0'])  # Remove any index column if present

# Check for any missing values
print(data_cleaned.isnull().sum())
# Calculate the conversion rates for each test group
conversion_rates = data_cleaned.groupby('test group')['converted'].mean() * 100
print("Conversion Rates by Test Group:")
print(conversion_rates)
# Define a function to perform the two-proportion z-test
def ab_test(conversions, observations):
    stat, p_value = proportions_ztest(conversions, observations)
    return stat, p_value

# Get the number of conversions and total observations for each group
conversions = data_cleaned.groupby('test group')['converted'].sum()
observations = data_cleaned['test group'].value_counts()

# Perform the z-test
stat, p_value = ab_test(conversions, observations)
print(f"Z-statistic: {stat}")
print(f"P-value: {p_value}")
# Visualize the conversion rates
plt.figure(figsize=(8, 6))
sns.barplot(x=conversion_rates.index, y=conversion_rates.values, palette="viridis")
plt.title('Conversion Rates by Test Group')
plt.xlabel('Test Group')
plt.ylabel('Conversion Rate (%)')
plt.show()
# Based on the z-test results and conversion rates, interpret the impact of the intervention
if p_value < 0.05:
    print("The difference in conversion rates is statistically significant.")
else:
    print("The difference in conversion rates is not statistically significant.")
    
print("Ad group conversion rate is higher than the PSA group, suggesting a positive impact from the intervention.")
import scipy.stats as stats
import numpy as np

def confidence_interval(data, confidence=0.95):
    """
    Calculate confidence intervals for the conversion rate.
    """
    mean = np.mean(data)
    n = len(data)
    stderr = stats.sem(data)
    margin_of_error = stderr * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean - margin_of_error, mean + margin_of_error

# Apply the function to each group
ci_ad = confidence_interval(data_cleaned[data_cleaned['test group'] == 'ad']['converted'])
ci_psa = confidence_interval(data_cleaned[data_cleaned['test group'] == 'psa']['converted'])

print(f"Confidence Interval for Ad group: {ci_ad}")
print(f"Confidence Interval for PSA group: {ci_psa}")
def cohen_h(p1, p2):
    """
    Calculate Cohen's h for effect size between two proportions.
    """
    return 2 * np.arcsin(np.sqrt(p1)) - 2 * np.arcsin(np.sqrt(p2))

# Calculate effect size
p1 = conversion_rates['ad'] / 100  # Conversion rate for Ad group
p2 = conversion_rates['psa'] / 100  # Conversion rate for PSA group
effect_size = cohen_h(p1, p2)

print(f"Cohen's h effect size: {effect_size}")
from statsmodels.stats.power import zt_ind_solve_power

# Conduct a power analysis
alpha = 0.05  # Significance level
power = 0.8  # Desired power

# Calculate the required sample size
required_sample_size = zt_ind_solve_power(effect_size=effect_size, alpha=alpha, power=power, ratio=(observations['psa'] / observations['ad']))

print(f"Required sample size per group: {required_sample_size}")
# Re-run the z-test with a different significance level (e.g., alpha = 0.01)
alpha_new = 0.01

stat_new, p_value_new = proportions_ztest(conversions, observations)
print(f"Z-statistic with alpha = 0.01: {stat_new}")
print(f"P-value with alpha = 0.01: {p_value_new}")
print("\n--- Summary of Findings ---")

print(f"Conversion rate (Ad group): {conversion_rates['ad']}%")
print(f"Conversion rate (PSA group): {conversion_rates['psa']}%")
print(f"Cohen's h effect size: {effect_size}")
print(f"Required sample size per group: {required_sample_size}")

if p_value < 0.05:
    print("The difference in conversion rates is statistically significant at alpha = 0.05.")
else:
    print("The difference in conversion rates is not statistically significant at alpha = 0.05.")
    
print("Based on the analysis, the Ad group shows a higher conversion rate, indicating that the intervention likely had a positive effect.")
