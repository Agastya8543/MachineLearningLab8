import pandas as pd
from scipy.stats import chi2_contingency

#Load the table data
table_data = {
    'age': ['<=30', '<=30', '31-40', '>40', '>40', '>40', '31-40', '<=30', '<=30', '>40', '<=30', '31-40', '31-40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

df = pd.DataFrame(table_data)

# Create a contingency table
contingency_table = pd.crosstab(index=df['age'], columns=[df['income'], df['student'], df['credit_rating']], margins=True)

# Perform the chi-squared test for independence
chi2, p, _, _ = chi2_contingency(contingency_table)

# Print the results
print(f"Chi-squared value: {chi2}")
print(f"P-value: {p}")

# Check if the null hypothesis (independence) is rejected
alpha = 0.05
if p < alpha:
    print("The features are not independent (reject the null hypothesis)")
else:
    print("The features are independent (fail to reject the null hypothesis)")
