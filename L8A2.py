import pandas as pd
import numpy as np
from scipy.stats import norm

def map_age_to_numeric(age_str):
    if '-' in age_str:
        return int(age_str.split('-')[0])
    elif '>' in age_str:
        return int(age_str.replace('>', '')) + 1  # Add 1 to represent an age greater than the upper limit
    elif '<=' in age_str:
        return int(age_str.replace('<=', ''))
    else:
        return int(age_str)

#Load the table data
table_data = {
    'age': ['<=30', '<=30', '31-40', '>40', '>40', '>40', '31-40', '<=30', '<=30', '>40', '<=30', '31-40', '31-40', '>40'],
    'income': ['high', 'high', 'high', 'medium', 'low', 'low', 'low', 'medium', 'low', 'medium', 'medium', 'medium', 'high', 'medium'],
    'student': ['no', 'no', 'no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no'],
    'credit_rating': ['fair', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'fair', 'fair', 'excellent', 'excellent', 'fair', 'excellent'],
    'buys_computer': ['no', 'no', 'yes', 'yes', 'yes', 'no', 'yes', 'no', 'yes', 'yes', 'yes', 'yes', 'yes', 'no']
}

df = pd.DataFrame(table_data)

# Calculate class conditional densities for age (assuming a normal distribution)
for cls in df['buys_computer'].unique():
    for age_range in df['age'].unique():
        subset = df[df['buys_computer'] == cls]
        age_values = subset['age'].apply(map_age_to_numeric)
        mean_age = age_values.mean()
        std_dev_age = age_values.std()
        numeric_age = map_age_to_numeric(age_range)
        density = norm.pdf(numeric_age, mean_age, std_dev_age)
        print(f'P(age={age_range} | buys_computer={cls}) = {density}')

# Calculate class conditional densities for categorical features
categorical_features = ['income', 'student', 'credit_rating']
for cls in df['buys_computer'].unique():
    for feature in categorical_features:
        crosstab = pd.crosstab(df[feature], df['buys_computer'], margins=True, normalize=True)
        conditional_density = crosstab[cls] / crosstab['All']
        print(f'P({feature} | buys_computer={cls}) = {conditional_density}')
