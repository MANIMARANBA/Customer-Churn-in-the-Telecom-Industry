import pandas as pd 

dataset_TSAP_path = r"D:\BA_Project_Portfolio\Telecom_Industry_Churn/Telecom_customer churn.csv"
dataset_TSAP = pd.read_csv(dataset_TSAP_path)
# Step 1: Examine the structure of the dataset
print(dataset_TSAP.head())
print(dataset_TSAP.info())
print(dataset_TSAP.describe())

# Step 2: Check for missing values and handle them
print(dataset_TSAP.isnull().sum())
dataset_TSAP.fillna(dataset_TSAP.mean(), inplace=True)

# Step 3: Identify and handle outliers
import seaborn as sns
sns.boxplot(data=dataset_TSAP)  # Visualize box plots for all numerical columns to identify outliers
# Handle outliers based on specific columns using appropriate techniques

# Step 4: Format the data if needed
# Example: Convert categorical variables to appropriate data types
dataset_TSAP['prizm_social_one'] = dataset_TSAP['prizm_social_one'].astype('category')


# Additional preprocessing steps based on column descriptions:
# Example: Apply log transformation to 'rev_Mean' and 'totrev'
import numpy as np
dataset_TSAP['rev_Mean'] = np.log(dataset_TSAP['rev_Mean'])
dataset_TSAP['totrev'] = np.log(dataset_TSAP['totrev'])


# Example: Handling missing values in specific columns
dataset_TSAP['crclscod'].fillna('unknown', inplace=True)

# Example: Binning a numerical variable into categories
bins = [0, 100, 200, np.inf]
labels = ['low', 'medium', 'high']
dataset_TSAP['totmou_bin'] = pd.cut(dataset_TSAP['totmou'], bins=bins, labels=labels)


import pandas as pd
import seaborn as sns

# Step 1: Perform descriptive statistics
descriptive_stats = dataset_TSAP.describe()
print(descriptive_stats)

# Step 2: Explore relationships between variables
correlation_matrix = dataset_TSAP.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')

# Step 3: Identify trends, patterns, and significant findings
# Example: Calculate churn rate
churn_rate = dataset_TSAP['churn'].mean() * 100
print(f"Churn Rate: {churn_rate:.2f}%")

# Example: Compare average revenue for churned and non-churned customers
avg_rev_churned = dataset_TSAP.loc[dataset_TSAP['churn'] == 1, 'rev_Mean'].mean()
avg_rev_non_churned = dataset_TSAP.loc[dataset_TSAP['churn'] == 0, 'rev_Mean'].mean()
print(f"Average Revenue (Churned): {avg_rev_churned:.2f}")
print(f"Average Revenue (Non-Churned): {avg_rev_non_churned:.2f}")

# Example: Plot distribution of a variable
sns.histplot(data=dataset_TSAP, x='mou_Mean', kde=True)

# Additional analysis and insights can be derived based on the dataset and specific objectives.

import matplotlib.pyplot as plt

# Example code for visualization 1: Bar plot of churn rate by customer category
customer_categories = ['Category A', 'Category B', 'Category C']
churn_rates = [0.25, 0.32, 0.15]

plt.bar(customer_categories, churn_rates)
plt.xlabel('Customer Category')
plt.ylabel('Churn Rate')
plt.title('Churn Rate by Customer Category')
plt.show()

# Example code for visualization 2: Line chart of churn over time
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
churn_counts = [100, 120, 90, 110, 80, 130]

plt.plot(months, churn_counts, marker='o')
plt.xlabel('Months')
plt.ylabel('Churn Count')
plt.title('Churn Trend Over Time')
plt.show()

# Example code for visualization 3: Histogram of customer tenure
tenure = [2, 5, 7, 4, 1, 3, 6, 2, 9, 8, 5, 4, 3, 7, 2, 1, 5, 3, 6, 4]

plt.hist(tenure, bins=range(1, 11))
plt.xlabel('Tenure (in years)')
plt.ylabel('Frequency')
plt.title('Distribution of Customer Tenure')
plt.show()
