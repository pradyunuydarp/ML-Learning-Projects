import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("train.csv")

# Columns in the dataset
columns = data.columns
target = 'Default'

# Split columns into numerical and categorical
numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns.drop([target])
categorical_columns = data.select_dtypes(include=['object']).columns

# Set up the plotting style
sns.set(style="whitegrid")

# Plot numerical features vs target
for col in numerical_columns:
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=data, x=target, y=col)
    plt.title(f"{col} vs {target}")
    plt.show()

# Plot categorical features vs target
for col in categorical_columns:
    plt.figure(figsize=(10, 5))
    
    # Limit to top 10 categories
    top_categories = data[col].value_counts().head(10).index
    filtered_data = data[data[col].isin(top_categories)]
    
    # Count plot with the 'Default' as hue
    sns.countplot(data=filtered_data, x=col, hue=target, order=top_categories)
    
    plt.title(f"{col} vs {target}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)
    plt.legend(title=target, loc='upper right')
    plt.show()