import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/healthcare-dataset-stroke-data.csv')

print(df.info())

plt.figure(figsize=(6, 4))
sns.countplot(x='stroke', data=df)
plt.title('Distribution of Stroke (0: No, 1: Yes)')
plt.savefig('eda_class_distribution.png')
plt.show()

plt.figure(figsize=(10, 8))
numerical_df = df.select_dtypes(include=['float64', 'int64'])
sns.heatmap(numerical_df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.savefig('eda_correlation.png')
plt.show()

sns.pairplot(df, hue='stroke', vars=['age', 'avg_glucose_level', 'bmi'])
plt.savefig('eda_pairplot.png')
plt.show()
