
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris


def load_and_explore_data():
    try:
        
        iris = load_iris()
        
        df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
        
        df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

       
        print("First 5 rows of the dataset:")
        print(df.head())

       
        print("\nDataset Information:")
        print(df.info())

       
        print("\nMissing Values:")
        print(df.isnull().sum())

        
        df.dropna(inplace=True)
        return df

    except FileNotFoundError:
        print("Error: Dataset file not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def basic_data_analysis(df):
    
    print("\nBasic Statistics of Numerical Columns:")
    print(df.describe())

    
    print("\nMean of Numerical Columns Grouped by Species:")
    grouped_means = df.groupby('species').mean()
    print(grouped_means)

    
    print("\nObservations:")
    print("1. Setosa species have the smallest measurements on average.")
    print("2. Virginica species tend to have the largest measurements on average.")
    print("3. Sepal length and petal length seem to have significant differences across species.")


def data_visualization(df):
    sns.set(style="whitegrid")

    
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['sepal length (cm)'], color='blue', label='Sepal Length (cm)')
    plt.title("Trend of Sepal Length Over Index")
    plt.xlabel("Index")
    plt.ylabel("Sepal Length (cm)")
    plt.legend()
    plt.show()

    
    plt.figure(figsize=(8, 5))
    sns.barplot(x='species', y='petal length (cm)', data=df)
    plt.title("Average Petal Length per Species")
    plt.xlabel("Species")
    plt.ylabel("Petal Length (cm)")
    plt.show()

    
    plt.figure(figsize=(8, 5))
    plt.hist(df['sepal width (cm)'], bins=15, color='green', edgecolor='black')
    plt.title("Distribution of Sepal Width")
    plt.xlabel("Sepal Width (cm)")
    plt.ylabel("Frequency")
    plt.show()

    
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=df)
    plt.title("Sepal Length vs Petal Length by Species")
    plt.xlabel("Sepal Length (cm)")
    plt.ylabel("Petal Length (cm)")
    plt.legend(title='Species')
    plt.show()


def main():
    
    df = load_and_explore_data()

    
    basic_data_analysis(df)

    
    data_visualization(df)

if __name__ == "__main__":
    main()
