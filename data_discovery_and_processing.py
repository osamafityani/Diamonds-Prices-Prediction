import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    # to show all columns of the data frame in the Run window
    pd.set_option('display.width', 320, 'display.max_columns', 12)

    # Data Discovery and Visualization
    diamonds = pd.read_csv('diamonds.csv')
    print(diamonds.head(), '\n\n')
    print(diamonds.info(), '\n\n')
    print(diamonds.describe(), '\n\n')
    # Remove the first column since it's just indexing and it may confuse the ML algorithm
    diamonds.drop('Unnamed: 0', axis=1, inplace=True)
    # checking for unknown numerical values (zeros)
    print(diamonds[(diamonds['x'] == 0) | (diamonds['y'] == 0) | (diamonds['z'] == 0)], '\n\n')
    # Remove all examples where any of the dimensions is 0.
    diamonds.drop(diamonds[(diamonds['x'] == 0) | (diamonds['y'] == 0) | (diamonds['z'] == 0)].index,
                  axis=0, inplace=True)

    # Exploring categorical features
    print(diamonds['cut'].value_counts(), '\n\n')
    print(diamonds['color'].value_counts(), '\n\n')
    print(diamonds['clarity'].value_counts(), '\n\n')

    print(diamonds.columns)

    sns.set_style('whitegrid')

    # a histogram for all numerical attributes
    diamonds.hist(bins=50, figsize=(14, 7))
    plt.title('Numerical Attributes plot')
    plt.show()

    # Looking for correlations
    corr_matrix = diamonds.corr()
    print(corr_matrix['price'].sort_values(ascending=False))
    sns.heatmap(corr_matrix, annot=True, cmap='summer')
    plt.title('Correlations plot')
    plt.show()

    sns.set_palette("summer")
    sns.jointplot('carat', 'price', diamonds, kind='reg', ratio=10, height=8)
    sns.jointplot('x', 'price', diamonds, kind='reg', ratio=10, height=8)
    sns.jointplot('y', 'price', diamonds, kind='reg', ratio=10, height=8)
    sns.jointplot('z', 'price', diamonds, kind='reg', ratio=10, height=8)
    sns.jointplot('depth', 'price', diamonds, kind='reg', ratio=10, height=8)
    sns.jointplot('table', 'price', diamonds, kind='reg', ratio=10, height=8)
    plt.show()

    # Checking outliers and remove examples where outliers appear
    print('\n', diamonds[diamonds['y'] > 30], '\n')
    print(diamonds[diamonds['z'] > 30], '\n')
    diamonds.drop(diamonds[(diamonds['z'] > 30) | (diamonds['y'] > 30)].index, axis=0, inplace=True)

    corr_matrix = diamonds.corr()
    print(corr_matrix['price'].sort_values(ascending=False))

    sns.set_palette("magma")
    sns.jointplot('y', 'price', diamonds, kind='reg', ratio=10, height=8)
    plt.xlabel('Y')
    plt.ylabel('Price')
    sns.jointplot('z', 'price', diamonds, kind='reg', ratio=10, height=8)
    plt.xlabel('Z')
    plt.ylabel('Price')
    plt.show()

    # introduce a new attribute 'dim_comb' which's a combination of the dimensions x, y, z and
    # is more correlated with the final price
    diamonds['dim_comb'] = diamonds['x'] * diamonds['y'] * diamonds['z']

    corr_matrix = diamonds.corr()
    print(corr_matrix['price'].sort_values(ascending=False))

    # Remove the separated dimensions.
    diamonds.drop(['x', 'y', 'z'], axis=1, inplace=True)

    # Visualizing correlations for one last time
    sns.heatmap(corr_matrix, annot=True, cmap='summer')
    plt.title('Correlations plot for last time')
    plt.show()

    print(diamonds.head())

    # Exploring the relations between the categorical attributes and the price
    sns.boxplot(x='cut', y='price', data=diamonds, order=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
    plt.xlabel('Cut')
    plt.ylabel('Price')
    plt.show()
    sns.boxplot(x='color', y='price', data=diamonds, order=['J', 'I', 'H', 'G', 'F', 'E', 'D'])
    plt.xlabel('Color')
    plt.ylabel('Price')
    plt.show()
    sns.boxplot(x='clarity', y='price', data=diamonds, order=['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    plt.xlabel('Clarity')
    plt.ylabel('Price')
    plt.show()

    # Encoding categorical attributes
    diamonds['cut_encoded'] = diamonds['cut'].map({'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4})
    diamonds.drop(['cut'], axis=1, inplace=True)
    diamonds['color'] = diamonds['color'].map({'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6})
    diamonds['clarity'] = diamonds['clarity'].map({'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4,
                                                   'VVS2': 5, 'VVS1': 6, 'IF': 7})
    print(diamonds.head())

else:
    # defining function preProcessing to call it from the modeling file
    def pre_processing():
        data = pd.read_csv('diamonds.csv')
        data.drop('Unnamed: 0', axis=1, inplace=True)
        data.drop(data[(data['x'] == 0) | (data['y'] == 0) | (data['z'] == 0)].index,
                  axis=0, inplace=True)
        data.drop(data[(data['z'] > 30) | (data['y'] > 30)].index, axis=0, inplace=True)
        data['dim_comb'] = data['x'] * data['y'] * data['z']
        data.drop(['x', 'y', 'z'], axis=1, inplace=True)
        data['cut_encoded'] = data['cut'].map({'Fair': 0, 'Good': 1, 'Very Good': 2, 'Premium': 3, 'Ideal': 4})
        data.drop(['cut'], axis=1, inplace=True)
        data['color'] = data['color'].map({'J': 0, 'I': 1, 'H': 2, 'G': 3, 'F': 4, 'E': 5, 'D': 6})
        data['clarity'] = data['clarity'].map({'I1': 0, 'SI2': 1, 'SI1': 2, 'VS2': 3, 'VS1': 4,
                                               'VVS2': 5, 'VVS1': 6, 'IF': 7})
        data['price_per_carat'] = data['price'] / data['carat']
        return data
