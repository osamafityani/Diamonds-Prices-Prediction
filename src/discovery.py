import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


if __name__ == "__main__":

    # to show all columns of the data frame in the Run window
    pd.set_option('display.width', 320, 'display.max_columns', 12)

    # Data Discovery and Visualization
    diamonds = pd.read_csv('D:/PRIEMERE/AI Club/New/Diamonds/diamonds.csv')
    print(diamonds.head(), '\n\n')
    print(diamonds.info(), '\n\n')
    print(diamonds.describe(), '\n\n')




    # Remove the first column since it's just indexing and it may confuse the ML algorithm
    diamonds.drop('Unnamed: 0', axis=1, inplace=True)
    # checking for unknown numerical values (zeros)
    print(diamonds[(diamonds['x'] == 0) | (diamonds['y'] == 0) | (diamonds['z'] == 0)], '\n\n')
    # Remove all examples where any of the dimensions is 0.
    diamonds.drop(diamonds[(diamonds['x'] == 0) | (diamonds['y'] == 0) | (diamonds['z'] == 0)].index, axis=0, inplace=True)

    # Exploring categorical features
    print(diamonds['cut'].value_counts(), '\n\n')
    print(diamonds['color'].value_counts(), '\n\n')
    print(diamonds['clarity'].value_counts(), '\n\n')

    print(diamonds.columns)

    sns.set_style('whitegrid')

    # a histogram for all numerical attributes
    diamonds.hist(bins=50, figsize=(14, 7))
    plt.show()

    # Looking for correlations
    corr_matrix = diamonds.corr()
    print(corr_matrix['price'].sort_values(ascending=False))
    sns.heatmap(corr_matrix, annot=True)
    plt.show()

    sns.jointplot('carat', 'price', diamonds, kind='reg')
    sns.jointplot('x', 'price', diamonds, kind='reg')
    sns.jointplot('y', 'price', diamonds, kind='reg')
    sns.jointplot('z', 'price', diamonds, kind='reg')
    sns.jointplot('depth', 'price', diamonds, kind='reg')
    sns.jointplot('table', 'price', diamonds, kind='reg')
    plt.show()

    # Checking outliers and remove examples where outliers appear
    print('\n', diamonds[diamonds['y'] > 30], '\n')
    print(diamonds[diamonds['z'] > 30], '\n')
    diamonds.drop(diamonds[(diamonds['z'] > 30) | (diamonds['y'] > 30)].index, axis=0, inplace=True)

    corr_matrix = diamonds.corr()
    print(corr_matrix['price'].sort_values(ascending=False))

    sns.jointplot('y', 'price', diamonds, kind='reg')
    sns.jointplot('z', 'price', diamonds, kind='reg')
    plt.show()

    # introduce a new attribute 'dim_comb' which's a combination of the dimensions x, y, z and
    # is more correlated with the final price
    diamonds['dim_comb'] = diamonds['x'] * diamonds['y'] * diamonds['z']

    corr_matrix = diamonds.corr()
    print(corr_matrix['price'].sort_values(ascending=False))

    # Remove the separated dimensions.
    diamonds.drop(['x', 'y', 'z'], axis=1, inplace=True)

    # Visualizing correlations for one last time
    sns.heatmap(corr_matrix, annot=True)
    plt.show()

    print(diamonds.head())

    # Exploring the relations between the categorical attributes and the price
    sns.barplot(x='cut', y='price', data=diamonds, order=['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'])
    plt.show()
    sns.barplot(x='color', y='price', data=diamonds, order=['J', 'I', 'H', 'G', 'F', 'E', 'D'])
    plt.show()
    sns.barplot(x='clarity', y='price', data=diamonds, order=['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    plt.show()

    # Encoding categorical attributes
    '''
    1- For the cut attribute, we realised that it is not really a categorical attribute. It represents the quality of
       the cut, so we decided to transform its values to [1, 2, 3, 4, ...] instead of a matrix of zeros and ones. So the 
       algorithm won't treat them like absolute categories. 
       '''
    encoder = LabelEncoder()
    encodedCut = encoder.fit_transform(diamonds['cut'].sort_values())
    encodedCutSeries = pd.Series(encodedCut, name='cutNew')

    '''
    2- The other attributes can be treated like absolute categories so we will just encode them to zeros and ones
       '''
    color = pd.get_dummies(diamonds['color'], drop_first=True)
    cut = pd.get_dummies(diamonds['cut'], drop_first=True)
    clarity = pd.get_dummies(diamonds['clarity'], drop_first=True)
    diamonds = pd.concat([diamonds.reset_index(drop=True),
                          color.reset_index(drop=True),
                          encodedCutSeries.reset_index(drop=True),
                          clarity.reset_index(drop=True)], axis=1)
    diamonds.drop(['color', 'cut', 'clarity'], axis=1, inplace=True)
else:
    def preProcessing():
        diamonds = pd.read_csv('D:/PRIEMERE/AI Club/New/Diamonds/diamonds.csv')
        # Remove the first column since it's just indexing and it may confuse the ML algorithm
        diamonds.drop('Unnamed: 0', axis=1, inplace=True)

        # Remove all examples where any of the dimensions is 0.
        diamonds.drop(diamonds[(diamonds['x'] == 0) | (diamonds['y'] == 0) | (diamonds['z'] == 0)].index, axis=0,
                      inplace=True)

        # Checking outliers and remove examples where outliers appear
        diamonds.drop(diamonds[(diamonds['z'] > 30) | (diamonds['y'] > 30)].index, axis=0, inplace=True)

        # introduce a new attribute 'dim_comb' which's a combination of the dimensions x, y, z and
        # is more correlated with the final price
        diamonds['dim_comb'] = diamonds['x'] * diamonds['y'] * diamonds['z']

        # Remove the separated dimensions.
        diamonds.drop(['x', 'y', 'z'], axis=1, inplace=True)

        # Encoding categorical attributes
        '''
        1- For the cut attribute, we realised that it is not really a categorical attribute. It represents the quality of
           the cut, so we decided to transform its values to [1, 2, 3, 4, ...] instead of a matrix of zeros and ones. So the 
           algorithm won't treat them like absolute categories. 
           '''
        encoder = LabelEncoder()
        encodedCut = encoder.fit_transform(diamonds['cut'].sort_values())
        encodedCutSeries = pd.Series(encodedCut, name='cutNew')

        diamonds['color'] = diamonds['color'].where(lambda c: c != 'J', 0)
        diamonds['color'] = diamonds['color'].where(lambda c: c != 'I', 1)
        diamonds['color'] = diamonds['color'].where(lambda c: c != 'H', 2)
        diamonds['color'] = diamonds['color'].where(lambda c: c != 'G', 3)
        diamonds['color'] = diamonds['color'].where(lambda c: c != 'F', 4)
        diamonds['color'] = diamonds['color'].where(lambda c: c != 'E', 5)
        diamonds['color'] = diamonds['color'].where(lambda c: c != 'D', 6)


        diamonds['clarity'] = diamonds['clarity'].where(lambda c: c != 'I1', 0)
        diamonds['clarity'] = diamonds['clarity'].where(lambda c: c != 'SI2', 1)
        diamonds['clarity'] = diamonds['clarity'].where(lambda c: c != 'SI1', 2)
        diamonds['clarity'] = diamonds['clarity'].where(lambda c: c != 'VS2', 3)
        diamonds['clarity'] = diamonds['clarity'].where(lambda c: c != 'VS1', 4)
        diamonds['clarity'] = diamonds['clarity'].where(lambda c: c != 'VVS2', 5)
        diamonds['clarity'] = diamonds['clarity'].where(lambda c: c != 'VVS1', 6)
        diamonds['clarity'] = diamonds['clarity'].where(lambda c: c != 'IF', 7)


        '''
        2- The other attributes can be treated like absolute categories so we will just encode them to zeros and ones
           '''

        diamonds = pd.concat([diamonds.reset_index(drop=True),
                              encodedCutSeries.reset_index(drop=True)], axis=1)
        diamonds.drop([ 'cut'], axis=1, inplace=True)

        diamonds['price_per_carat'] = diamonds['price'] / diamonds['carat']
        # diamonds.drop('dim_comb', axis=1, inplace=True)
        return diamonds
