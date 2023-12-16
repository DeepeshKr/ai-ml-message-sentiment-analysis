# libraries for data manipulation
import pandas as pd
import numpy as np

# libraries for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# importing LLM models and tokenizers
from transformers import T5Tokenizer, T5ForConditionalGeneration

# importing metric functions
from sklearn.metrics import confusion_matrix, accuracy_score

# importing library to split the data
from sklearn.model_selection import train_test_split

# library for model deployment
import gradio as gr

from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large", load_in_8bit=True, device_map="auto")


data = None
def read_data():
    ''' read data file'''
    df = pd.read_csv("US_Airways.csv")

    output = df.sample(10, random_state=1)
    print(output)
    
    # creating a copy of the data to avoid any changes to original data
    data = df.copy()

    # checking the statistical summary of the data
    print(data.describe().T)
    return data


def labeled_barplot(data, feature, perc=False, n=None):
    """
    Barplot with percentage at the top

    data: dataframe
    feature: dataframe column
    perc: whether to display percentages instead of count (default is False)
    n: displays the top n category levels (default is None, i.e., display all levels)
    """

    total = len(data[feature])  # length of the column
    count = data[feature].nunique()
    if n is None:
        plt.figure(figsize=(count + 1, 5))
    else:
        plt.figure(figsize=(n + 1, 5))

    plt.xticks(rotation=90, fontsize=15)
    ax = sns.countplot(
        data=data,
        x=feature,
        hue=feature,  # Assign x variable to hue
        palette="Paired",
        order=data[feature].value_counts().index[:n].sort_values(),
        legend=False  # Set legend to False
    )


    for p in ax.patches:
        if perc == True:
            label = "{:.1f}%".format(
                100 * p.get_height() / total
            )  # percentage of each class of the category
        else:
            label = p.get_height()  # count of each level of the category

        x = p.get_x() + p.get_width() / 2  # width of the plot
        y = p.get_height()  # height of the plot

        ax.annotate(
            label,
            (x, y),
            ha="center",
            va="center",
            size=12,
            xytext=(0, 5),
            textcoords="offset points",
        )  # annotate the percentage

    plt.show()  # show the plot


def show_graphs(data):
    """#### Distribution of sentiments across all the tweets"""

    labeled_barplot(data, "airline_sentiment", perc=True)

    """- **Majority of the tweets are negative (78%), followed by neutral tweets (13%), and then the positive tweets (9%).**

    #### Plot of all the negative reasons
    """

    labeled_barplot(data, "negativereason", perc=True)

    """**The predominant causes of negative tweets appear to be customer service issues(28%) and late flights(16%), as indicated by the graph**"""

    sns.histplot(data, x='retweet_count', bins =50)

    """- A majority of the customer tweets are not re-tweeted
    - The lower number of retweet counts suggests that customers are less inclined to retweet
    """

    sns.barplot(data, y='retweet_count', x='airline_sentiment', errorbar=('ci', False))

def model_training(data):
    """## EDA Results
    `airline sentiment`
    ## **Model Training and Evalution**
    Training an AI model is important because it allows machines to learn and perform tasks without explicit programming. It enables the following:

    - **Learning from Data**
    - **Generalization and Adaptability**
    - **Optimization and Performance Improvement**

    We'll be using a pre-trained large language model (LLM) here. So, we don't need to train the model. We'll directly import the pre-trained model and then use it for predictions.

    - We'll use the Google FLAN-T5 model (large variant) for illustration

    ## Data Preprocessing
    """

    # Specify the features (X) and the target variable (y)
    X = data.drop('airline_sentiment', axis=1)  # Replace 'target_variable' with the actual name of your target column
    y = data['airline_sentiment']

    # Further split the temporary data into validation and test sets
    X_validation, X_test, y_validation, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    text_column_validation = X_validation['text'].copy()
    actual_sentiment_validation = y_validation.copy()

    text_column_test = X_test['text'].copy()
    actual_sentiment_test = y_test.copy()


def main():
    data = read_data()
    show_graphs(data)


if __name__ == "__main__":
    main()
