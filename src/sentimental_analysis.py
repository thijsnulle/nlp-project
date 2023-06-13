import numpy as np
import nltk
nltk.download('vader_lexicon')
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def sentiment_analysis_word(word: str) -> float:
    """
    This function takes a word as input and returns the sentiment of the word. It does this using the NLTK package.
    :param word: string
    :return: float
    """
    sid = SentimentIntensityAnalyzer()

    # Lemamtize words before checking the polarity score.
    word2 = lammatize_word(word)
    sentiment =  sid.polarity_scores(word2)['compound']
    return sentiment


def sentiment_analysis_sentence(sentence: [str]) -> (float, float):
    """
    This function takes a sentence as input and returns the sentiment of the sentence.
    :param sentence: string
    :return: float
    """

    sentiment = np.zeros(len(sentence))

    for i, word in enumerate(sentence):
        sentiment[i] = sentiment_analysis_word(word)

    return (sentiment, np.mean(sentiment))


def only_low_sentiment_words(sentence: [str], delta: float) -> str:
    """
    This function takes a sentence as input and returns the words with a sentiment lower than delta.
    :param sentence: string, which will be filtered on sentiment
    :param delta: float, the threshold for the sentiment. The absolute value will be taken.
    :return: string, without the words with a sentiment higher than delta
    """

    returned_string = ""
    for i, word in enumerate(sentence):

        s = sentiment_analysis_word(word)
        if(np.abs(s) <= np.abs(delta)):

            returned_string += word + " "
    return returned_string


def lammatize_word(word: str) -> str:
    """
    This function takes a word as input and returns the base word of the word. It does this using the NLTK package.
    :param word: string
    :return: string
    """
    lemmatizer = WordNetLemmatizer()

    # Hashtags are not recognised by the polarity_scores function.
    word = word.replace("#", "")
    base_word =  lemmatizer.lemmatize(word)

    return base_word


if __name__ == '__main__':
    sentence = "I fucking love and hate you death"

    # change string to a list of words
    sentence = sentence.split()

    print("Low sentiment", sentiment_analysis_sentence(sentence))
    # print("this is s, s")