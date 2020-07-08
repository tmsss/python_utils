import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
import string

# Tokenização, lemmatização, remoção de stop words e pontuação

class Preprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, stopwords=None, punct=None, lower=True, strip=True):
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = set(stopwords) if stopwords else set(stop_words)
        self.punct      = set(punct) if punct else set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def transform(self, X):

        X = X.map(self.map_df)

        return X

    def tokenize(self, document):

        # Break the document into sentences
        for sent in nltk.sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in nltk.pos_tag(nltk.wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token

                # If punctuation or stopword, ignore token and continue
                if token in self.stopwords or all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        """
        Converts the Penn Treebank tag to a WordNet POS tag, then uses that
        tag to perform much more accurate WordNet lemmatization.
        """
        tag = {
            'N': wn.NOUN,
            'V': wn.VERB,
            'R': wn.ADV,
            'J': wn.ADJ
        }.get(tag[0], wn.NOUN)

        return self.lemmatizer.lemmatize(token, tag)

    def map_df(self, document):
        text = list(self.tokenize(document))
        return ' '.join(text)


def accuracy_summary(classifier, X, y):

    # Separação em training e test sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=555)

    # Codificação das categorias
    labels = LabelEncoder()
    y = labels.fit_transform(y)

    # Ajustamento do modelo
    model = classifier.fit(x_train, y_train)

    # Previsão com base no modelo
    y_pred = model.predict(x_test)

    # Avaliação do modelo
    print("Mislabeled points: %d out of %d" % (np.sum(np.array(y_test) != np.array(y_pred)), len(y_test)))
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("-"*80)


def feature_variation(start, end, interval, model, X, y):
    # Lista com valores de features no intervalo de 1000 a 10000
    n_features = np.arange(start, end, interval)

    # Iteração do modelo de acordo com a variação do número de features
    for n in n_features:
        classifier = Pipeline([
            ('vectorizer', CountVectorizer(max_features=n, analyzer='word')),
            ('transformer', TfidfTransformer()),
            ('classifier', model),
        ])
        print("Validation result for {} features".format(n))
        accuracy_summary(classifier, X, y)

def n_gram_variation(model, X, y):

    # Lista com os parâmetros de teste
    n_gram = [('unigram', (1,1)), ('bigram', (1,2)), ('trigram', (1,3)), ('quadrigram', (1,4))]

    # Iteração do modelo de acordo com a variação de n-grams
    for label, n in n_gram:
        classifier = Pipeline([
                ('vectorizer', CountVectorizer(max_features=3000, analyzer='word', ngram_range=n, lowercase=True)),
                ('transformer', TfidfTransformer()),
                ('classifier', model),
            ])
        print("Validation result for {}".format(label))
        accuracy_summary(classifier, X, y)


def best_model(model, X, y):

    classifier = Pipeline([
            ('preprocessor', Preprocessor()),
            ('vectorizer', CountVectorizer(max_features=3000, analyzer='word', ngram_range=(1,1))),
            ('transformer', TfidfTransformer()),
            ('classifier', model),
        ])
    print("Validation result for preprocessing")
    accuracy_summary(classifier, X, y)
