import pandas as pd
from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics                                                 
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import spacy


def evaluate_model(model, x_test, y_test, cm=False):
    prediction = model.predict(x_test)

    accuracy = metrics.accuracy_score(y_test, prediction)
    precision = metrics.precision_score(y_test, prediction)
    recall = metrics.recall_score(y_test, prediction)
    f1 = metrics.f1_score(y_test, prediction)

    # Wyświetlanie metryk
    print("Accuracy:   %0.3f" % (accuracy*100))
    print("Precision:   %0.3f" % (precision*100))
    print("Recall:   %0.3f" % (recall*100))
    print("F1 score:   %0.3f" % (f1*100))
    
    if cm:
        cm = metrics.confusion_matrix(y_test, prediction, labels=[0,1])

        fig, ax = plot_confusion_matrix(conf_mat=metrics.confusion_matrix(y_test, prediction),
                                        show_absolute=True,
                                        show_normed=True,
                                        colorbar=True)
        plt.show()

    return accuracy, precision, recall, f1

def tfidf_vectorize(sentences: pd.Series, **kwargs) -> pd.DataFrame:
    """
    Creates a DataFrame with vector representation from text documents.

    :param sentences: corpus of documents that needs to be represented by vectors
    :return: DataFrame containing tf-idf vectors with feature names as columns
    """
    vectorizer = TfidfVectorizer(**kwargs)
    document_term_matrix = vectorizer.fit_transform(sentences)
    feature_names = vectorizer.get_feature_names_out()

    return pd.DataFrame(document_term_matrix.toarray(), columns=feature_names)


def spacy_embedding(
    data: pd.DataFrame, text_column: str, columns_to_include: list, nlp
) -> pd.DataFrame:
    """
    Creates numerical representation of text using spacy.
    :param data: Pandas DataFrame containing input data with 'text_column'
    :param text_columns: name of column containing text
    :param columns_to_include: list of columns from data
    :return: Pandas DataFrame containing numeric values representing text
    """
    # 96 is length of normal nlp pipe vector
    embedded_text = [
        doc.vector if doc else [0] * 96 for doc in nlp.pipe(data[text_column])
    ]

    embedded_text = pd.DataFrame(embedded_text)

    embedded_text[columns_to_include] = data[columns_to_include]

    return embedded_text