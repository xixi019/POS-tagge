from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
import sklearn_crfsuite
from sklearn_crfsuite import scorers
from sklearn.model_selection import cross_val_score
from .pre_process import token_to_features
import numpy as np
import pickle
import spacy
import time
import os

nlp = spacy.load("en_core_web_sm")

def fit_and_report_crf(X, Y, cross_val = True, n_folds=5):
    crf = sklearn_crfsuite.CRF(
                                algorithm='lbfgs',
                                c1=0.1,
                                c2=0.1,
                                max_iterations=100,
                                all_possible_transitions=True)
    if cross_val:
        from sklearn.model_selection import cross_val_score
        print(f"Doing {n_folds}-fold cross-validation.")
        scores = cross_val_score(crf, X, Y, cv=n_folds)
        print(f"{n_folds}-fold cross-validation results over training set:\n")
        print("Fold\tScore".expandtabs(15))
        for i in range(n_folds):
            print(f"{i+1}\t{scores[i]:.3f}".expandtabs(15))
        print(f"Average\t{np.mean(scores):.3f}".expandtabs(15))
    print("Fitting model.")
    start_time = time.time()
    crf.fit(X, Y)
    end_time = time.time()
    print(f"Took {int(end_time - start_time)} seconds.")
    return crf

def tag_sentence_crf(sentence, model):
    doc = nlp(sentence)
    tokenized_sent = [token.text for token in doc]
    featurized_sent = []
    for i in range(len(tokenized_sent)):
        featurized_sent.append(token_to_features(tokenized_sent, i))
    labels = model.predict([featurized_sent])
    tagged_sent = list(zip(tokenized_sent, labels[0]))
    return tagged_sent


def tag_eval_sentence_crf(tokenized_sent, model):
    featurized_sent = []
    for i in range(len(tokenized_sent)):
        featurized_sent.append(token_to_features(tokenized_sent, i))
    labels = model.predict([featurized_sent])
    tagged_sent = list(zip(tokenized_sent, labels[0]))
    return tagged_sent


def print_tagged_sent_crf(tagged_sent):
    for token in tagged_sent:
        print(f"{token[0]}\t{token[1]}".expandtabs(15))

def return_tagged_sent_crf(tagged_sent):
    for token in tagged_sent:
        yield f"{token[0]}\t{token[1]}".expandtabs(15)

def save_model_crf(model, output_file):
    print(f"Saving model to {output_file}.")
    with open(output_file, "wb") as outfile:
        pickle.dump(model, outfile)

def load_model_crf(output_file):
    print(f"Loading model from {output_file}.")
    with open(output_file, "rb") as infile:
        model = pickle.load(infile)
    return model