from sklearn.feature_extraction import DictVectorizer
from nltk.tokenize.treebank import TreebankWordDetokenizer
import os


def load_txt(file):
    X, Y = [], []
    with open(file, "r", encoding="utf-8") as infile:
        sents = infile.read().split("\n\n")
        if sents[-1] == "":
            sents = sents[:-1]
        for sent in sents:
            words, tags = [], []
            lines = sent.split("\n")
            for line in lines:
                line = line.strip().split("\t")
                if len(line) != 2:
                    raise TabError("Tried to read .txt file, but did not find two columns.")
                else:
                    words.append(line[0])
                    tags.append(line[1])
            X.append(words)
            Y.append(tags)
    return X, Y


def load_conllu(file):
    X, Y = [], []
    with open(file, "r", encoding="utf8") as infile:
        sents = infile.read().split("\n\n")
        if sents[-1] == "":
            sents = sents[:-1]
        for sent in sents:
            words, tags = [], []
            lines = sent.split("\n")
            for line in lines:
                if line.startswith("#"):
                    continue
                line = line.strip().split("\t")
                if len(line) != 10:
                    raise TabError("Tried to read .txt file, but did not find ten columns.")
                else:
                    words.append(line[1])
                    tags.append(line[3])
            X.append(words)
            Y.append(tags)
    return X, Y


def load_dataset(file):
    if file.endswith(".conllu"):
        try:
            X, Y = load_conllu(file)
            return X, Y
        except TabError:
            print("Tried to read .txt file, but did not find ten columns.")
    else:
        try:
            X, Y = load_txt(file)
            return X, Y
        except TabError:
            print("Tried to read .txt file, but did not find two columns.")


# Add more features
def token_to_features(sent, i):
    word = sent[i]

    features = {
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        # Take the first two and three letter of words as feature
        'word[:2]': word[:2],
        'word[:3]': word[:3],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
    }
    if i > 0:
        word1 = sent[i-1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True
# Try to take the word two words before
# the word we are looking at as the feature,

    if i > 1:
        word1 = sent[i-2]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    elif i == 0:
        pass
    else:
        features['Secondfirst_of_BOS'] = True

# Take the words two words after the token
# we are looking at into account as the feature.
    if i < len(sent)-2:
        word1 = sent[i+2]
        features.update({
            '+2:word.lower()': word1.lower(),
            '+2:word.istitle()': word1.istitle(),
            '+2:word.isupper()': word1.isupper(),
        })
    elif i == len(sent)-1:
        pass
    else:
        features['SecondLast_of_EOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True
    return features


# Each sentence is processed to be a list of strings,
# and the whole corpus becomes a list of lists(which store the sentences.).
# same applys to the tags.
def prepare_data_for_training_crf(X, Y):
    X_out, Y_out = [], []
    for i, sent in enumerate(X):
        X_out1, Y_out1 = [], []
        for j in range(len(sent)):
            features = token_to_features(sent, j)
            X_out1.append(features)
            Y_out1.append(Y[i][j])
        X_out.append(X_out1)
        Y_out.append(Y_out1)
    return X_out, Y_out
