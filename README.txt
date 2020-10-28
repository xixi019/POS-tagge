This is a Python script to do part of speech tagging using crfsuite.

Configuration
-------------
Requirement:
argparse, os, json, sklearn, numpy , pandas, seaborn, 
matplotlib, sklearn_crfsuite, pickle, spacy, time, nltk.


Run the code(The code is fully commented. Refer to the code for detail.)
------------

Three differents mode:
1. Train:
$ python run.py --mode train --config.yaml

This will automatically train and fit the model on the file as specified in the yaml file. 
The accuracy of the model on the training data will be printed.
Then the model will be saved on a pickle file in the same directory.
 The output is like this:
-----------------------
Training a model on en_ewt/en_ewt-ud-train.conllu
5-fold cross-validation results over training set:

Fold           Score
1              0.930
2              0.922
3              0.930
4              0.928
5              0.934
Average        0.929
Fitting model.
Took 93 seconds.
Saving model to ./svm_tagger.pickle.

--------------
2. Tag(can be run on a file or a single text, lets take the sentence as example.)
$ python run.py --mode tag --text 'I like to eat banana.'--config.yaml

This will tag the sentence, print the result  and save the output in a file.
----------------
output:
Tagging text using pretrained model: ./en_ewt/en_ewt-ud-train.conllu.
Loading model from ./svm_tagger.pickle.
I              PRON
like           VERB
to             PART
eat            VERB
banana         NOUN
.              PUNCT
--------------
3. eval 
This mode will take the file as specified in the gold parameter, 
load the model as saved in the pickle file.
Then it shall print the matrix of the accuracy of the model.
$ python run.py --mode eval --gold "en_ewt-ud-dev.conllu" --config config.yaml
output:
----------------
Loading model from ./svm_tagger.pickle.
              precision    recall  f1-score   support

         ADJ       0.89      0.88      0.88      1789
         ADP       0.92      0.97      0.95      2021
         ADV       0.90      0.87      0.88      1265
         AUX       0.95      0.97      0.96      1512
       CCONJ       0.99      0.99      0.99       780
         DET       0.98      0.99      0.98      1894
        INTJ       0.86      0.66      0.75       115
        NOUN       0.89      0.93      0.91      4195
         NUM       0.98      0.93      0.95       378
        PART       0.94      0.94      0.94       630
        PRON       0.98      0.98      0.98      2219
       PROPN       0.89      0.85      0.87      1879
       PUNCT       0.99      1.00      1.00      3083
       SCONJ       0.89      0.81      0.85       403
         SYM       0.94      0.70      0.80        70
        VERB       0.92      0.91      0.91      2762
           X       0.81      0.56      0.66       155

    accuracy                           0.93     25150
   macro avg       0.92      0.88      0.90     25150
weighted avg       0.93      0.93      0.93     25150

Code:
1. We decided to use crfsuite to improve the performance of our model, 
which achieves 93% on the development data.
2. Also some more features are added to featurize the sentences. 
Refer to the code in pro_process for detail.

---------Note
Actually we have tried different models, 
Multi-layer Perceptron(93%), 
PassiveAggressiveClassifier(random_state=0) 92%
Decision tree(89% along with long time)
Random tree(90.9% along with insanely long time)
Adaboost classifier(37.7% and insanely long timr)
but none of them improve accuracy to a large extent.
Ridge classifier has accuracy of 0.91, just like stochatic gradient descent
(give a lot of warning about not converging).
Accuracy drops to 72 in xgboost.


