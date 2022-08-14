import DataPrep
import FeatureSelection
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, f1_score, classification_report
import pandas as pd
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Bidirectional, LSTM, Activation, Dense, Dropout, Input, Embedding
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence

#string to test
#BAG OF WORDS

nb_pipeline = Pipeline([
        ('NBCV',FeatureSelection.countV),
        ('nb_clf',MultinomialNB())])

nb_pipeline.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_nb = nb_pipeline.predict(DataPrep.test_news['Statement'])
np.mean(predicted_nb == DataPrep.test_news['Label'])


svm_pipeline = Pipeline([
        ('svmCV',FeatureSelection.countV),
        ('svm_clf',svm.LinearSVC())
        ])

svm_pipeline.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_svm = svm_pipeline.predict(DataPrep.test_news['Statement'])
np.mean(predicted_svm == DataPrep.test_news['Label'])


sgd_pipeline = Pipeline([
        ('svm2CV',FeatureSelection.countV),
        ('svm2_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5))
        ])

sgd_pipeline.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_sgd = sgd_pipeline.predict(DataPrep.test_news['Statement'])
np.mean(predicted_sgd == DataPrep.test_news['Label'])



def build_confusion_matrix(classifier):

    k_fold = KFold(n_splits=5)
    scores = []
    confusion = np.array([[0,0],[0,0]])

    for train_ind, test_ind in k_fold.split(DataPrep.train_news):
        train_text = DataPrep.train_news.iloc[train_ind]['Statement']
        train_y = DataPrep.train_news.iloc[train_ind]['Label']

        test_text = DataPrep.train_news.iloc[test_ind]['Statement']
        test_y = DataPrep.train_news.iloc[test_ind]['Label']

        classifier.fit(train_text,train_y)
        predictions = classifier.predict(test_text)

        confusion += confusion_matrix(test_y,predictions)
        score = f1_score(test_y,predictions)
        scores.append(score)

    return (print('Total statements classified:', len(DataPrep.train_news)),
    print('Score:', sum(scores)/len(scores)),
    print('Confusion matrix:'),
    print(confusion))

print("Naive Bayes : ")
build_confusion_matrix(nb_pipeline)
print("SVM : ")
build_confusion_matrix(svm_pipeline)
print("SGD : ")
build_confusion_matrix(sgd_pipeline)


#Now using n-grams
#naive-bayes classifier

print("Using n-grams: \n\n\n")

nb_pipeline_ngram = Pipeline([
        ('nb_tfidf',FeatureSelection.tfidf_ngram),
        ('nb_clf',MultinomialNB())])

nb_pipeline_ngram.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_nb_ngram = nb_pipeline_ngram.predict(DataPrep.test_news['Statement'])
np.mean(predicted_nb_ngram == DataPrep.test_news['Label'])


#linear SVM classifier
svm_pipeline_ngram = Pipeline([
        ('svm_tfidf',FeatureSelection.tfidf_ngram),
        ('svm_clf',svm.LinearSVC())
        ])

svm_pipeline_ngram.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_svm_ngram = svm_pipeline_ngram.predict(DataPrep.test_news['Statement'])
np.mean(predicted_svm_ngram == DataPrep.test_news['Label'])


#sgd classifier
sgd_pipeline_ngram = Pipeline([
         ('sgd_tfidf',FeatureSelection.tfidf_ngram),
         ('sgd_clf',SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5))
         ])

sgd_pipeline_ngram.fit(DataPrep.train_news['Statement'],DataPrep.train_news['Label'])
predicted_sgd_ngram = sgd_pipeline_ngram.predict(DataPrep.test_news['Statement'])
np.mean(predicted_sgd_ngram == DataPrep.test_news['Label'])



print("Naive Bayes : ")
build_confusion_matrix(nb_pipeline_ngram)
print("SVM : ")
build_confusion_matrix(svm_pipeline_ngram)
print("SGD : ")
build_confusion_matrix(sgd_pipeline_ngram)
print("\n\n")
print("Naive Bayes : ")
print(classification_report(DataPrep.test_news['Label'], predicted_nb_ngram))
print("SVM : ")
print(classification_report(DataPrep.test_news['Label'], predicted_svm_ngram))



# # Bi-LSTM
# train = pd.DataFrame()
# validate = pd.DataFrame()
# test = pd.DataFrame()
#
# read_train = pd.read_csv("train.tsv", delimiter="\t", header = None)
# read_test = pd.read_csv("test.tsv", delimiter="\t", header = None)
# read_valid = pd.read_csv("valid.tsv", delimiter="\t", header = None)
#
#
# def map_to_int(x):
#     if x == 'true' or x == 'True' or x == "half-true" or x == 'mostly-true':
#         return 0
#     else:
#         return 1
#
# train['text'] = read_train[2]
# train['label'] = read_train[1].apply(map_to_int)
#
# validate['text'] = read_valid[2]
# validate['label'] = read_valid[1].apply(map_to_int)
#
# test['text'] = read_test[2]
# test['label'] = read_test[1].apply(map_to_int)
#
# print(len(train), len(validate), len(test))
#
#
# X_train = pd.concat([train,validate])['text']
# Y_train = pd.concat([train,validate])['label']
# X_test = test['text']
# Y_test = test['label']
#
# # Plot for checking skewness
# # sns.countplot(Y_train)
# # plt.xlabel('Label')
# # plt.title('Number of real and fake news')
#
# max_words = 1000
# max_len = 150
# # Convert to tokens
# tok = Tokenizer(num_words=max_words)
# # Update Vocabulary by adding appropriate text data. (Word index lis\ created by providing a
# # unique value to each word got from the Tokenizer)
# tok.fit_on_texts(X_train)
# # Convert each text in tok to sequence of integers. (Same as above but does it for the input, as compared to
# # doing it on the vocabulary) (NOTE : Here, both the inputs are the same. No external pre-processed Vocabulary used.
# # So, the input is used in both phases)
# sequences = tok.texts_to_sequences(X_train)
# # Evens the length of all the sequences created from above. (Zero padding - zero at the beginning until it reaches the
# # same length as the max_length (Here, it is cut off at 150))
# sequences_matrix = sequence.pad_sequences(sequences, maxlen=max_len)
#
# # Bi-LSTM Function
# def BiLSTM():
#     '''
#     Builds the Bidirectional LSTM Network.
#
#     The function builds Bidirectional LSTM network by taking word embeddings as input and
#     using Bidirectional, LSTM, Dense, Dropout, Dense layer in the specified order. 'ReLU'
#     is used as activation function in the network and 'sigmoid' is used as
#     activation function for the classification layer.
#
#     Parameters
#     ----------
#     None
#     '''
#     inputs = Input(name='inputs', shape=[max_len])
#     layer = Embedding(max_words, 50, input_length=max_len)(inputs)
#     layer = Bidirectional(LSTM(64))(layer)
#     # Fully connected
#     layer = Dense(256, name='FC1')(layer)
#     layer = Activation('relu')(layer)
#     # Drops a few layers(randomized) (Arguments - Fraction of input units to drop [0,1] - zero to 1 ) , to speed up training
#     # by weight updation only when there is a change.
#     layer = Dropout(0.5)(layer)
#     layer = Dense(1, name='out_layer')(layer)
#     layer = Activation('sigmoid')(layer)
#     model = Model(inputs=inputs, outputs=layer)
#     return model
#
# # Model summary
# model = BiLSTM()
# model.summary()
# # entropy loss = -y log (y^hat), RMSprop() - Step size of gd defined for each weight separately.
# model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
#
# # Train in batches, use the last few values in train to validate in each epoch(validation_split)
# model.fit(sequences_matrix,Y_train,batch_size=128,epochs=40,validation_split = 0.2)
#
# # Text
# X_test = X_test[:1267]
# # Label
# Y_test = Y_test[:1267]
#
# # Convert to text to sequences.
# test_sequences = tok.texts_to_sequences(X_test)
# # Zero padding to equal length
# test_sequences_matrix = sequence.pad_sequences(test_sequences,maxlen=max_len)
#
# # Predict on sequence matrix
# y_pred = model.predict(test_sequences_matrix)
#
# # ON TEST
# result = model.evaluate(test_sequences_matrix,Y_test, batch_size = 128)
# print("Test loss, Test accuracy : ",result)
# # Take prediction values(0 or 1) for each test instance (-1 means last value - which has the final prediction output)
# y_pred = y_pred.argmax(axis=-1)
# # Check this out
# # (https://datascience.stackexchange.com/questions/64441/how-to-interpret-classification-report-of-scikit-learn)
# print(classification_report(test['label'], y_pred))





