import os
import pprint
import numpy as np
import pandas as pd
import pickle
import timeit
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# from nltk.stem.snowball import SnowballStemmer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

import sklearn.model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from sklearn.utils import shuffle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import ShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# from sklearn.externals import joblib
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score

import torch

from option import Options
from utils.training_helper import create_logger


# https://www.kaggle.com/samuelrod/sdgs-multi-label-text-classifier-baseline/comments
def grid_search(train_x, train_y, test_x, test_y, labels, parameters, pipeline):
    '''Train pipeline, test and print results'''
    grid_search_tune = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3, verbose=10)
    grid_search_tune.fit(train_x, train_y)

    print()
    print("Best parameters set:")
    print(grid_search_tune.best_estimator_.steps)
    print()

    # measuring performance on test set
    print("Applying best classifier on test data:")
    best_clf = grid_search_tune.best_estimator_
    predictions = best_clf.predict(test_x)

    print(classification_report(test_y, predictions, target_names=labels))
    print("ROC-AUC:", roc_auc_score(test_y, predictions))


def split_train_val(train_df, logger):
    logger.info('total: {}'.format(len(train_df)))
    logger.info('train shape: {}\n'.format(train_df.shape))

    logger.info('unique posting id: {}'.format(len(train_df['posting_id'].unique())))
    logger.info('unique image: {}'.format(len(train_df['image'].unique())))
    logger.info('unique image phash: {}'.format(len(train_df['image_phash'].unique())))
    logger.info('unique title: {}'.format(len(train_df['title'].unique())))
    logger.info('unique label group: {}'.format(len(train_df['label_group'].unique())))  # 11014
    logger.info('\n')
    logger.info('image counts:')
    logger.info(train_df['image'].value_counts())
    logger.info('\n\n')

    gkf = GroupKFold(n_splits=5)
    x_shuffled, y_shuffled, groups_shuffled = \
        shuffle(train_df, train_df['label'], train_df['posting_id'].tolist(), random_state=8)

    results = []
    for idx, (train_idx, val_idx) in enumerate(gkf.split(x_shuffled, y_shuffled, groups=groups_shuffled)):
        train_fold = train_df.iloc[train_idx]
        val_fold = train_df.iloc[val_idx]
        results.append((train_fold, val_fold))

    return results


def clean_text(title_lst):
    # print('original text: ', text)
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()

    # nltk.download('punkt')
    # nltk.download('stopwords')

    cleaned_title = []
    for i, title in enumerate(title_lst):
        words = word_tokenize(title)
        clean_words = []
        for word in words:
            word = word.lower()
            if word not in stopwords.words('indonesian'):  # 불용어 제거
                # stemmer = SnowballStemmer('indonesian')
                # word = stemmer.stem(word) #어간 추출
                word = stemmer.stem(word)
                clean_words.append(word)
        # print(clean_words)
        new_title = ' '.join(clean_words)
        cleaned_title.append(new_title)

    return cleaned_title


# TODO:
def save_model(ml_obj, name):
    # save the model to disk
    filename = os.path.join('./experiments/shopee-product-matching/ml', name)
    pickle.dump(ml_obj, open(filename, 'wb'))

    # # load the model from disk
    # loaded_model = pickle.load(open(filename, 'rb'))
    # result = loaded_model.score(X_test, Y_test)
    # print(result)

def main():
    args, args_desc = Options().parse()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    print(args)

    logger, log_file, final_output_dir, tb_log_dir, create_at = create_logger(args, args_desc)
    logger.info('-------------- params --------------')
    logger.info(pprint.pformat(args.__dict__))

    train_df = pd.read_csv(os.path.join(args.dataset_root, 'train.csv'))

    # TODO: clean_text 함수 사용 고민중.. 시간이 많이 걸림.
    # train_df['title_clean'] = clean_text(train_df['title'])
    # title_to_use = cudf.DataFrame(df).title_clean

    # #####################################################################################################
    # sample
    # from sklearn.datasets import fetch_20newsgroups
    # twenty_train = fetch_20newsgroups(subset='train', shuffle=True)
    # print(twenty_train.target_names)
    #
    # # twenty_train -> list of text(news)
    # print("\n".join(twenty_train.data[0].split("\n")[:3]))  # prints first line of the first data file
    # #####################################################################################################

    # https://www.kaggle.com/nadintamer/titanic-survival-predictions-beginner
    # predictors = train_df.drop(['label_group', 'label'], axis=1)
    predictors = train_df.drop(['posting_id', 'image', 'image_phash', 'label_group', 'label'], axis=1)
    target = train_df["label"]
    x_train, x_val, y_train, y_val = train_test_split(predictors, target, test_size=0.22, random_state=8)

    count_vect = CountVectorizer(stop_words='english',)
                                 # binary=True,
                                 # lowercase=True,
                                 # # max_df = 0.5,
                                 # # min_df = 2,
                                 # max_features=25000)
    X_train_counts = count_vect.fit_transform(x_train['title']).toarray().astype(np.float32)
    print('X_train_counts.shap:', X_train_counts.shape)

    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts).toarray().astype(np.float32)
    print('X_train_tfidf.shape:', X_train_tfidf.shape)

    count_vect2 = CountVectorizer(stop_words='english',)
                                 # binary=True,
                                 # lowercase=True,
                                 # # max_df = 0.5,
                                 # # min_df = 2,
                                 # max_features=25000)
    x_val_counts = count_vect2.fit_transform(x_val['title']).toarray().astype(np.float32)
    print('x_val_counts.shap:', x_val_counts.shape)

    tfidf_transformer2 = TfidfTransformer()
    x_val_tfidf = tfidf_transformer2.fit_transform(x_val_counts).toarray().astype(np.float32)
    print('x_val_tfidf.shape:', x_val_tfidf.shape)

    # ---------
    # Failure
    # ---------
    # text_clf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', MultinomialNB())])
    # text_clf = text_clf.fit(x_train, y_train)
    # y_pred = text_clf.predict(x_val_tfidf)
    # acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
    # print(acc_gbk)

    # gaussian = GaussianNB()
    # gaussian.fit(X_train_tfidf, y_train)
    # y_pred = gaussian.predict(x_val_tfidf)
    # acc_gaussian = round(accuracy_score(y_pred, y_val) * 100, 2)
    # print(acc_gaussian)

    # gbk = GradientBoostingClassifier(verbose=True)
    # gbk.fit(X_train_tfidf, y_train)
    # y_pred = gbk.predict(x_val_tfidf)
    # acc_gbk = round(accuracy_score(y_pred, y_val) * 100, 2)
    # print(acc_gbk)

    start = timeit.default_timer()
    svc = SVC()
    svc.fit(X_train_tfidf, y_train)
    y_pred = svc.predict(x_val_tfidf)
    acc_svc = round(accuracy_score(y_pred, y_val) * 100, 2)
    end = timeit.default_timer()
    logger.info('elapsed time: %d' % np.int((end - start) / 60))
    print(acc_svc)
    save_model(svc, 'svc_model.m')

    # #############
    # Naive Bayes
    # #############
    # pipeline = Pipeline([
    #     ('tfidf', TfidfVectorizer(stop_words='english')),
    #     ('clf', OneVsRestClassifier(MultinomialNB(
    #         fit_prior=True, class_prior=None))),
    # ])
    # from sklearn.feature_extraction.text import CountVectorizer
    # from sklearn.feature_extraction.text import TfidfTransformer
    # from sklearn.linear_model import SGDClassifier
    #
    # pipeline = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
    #                      ('clf-svm',
    #                       SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, max_iter=5, random_state=42))])
    #
    # params = sklearn.estimator.get_params().keys()
    #
    # parameters = {
    #     'tfidf__max_df': (0.25, 0.5, 0.75),
    #     'tfidf__ngram_range': [(1, 1), (1, 2), (1, 3)],
    #     'clf__estimator__alpha': (1e-2, 1e-3)
    # }
    # grid_search(x_train, y_train, x_val, y_val, 'label_group', parameters, pipeline)

    model = NearestNeighbors(n_neighbors=100, metric='cosine')


if __name__ == '__main__':
    main()
