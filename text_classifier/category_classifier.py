import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from collections import Counter
from pyvi import ViTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import confusion_matrix

import numpy as np

import pickle

from sklearn.linear_model import SGDClassifier

class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.tokenizer = ViTokenizer

    def fit(self, *_):
        return self

    def transform(self, X, y=None, **fit_params):
        result = X.apply(lambda text: self.tokenizer.tokenize(text))
        return result



if __name__ == '__main__':
    df = pd.read_csv('out.csv')
    df = df[['title', 'category']]

    #Clean NaN
    df = df[pd.notnull(df['title'])]
    df = df[pd.notnull(df['category'])]

    for index, row in df.iterrows():
        if row['category'].isnumeric():
            row['category'] = None
    df = df[pd.notnull(df['category'])]
    X_train_content, x_test_content, Y_train, y_test = train_test_split(df['title'], df['category'],
                                                                        test_size=0.2)

    train = pd.DataFrame({'content': X_train_content.tolist(), 'category': Y_train.tolist()})
    train.to_csv('train.csv')

    test = pd.DataFrame({'content': x_test_content.tolist(), 'category': y_test.tolist()})
    test.to_csv('test.csv')

    #Word segmentation
    word_segmentation = FeatureTransformer()
    data_segmented_train = word_segmentation.fit_transform(train['content'])
    data_segmented_test = word_segmentation.transform(test['content'])
    #print(data_segmented)


    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    Data_BoW_content = count_vect.fit_transform(data_segmented_train.values.astype('U'))
    Data_tfidf_content_train = tfidf_transformer.fit_transform(Data_BoW_content)

    clf = MultinomialNB()
    clf.fit(Data_tfidf_content_train, Y_train)



    Data_BoW_test = count_vect.transform(data_segmented_test.values.astype('U'))
    Data_tfidf_content_test = tfidf_transformer.transform(Data_BoW_test)
    y_score = clf.predict(Data_tfidf_content_test)

    test_content = pd.DataFrame(["Thẻ Nhớ Adata Micro SDXC 64GB Class 10"], columns= ['test_data'])
    count_test = count_vect.transform(test_content['test_data'].values.astype('U'))
    testcase = tfidf_transformer.transform(count_test)
    print(clf.predict(testcase))

    multi_result = pd.DataFrame({'category': clf.classes_, 'proba': clf.predict_proba(testcase)[0]})
    print(multi_result.sort_values(by=['proba'], ascending = False))

    input_file = pd.read_csv('final_input_2.csv', low_memory=False)
    input_file = input_file[['title', 'tiki_cat']]
    #input_file = input_file[not input_file['title'].isnumeric()]
    input_segmentation = word_segmentation.transform(input_file['title'])
    input_BoW = count_vect.transform(input_segmentation.values.astype('U'))
    input_tfidf = tfidf_transformer.transform(input_BoW)


    output_temp = pd.DataFrame(columns = ['pred_1', 'prob_1' ,'pred_2', 'prob_2'])
    data = []

    for elem in input_tfidf:
        try:
            clf.predict(elem)
            multi_results = pd.DataFrame({'category': clf.classes_, 'proba': clf.predict_proba(elem)[0]})
            multi_results = multi_results.sort_values(by=['proba'], ascending=False)
            datum = [multi_results['category'].iloc[0], multi_results['proba'].iloc[0],
                    multi_results['category'].iloc[1], multi_results['proba'].iloc[1]]
            data.append(datum)
        except(RuntimeError, TypeError, NameError, KeyError):
            pass

    output_temp = pd.DataFrame(data=data)
    output_temp = output_temp.rename(columns={output_temp.columns[0]: 'pred_1', output_temp.columns[1]: 'prob_1',
                                output_temp.columns[2]: 'pred_2', output_temp.columns[3]: 'prob_2'})
    output = pd.DataFrame({'title': input_file['title'], 'actual': input_file['tiki_cat'],
                           'pred_1': output_temp['pred_1'], 'prob_1': output_temp['prob_1'],
                           'pred_2': output_temp['pred_2'], 'prob_2': output_temp['prob_2']})
    output.to_csv('final_result.csv')


    #############

    result = {'content': x_test_content, 'y_pred': y_score, 'y_actual': y_test}
    results = pd.DataFrame(data=result)

    f = open('false_results.csv', 'w')

    f.write('content,y_pred,y_actual')
    f.write('\n')

    for index, row in results.iterrows():
        if row[1] != row[2]:
            row[0] = str(row[0]).replace(',', '-')
            row[1] = str(row[1]).replace(',', '-')
            row[2] = str(row[2]).replace(',', '-')
            f.write(str(row[0])+ "," + row[1] + "," + row[2])
            f.write('\n')

    n_right = 0
    i = 0
    for index, row in test.iterrows():
        if y_score[i] == row['category']:
            n_right += 1
        i += 1

    results_all = pd.DataFrame({'content':test['content'], 'y_pred': y_score ,'y_actual': test['category']})
    results_all.to_csv('result.csv')

    print("Accuracy: %.2f%%" % ((n_right / float(len(y_test)) * 100)))
