PATH = '/Users/siddhantthakur/pizza req/'
from tqdm import tqdm
import json
import pandas as pd
import numpy as np
from statistics import mean
import shap

from prettytable import PrettyTable
import matplotlib.pyplot as plt

import nltk
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC

from gensim.models import KeyedVectors
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

import warnings
warnings.filterwarnings("ignore")

# setting a random state to keep the output consistent
RANDOM_STATE = 4

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# loading the json data
with open(PATH + 'pizza_request_dataset/pizza_request_dataset.json', 'r') as f:
    data = json.load(f)

# getting the success rate of receiving a pizza
success = [d['requester_received_pizza'] for d in data]
print("Average Success Rate",str(sum(success)/len(success)*100))

# creating features like upvotes-downvotes
for d in data:
    d['upvotes_minus_downvotes'] = d['number_of_upvotes_of_request_at_retrieval'] - d['number_of_downvotes_of_request_at_retrieval']

# combining the title and textual part of the request
for d in data:
    d['final_request_text'] = d['request_title'] + " " + d['request_text_edit_aware']

# converting json to dataframe
df = pd.json_normalize(data)
df['requester_received_pizza'] = df['requester_received_pizza'].astype(int)
df = df.sample(frac=1).reset_index(drop=True)
df.head()

# printing the null values in the dataset
print(df['requester_received_pizza'].isna().sum())

# text preprocessing
stopwords_eng = stopwords.words('english')
lemmatizer = WordNetLemmatizer()

def process_text(text):
    text = text.replace("\n"," ").replace("\r"," ")
    text = re.sub(r'“', " '' ", text)
    text = re.sub(r'”', " '' ", text)
    text = re.sub(r'"', " '' ", text)
    text = re.sub(r'(\S)(\'\')', r'\1 \2 ', text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'s", " is", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'t", " not", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'m", " am", text)
    
    punc_list = '!"#$%()*+,-./:;<=>?@^_{|}~[]'
    t = str.maketrans(dict.fromkeys(punc_list," "))
    text = text.translate(t)
    
    t = str.maketrans(dict.fromkeys("'`",""))
    text = text.translate(t)
    
    text = text.lower()
    
    tokens = regexp_tokenize(text,pattern='\s+',gaps=True)
    cleaned_tokens = []
    
    for t in tokens:
        if t not in stopwords_eng:
            l = lemmatizer.lemmatize(t)
            cleaned_tokens.append(l)
    
    return cleaned_tokens

# 20% split for train and test data
train_set = df.loc[:4536,:]
test_set = df.loc[4537:,:].reset_index(drop=True)
print(train_set.shape)
print(test_set.shape)

# engineering more features using past literature
final_df = df.loc[:,['upvotes_minus_downvotes','requester_account_age_in_days_at_request','request_number_of_comments_at_retrieval']]
final_df['length_of_text'] = df.apply(lambda r: len(r['final_request_text']), axis = 1)
final_df['evidence_link'] = df.apply(lambda r: 1 if re.findall(r'(?:http\:|https\:)?\/\/.*\.',r['final_request_text']) else 0, axis = 1)
final_df['requester_posted_before'] = df.apply(lambda r: 0 if r['requester_days_since_first_post_on_raop_at_request']==0 else 1, axis = 1)
final_df['requester_received_pizza'] = df.loc[:,'requester_received_pizza']

# splitting the final train and test set
final_train_set = final_df.loc[:4536,:]
final_test_set = final_df.loc[4537:,:].reset_index(drop=True)
print(final_train_set.shape)
print(final_test_set.shape)

# Random Forest based TF-IDF Model
new_tfidf_vec = TfidfVectorizer(analyzer=process_text)
new_tfidf_df = new_tfidf_vec.fit_transform(train_set['final_request_text'])

new_tfidf_labels = train_set['requester_received_pizza']

print(new_tfidf_df.shape)

new_skfold_tfidf = StratifiedKFold(n_splits = 10, shuffle=True, random_state=RANDOM_STATE)
new_tfidf_model_lr = LogisticRegression(class_weight = {0:1, 1:5}, random_state=RANDOM_STATE)
new_tfidf_model_nbg = GaussianNB()
new_tfidf_model_nbm = MultinomialNB()
new_tfidf_model_rf = RandomForestClassifier(n_estimators=590, min_samples_leaf=2, min_samples_split=2, max_features='sqrt',max_depth=50, bootstrap=False, random_state=RANDOM_STATE)

new_tfidf_model_lr_accuracy_score = []
new_tfidf_model_nbg_accuracy_score = []
new_tfidf_model_nbm_accuracy_score = []
new_tfidf_model_rf_accuracy_score = []

for train_index, test_index in new_skfold_tfidf.split(new_tfidf_df, new_tfidf_labels):
    X_train, X_test = new_tfidf_df[train_index], new_tfidf_df[test_index]
    y_train, y_test = new_tfidf_labels[train_index], new_tfidf_labels[test_index]
    
    new_tfidf_model_lr.fit(X_train,y_train)
    new_tfidf_model_lr_accuracy_score.append(accuracy_score(y_test,new_tfidf_model_lr.predict(X_test)))

    new_tfidf_model_nbg.fit(X_train.toarray(),y_train)
    new_tfidf_model_nbg_accuracy_score.append(accuracy_score(y_test,new_tfidf_model_nbg.predict(X_test.toarray())))
    
    new_tfidf_model_nbm.fit(X_train,y_train)
    new_tfidf_model_nbm_accuracy_score.append(accuracy_score(y_test,new_tfidf_model_nbm.predict(X_test)))
    
    new_tfidf_model_rf.fit(X_train,y_train)
    new_tfidf_model_rf_accuracy_score.append(accuracy_score(y_test,new_tfidf_model_rf.predict(X_test)))

new_final_tfidf_rf_df = final_train_set.copy()
new_tfidf_rf_est_prob = new_tfidf_model_rf.predict_proba(new_tfidf_df)[:,1]
new_final_tfidf_rf_df.insert(5,'prob_from_text',new_tfidf_rf_est_prob)
new_final_tfidf_rf_df

new_final_tfidf_rf_df = new_final_tfidf_rf_df.sample(frac=1).reset_index(drop=True)
new_final_tfidf_rf_labels = new_final_tfidf_rf_df.iloc[:,-1]
new_final_tfidf_rf_df = new_final_tfidf_rf_df.iloc[:,:-1]
new_final_tfidf_rf_labels

new_skfold_final_tfidf_rf = StratifiedKFold(n_splits = 10, shuffle=True, random_state=RANDOM_STATE)
new_final_tfidf_rf_model_lr = LogisticRegression(class_weight = {0:1, 1:5}, random_state=RANDOM_STATE)
new_final_tfidf_rf_model_nbg = GaussianNB()
new_final_tfidf_rf_model_rf = RandomForestClassifier(n_estimators=590, min_samples_leaf=2, min_samples_split=2, max_features='sqrt',max_depth=50, bootstrap=False, random_state=RANDOM_STATE)

new_final_tfidf_rf_model_lr_accuracy_score = []
new_final_tfidf_rf_model_nbg_accuracy_score = []
new_final_tfidf_rf_model_rf_accuracy_score = []

new_final_tfidf_rf_model_lr_precision_score = []
new_final_tfidf_rf_model_nbg_precision_score = []
new_final_tfidf_rf_model_rf_precision_score = []

new_final_tfidf_rf_model_lr_recall_score = []
new_final_tfidf_rf_model_nbg_recall_score = []
new_final_tfidf_rf_model_rf_recall_score = []

new_final_tfidf_rf_model_lr_f1_score = []
new_final_tfidf_rf_model_nbg_f1_score = []
new_final_tfidf_rf_model_rf_f1_score = []

for train_index, test_index in new_skfold_final_tfidf_rf.split(new_final_tfidf_rf_df, new_final_tfidf_rf_labels):
    X_train, X_test = new_final_tfidf_rf_df.iloc[train_index,:], new_final_tfidf_rf_df.iloc[test_index,:]
    y_train, y_test = new_final_tfidf_rf_labels[train_index], new_final_tfidf_rf_labels[test_index]
    
    new_final_tfidf_rf_model_lr.fit(X_train,y_train)
    new_final_tfidf_rf_model_lr_accuracy_score.append(accuracy_score(y_test,new_final_tfidf_rf_model_lr.predict(X_test)))
    new_final_tfidf_rf_model_lr_precision_score.append(precision_score(y_test,new_final_tfidf_rf_model_lr.predict(X_test)))
    new_final_tfidf_rf_model_lr_recall_score.append(recall_score(y_test,new_final_tfidf_rf_model_lr.predict(X_test)))
    new_final_tfidf_rf_model_lr_f1_score.append(f1_score(y_test,new_final_tfidf_rf_model_lr.predict(X_test)))
    
    new_final_tfidf_rf_model_nbg.fit(X_train,y_train)
    new_final_tfidf_rf_model_nbg_accuracy_score.append(accuracy_score(y_test,new_final_tfidf_rf_model_nbg.predict(X_test)))
    new_final_tfidf_rf_model_nbg_precision_score.append(precision_score(y_test,new_final_tfidf_rf_model_nbg.predict(X_test)))
    new_final_tfidf_rf_model_nbg_recall_score.append(recall_score(y_test,new_final_tfidf_rf_model_nbg.predict(X_test)))
    new_final_tfidf_rf_model_nbg_f1_score.append(f1_score(y_test,new_final_tfidf_rf_model_nbg.predict(X_test)))
    
    new_final_tfidf_rf_model_rf.fit(X_train,y_train)
    new_final_tfidf_rf_model_rf_accuracy_score.append(accuracy_score(y_test,new_final_tfidf_rf_model_rf.predict(X_test)))
    new_final_tfidf_rf_model_rf_precision_score.append(precision_score(y_test,new_final_tfidf_rf_model_rf.predict(X_test)))
    new_final_tfidf_rf_model_rf_recall_score.append(recall_score(y_test,new_final_tfidf_rf_model_rf.predict(X_test)))
    new_final_tfidf_rf_model_rf_f1_score.append(f1_score(y_test,new_final_tfidf_rf_model_rf.predict(X_test)))

# using shapley values to understand importance of each feature
tfidf_explainer = shap.TreeExplainer(new_final_tfidf_rf_model_rf)
tfidf_shap_values = tfidf_explainer.shap_values(X_test)
shap.summary_plot(tfidf_shap_values, X_test, plot_type="bar")
plt.savefig('eps/FeatureImportanceShapley_rf.eps',dpi=1200,bbox_inches='tight')

test_tfidf_df = new_tfidf_vec.transform(test_set['final_request_text'])
print(test_tfidf_df.shape)

test_final_tfidf_rf_df = final_test_set.copy()
test_tfidf_rf_est_prob = new_tfidf_model_rf.predict_proba(test_tfidf_df)[:,1]
test_final_tfidf_rf_df.insert(5,'prob_from_text',test_tfidf_rf_est_prob)
test_final_tfidf_rf_df

test_final_tfidf_rf_df = test_final_tfidf_rf_df.sample(frac=1).reset_index(drop=True)
test_final_tfidf_rf_labels = test_final_tfidf_rf_df.iloc[:,-1]
test_final_tfidf_rf_df = test_final_tfidf_rf_df.iloc[:,:-1]
test_final_tfidf_rf_labels

# testing performance
final_accuracy = pd.DataFrame({'Count Model':[
                             100*accuracy_score(test_final_count_rf_labels, new_final_count_rf_model_lr.predict(test_final_count_rf_df)),
                             100*accuracy_score(test_final_count_rf_labels, new_final_count_rf_model_nbg.predict(test_final_count_rf_df)),
                             100*accuracy_score(test_final_count_rf_labels, new_final_count_rf_model_rf.predict(test_final_count_rf_df))
                         ], 
                         'TF-IDF':[
                             100*accuracy_score(test_final_tfidf_rf_labels, new_final_tfidf_rf_model_lr.predict(test_final_tfidf_rf_df)),
                             100*accuracy_score(test_final_tfidf_rf_labels, new_final_tfidf_rf_model_nbg.predict(test_final_tfidf_rf_df)),
                             100*accuracy_score(test_final_tfidf_rf_labels, new_final_tfidf_rf_model_rf.predict(test_final_tfidf_rf_df))
                         ], 
                         'Min-Max Doc2Vec':[
                             100*accuracy_score(test_final_minmax_word2vec_rf_labels, new_final_minmax_word2vec_model_lr.predict(test_final_minmax_word2vec_rf_df)),
                             100*accuracy_score(test_final_minmax_word2vec_rf_labels, new_final_minmax_word2vec_model_nbg.predict(test_final_minmax_word2vec_rf_df)),
                             100*accuracy_score(test_final_minmax_word2vec_rf_labels, new_final_minmax_word2vec_model_rf.predict(test_final_minmax_word2vec_rf_df))
                         ],
                         'Doc2Vec 100':[
                             100*accuracy_score(test_final_doc2vec_100_rf_labels, new_final_doc2vec_100_model_lr.predict(test_final_doc2vec_100_rf_df)),
                             100*accuracy_score(test_final_doc2vec_100_rf_labels, new_final_doc2vec_100_model_nbg.predict(test_final_doc2vec_100_rf_df)),
                             100*accuracy_score(test_final_doc2vec_100_rf_labels, new_final_doc2vec_100_model_rf.predict(test_final_doc2vec_100_rf_df))
                         ],
                         'Doc2Vec 300':[
                             100*accuracy_score(test_final_doc2vec_300_rf_labels, new_final_doc2vec_300_model_lr.predict(test_final_doc2vec_300_rf_df)),
                             100*accuracy_score(test_final_doc2vec_300_rf_labels, new_final_doc2vec_300_model_nbg.predict(test_final_doc2vec_300_rf_df)),
                             100*accuracy_score(test_final_doc2vec_300_rf_labels, new_final_doc2vec_300_model_rf.predict(test_final_doc2vec_300_rf_df))
                         ]},
                        index = ['LogReg','Gaussian NB','RF'])
final_precision = pd.DataFrame({'Count Model':[
                             100*precision_score(test_final_count_rf_labels, new_final_count_rf_model_lr.predict(test_final_count_rf_df)),
                             100*precision_score(test_final_count_rf_labels, new_final_count_rf_model_nbg.predict(test_final_count_rf_df)),
                             100*precision_score(test_final_count_rf_labels, new_final_count_rf_model_rf.predict(test_final_count_rf_df))
                         ], 
                         'TF-IDF':[
                             100*precision_score(test_final_tfidf_rf_labels, new_final_tfidf_rf_model_lr.predict(test_final_tfidf_rf_df)),
                             100*precision_score(test_final_tfidf_rf_labels, new_final_tfidf_rf_model_nbg.predict(test_final_tfidf_rf_df)),
                             100*precision_score(test_final_tfidf_rf_labels, new_final_tfidf_rf_model_rf.predict(test_final_tfidf_rf_df))
                         ], 
                         'Min-Max Doc2Vec':[
                             100*precision_score(test_final_minmax_word2vec_rf_labels, new_final_minmax_word2vec_model_lr.predict(test_final_minmax_word2vec_rf_df)),
                             100*precision_score(test_final_minmax_word2vec_rf_labels, new_final_minmax_word2vec_model_nbg.predict(test_final_minmax_word2vec_rf_df)),
                             100*precision_score(test_final_minmax_word2vec_rf_labels, new_final_minmax_word2vec_model_rf.predict(test_final_minmax_word2vec_rf_df))
                         ],
                         'Doc2Vec 100':[
                             100*precision_score(test_final_doc2vec_100_rf_labels, new_final_doc2vec_100_model_lr.predict(test_final_doc2vec_100_rf_df)),
                             100*precision_score(test_final_doc2vec_100_rf_labels, new_final_doc2vec_100_model_nbg.predict(test_final_doc2vec_100_rf_df)),
                             100*precision_score(test_final_doc2vec_100_rf_labels, new_final_doc2vec_100_model_rf.predict(test_final_doc2vec_100_rf_df))
                         ],
                         'Doc2Vec 300':[
                             100*precision_score(test_final_doc2vec_300_rf_labels, new_final_doc2vec_300_model_lr.predict(test_final_doc2vec_300_rf_df)),
                             100*precision_score(test_final_doc2vec_300_rf_labels, new_final_doc2vec_300_model_nbg.predict(test_final_doc2vec_300_rf_df)),
                             100*precision_score(test_final_doc2vec_300_rf_labels, new_final_doc2vec_300_model_rf.predict(test_final_doc2vec_300_rf_df))
                         ]},
                        index = ['LogReg','Gaussian NB','RF'])
final_recall = pd.DataFrame({'Count Model':[
                             100*recall_score(test_final_count_rf_labels, new_final_count_rf_model_lr.predict(test_final_count_rf_df)),
                             100*recall_score(test_final_count_rf_labels, new_final_count_rf_model_nbg.predict(test_final_count_rf_df)),
                             100*recall_score(test_final_count_rf_labels, new_final_count_rf_model_rf.predict(test_final_count_rf_df))
                         ], 
                         'TF-IDF':[
                             100*recall_score(test_final_tfidf_rf_labels, new_final_tfidf_rf_model_lr.predict(test_final_tfidf_rf_df)),
                             100*recall_score(test_final_tfidf_rf_labels, new_final_tfidf_rf_model_nbg.predict(test_final_tfidf_rf_df)),
                             100*recall_score(test_final_tfidf_rf_labels, new_final_tfidf_rf_model_rf.predict(test_final_tfidf_rf_df))
                         ], 
                         'Min-Max Doc2Vec':[
                             100*recall_score(test_final_minmax_word2vec_rf_labels, new_final_minmax_word2vec_model_lr.predict(test_final_minmax_word2vec_rf_df)),
                             100*recall_score(test_final_minmax_word2vec_rf_labels, new_final_minmax_word2vec_model_nbg.predict(test_final_minmax_word2vec_rf_df)),
                             100*recall_score(test_final_minmax_word2vec_rf_labels, new_final_minmax_word2vec_model_rf.predict(test_final_minmax_word2vec_rf_df))
                         ],
                         'Doc2Vec 100':[
                             100*recall_score(test_final_doc2vec_100_rf_labels, new_final_doc2vec_100_model_lr.predict(test_final_doc2vec_100_rf_df)),
                             100*recall_score(test_final_doc2vec_100_rf_labels, new_final_doc2vec_100_model_nbg.predict(test_final_doc2vec_100_rf_df)),
                             100*recall_score(test_final_doc2vec_100_rf_labels, new_final_doc2vec_100_model_rf.predict(test_final_doc2vec_100_rf_df))
                         ],
                         'Doc2Vec 300':[
                             100*recall_score(test_final_doc2vec_300_rf_labels, new_final_doc2vec_300_model_lr.predict(test_final_doc2vec_300_rf_df)),
                             100*recall_score(test_final_doc2vec_300_rf_labels, new_final_doc2vec_300_model_nbg.predict(test_final_doc2vec_300_rf_df)),
                             100*recall_score(test_final_doc2vec_300_rf_labels, new_final_doc2vec_300_model_rf.predict(test_final_doc2vec_300_rf_df))
                         ]},
                        index = ['LogReg','Gaussian NB','RF'])
final_f1_score = pd.DataFrame({'Count Model':[
                             100*f1_score(test_final_count_rf_labels, new_final_count_rf_model_lr.predict(test_final_count_rf_df)),
                             100*f1_score(test_final_count_rf_labels, new_final_count_rf_model_nbg.predict(test_final_count_rf_df)),
                             100*f1_score(test_final_count_rf_labels, new_final_count_rf_model_rf.predict(test_final_count_rf_df))
                         ], 
                         'TF-IDF':[
                             100*f1_score(test_final_tfidf_rf_labels, new_final_tfidf_rf_model_lr.predict(test_final_tfidf_rf_df)),
                             100*f1_score(test_final_tfidf_rf_labels, new_final_tfidf_rf_model_nbg.predict(test_final_tfidf_rf_df)),
                             100*f1_score(test_final_tfidf_rf_labels, new_final_tfidf_rf_model_rf.predict(test_final_tfidf_rf_df))
                         ], 
                         'Min-Max Doc2Vec':[
                             100*f1_score(test_final_minmax_word2vec_rf_labels, new_final_minmax_word2vec_model_lr.predict(test_final_minmax_word2vec_rf_df)),
                             100*f1_score(test_final_minmax_word2vec_rf_labels, new_final_minmax_word2vec_model_nbg.predict(test_final_minmax_word2vec_rf_df)),
                             100*f1_score(test_final_minmax_word2vec_rf_labels, new_final_minmax_word2vec_model_rf.predict(test_final_minmax_word2vec_rf_df))
                         ],
                         'Doc2Vec 100':[
                             100*f1_score(test_final_doc2vec_100_rf_labels, new_final_doc2vec_100_model_lr.predict(test_final_doc2vec_100_rf_df)),
                             100*f1_score(test_final_doc2vec_100_rf_labels, new_final_doc2vec_100_model_nbg.predict(test_final_doc2vec_100_rf_df)),
                             100*f1_score(test_final_doc2vec_100_rf_labels, new_final_doc2vec_100_model_rf.predict(test_final_doc2vec_100_rf_df))
                         ],
                         'Doc2Vec 300':[
                             100*f1_score(test_final_doc2vec_300_rf_labels, new_final_doc2vec_300_model_lr.predict(test_final_doc2vec_300_rf_df)),
                             100*f1_score(test_final_doc2vec_300_rf_labels, new_final_doc2vec_300_model_nbg.predict(test_final_doc2vec_300_rf_df)),
                             100*f1_score(test_final_doc2vec_300_rf_labels, new_final_doc2vec_300_model_rf.predict(test_final_doc2vec_300_rf_df))
                         ]},
                        index = ['LogReg','Gaussian NB','RF'])
final_auc_roc = pd.DataFrame({'Count Model':[
                             100*roc_auc_score(test_final_count_rf_labels, new_final_count_rf_model_lr.predict_proba(test_final_count_rf_df)[:, 1]),
                             100*roc_auc_score(test_final_count_rf_labels, new_final_count_rf_model_nbg.predict_proba(test_final_count_rf_df)[:, 1]),
                             100*roc_auc_score(test_final_count_rf_labels, new_final_count_rf_model_rf.predict_proba(test_final_count_rf_df)[:, 1])
                         ], 
                         'TF-IDF':[
                             100*roc_auc_score(test_final_tfidf_rf_labels, new_final_tfidf_rf_model_lr.predict_proba(test_final_tfidf_rf_df)[:, 1]),
                             100*roc_auc_score(test_final_tfidf_rf_labels, new_final_tfidf_rf_model_nbg.predict_proba(test_final_tfidf_rf_df)[:, 1]),
                             100*roc_auc_score(test_final_tfidf_rf_labels, new_final_tfidf_rf_model_rf.predict_proba(test_final_tfidf_rf_df)[:, 1])
                         ],
                         'Min-Max Doc2Vec':[
                             100*roc_auc_score(test_final_minmax_word2vec_rf_labels, new_final_minmax_word2vec_model_lr.predict_proba(test_final_minmax_word2vec_rf_df)[:, 1]),
                             100*roc_auc_score(test_final_minmax_word2vec_rf_labels, new_final_minmax_word2vec_model_nbg.predict_proba(test_final_minmax_word2vec_rf_df)[:, 1]),
                             100*roc_auc_score(test_final_minmax_word2vec_rf_labels, new_final_minmax_word2vec_model_rf.predict_proba(test_final_minmax_word2vec_rf_df)[:, 1])
                         ],
                         'Doc2Vec 100':[
                             100*roc_auc_score(test_final_doc2vec_100_rf_labels, new_final_doc2vec_100_model_lr.predict_proba(test_final_doc2vec_100_rf_df)[:, 1]),
                             100*roc_auc_score(test_final_doc2vec_100_rf_labels, new_final_doc2vec_100_model_nbg.predict_proba(test_final_doc2vec_100_rf_df)[:, 1]),
                             100*roc_auc_score(test_final_doc2vec_100_rf_labels, new_final_doc2vec_100_model_rf.predict_proba(test_final_doc2vec_100_rf_df)[:, 1])
                         ],
                         'Doc2Vec 300':[
                             100*roc_auc_score(test_final_doc2vec_300_rf_labels, new_final_doc2vec_300_model_lr.predict_proba(test_final_doc2vec_300_rf_df)[:, 1]),
                             100*roc_auc_score(test_final_doc2vec_300_rf_labels, new_final_doc2vec_300_model_nbg.predict_proba(test_final_doc2vec_300_rf_df)[:, 1]),
                             100*roc_auc_score(test_final_doc2vec_300_rf_labels, new_final_doc2vec_300_model_rf.predict_proba(test_final_doc2vec_300_rf_df)[:, 1])
                         ]},
                        index = ['LogReg','Gaussian NB','RF'])

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 12}

plt.rc('font', **font)
fig, ax = plt.subplots(2,2, figsize=(15,14))
ax[0][0].plot(final_accuracy.index,final_accuracy.loc[final_accuracy.index,:], marker = 'o')
ax[0][0].legend(final_accuracy.columns)
ax[0][0].set_ylim(0,100)
ax[0][0].set_title('Accuracy')
ax[0][1].plot(final_precision.index,final_precision.loc[final_precision.index,:], marker = 'o')
ax[0][1].legend(final_precision.columns)
ax[0][1].set_ylim(0,100)
ax[0][1].set_title('Precision')
ax[1][0].plot(final_recall.index,final_recall.loc[final_recall.index,:], marker = 'o')
ax[1][0].legend(final_recall.columns)
ax[1][0].set_ylim(0,100)
ax[1][0].set_title('Recall')
ax[1][1].plot(final_f1_score.index,final_f1_score.loc[final_f1_score.index,:], marker = 'o')
ax[1][1].legend(final_f1_score.columns)
ax[1][1].set_ylim(0,100)
ax[1][1].set_title('F-Score')
fig.show()
fig.savefig('eps/Final_Model_Performance.eps',dpi=1200, bbox_inches = 'tight')

plt.figure(figsize = (12,10))
plt.plot(final_auc_roc.index,final_auc_roc.loc[final_auc_roc.index,:], marker = 'o')
plt.legend(final_auc_roc.columns)
plt.ylim(0,100)
plt.title('AUC ROC Score')
plt.savefig('eps/Auc_Roc_Score_metric.eps', dpi = 1200, bbox_inches = 'tight')
plt.show()

# Graph for Text Models
text_accuracy_score = {'Count Model':[accuracy_score(final_test_set['requester_received_pizza'],new_count_model_lr.predict(test_count_df))*100,accuracy_score(final_test_set['requester_received_pizza'],new_count_model_nbg.predict(test_count_df.toarray()))*100,accuracy_score(final_test_set['requester_received_pizza'],new_count_model_rf.predict(test_count_df))*100], 
                       'TF-IDF':[accuracy_score(final_test_set['requester_received_pizza'],new_tfidf_model_lr.predict(test_tfidf_df))*100,accuracy_score(final_test_set['requester_received_pizza'],new_tfidf_model_nbg.predict(test_tfidf_df.toarray()))*100,accuracy_score(final_test_set['requester_received_pizza'],new_tfidf_model_rf.predict(test_tfidf_df))*100],
                       'MinMax Word2Vec':[accuracy_score(final_test_set['requester_received_pizza'],new_minmax_word2vec_model_lr.predict(test_minmax_word2vec_df))*100,accuracy_score(final_test_set['requester_received_pizza'],new_minmax_word2vec_model_rf.predict(test_minmax_word2vec_df))*100,accuracy_score(final_test_set['requester_received_pizza'],new_minmax_word2vec_model_rf.predict(test_minmax_word2vec_df))*100],
                       'Doc2Vec 100':[accuracy_score(final_test_set['requester_received_pizza'],new_doc2vec_100_model_lr.predict(test_doc2vec_100_df))*100,accuracy_score(final_test_set['requester_received_pizza'],new_doc2vec_100_model_nbg.predict(test_doc2vec_100_df))*100,accuracy_score(final_test_set['requester_received_pizza'],new_doc2vec_100_model_rf.predict(test_doc2vec_100_df))*100],
                       'Doc2Vec 300':[accuracy_score(final_test_set['requester_received_pizza'],new_doc2vec_300_model_lr.predict(test_doc2vec_300_df))*100,accuracy_score(final_test_set['requester_received_pizza'],new_doc2vec_300_model_nbg.predict(test_doc2vec_300_df))*100,accuracy_score(final_test_set['requester_received_pizza'],new_doc2vec_300_model_rf.predict(test_doc2vec_300_df))*100]}
plot_data = pd.DataFrame(text_accuracy_score, index=['Logistic Regression','Gaussian Naive Bayes','Random Forest'])

fig = plt.figure(figsize=(10,5))
plt.plot(plot_data.index,plot_data.loc[plot_data.index,:], marker = 'o')
plt.ylim(40,100)
plt.title('Comparison of Accuracy Score using different Text Embedding Methods')
plt.xlabel('Models/Algorithms Used')
plt.ylabel('Accuracy Score')
plt.legend(plot_data.columns)
plt.savefig('eps/AccScoreTextModel.eps', dpi=1200,bbox_inches='tight')
plt.show()