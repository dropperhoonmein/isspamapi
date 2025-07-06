import numpy as np
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from preprocessor import PreprocessedText



df1 = pd.read_csv("hf://datasets/TrainingDataPro/email-spam-classification/email_spam.csv")
df1['type'] = ['spam' if x == 'spam' else 'ham' for x in df1['type']]
df1['text']= "Subject"+": "+df1['title']+" "+df1['text']


df2 = pd.read_csv('datasets\\spam_ham_dataset.csv')
df3 = pd.read_csv('datasets\\CEAS_08.csv')
df3['body'] = "Subject"+": "+df3['subject']+" "+df3['body']
df3['label'] = ['spam' if x == 1 else 'ham' for x in df3['label']]
df4 = pd.read_csv('datasets\\SpamAssasin.csv')
df4['body'] = "Subject"+": "+df4['subject']+" "+df4['body']
df4['label'] = ['spam' if x == 1 else 'ham' for x in df4['label']]
df = pd.DataFrame()
df['text'] = pd.concat([df1['text'], df2['text'],df3['body'],df4['body']], axis=0)
df['label']=pd.concat([df1['type'], df2['label'],df3['label'],df4['label']], axis=0)
df = df.fillna("")

Y = df['label']
X = df['text']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1,stratify=Y,random_state=2)
spam_pipeline = Pipeline([
   ('stemming',PreprocessedText()),
   ('vectorize',TfidfVectorizer()),
   ('model', LogisticRegression())
])


spam_pipeline.fit(X_train,Y_train)
Y_predict = spam_pipeline.predict(X_test)
print(accuracy_score(Y_test,Y_predict))
email = '''Subject: Meeting Reminder for Tomorrow

Hey John,

Just a quick reminder about our team meeting scheduled for tomorrow at 10:30 AM in Conference Room B.

We'll be going over the Q3 progress, upcoming deadlines, and assigning action items. Please bring your updated reports and let me know if you'd like to add anything to the agenda.

Looking forward to a productive session!

Best,
Sarah
'''
print(spam_pipeline.predict([email]))
with open('trained_pipeline-0.1.0.sav','wb') as f:
  pickle.dump(spam_pipeline,f)