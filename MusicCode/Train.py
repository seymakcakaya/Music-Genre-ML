#!/usr/bin/env python
# coding: utf-8

# In[13]:


def getmetadata(filename):
    import librosa
    import numpy as np


    y, sr = librosa.load(filename)
    #fetching tempo

    onset_env = librosa.onset.onset_strength(y, sr)
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)

    #fetching beats

    y_harmonic, y_percussive = librosa.effects.hpss(y)
 
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,sr=sr)

    #chroma_stft

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)

    #rmse

    rmse = librosa.feature.rms(y=y)

    #fetching spectral centroid

    spec_centroid = librosa.feature.spectral_centroid(y, sr=sr)[0]

    #spectral bandwidth

    spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)

    #fetching spectral rolloff

    spec_rolloff = librosa.feature.spectral_rolloff(y+0.01, sr=sr)[0]

    #zero crossing rate

    zero_crossing = librosa.feature.zero_crossing_rate(y)

    #mfcc

    mfcc = librosa.feature.mfcc(y=y, sr=sr)

    #metadata dictionary

    metadata_dict = {'chroma_stft':np.mean(chroma_stft),'chroma_var':np.var(chroma_stft),'rmse':np.mean(rmse),'rmse_var':np.var(rmse),
                     'spectral_centroid':np.mean(spec_centroid),'spectral_centroid_var':np.var(spec_centroid),'spectral_bandwidth':np.mean(spec_bw), 
                     'spectral_bandwidth_var':np.var(spec_bw),'rolloff':np.mean(spec_rolloff),'rolloff_var':np.var(spec_rolloff), 'zero_crossing_rates':np.mean(zero_crossing),'zero_crossing_rates_var':np.var(zero_crossing)
                     ,'harmony_mean':np.mean(y_harmonic),'harmony_var':np.var(y_harmonic),'perceptr_mean':np.mean(y_percussive),'perceptr_var':np.var(y_percussive),'tempo':tempo
                    }

   
    for i in range(1,21):
        metadata_dict.update({'mfcc'+str(i):np.mean(mfcc[i-1])})
        metadata_dict.update({'mfcc'+str(i)+'_var':np.var(mfcc[i-1])})
   
    return list(metadata_dict.values())




# In[15]:



general_path = 'C:/Users/user/Desktop/Music Genre/dataset/Data'
b=(f'{general_path}/genres_original/blues/blues.00000.wav')


# In[16]:


# Usual Libraries
import pandas as pd

import numpy as np
general_path = 'C:/Users/user/Desktop/Music Genre/dataset/Data'

data = pd.read_csv(f'{general_path}/features_30_sec.csv')


# In[17]:


# Usual Libraries
import pandas as pd
from joblib import dump
import numpy as np
from importlib.metadata import metadata

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE

import sklearn
import importlib
# Librosa (the mother of audio files)
import librosa
import librosa.display
import IPython.display as ipd
import pandas
import warnings
warnings.filterwarnings('ignore')
import os
general_path = 'C:/Users/user/Desktop/Music Genre/dataset/Data'

data = pd.read_csv(f'{general_path}/features_3_sec.csv')
data = data.iloc[0:, 1:]

data.drop('length',axis=1, inplace=True)
data.head()


# In[ ]:





# In[18]:



y = data['label'] # genre variable.
X = data.loc[:, data.columns != 'label']
#select all columns but not the labels

#### NORMALIZE X ####

# Normalize so everything is on the same scale.

cols = X.columns

min_max_scaler = preprocessing.MinMaxScaler()

np_scaled = min_max_scaler.fit_transform(X)

X = pd.DataFrame(np_scaled, columns = cols)

print(len(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def model_assess(model, title = "Default"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    #print(confusion_matrix(y_test, preds))
    print('Accuracy', title, ':', round(accuracy_score(y_test, preds), 5), '\n')




knn = KNeighborsClassifier(n_neighbors=3)
model_assess(knn, "KNN")


# In[19]:


data['label'] = data['label'].astype('category')
data['class_label'] = data['label'].cat.codes
lookup_genre_name = dict(zip(data.class_label.unique(), data.label.unique()))   
lookup_genre_name


# In[ ]:


lookup_genre_name = dict(zip(df.class_label.unique(), df.class_name.unique()))   
lookup_genre_name


# In[26]:



general_path = 'C:/Users/user/Desktop/Music Genre/dataset/Data'
b=(f'{general_path}/genres_original/country/country.00036.wav')
a=getmetadata(b)
d1 =np.array(a)
data1 = min_max_scaler.transform([d1])
genre_prediction = knn.predict(data1)
genre_prediction
print(genre_prediction[0])



# In[28]:


import pickle
pick1 = {
    'norma':min_max_scaler,
    'svmp':knn,
    
}
pickle.dump( pick1, open( 'models' + ".p", "wb" ) )


# In[27]:


genre_prediction


# In[38]:


pip install django


# In[41]:


def predict_gen(meta1):
    import pickle
    import os
    file = open("models.p",'rb')
    
    data = pickle.load(file)
    svmp = data['svmp']
    norma = data['norma']
    x = norma.transform([meta1])
    pred = svmp.predict(x)
    return(pred[0])


# In[ ]:





# In[42]:



general_path = 'C:/Users/user/Desktop/Music Genre/dataset/Data'
b=(f'{general_path}/genres_original/country/country.00036.wav')

meta = getmetadata(b)
genre = predict_gen(meta)
print(genre)


# In[ ]:




