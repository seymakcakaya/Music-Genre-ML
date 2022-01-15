#!/usr/bin/env python
# coding: utf-8

# In[1]:


import librosa as lbr
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import pathlib
import csv
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import sqlite3
from scipy.stats import kurtosis
from scipy.stats import skew
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot
import sys
from PyQt5.QtGui import QIcon
import pygame


import joblib
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtCore import pyqtSlot
import sys
from PyQt5.QtGui import QIcon



import joblib
from joblib import dump, load
from sklearn.preprocessing import StandardScaler

import pickle
import os

file = open("modelsNB.p",'rb')    
NaiveBayes =pickle.load(file)

file = open("modelsKNN.p",'rb')    
KNN = pickle.load(file)

file = open("modelsRF.p",'rb')    
RandomForest = pickle.load(file)

file = open("modelsSVM.p",'rb')    
SupportVectorMachine = pickle.load(file)
file = open("modelsLG.p",'rb')    
LogisticRegression = pickle.load(file)

file = open("modelsGRB.p",'rb')    
GradientBoostingClassifier = pickle.load(file)

file = open("modelsADA.p",'rb')    
AdaBoostClassifier =pickle.load(file)

file = open("modelsXGB.p",'rb')    
CrossGradientBooster = pickle.load(file)
wavPath = ''


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(1200, 1000)
        #Form.setStyleSheet("background-color: rgb(209,238,238);")
        # Form.setFixedSize(1123, 823)
        self.showWavPath = QtWidgets.QTextEdit(Form)
        self.showWavPath.setStyleSheet("background-color: rgb(225, 235, 238);"
                                 "border-style: outset;"
                                 "border-width: 2px;"
                                 # "font:  40px;"
                                 "border-radius: 10px;"
                                 "padding: 6px;"
                                 "border-color: rgb(0, 95, 105);")
        self.showWavPath.setGeometry(QtCore.QRect(125, 175, 350, 50))
        self.showWavPath.setObjectName("showWavPath")
        self.uploadButton = QtWidgets.QPushButton(Form)
        #self.uploadButton.setGeometry(QtCore.QRect(485, 175, 50, 50))
        self.uploadButton.setStyleSheet("background-color: rgb(150,205,205);"
                        "border-style: outset;"
                                        "font:  40px;"
                        "border-width: 2px;"
                        "border-radius: 10px;"
                        "padding: 6px;"
                        "border-color: rgb(102,139,139);")
        self.uploadButton.setGeometry(QtCore.QRect(485, 175, 50, 50))
        self.uploadButton.setObjectName("uploadButton")

        self.label_3 = QtWidgets.QLabel(Form)
        self.label_3.setStyleSheet("color: 	teal")
        self.label_3.setGeometry(QtCore.QRect(740, 125, 400, 50))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")

        self.playButton = QtWidgets.QPushButton(Form)

        self.playButton.setStyleSheet("background-color: rgb(150,205,205);"
                                        "border-style: outset;"
                                        "border-width: 2px;"
                                         "font: bold 50px;"
                                        "border-radius: 10px;"
                                        "padding: 6px;"
                                        "border-color: rgb(102,139,139);")
        self.playButton.setGeometry(QtCore.QRect(535, 175, 50, 50))
        self.playButton.setObjectName("playButton")
        self.classifyButton = QtWidgets.QPushButton(Form)
        self.classifyButton.setStyleSheet("background-color: rgb(150,205,205);"
                                        "border-style: outset;"
                                        "border-width: 2px;"
                                       #"font:  40px;"
                                        "border-radius: 10px;"
                                        "padding: 6px;"
                                        "border-color: rgb(102,139,139);")
        self.classifyButton.setGeometry(QtCore.QRect(175, 680, 250, 50))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.classifyButton.setFont(font)
        self.classifyButton.setObjectName("classifyButton")
        self.label = QtWidgets.QLabel(Form)
        self.label.setGeometry(QtCore.QRect(200, 125, 400, 50))
        self.label.setStyleSheet("color: teal")

        font = QtGui.QFont()
        font.setPointSize(15)
        self.label.setFont(font)
        self.label.setObjectName("label")

        self.label_4 = QtWidgets.QLabel(Form)
        self.label_4.setStyleSheet("color: 	teal")
        self.label_4.setGeometry(QtCore.QRect(150, 260, 550, 50))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_4.setFont(font)
        self.label_4.setObjectName("label_4")

        self.accuracy = QtWidgets.QLabel(Form)
        self.accuracy.setGeometry(QtCore.QRect(125, 400, 550, 50))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.accuracy.setStyleSheet("color: 	teal")
        self.accuracy.setFont(font)
        self.accuracy.setObjectName("accuracy")

        self.label_2 = QtWidgets.QLabel(Form)
        self.label_2.setGeometry(QtCore.QRect(760, 490, 400, 50))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.label_2.setStyleSheet("color: 	teal")
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.mlBox = QtWidgets.QComboBox(Form)
        self.mlBox.setGeometry(QtCore.QRect(125, 325, 350, 50))
        self.mlBox.setObjectName("mlBox")
        self.mlBox.setStyleSheet("background-color: rgb(225, 235, 238);"
                                        "border-style: outset;"
                                        "border-width: 2px;"
                                       #"font:  40px;"
                                        "border-radius: 10px;"
                                        "padding: 6px;"
                                        "border-color: rgb(0, 95, 105);")
        self.baslik = QtWidgets.QLabel(Form)
        self.baslik.setStyleSheet("color: 		rgb(0, 95, 105)")
        self.baslik.setGeometry(QtCore.QRect(250, 50, 1000, 50))
        font = QtGui.QFont()
        font.setPointSize(20)
        self.baslik.setFont(font)
        self.baslik.setObjectName("baslik")
        self.pauseButton = QtWidgets.QPushButton(Form)
        self.pauseButton.setFont(font)
        self.pauseButton.setStyleSheet("background-color: rgb(150,205,205);"
                                        "border-style: outset;"
                                        "border-width: 2px;"
                                       "font:  40px;"
                                        "border-radius: 10px;"
                                        "padding: 6px;"
                                        "border-color: rgb(102,139,139);")
        self.pauseButton.setGeometry(QtCore.QRect(585, 175, 50, 50))
        self.pauseButton.setObjectName("pauseButton")

        self.showGenreText = QtWidgets.QTextEdit(Form)
        self.showGenreText.setStyleSheet("background-color: rgb(225, 235, 238);;"
                                 "border-style: outset;"
                                 "border-width: 2px;"
                                 # "font:  40px;"
                                 "border-radius: 10px;"
                                 "padding: 6px;"
                                 "border-color: rgb(0, 95, 105);")
        self.showGenreText.setGeometry(QtCore.QRect(700, 175, 350, 300))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.showGenreText.setFont(font)
        self.showGenreText.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.showGenreText.setObjectName("showGenreText")

        self.showAccuracy = QtWidgets.QTextEdit(Form)
        self.showAccuracy.setStyleSheet("background-color: rgb(225, 235, 238);"
                                 "border-style: outset;"
                                 "border-width: 2px;"
                                 # "font:  40px;"
                                 "border-radius: 10px;"
                                 "padding: 6px;"
                                "border-color: rgb(0, 95, 105);")
        self.showAccuracy.setGeometry(QtCore.QRect(125, 450, 350, 200))
        font = QtGui.QFont()
        font.setPointSize(15)
        self.showAccuracy.setFont(font)
        self.showAccuracy.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.showAccuracy.setObjectName("showAccuracy")

        self.songList = QtWidgets.QListWidget(Form)
        self.songList.setStyleSheet("background-color: rgb(225, 235, 238);"
                                 "border-style: outset;"
                                 "border-width: 2px;"
                                 # "font:  40px;"
                                 "border-radius: 10px;"
                                 "padding: 6px;"
                                 "border-color: rgb(0, 95, 105);")
        self.songList.setGeometry(QtCore.QRect(700, 550, 350, 300))
        self.songList.setObjectName("songList")
        self.onerButton = QtWidgets.QPushButton(Form)
        self.onerButton.setStyleSheet("background-color: rgb(150,205,205);"
                                        "border-style: outset;"
                                        "border-width: 2px;"
                                        "font-color:  teal;"
                                        "border-radius: 10px;"
                                        "padding: 6px;"
                                        "border-color: rgb(102,139,139);")
        self.onerButton.setGeometry(QtCore.QRect(175, 760, 250, 50))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.onerButton.setFont(font)
        self.onerButton.setObjectName("onerButton")

        self.retranslateUi(Form)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("Form", "Music Genre Classifier and Recommender"))
        MainWindow.setStyleSheet("background-color: aliceblue;")
        self.uploadButton.setText(_translate("Form", "♬"))
        self.label_3.setText(_translate("Form", "Seçilen Müziğin Türü"))
        self.playButton.setText(_translate("Form", "▶"))
        self.classifyButton.setText(_translate("Form", "Sınıflandır"))
        self.label.setText(_translate("Form", "Müzik Dosyasını Yükleyiniz"))
        self.label_4.setText(_translate("Form", "Makine Öğrenmesi Yöntemi Seçiniz"))
        self.accuracy.setText(_translate("Form", "Seçtiğiniz Yöntemin Doğruluk Oranı"))
        self.label_2.setText(_translate("Form", "Önerilen Şarkılar"))
        self.baslik.setText(_translate("Form", "Müzik Türü Sınıflandırma Ve Öneri Sistemi"))
        self.pauseButton.setText(_translate("Form", "‖"))
        self.showGenreText.setHtml(_translate("Form",
                                              "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                              "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                              "p, li { white-space: pre-wrap; }\n"
                                              "</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:26pt; font-weight:400; font-style:normal;\">\n"
                                              "<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
                                              "<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.showAccuracy.setHtml(_translate("Form",
                                              "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
                                              "<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
                                              "p, li { white-space: pre-wrap; }\n"
                                              "</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:26pt; font-weight:400; font-style:normal;\">\n"
                                              "<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p>\n"
                                              "<p align=\"center\" style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><br /></p></body></html>"))
        self.onerButton.setText(_translate("Form", "Şarkı Öner"))

        self.mlBox.addItem("KNN")
        self.mlBox.addItem("Cross Gradient Boosting")
        self.mlBox.addItem("Support Vector Machine")
        self.mlBox.addItem("Random Forest")
        self.mlBox.addItem("Gradient Boosting")
        self.mlBox.addItem("Logistic Regression")
        self.mlBox.addItem("Naive Bayes")
        self.mlBox.addItem("Ada Boost")

        self.playButton.clicked.connect(self.play)
        self.pauseButton.clicked.connect(self.stop)
        self.uploadButton.clicked.connect(self.browseFile)
        self.classifyButton.clicked.connect(self.classify)
        self.onerButton.clicked.connect(self.recommend)

    def browseFile(self):
        filter = "wav(*.wav)"
        global wavPath
        fpath = QtWidgets.QFileDialog.getOpenFileName(None, caption='Select .wav File',
                                                      directory='C:/Users/user/Music\\', filter=filter)
        self.showWavPath.setText(fpath[0])
        print("file path")
        print(fpath[0])
        wavPath = fpath[0]

    def getmetadata(self, filename):

        y, sr = lbr.load(filename)
        # fetching tempo

        onset_env = lbr.onset.onset_strength(y, sr)
        tempo = lbr.beat.tempo(onset_envelope=onset_env, sr=sr)

        # fetching beats

        y_harmonic, y_percussive = lbr.effects.hpss(y)

        tempo, beat_frames = lbr.beat.beat_track(y=y_percussive, sr=sr)

        # chroma_stft

        chroma_stft = lbr.feature.chroma_stft(y=y, sr=sr)

        # rmse

        rmse = lbr.feature.rms(y=y)

        # fetching spectral centroid

        spec_centroid = lbr.feature.spectral_centroid(y, sr=sr)[0]

        # spectral bandwidth

        spec_bw = lbr.feature.spectral_bandwidth(y=y, sr=sr)

        # fetching spectral rolloff

        spec_rolloff = lbr.feature.spectral_rolloff(y + 0.01, sr=sr)[0]

        # zero crossing rate

        zero_crossing = lbr.feature.zero_crossing_rate(y)

        # mfcc

        mfcc = lbr.feature.mfcc(y=y, sr=sr)

        # metadata dictionary

        metadata_dict = {'chroma_stft': np.mean(chroma_stft), 'chroma_var': np.var(chroma_stft), 'rmse': np.mean(rmse),
                         'rmse_var': np.var(rmse),
                         'spectral_centroid': np.mean(spec_centroid), 'spectral_centroid_var': np.var(spec_centroid),
                         'spectral_bandwidth': np.mean(spec_bw),
                         'spectral_bandwidth_var': np.var(spec_bw), 'rolloff': np.mean(spec_rolloff),
                         'rolloff_var': np.var(spec_rolloff), 'zero_crossing_rates': np.mean(zero_crossing),
                         'zero_crossing_rates_var': np.var(zero_crossing)
            , 'harmony_mean': np.mean(y_harmonic), 'harmony_var': np.var(y_harmonic),
                         'perceptr_mean': np.mean(y_percussive), 'perceptr_var': np.var(y_percussive), 'tempo': tempo
                         }

        for i in range(1, 21):
            metadata_dict.update({'mfcc' + str(i): np.mean(mfcc[i - 1])})
            metadata_dict.update({'mfcc' + str(i) + '_var': np.var(mfcc[i - 1])})

        return list(metadata_dict.values())

    def predict_gen(self, meta1, data):
        print("predivtte")
        svmp = data['clc']
        norma = data['norma']
        x = norma.transform([meta1])
        pred = svmp.predict(x)
        return (pred[0])

    # Siniflandirma tusu butonu metodu

    def classify(self):
        self.showGenreText.setText(" ")

        if (wavPath != ''):

            meta = self.getmetadata(wavPath)
            if (self.mlBox.currentText() == 'Naive Bayes'):
                genre = self.predict_gen(meta, NaiveBayes)
                self.showGenreText.setText(genre)
                self.showAccuracy.setText("Naive Bayes Algoritması ile %52 oranında doğru sınıflandırıldı.")

                pass

            if (self.mlBox.currentText() == 'KNN'):
                genre = self.predict_gen(meta, KNN)
                self.showGenreText.setText(genre)
                self.showAccuracy.setText("KNN Algoritması ile %91 oranında doğru sınıflandırıldı.")

                pass

            if (self.mlBox.currentText() == 'Support Vector Machine'):
                genre = self.predict_gen(meta, SupportVectorMachine)
                self.showGenreText.setText(genre)
                self.showAccuracy.setText("Support Vector Machine Algoritması ile %75 oranında doğru sınıflandırıldı.")
                pass

            if (self.mlBox.currentText() == 'Random Forest'):
                genre = self.predict_gen(meta, RandomForest)
                self.showGenreText.setText(genre)
                self.showAccuracy.setText("Random Forest Algoritması ile %89 oranında doğru sınıflandırıldı.")

                pass

            if (self.mlBox.currentText() == 'Ada Boost'):
                genre = self.predict_gen(meta, AdaBoostClassifier)
                self.showGenreText.setText(genre)
                self.showAccuracy.setText("Ada Boost Algoritması ile %49 oranında doğru sınıflandırıldı.")

                pass

            if (self.mlBox.currentText() == 'Gradient Boosting'):
                genre = self.predict_gen(meta, GradientBoostingClassifier)
                self.showGenreText.setText(genre)
                self.showAccuracy.setText("Gradient Boost Algoritması ile %83 oranında doğru sınıflandırıldı.")

                pass

            if (self.mlBox.currentText() == 'Logistic Regression'):
                genre = self.predict_gen(meta, LogisticRegression)
                self.showGenreText.setText(genre)
                self.showAccuracy.setText("Logistic Regression Algoritması ile %70 oranında doğru sınıflandırıldı.")

                pass

            if (self.mlBox.currentText() == 'Cross Gradient Boosting'):
                genre = self.predict_gen(meta, CrossGradientBooster)
                self.showGenreText.setText(genre)
                self.showAccuracy.setText("Cross Gradient Boosting Algoritması ile %91 oranında doğru sınıflandırıldı.")

                pass

    # Sarki onerme tusu butonu
    def recommend(self):
        import pickle
        import os

        file = open("C:/Users/user/PycharmProjects/flaskProject/model_weigths/models.p", 'rb')
        veri = pickle.load(file)
        norma = veri['norma']

        self.songList.clear()

        from sklearn.neighbors import NearestNeighbors
        con = sqlite3.connect("data.db")
        df = pd.read_sql_query("SELECT * from {}".format('OZELLIK'), con)
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(df['label'])
        labelEncoded = le.transform(df['label'])
        df['LabelEncoded'] = labelEncoded
        y = df['LabelEncoded']
        X = df.drop(['LabelEncoded', 'label', 'length'], axis=1)
        X.set_index('filename', inplace=True)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        ftrarr = self.getmetadata(wavPath)

        ftrarr_scaled = scaler.transform([ftrarr])
        bestSelect = SelectKBest(k=50)
        X_new = bestSelect.fit_transform(X_scaled, y)
        ftr_new = bestSelect.transform(ftrarr_scaled)
        neighbours = NearestNeighbors(n_neighbors=5)
        neighbours.fit(X_new)
        indices = neighbours.kneighbors(ftr_new, return_distance=False)
        indices = indices.reshape(-1)
        X['filename'] = X.index
        for i in range(0, 5):
            self.songList.addItem(X['filename'][indices].values[i])
        con.close()

    # Muzik oynatma tusu metodu
    def play(self):
        if (wavPath != ''):
            pygame.init()
            pygame.mixer.init()
            channel1 = pygame.mixer.Channel(0)
            click = pygame.mixer.Sound(wavPath)
            channel1.play(click)

    # Muzik durdurma tusu metodu
    def stop(self):
        pygame.mixer.pause()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)
    Form = QtWidgets.QWidget()
    ui = Ui_Form()
    ui.setupUi(Form)
    Form.show()
    sys.exit(app.exec_())


# In[ ]:





# In[ ]:




