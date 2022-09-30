import math
import csv
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm

def load_data(filename):
    return pd.read_csv(filename, header=None)

'''
def segment_gesture(df):
    return_df = pd.DataFrame()
    
    pos_mean = df.iloc[:60]
    pos_std = df.iloc[60:120]
    angle_mean = df.iloc[120:180]
    angle_std = df.iloc[180:240]
    gesture_name = df[240]
    gesture_number = df[241]

    return_df["pos_mean"] = pos_mean.tolist()
    return_df["gesture_name"] = gesture_name

    return return_df
'''

def visualize_points_gesture(df):
    x = df[::3]
    y = df[1::3]
    z = df[2::3]
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    
    ax.scatter(x,y,z) # plot the point (2,3,4) on the figure
    #ax.plot(x,y,z)
    plt.show()

def visualize_labels(df):
    df.value_counts().plot(kind='bar')
    plt.show()
    
def train_decision_tree(train_data, train_labels):
    clf = tree.DecisionTreeClassifier().fit(train_data, train_labels)
    return clf

def train_random_forest(train_data, train_labels):
    clf = RandomForestClassifier(max_depth=13, random_state=0).fit(train_data, train_labels)
    return clf

def train_MLP(train_data, train_labels):
    clf = MLPClassifier(random_state=2, max_iter=3000).fit(train_data, train_labels)
    return clf

def train_SVM(train_data, train_labels):
    clf = svm.SVC(kernel = 'linear',gamma = 'scale', shrinking = False,).fit(train_data, train_labels)
    return clf

def train_kNN(train_data, train_labels):
    neigh = KNeighborsClassifier(n_neighbors=3).fit(train_data, train_labels)
    return neigh


if __name__ == '__main__':
    #Loading data
    data = load_data("train-final.csv")
    test_data = load_data("test-final.csv")
    
    #Removing gestures with missing values
    train_data = data.dropna(axis=0)
    test_data = test_data.dropna(axis=0)
    
    #Preprocessing data for algoritms
    train_labels = train_data[240]
    train_labels_num = train_data[241]
    labels_to_num = dict(zip(train_labels.tolist(), train_labels_num.tolist()))
    train_data = train_data.drop([240, 241], axis=1)
    
    test_labels = test_data[240]
    test_labels_num = test_data[241]
    test_data = test_data.drop([240, 241], axis=1)
    
    #Ploting the points of one gesture from the train dataset
    visualize_points_gesture(train_data.iloc[3])
    visualize_labels(train_labels)
    
    #Using different classifying algoritms and printing their score
    #Decision Tree
    decision_tree = train_decision_tree(train_data, train_labels_num)
    score = decision_tree.score(test_data, test_labels_num)
    print(f"Score from Decision Tree is: {score}")
    
    #Random Forest
    random_forest = train_random_forest(train_data, train_labels_num)
    score = random_forest.score(test_data, test_labels_num)
    print(f"Score from Random forest is: {score}")
    
    #MLP
    MLP = train_MLP(train_data, train_labels_num)
    score = MLP.score(test_data, test_labels_num)
    print(f"Score from MLP is: {score}")
    
    #SVM
    SVM = train_SVM(train_data, train_labels_num)
    score = SVM.score(test_data, test_labels_num)
    print(f"Score from SVM is: {score}")
    
    #kNN
    kNN = train_kNN(train_data, train_labels_num)
    score = kNN.score(test_data, test_labels_num)
    print(f"Score from kNN is: {score}")
   