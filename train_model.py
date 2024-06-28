import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
import tensorflow as tf

def train_model(X_train, y_train):
   num_features = len(X_train[0])
   epochs = 10 # 100
   dropout_prob = 0.2 # 0
   lr = 0.001 # 0.01, 0.005
   num_nodes = 16 # 32, 64
   batch_size = 128 # 32, 64
   
   # based on classification model example: colab.research.google.com/drive/16w3TDn_tAku17mum98EWTmjaLHAJcsk0
   nn_model = tf.keras.Sequential([
       tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(num_features,)),
       tf.keras.layers.Dropout(dropout_prob),
       #tf.keras.layers.Dense(num_nodes, activation='relu'),
       #tf.keras.layers.Dropout(dropout_prob),
       tf.keras.layers.Dense(1, activation='sigmoid')
   ])

   nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy', metrics=['accuracy'])

   return nn_model


def train_data():
    cols = ["z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7", "class"]
    df = pd.read_csv("output/combined_results.data", names=cols)
    #cols = ["left_z0", "left_z1", "left_z2", "left_z3", "left_z4", "left_z5", "left_z6", "left_z7",
    #        "right_z0", "right_z1", "right_z2", "right_z3", "right_z4", "right_z5", "right_z6", "right_z7", 
    #        "r","g","b",  "ratio",   "class"]
    df = pd.read_csv("output/combined_results_more_features.data", names=cols)
    label1 = "dolphin"
    label2 = "shark"
    df["class"] = (df["class"] == "d").astype(int) # d for dolphin

    for label in cols[:-1]:
        plt.hist(df[df["class"]==1][label], color="blue", label=label1, alpha=0.7, density=True)
        plt.hist(df[df["class"]==0][label], color="red", label=label2, alpha=0.7, density=True)
        plt.title(label)
        plt.ylabel("Probability")
        plt.xlabel(label)
        plt.legend()  
        #plt.show()     

    train, test = np.split(df.sample(frac=1), [int(0.8*len(df))])

    # clean up data
    scaler = StandardScaler() # removes the mean and scales the data to unit variance
    X_train = train[train.columns[:-1]].values
    y_train = train[train.columns[-1]].values
    X_train = scaler.fit_transform(X_train)
    ros = RandomOverSampler()
    X_train, y_train = ros.fit_resample(X_train, y_train)

    X_test = test[test.columns[:-1]].values
    y_test = test[test.columns[-1]].values
    X_test = scaler.fit_transform(X_test)

    # KNN model
    #knn_model = KNeighborsClassifier(n_neighbors=5)
    #knn_model.fit(X_train, y_train)
    #y_pred = knn_model.predict(X_test)
    #print(" ****** KNN classification report ****** ")
    #print(classification_report(y_test, y_pred))
    
    # Neural network model
    model = train_model(X_train,y_train)
    y_pred = model.predict(X_test)

    y_pred = (y_pred > 0.5).astype(int).reshape(-1,)
    print(" ****** Neural Net classification report ****** ")
    print(classification_report(y_test, y_pred))
    print(model.evaluate(X_test, y_test))

if __name__ == '__main__':
    train_data()