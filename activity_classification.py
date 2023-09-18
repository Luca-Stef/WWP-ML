# 0 - standing
# 1 - walking
# 2 - turning
# 3 - sitting
# 4 - other
# 5 - not in use (faulty data)

## ABSTRACT PIPELINE ##
# Unpack json into dataframe of shape (measurements, features) -> impute -> drop duplicates -> drop bad data -> (checkpoint) 
# -> Rescale sensor readings -> Create sliding set of windows each labeled with an activity associated with the final or modal 
# measurement in the window. Must not create discontinuities by creating any one window across different files. This is why
# data from different files must be kept separate until windows are created. After that they can all be combined into the 
# training and test data.
# -> Compile all windows from all .js files into one dataset -> One-Hot encode activity label -> train-test split and 
# shuffle windows -> define model -> fit model -> evaulate on test set -> plot loss and confusion -> save model -> 
# convert to tflite/c file (tflite includes additional optimisation using float quantisation)

## HEURISTICS ##
# Average time between readings is 19.5 ms, 50 readings is roughly a second of real time
# A cadence of 99 steps/min means ~606 ms (31 measurements) per step so 60 measurements could roughly correspond to one 
# stride/gait cycle. Average duration of a gait phase is 150 ms (~7.5 measurements)
# 100 measurements could correspond to ~2 seconds of usage.
# Model memory size is input size + weights size = 8*180*4/5 + 621*4 = 3636 bytes

# Current best is 97.5% accuracy with a 316 parameter LSTM trained for 40 epochs with a window size of 180 and down sample 
# factor of 5. More training time and more parameters should help slightly.

## TODO ## 
# There are many improvements to be made. One might play around blindly with the preprocessing and feature engineering 
# (crucial in the pipeline), and look for changes in accuracy and memory. Also can tweak the topology of the neural network. 
# There is still some reading to be done, I made notes but to me it is still unclear what the best possible solution is.
# Some suggestions for future work: feature engineering improvements, hybrid approach to classification (machine learning 
# combined with the old threshold methods), try more tuning, add staggered windows, add moving average for each window, 
# add recurrent neural network as input to another neural network which combines other summary features of windows.

import json
import os
import pandas as pd
import scipy
import numpy as np
import subprocess as sp
from sklearn import svm, tree
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
import xgboost as xgb
from tensorflow import keras
import tensorflow as tf
from micromlgen import port
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

prefeatures = []

class_domains = ['segmClass.classFlag', 'segmClass.stepFlag', 'segmClass.fase',
       'segmClass.IC', 'segmClass.MS_end', 'segmClass.MS_start',
       'segmClass.TO', 'segmClass.IC_prev']

features = ['accx', 'accy', 'accz', "ps", 'quat0', 'quat1', 'quat2', 'quat3']

window_length = 180
sample_rate = 5
stagger = 0
onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
scaler = RobustScaler()

def create_windows(X, y, window_length=20, shift=1, sample_rate=1, stagger=0):
    """
    Create windows of length window_length. shift is the interval between each window and sample rate is the interval 
    between each reading within a window. For example you can use a window of 60 readings but downsample by 2 so you 
    effectively get only 30 measurements in the same period of time. Stagger is the space between two windows if 
    choosing to use staggered pairs of windows.
    """
    if window_length == 1:
        return X, y
    
    Xs, ys = [], []
    if not stagger == 0:
        for i in range(0, len(X) - window_length - stagger, shift):
            Xs.append(np.r_[X.iloc[i:i + window_length:sample_rate],X.iloc[i + stagger:i + window_length + stagger:sample_rate]])
            ys.append(scipy.stats.mode(y.iloc[i: i + window_length + stagger].values)[0])

    elif stagger == 0:
        for i in range(0, len(X) - window_length, shift):
            Xs.append(X.iloc[i:(i + window_length):sample_rate].values)
            ys.append(scipy.stats.mode(y.iloc[i: i + window_length].values)[0])

    return np.array(Xs), np.array(ys).reshape(-1, 1)

def preprocess(scaler, window_length):
    """
    Preprocessing with a checkpoint to avoid repeating redundant computations. First Unpack json into dataframe of 
    shape (measurements, features) -> impute -> drop duplicates -> drop bad data -> Rescale sensor readings. Then save as
    separate csv files. This stage is always the same. Then read in csv files and create sliding windows for each file. 
    You can play around with parameters of the sliding windows without having to repeat the first stage of preprocessing.
    """
    
    files = sp.getoutput('ls data/*.js').split("\n")
    data = []
    windows = []
    labels = []
    
    for file in files:
        
        if os.path.exists(f"{file[:-3]}.csv"): # This part skips the creation of the intermediate .csv files.
            print(f"Loading {file[:-3]}.csv")
            data.append(pd.read_csv(file[:-3] + ".csv"))

        else: # This part loads json and saves an intermediate .csv with clean label to measurement correspondence. This is before window creation.
            print(f"Loading {file}")
            with open(file, 'r') as read_file:
                raw_data = json.loads(read_file.read())

            df = pd.json_normalize(raw_data["data"], max_level=1)
            
            # Drop preceding labels with no corresonding measurements
            first_valid_index = df["semiRawImuPs.acc"].first_valid_index()
            df = df.iloc[df.index >= first_valid_index]

            dfacc = df["semiRawImuPs.acc"].dropna().apply(pd.Series).rename(columns={"x": "accx", "y": "accy", "z": "accz"})
            df = df.drop("semiRawImuPs.acc", axis=1).assign(**dfacc)

            dfgyro = df["semiRawImuPs.gyro"].dropna().apply(pd.Series).rename(columns={"x": "gyrox", "y": "gyroy", "z": "gyroz"})
            df = df.drop("semiRawImuPs.gyro", axis=1).assign(**dfgyro)

            dfps = df["semiRawImuPs.ps"].dropna().apply(pd.Series).rename(columns={k:"ps"+str(k) for k in range(9)})
            df = df.drop("semiRawImuPs.ps", axis=1).assign(**dfps)

            dfquat = df["semiRawImuPs.quat"].dropna().apply(pd.Series).rename(columns={k:"quat"+str(k) for k in range(4)})
            df = df.drop("semiRawImuPs.quat", axis=1).assign(**dfquat)

            # Drop redundant segmClass.ts and tempParam features
            df["ts"] = df["ts"].combine_first(df["segmClass.ts"])
            df.drop("segmClass.ts", axis=1, inplace=True)
            df = df.drop(['tempParam.NS', 'tempParam.CD', 'tempParam.CD_std', 'tempParam.CD_sumx2', 'tempParam.Swing', 'tempParam.Pushing', 'tempParam.FF'], axis=1)
            
            df["ps"] = df[['ps0', 'ps1', 'ps2', 'ps3', 'ps4', 'ps5', 'ps6', 'ps7', 'ps8']].sum(axis=1)
            df.loc[:,"segmClass.classFlag"] = df.loc[:,"segmClass.classFlag"].bfill()
            df.dropna(subset=features, inplace=True)
            df = df[~(df['segmClass.classFlag'] == 5)]

            df.to_csv(file[:-3] + ".csv")
            data.append(df)
    
    scaler = scaler.fit(pd.concat(data, axis=0)[features])
    
    for df in data:
        df[features] = scaler.transform(df[features])
        X, y = create_windows(df[features], df['segmClass.classFlag'], window_length=window_length, sample_rate=sample_rate, stagger=stagger)
        windows.append(X); labels.append(y)
    
    X = np.concatenate(windows)
    y = np.concatenate(labels)
    
    return X, ((y == 1) | (y == 2))

def feature_engineer(windows):
    """
    If given a set of windows or staggered pairs of windows, calculate summary statistics on each window. 
    """
    if not stagger == 0:
        window1 = windows[:,:window_length,:]
        window2 = windows[:,window_length:,:]

        mean1 = windows1.mean(axis=1)
        std1 = windows1.std(axis=1)
        spread1 = windows1.max(axis=1) - windows1.min(axis=1)
        diff1 = windows1[:,-1,:] - windows1[:,0,:]
        curv1 = 2*windows1[:,windows1.shape[1]//2,:] - windows1[:,0,:] - windows1[:,-1,:]

        mean2 = windows2.mean(axis=1)
        std2 = windows2.std(axis=1)
        spread2 = windows2.max(axis=1) - windows2.min(axis=1)
        diff2 = windows2[:,-1,:] - windows2[:,0,:]
        curv2 = 2*windows2[:,windows2.shape[1]//2,:] - windows2[:,0,:] - windows2[:,-1,:]

        return np.concatenate((mean1, std1, spread1, diff1, curv1, mean2, std2, spread2, diff2, curv2), axis=1)

    else:
        mean = windows.mean(axis=1)
        std = windows.std(axis=1)
        spread = windows.max(axis=1) - windows.min(axis=1)
        diff = windows[:,-1,:] - windows[:,0,:]
        curv = 2*windows[:,windows.shape[1]//2,:] - windows[:,0,:] - windows[:,-1,:]
        return np.concatenate((mean, std, spread, diff, curv), axis=1)

def perceptron_model():
    """
    Define multi layer perceptron model and topology. If you want to change from binary to multi-class classification you 
    need to change the loss from binary crossentropy to categorical crossentropy, as well as the output shape and the 
    activation of the final layer.
    """
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=5, input_shape=[X_train.reshape(X_train.shape[0], -1).shape[1]], activation='relu'))
    model.add(keras.layers.Dense(units=5, activation='relu'))
    model.add(keras.layers.Dense(y_train.reshape(-1,1).shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def fit_evaluate_perc(X_train, y_train, X_test, y_test, name):
    """
    Fit training data for 20 epochs. For a final model please change this to at least 100 epochs to ensure it converges to maximum
    possible accuracy. Also does inference on test set, and uses the output and the ground truth to calculate accuracy and plot confusion.
    Also plots training and validation loss as a function of number of iterations.
    """

    history = model.fit(X_train.reshape(X_train.shape[0], -1), y_train.reshape(-1,1), epochs=20, validation_split=0.1, batch_size=64)
    loss, accuracy = model.evaluate(X_test.reshape(X_test.shape[0], -1), y_test.reshape(-1,1))
    y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))

    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.title(f"Final loss {loss}")
    plt.savefig(f"figures/{name}_loss.png")

    cm = confusion_matrix(y_test, np.round(y_pred))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure()
    cm_display.plot()
    cm_display.ax_.set_title(f"Accuracy {accuracy}")
    plt.savefig(f"figures/{name}_cm.png")
    print(f"Trained, evaulated {name} model and created confusion matrix at figures/{name}_cm.png\n")

def LSTM_model():
    """
    Define LSTM model and topology. If you want to change from binary to multi-class classification you need to change the loss
    from binary crossentropy to categorical crossentropy, as well as the output shape and the activation of the final layer.
    """
    model = keras.Sequential()
    #model.add(keras.layers.Bidirectional(keras.layers.LSTM(units=5, input_shape=[X_train.shape[1], X_train.shape[2]])))
    model.add(keras.layers.LSTM(units=5, input_shape=[X_train.shape[1], X_train.shape[2]]))
    model.add(keras.layers.Dropout(rate=0.5))
    model.add(keras.layers.Dense(units=5, activation='relu'))
    model.add(keras.layers.Dense(y_train.shape[1], activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    return model

def fit_evaluate_LSTM(X_train, y_train, X_test, y_test, name):
    """
    Fit training data for 20 epochs. For a final model please change this to at least 100 epochs to ensure it converges to maximum
    possible accuracy. Also does inference on test set, and uses the output and the ground truth to calculate accuracy and plot confusion.
    Also plots training and validation loss as a function of number of iterations.
    """

    history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1)
    loss, accuracy = model.evaluate(X_test, y_test)
    y_pred = model.predict(X_test)

    plt.figure()
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.title(f"Final loss {loss}")
    plt.savefig(f"figures/{name}_loss.png")

    cm = confusion_matrix(y_test, np.round(y_pred))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure()
    cm_display.plot()
    cm_display.ax_.set_title(f"Accuracy {accuracy}")
    plt.savefig(f"figures/{name}_cm.png")
    print(f"Trained, evaulated {name} model and created confusion matrix at figures/{name}_cm.png\n")

def evaluate(model_name, y_pred, y_test):
    """
    Calculate accuracy and plot confusion matrix for a set of predictions and ground truths. Sometimes a model spits out
    raw probabilities instead of predictions. For binary classification a lot of models will do y_pred > 0.5 but with 
    multi-class you may not have any probability over 0.5 for a given example, some models will fail to give a prediction 
    in this case. For example with probabilities [0.2,0.35,0.45] there is no class over 0.5 so the model may output [0,0,0]. 
    In this case you have to manually extract the maximum probability if that's what you want to interpret as the output.
    """
     
    if len(y_pred.shape) == 2:
        if y_pred.shape[1] == 6:
            y_pred = np.argmax(y_pred, axis=-1)

    accuracy = accuracy_score(y_test, y_pred.reshape(y_pred.shape[0],-1))
    cm = confusion_matrix(y_test, y_pred.reshape(y_pred.shape[0],-1))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure()
    cm_display.plot()
    cm_display.ax_.set_title(f"Accuracy {accuracy}")
    plt.savefig(f"figures/{model_name}_cm.png")
    print(f"Trained, evaulated {model_name} model and created confusion matrix at figures/{model_name}_cm.png\n")

def to_tflite(model, name, n_params, accuracy):
    """ 
    Save the model, convert to tflite, and save it as tflite format as well. There is also some memory optimisation.
    If you want the model as a C array, you can do `xxd -i model.tflite > model.cc`. The alternative to tflite is to save 
    a .h5 format model and then convert to .c and .h files using keras2c and also copy in the includes from the keras2c github.
    """
    model.save(f'models/{name}_{n_params}_{accuracy}')
    converter = tf.lite.TFLiteConverter.from_saved_model(f'models/{name}_{n_params}_{accuracy}') # path to the SavedModel directory
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_lower_tensor_list_ops = False
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    tflite_model = converter.convert()
    with open(f"models/{name}_{n_params}_{accuracy}.tflite", 'wb') as f:
        f.write(tflite_model)

## PREPROCESSING ##
X, y = preprocess(scaler, window_length)
onehot = onehot.fit(y.reshape(-1,1)) # Some models require onehot encoding for multiple classes!! For binary classification this is never necessary
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print(f"Proceeding with training data of shape {X_train.shape}")


## MODELS SANDBOX ##
"""# Support vector machine
model = svm.SVC().fit(X_train.reshape(X_train.shape[0], -1), y_train)
y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
evaluate("SVM", y_pred, y_test)

# single decision tree
model = tree.DecisionTreeClassifier(max_depth=8).fit(X_train.reshape(X_train.shape[0], -1), onehot.transform(y_train))
y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
evaluate("Tree", y_pred, y_test); print(model.tree_.node_count, " nodes ", model.tree_.max_depth, "depth")
with open('models/tree.h', 'w') as file:
    file.write(port(model))

# tensor flow random forest (conflicts with main tensorflow library so use a different evironment) 
model = tfdf.keras.RandomForestModel(num_trees=3, max_num_nodes=53, growing_strategy="BEST_FIRST_GLOBAL")
history = model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
evaluate("Forest", y_pred, y_test)

# gradient boosting from sklearn, gradient boost is somewhat similar to random forest
model = GradientBoostingClassifier(verbose=1).fit(X_train.reshape(X_train.shape[0], -1), y_train)
y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
evaluate("Boost", y_pred, y_test)

# Histogram based gradient boosting from sklearn
model = HistGradientBoostingClassifier(verbose=1).fit(X_train.reshape(X_train.shape[0], -1), y_train)
y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
evaluate("HistBoost", y_pred, y_test)

# extreme gradient boost
model = xgb.XGBClassifier(sampling_method="gradient_based", verbosity=2).fit(X_train.reshape(X_train.shape[0], -1), y_train)
y_pred = model.predict(X_test.reshape(X_test.shape[0], -1))
evaluate("xgb", y_pred, y_test)"""
breakpoint()
# long-short term memory is a recurrent neural network architecture, there are other times of RNN which may also be good
model = LSTM_model()
fit_evaluate_LSTM(X_train, y_train, X_test, y_test, "LSTM")

# perceptron is the simplest neural network, a single layer perceptron is almost the same as least squares regression 
model = perceptron_model()
fit_evaluate_perc(X_train, y_train, X_test, y_test, "Perceptron")