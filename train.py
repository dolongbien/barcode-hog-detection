import numpy as np
import matplotlib.pyplot as plt
import itertools
import argparse
import pickle
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


if __name__ == "__main__":
    np.random.seed(2402)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--load-path", required=False, help="Path to the directory containing train, validate and test set")
    ap.add_argument("-n", "--name", required=False, help="Path to save model")

    args = vars(ap.parse_args())

    if args.get("load-path"):
        DIR_DATA = (args["load-path"])
    else:
        DIR_DATA = "./Data/"

    X_train = np.load(DIR_DATA + "X_train.npy")
    y_train = np.load(DIR_DATA + "y_train.npy")

    X_val = np.load(DIR_DATA + "X_val.npy")
    y_val = np.load(DIR_DATA + "y_val.npy")

    X_test = np.load(DIR_DATA + "X_test.npy")
    y_test = np.load(DIR_DATA + "y_test.npy")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    svm = SVC(C=9.0, gamma=0.001, probability=True)
    svm.fit(X_train_scaled, y_train)

    y_pred = svm.predict(X_val_scaled)
    print("Accuracy on the validation set:", accuracy_score(y_val, y_pred))
    plot_confusion_matrix(confusion_matrix(y_pred, y_val), classes=[0, 1], title='Precision-Recal on validation set')

    y_pred = svm.predict(X_test_scaled)
    print("Accuracy on the test set:", accuracy_score(y_test, y_pred))
    plot_confusion_matrix(confusion_matrix(y_pred, y_test), classes=[0, 1], title='Precision-Recal on test set')

    if args.get("name", None):
        name = args["name"]
        joblib.dump(svm, name + '.joblib')
        joblib.dump(scaler, name + "_scaler.joblib")
