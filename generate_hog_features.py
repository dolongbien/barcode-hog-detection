import numpy as np
import argparse
from skimage.feature import hog
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    np.random.seed(2402)

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", required=False, help="Path to load positive_images.npy and negative_image.npy")
    ap.add_argument("-n", "--name", required=False, help="Path to save X.npy and y.npy")

    args = vars(ap.parse_args())

    if args.get("path"):
        DIR_DATA = (args["path"])
    else:
        DIR_DATA = "./Data/"

    print("Generate HOG features for positive set")
    positives = np.load(DIR_DATA + "positive_images.npy")
    features_list = []
    for i in range(positives.shape[0]):
        print("Generate HOG feature for image", i+1)
        image = positives[i, :, :]
        features, hog_image = hog(image, orientations=180, pixels_per_cell=(150, 250),
                        cells_per_block=(1, 1), block_norm="L2-Hys", visualize=True)
        features_list.append(features)
    X_pos = np.array(features_list)
    y_pos = np.ones(X_pos.shape[0])

    print("Generate HOG features for negative set")
    negatives = np.load(DIR_DATA + "positive_images.npy")
    features_list = []
    for i in range(negatives.shape[0]):
        print("Generate HOG feature for image", i+1)
        image = negatives[i, :, :]
        features, hog_image = hog(image, orientations=180, pixels_per_cell=(150, 250),
                        cells_per_block=(1, 1), block_norm="L2-Hys", visualize=True)
        features_list.append(features)
    X_neg = np.array(features_list)
    y_neg = np.zeros(X_neg.shape[0])

    X = np.concatenate((X_pos, X_neg))
    y = np.concatenate((y_pos, y_neg))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test)

    print("Train set:", X_train.shape, y_train.shape)
    print("Validate set:", X_val.shape, y_val.shape)
    print("Test set:", X_test.shape, y_test.shape)

    # print("Saving X and y")
    # np.save(DIR_DATA + "X_train", X_train)
    # np.save(DIR_DATA + "y_train", y_train)
    # np.save(DIR_DATA + "X_val", X_val)
    # np.save(DIR_DATA + "y_val", y_val)
    # np.save(DIR_DATA + "X_test", X_test)
    # np.save(DIR_DATA + "y_test", y_test)
