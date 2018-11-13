import numpy as np
import cv2
import argparse
from sklearn.externals import joblib
from detection import Detection


if __name__ == "__main__":
    np.random.seed(2402)

    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to image")
    ap.add_argument("-m", "--model", required=False, help="Path to the scikit-learn classifier")
    ap.add_argument("-s", "--scaler", required=False, help="Path to the scikit-learn scaler")
    args = vars(ap.parse_args())

    model_path: str = args.get("model", None)
    if not model_path:
        model_path = "./Model/svm.joblib"
    model = joblib.load(model_path)

    scaler_path: str = args.get("scaler", None)
    if not scaler_path:
        scaler_path = "./Model/svm_scaler.joblib"
    scaler = joblib.load(scaler_path)

    image = cv2.imread(args["image"])
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    detect = Detection(model, scaler)
    # Detected barcode image
    barcode: np.ndarray = detect.detect(image, window_size=(150, 250), orientation=180,
                                    pixels_per_cell=(150, 250), cells_per_block=(1, 1),
                                    threshold_proba=0.98, threshold_overlap=0.15)
