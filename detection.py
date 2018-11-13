import numpy as np
from skimage.transform import pyramid_gaussian
from skimage.feature import hog


class Detection:
    def __init__(self, model, scaler=None):
        self.model = model
        self.scaler = scaler
        self.barcode: np.ndarray = None

    def detect(self, image: np.ndarray, window_size: tuple, orientation: int, pixels_per_cell: tuple, cells_per_block: tuple, threshold_proba=0.90, threshold_overlap=0.3) -> list:
        boxes = []
        for scale in pyramid_gaussian(image, max_layer=-1, downscale=1.5, multichannel=True):
            if scale.shape[0] < window_size[0] or scale.shape[1] < window_size[1]:
                break

            for (x, y, window) in Detection.__sliding_window(scale, step_size=20, window_size=window_size):
                # Processing the window here
                gray = scale[y:y+window_size[0], x:x+window_size[1], :]
                gray = Detection.__rgb2gray(gray)

                features: np.ndarray = hog(gray, orientations=orientation, pixels_per_cell=pixels_per_cell,
                                            cells_per_block=cells_per_block, block_norm="L2-Hys", feature_vector=True)
                if features.size == 0:
                    continue

                X = np.expand_dims(features, 0)
                if self.scaler:
                    X = self.scaler.transform(X)

                y_pred_proba = self.model.predict_proba(X)
                y_pred = np.argmax(y_pred_proba)

                if y_pred == 1 and np.max(y_pred_proba) >= threshold_proba:
                    boxes.append((x, y, x+window_size[1], y+window_size[0], np.max(y_pred_proba)))

        boxes_surpressed: np.ndarray = self.__non_max_surpression(boxes, overlapThresh=threshold_overlap, use_proba=True)
        x1: int = np.min(boxes_surpressed[:, 0])
        y1: int = np.min(boxes_surpressed[:, 1])
        x2: int = np.max(boxes_surpressed[:, 2])
        y2: int = np.max(boxes_surpressed[:, 3])

        barcode: np.ndarray = image[y1:y2, x1:x2, :]
        self.barcode = barcode
        return self.barcode

    def __non_max_surpression(self, boxes: np.ndarray, overlapThresh: int, use_proba=True):
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        boxes = np.array(boxes, dtype=np.float)

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        proba = boxes[:, 4]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        if use_proba:
            idxs = np.argsort(proba)

        # keep looping while some indexes still remain in the indexes list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked using the integer data type
        return boxes[pick].astype("int")

    @classmethod
    def __rgb2gray(cls, image: np.ndarray):
        gray: np.ndarray = None
        if len(image.shape) == 2:
            gray = image
        else:
            if np.all(image[:, :, 0] == image[:, :, 1]) and np.all(image[:, :, 0] == image[:, :, 2]):
                gray = image[:, :, 0]
            else:
                R, G, B = image[:, :, 0], image[:, :, 1], image[:, :, 2]
                gray = 0.289 * R + 0.587 * G + 0.114 * B
        return gray

    @classmethod
    def __sliding_window(cls, image: np.ndarray, step_size: int, window_size: tuple):
        # Slide the window across the image
        for y in range(0, image.shape[0], step_size):
            for x in range(0, image.shape[1], step_size):
                # Yield the current window
                yield (x, y, image[y:y+window_size[1], x:x+window_size[0]])
