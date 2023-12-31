import logging
# for some reasons, sklearn needed to be imported before other packages
from sklearn.cluster import MiniBatchKMeans

import cv2
import torch
import base64
import numpy as np
from typing import List

def setup_logger():
    FORMAT = logging.Formatter(
    '%(asctime)s | %(levelname)-8s | %(filename)s:%(lineno)-3d | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    print(f'Created logger with name {__name__}')
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setFormatter(FORMAT)
    logger.addHandler(ch)
    return logger

def bin_mask_from_cls_idx(full_mask: torch.Tensor, cls_idx_list: List[int]) -> torch.Tensor:
    mask = full_mask.clone()
    for i in cls_idx_list:
        mask[mask==i] = 255
    mask[mask!=255] = 0
    return mask

def array_to_encoded_str(image: np.ndarray) -> str:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    result, byte_data = cv2.imencode(".jpg", image, encode_param)
    img_str = base64.encodebytes(byte_data).decode('utf-8')
    return img_str

def quantize_img_color(img: np.ndarray, n_clusters: int=16) -> np.ndarray:
    image = img.copy()
    (h, w) = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # reshape the image into a feature vector so that k-means
    # can be applied
    image = image.reshape((image.shape[0] * image.shape[1], 3))
    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters = n_clusters)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]
    # reshape the feature vectors to images
    quant = quant.reshape((h, w, 3))
    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    return quant

def cartoonify_img(image: np.ndarray, option: str='disable', cartoonifier = None) -> np.ndarray:
    if option == 'opencv':
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # applying median blur to smoothen an image
        smooth_gray = cv2.medianBlur(gray, 5)
        # retrieving the edges for cartoon effect
        edge_img = cv2.adaptiveThreshold(smooth_gray, 255, 
                                        cv2.ADAPTIVE_THRESH_MEAN_C, 
                                        cv2.THRESH_BINARY, 15, 9)
        # applying bilateral filter to remove noise
        color_img = cv2.bilateralFilter(image, 9, 50, 50)
        color_img = quantize_img_color(color_img)
        # masking edged image with our "BEAUTIFY" image
        cartoonified = cv2.bitwise_and(color_img, color_img, mask=edge_img)
        return cartoonified
    elif option == 'cartoongan' and cartoonifier is not None:
        ori_size = image.shape[:2]
        cartoonified = cartoonifier.predict(image)
        cartoonified = cv2.resize(cartoonified, (ori_size[1], ori_size[0]))
        return cartoonified
    else:
        return image