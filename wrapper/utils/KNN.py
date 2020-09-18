import cv2
import numpy as np

def get_colors(src_img,K):
    """
    Sample of colors from image using cv2 K-means color extraction.
    """
    Z = src_img.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret,label,center = cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    # Dominant colors list from kmeans calc
    colors = np.uint8(center)
    return colors