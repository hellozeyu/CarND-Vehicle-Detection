from skimage.feature import hog
from skimage import color, exposure
from vehicle_detect import *
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
from vehicle_detect import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time

box_history = []

def carDetectionPipeline(image):
    global box_history

    # load the model from the pickle (or cache)
    dist_pickle = pickle.load( open('svc_pickle.p', "rb" ) )
    svc = dist_pickle["svc"]
    X_scaler = dist_pickle["scaler"]
    orient = dist_pickle["orient"]
    pix_per_cell = dist_pickle["pix_per_cell"]
    cell_per_block = dist_pickle["cell_per_block"]
    spatial_size = dist_pickle["spatial_size"]
    hist_bins = dist_pickle["hist_bins"]
    color_space = dist_pickle["color_space"]
    accuracy_score = dist_pickle["accuracy_score"]
    hog_channel = "ALL"
    y_start = 380
    y_stop = 600

    copy_image = np.copy(image)
    copy_image = copy_image.astype(np.float32) / 255
    box_list = find_cars(copy_image, y_start, y_stop, svc, X_scaler, orient, pix_per_cell,
                                    cell_per_block, spatial_size, hist_bins,color_space, hog_channel)

    heat = np.zeros_like(image[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,box_list)

    box_history.append(heat)
    if (len(box_history) > 15):
         del box_history[0]

    heat_sum = np.sum(box_history, axis=0)

    heat_sum = apply_threshold(heat_sum,15)
    heat_sum = np.clip(heat_sum, 0, 255)


    # Find final boxes from heatmap using label function
    labels = label(heat_sum)
    draw_img_sum = draw_labeled_bboxes(image, labels)

    return draw_img_sum
