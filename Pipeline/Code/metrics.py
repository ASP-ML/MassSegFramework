import cv2
import os
import numpy as np
from scipy.spatial.distance import cdist
from tensorflow.keras import backend as K

smooth = 100

def dice_coef(y_true, y_pred):
    y_truef = K.flatten(y_true)
    y_predf = K.flatten(y_pred)
    And = K.sum(y_truef * y_predf)
    return ((2 * And + smooth) / (K.sum(y_truef) + K.sum(y_predf) + smooth))

def iou(y_true, y_pred):
    intersection = K.sum(y_true * y_pred)
    sum_ = K.sum(y_true + y_pred)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def get_contour_points(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return np.empty((0, 2))
    points = np.vstack(contours).squeeze()
    if points.ndim == 1:
        points = np.expand_dims(points, axis=0)
        
    return points

def hd95(y_true, y_pred):
    y_true = np.array(y_true, dtype=np.uint8)
    y_pred = np.array(y_pred, dtype=np.uint8)
    # Extract contour points from both masks
    true_points = get_contour_points(y_true)
    pred_points = get_contour_points(y_pred)

    # Handles cases where there are no points in either mask
    if len(true_points) == 0 or len(pred_points) == 0:
        return np.nan
    
    distances = cdist(true_points, pred_points)
    d_true_pred = np.min(distances, axis=1)
    d_pred_true = np.min(distances, axis=0)

    # Using the 95th percentile to calculate the Hausdorff distance
    hd95_1 = np.percentile(d_true_pred, 95)
    hd95_2 = np.percentile(d_pred_true, 95)

    return max(hd95_1, hd95_2)

def adjust_data(mask):
    mask = mask / 255
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return (mask)

def metrics(input_folder_gt, input_folder_pl, output_folder, name):
    masks = os.listdir(input_folder_pl)
    imgIdArray = []
    iouArray = []
    diceArray = []
    hd95Array = []

    for mask in masks:
        name_1 = input_folder_pl + mask
        gray1 = cv2.imread(name_1, cv2.IMREAD_GRAYSCALE)
        ret, img1 = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)
        imgPL = adjust_data(img1)

        mask_id = str(mask).split('F_')[1].split('_f')[0]

        name_2 = input_folder_gt + mask_id + '.jpg'
        gray2 = cv2.imread(name_2, cv2.IMREAD_GRAYSCALE)
        ret, img2 = cv2.threshold(gray2, 127, 255, cv2.THRESH_BINARY)
        imgGT = adjust_data(img2)

        imgIdArray.append(mask_id)
        iouArray.append((iou(imgGT, imgPL)).numpy())
        diceArray.append((dice_coef(imgGT, imgPL)).numpy())
        hd95Array.append(hd95(imgGT, imgPL))

    file_name= 'Masks_ID_' + name
    output_file = os.path.join(output_folder, file_name + '.txt')

    file_name_1 = 'IOU_CV_' + name
    output_file_1 = os.path.join(output_folder, file_name_1 + '.txt')

    file_name_2 = 'DICE_CV_' + name
    output_file_2 = os.path.join(output_folder, file_name_2 + '.txt')

    file_name_3 = 'HD95_CV_' + name
    output_file_3 = os.path.join(output_folder, file_name_3 + '.txt')

    with open(output_file, 'w') as file:
        for id in imgIdArray:
            file.write(str(id) + "\n")

    with open(output_file_1, 'w') as file:
        for metricI in iouArray:
            file.write(str(metricI) + "\n")

    with open(output_file_2, 'w') as file:
        for metricD in diceArray:
            file.write(str(metricD) + "\n")

    with open(output_file_3, 'w') as file:
        for metricH in hd95Array:
            file.write(str(metricH) + "\n")

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''#
##Change according to were you are running the code
investigation_fles_path = 'C:/Users/camiz/'
running_path = investigation_fles_path + 'Breast_Cancer_Investigation/Pipeline/'

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''#
##Choose a dataset to calculate the metrics on:

##INbreastDataset
#dataset = 'INbreastDataset'
#name = 'INb'

##CBIS-DDSMDataset
dataset = 'CBIS-DDSMDataset'
name = 'CBIS'

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''#
##For calculating the metrics
input_folder_gt = running_path + dataset + '/InputImages/OriginalTestImages/groundTrue/'
input_folder_pl = running_path + dataset + '/Results/joinedMasks/'
output_folder = running_path + dataset + '/Results/metrics/'
metrics(input_folder_gt, input_folder_pl, output_folder, name)