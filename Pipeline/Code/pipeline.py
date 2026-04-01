import torchvision.transforms as transforms
from ultralytics import YOLO
from PIL import Image
import matlab.engine
import cv2
import os
import cropImage as crop
import activeCountours as AC
import time

def grayscale(inputPath, outputPath):
    img_rgb = Image.open(inputPath)

    img_grayscale = img_rgb.convert('L')
    img_grayscale.save(outputPath)

    print("Grayscale completed")

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''#
##Change according to were you are running the code
investigation_fles_path = "C:/Users/camiz/"
running_path = investigation_fles_path + "Breast_Cancer_Investigation/"

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''#
model = "n"

## Chose the dataset trained YOLOv11 model:
## INbreast YOLO model
#threshold = 0.4
#datasetYOLO = "INbreast"

## CBIS YOLO model
threshold = 0.3
datasetYOLO = "CBIS-DDSM"

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''#
## Chose the test dataset:
#dataset = "INbreastDataset"
#filecoordinatesName = "bbResultcoordinatesIN"

dataset = "CBIS-DDSMDataset"
filecoordinatesName = "bbResultcoordinatesCBIS"

#dataset = "miniMIASDataset"
## Depending on the YOLO model used, chose an option:
#filecoordinatesName = "bbResultcoordinates_usingIN"
#filecoordinatesName = "bbResultcoordinates_usingCBIS"

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''#
## For INbreast and CBIS-DDSM test datasets:
input_original_image_folder = running_path + "Pipeline/" +  dataset + "/OriginalTestImages/"
input_folder_testing_images = running_path + 'Pipeline/' +  dataset + "/InputImages/OriginalTestImages/images/"
output_crop = running_path + "Pipeline/" +  dataset + "/InputImages/PreprocessImages/crop/"
output_gray = running_path + "Pipeline/" +  dataset + "/InputImages/PreprocessImages/grayscale/"
output_filtered = running_path + "Pipeline/" +  dataset + "/InputImages/PreprocessImages/filtered/"
output_AC = running_path + "Pipeline/" +  dataset + "/Results/masks/crop/"
path_wholeMaskOutputAC = running_path + "Pipeline/" +  dataset + "/Results/masks/whole/"

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''#
## For Mini-MIAS test dataset:
#input_original_image_folder = running_path + "Pipeline/" + dataset + "/OriginalTestImages/"
#input_folder_testing_images = running_path + "Pipeline/" + dataset + "/InputImages/OriginalTestImages/images/"
#output_crop = running_path + "Pipeline/" + dataset + "/InputImages/PreprocessImages/" + datasetYOLO + "/crop/"
#output_gray = running_path + "Pipeline/" + dataset + "/InputImages/PreprocessImages/" + datasetYOLO + "/grayscale/"
#output_filtered = running_path + "Pipeline/" + dataset + "/InputImages/PreprocessImages/" + datasetYOLO + "/filtered/"
#output_AC = running_path + "Pipeline/" +  dataset + "/Results/masks/" + datasetYOLO + "/crop/"
#path_wholeMaskOutputAC = running_path + "Pipeline/" + dataset + "/Results/masks/" + datasetYOLO + "/whole/"

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''#
yoloModelPath = running_path + "DetectionModels/runs_test/" + datasetYOLO + "/yolo11" + model + ".pt/test/train/weights/best.pt"
bbCoordDictionary = {}
imagesOriginal = os.listdir(input_folder_testing_images)

# Import the trained YOLO model
model = YOLO(yoloModelPath)

start_time = time.time() 

for imageOriginal in imagesOriginal:

    image_path = os.path.join(input_folder_testing_images, imageOriginal)

    image = Image.open(image_path)
    # Resize a copy of the original image to fit the YOLO mode input
    transform = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])
    YOLO_input_image = transform(image).unsqueeze(0)
    shape = YOLO_input_image.shape
    
    # There are some images that come out of the resize with only 1 channel
    if shape[1] != 3:
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        rgb_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)
        cv2.imwrite(image_path, rgb_image)

        # Do again the resize
        image = Image.open(image_path)
        transform = transforms.Compose([transforms.Resize((640, 640)), transforms.ToTensor()])
        YOLO_input_image = transform(image).unsqueeze(0)

    # Made detections in the resize image with the YOLO model
    detections = model(YOLO_input_image, conf=threshold)
    if len(detections) != 0:
        nameImage = str(imageOriginal).split(".")[0]

        # Obtain the size of the original image
        originalImage = os.path.join(input_original_image_folder, imageOriginal)
        im = cv2.imread(originalImage)
        sizeh, sizew, channels = im.shape
        ratioh, ratiow = sizeh / 640, sizew / 640

        # Obtain the coordinates of the bounding boxes obtained from the YOLO model
        for i in range(len(detections[0].boxes.xyxyn)):
            idImage = nameImage + "_" + str(i + 1)
            cropImage = os.path.join(output_crop, idImage + "_crop.jpg")
            grayImage = os.path.join(output_gray, idImage + "_grayscale.jpg")
            filteredImage = os.path.join(output_filtered, idImage + "_filtered.jpg")

            bbooxModel = detections[0].boxes.xyxyn[i]
            bbooxcoordinatesTensor = [bbooxModel[0], bbooxModel[1], bbooxModel[2], bbooxModel[3]]
            bbooxcoordinatesPython = []

            # Change the bounding box coordinates from tensorflow obtects to float
            for coordinate in bbooxcoordinatesTensor:
                numpy_array = coordinate.numpy()
                python_float = float(numpy_array)
                bbooxcoordinatesPython.append(python_float)

            # Obtain the bounding box coordinates in terms of the original image scale
            top_left_YOLO = (bbooxcoordinatesPython[0] * 640, bbooxcoordinatesPython[1] * 640)
            bottom_right_YOLO = (bbooxcoordinatesPython[2] * 640, bbooxcoordinatesPython[3] * 640)
            top_left_Original = [int(top_left_YOLO[0] * ratiow), int(top_left_YOLO[1] * ratioh)]
            bottom_right_Original = [int(bottom_right_YOLO[0] * ratiow), int(bottom_right_YOLO[1] * ratioh)]

            # Save the bounding box coordinates in a dictionary
            bbCoordDictionary[idImage] = [top_left_Original[0], top_left_Original[1], bottom_right_Original[0], bottom_right_Original[1]]
            
            # Crop the ROI obtained from the detections in the YOLO model from the original image
            crop.crop_and_save_image(originalImage, cropImage, top_left_Original, bottom_right_Original)
            
            # Pass the crop image to grascale
            grayscale(cropImage, grayImage)

            # Pass the image through the Median filter to reduce noise
            eng = matlab.engine.start_matlab()
            eng.filtered_oneImage(grayImage, filteredImage, nargout=0)
            eng.quit()

            # Preform the segmentation with Active Countour models (geodesic and chan-vase)
            image_input_AC = Image.open(filteredImage)
            width, height = image_input_AC.size
            centerx = width / 2
            centery = height / 2
            centerPoint = [centerx, centery]
            AC.pipLine(centerPoint, filteredImage, output_AC)
            print("AC completed")

            path_originalMask = os.path.join(input_original_image_folder, nameImage + ".jpg")
            path_inputACMask = os.path.join(output_AC, "mask_CVBWCF_" + idImage + ".jpg")
            path_outputWholeACMask = os.path.join(path_wholeMaskOutputAC, "mask_CVBWCF_" + idImage + "_final.jpg")

            # Obtain the masks in terms of the original image
            eng = matlab.engine.start_matlab()
            eng.cropToOriginalMaskCV(path_originalMask, path_inputACMask, path_outputWholeACMask, top_left_Original, bottom_right_Original, nargout=0)
            eng.quit()
    else:
        print("No detections were made in " + imageOriginal)

end_time = time.time()
execution_time = end_time - start_time
print(f"Total Pipeline took {execution_time:.4f} seconds to run.")

# Save the bounding box coordinates in a txt file
file = open(running_path + "Pipeline/" + dataset + "/Results/coordinates/" + filecoordinatesName + ".txt", "w")
for key, value in bbCoordDictionary.items():
    file.write("%s:%s\n" % (key, value))
