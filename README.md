# MassSeg-Framework: A Breast Mass Detection and Segmentation Framework Based on Deep Learning and Active Contour Model

## Project Objective:
Implement MassSeg, a new automatic two-step breast mass segmentation method that combines the YOLOv11 architecture with a Chan-Vesse active contour model to maximize the lesion’s detection and enhance the lesion’s segmentation on mammography images.

## Table of Contents:
- [Installation](#installation)
    - [MATLAB Setup](#matlab-setup)
    - [Setup Running Environment](#setup-running-environment)
- [Project Structure](#project-structure)
- [Running the Pipeline](#running-pipeline)
    - [Step 1: Main Pipeline](#step-1-main-pipeline)
    - [Step 2: Mask Post-processing](#step-2-mask-post-processing)
    - [Step 3: Metrics Computation](#step-3-metrics-computation)
    - [Step 4: Results Analysis](#step-4-results-analysis)
- [Datasets](#datasets)
- [Training YOLO Models (Google Colab)](#train-yolo)
- [Troubleshooting](#troubleshooting)

## Installation
### Prerequisites
- Python 3.9
- Conda
- Your preferred IDE (Visual Studio Code recommended)
- MATLAB R2024 with these add-ons:
    - Image Processing Toolbox
    - MATLAB Coder
    - MATLAB Compiler
    - MATLAB SDK
    - MATLAB Test
 
### Clone the repository
```bash
git clone https://github.com/ASP-ML/MassSeg.git
cd MassSeg
```

### MATLAB Setup
1. Install the MATLAB Add-On for Python Integration
    - Open MATLAB.
    - Click on Add-Ons, then on Get Add-Ons.
    - Search for "Using MATLAB with Python" and install the version provided by Sebastian Castro.
2. Set Up MATLAB Path
    - In MATLAB, click Set Path.
    - Click Add Folder, then add the path to the Pipeline Matlab folder (e.g., C:\Users\user\MassSeg\Pipeline\Matlab).
    - Select the newly added path, click Move to Top, then click Save.

### Setup Running Environment
1. Setup the Python Environment
    - In your IDE's terminal (e.g., Visual Studio Code), run the install_conda_PL.ps1 script.   
2. Install MATLAB Engine API for Python
    - Follow the official [MATLAB Engine API installation steps](https://la.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
        - Before doing the installation, navigate to the MATLAB installation directory using the terminal (Windows-Specific Command). Be sure to be in the C:\ folder: 
            ```bash
            $ cd '.\Program Files\MATLAB\R2024a'
            ```
        - Then, when installing, replace in the command "matlabroot" with "." (ex., below): 
            ```bash
            $ cd '.\extern\engines\python'
            ```
        - Install engine API version 24.1.2 (ensure the BCIenv environment is activated in your IDE's terminal before running the command): 
            ```bash
            $ python -m pip install matlabengine==24.1.2
            ```
## Project Structure
```
MassSeg/  
├── DetectionModels/  
├── Pipeline/  
│   ├── CBIS-DDSMDataset/  
│   │   ├── InputImages/  
│   │   │   ├── OrignalTestImages/   
│   │   │   │   ├── groundTrue/     # Ground truth mask  
│   │   │   │   └── images/         # Original mammograms used for the pipeline  
│   │   │   ├── PreprocessImages/    
│   │   │   │   ├── crop/           # Crop images of the mass detected  
│   │   │   │   ├── filtered/       # Crop images after the CLAHE filter  
│   │   │   │   └── grayscale/      # Crop filtered images converted into grayscale  
│   │   ├── OriginalTestImages/     # Original mammograms use as ground truth  
│   │   ├── Results/  
│   │   │   ├── coordinates/        # Detected bounding boxes coordinates 
│   │   │   ├── joinedMasks/        # Combined segmented mass masks the size of the original mammogram images   
│   │   │   ├── masks/              # Individual segmented mass masks  
│   │   │   │  ├── crop/            # Crop segmented mass masks  
│   │   │   │  └── whole/           # Segmented mass masks the size of the original mammogram images 
│   │   │   └── metrics/            # Evaluation metrics (DICE, IOU, HD95) 
│   │   └── TestLabels/             # Ground truth bounding boxes coordinates
│   ├── Code/    
│   │   ├── activeCountours.py    
│   │   ├── pipeline.py  
│   │   ├── cropImage.py  
│   │   ├── joinMasks.py  
│   │   ├── metrics.py  
│   │   ├── morphsnakes_v1.py  
│   │   ├── morphsnakes.py  
│   │   ├── pipeline.py  
│   │   ├── resultAnalysis.ipynb  
│   │   ├── setup.py  
│   │   └── test_morphsnakes.py  
│   ├── INbreastDataset/  
│   │   ├── InputImages/  
│   │   │   ├── OrignalTestImages/   
│   │   │   │   ├── groundTrue/  
│   │   │   │   └── images/  
│   │   │   ├── PreprocessImages/    
│   │   │   │   ├── crop/  
│   │   │   │   ├── filtered/  
│   │   │   │   └── grayscale/  
│   │   ├── OriginalTestImages/  
│   │   ├── Results/  
│   │   │   ├── coordinates/  
│   │   │   ├── joinedMasks/  
│   │   │   ├── masks/  
│   │   │   │  ├── crop/  
│   │   │   │  └── whole/  
│   │   │   └── metrics/  
│   │   └── TestLabels/  
│   ├── Matlab/  
│   │   ├── cropToOriginalMaskCV.m  # Script to resize the segmented mask to its original size 
│   │   └── filtered_oneImage.m  
│   ├── miniMIASDataset/  
│   │   ├── InputImages/    
│   │   │   ├── OrignalTestImages/   
│   │   │   │   ├── images/  
│   │   │   │   └── massInfo.txt  
│   │   │   ├── PreprocessImages/    
│   │   │   │   ├── CBIS-DDSM/      # Crop images of the mass detected using the model trained with the CBIS-DDSM dataset
│   │   │   │   │   ├── crop/  
│   │   │   │   │   ├── filtered/  
│   │   │   │   │   └── grayscale/  
│   │   │   │   ├── INbreast/       # Crop images of the mass detected using the model trained with the INbreast dataset
│   │   │   │   │   ├── crop/  
│   │   │   │   │   ├── filtered/  
│   │   │   │   │   └── grayscale/  
│   │   ├── OriginalTestImages/  
│   │   ├── Results/  
│   │   │   ├── coordinates/  
│   │   │   ├── joinedMasks/  
│   │   │   │   ├── CBIS-DDSM/  
│   │   │   │   └── INbreast/    
│   │   │   ├── masks/  
│   │   │   │   ├── CBIS-DDSM/  
│   │   │   │   │   ├── crop/  
│   │   │   │   │   └── whole/   
│   │   │   │   ├── INbreast/     
│   │   │   │   │   ├── crop/  
│   │   │   │   │   └── whole/   
│   │   └── processDataset.ipynb    # Script to rename the original files
│   ├── instal_conda_PL.ps1         # File to set up the Python Environment
│   └── requirements_PL.txt  
└── README.md
```
### Running the Pipeline:
1. Configure and Run `pipeline.py`
    - Open the pipeline.py file located in the Pipeline/Code folder.
    - Modify the script as follows:
        - Path Configuration: Update the path in line 21 to your computer's path.
        - Model Selection:
            - Model train with INbreast dataset: Uncomment line 30 and comment line 34. Run line 29 in the terminal with the BCIenv environment activated.  
            - Model train with CBIS-DDSM dataset: Uncomment line 34 and comment line 30. Run line 33 in the terminal with the BCIenv environment activated.
        - Dataset Selection:  
            - INbreast dataset: Uncomment lines 38 and 39; comment out everything else in this section.  
            - CBIS-DDSM dataset: Uncomment lines 41 and 42, and comment out everything else in this section.  
            - mini-MIAS dataset: Uncomment line 44. Depending on the model selected above:  
                - For INbreast-trained model: Uncomment line 46.    
                - For CBIS-DDSM-trained model: Uncomment line 47.  
        - Dataset Path Selection:
            - If using INbreast or CBIS-DDSM datasets
                - Comment lines 61 to 67.
                - Uncomment lines 51 to 57.     
            - If using mini-MIAS dataset
                - Comment lines 51 to 57.
                - Uncomment lines 61 to 67.
    - Clean the following folders in the dataset you plan to use: (Esto no se si es necesario o si es mejor que agreguen algo extra al nombre de los archivos que se generan para que tengan nuestros resultados y los que ellos corran)
        - INbreast dataset: Clean folders under INbreastDataset.
        - CBIS-DDSM dataset: Clean folders under CBIS-DDSMDataset.
        - mini-MIAS dataset: Clean folders under miniMIASDataset.
        - Paths to clean include:
            - For INbreast and CBIS-DDSM:
                - InputImages/PreprocessImages/crop
                - InputImages/PreprocessImages/filtered
                - InputImages/PreprocessImages/grayscale
                - Results/coordinates
                - Results/joinedMasks
                - Results/masks/crop
                - Results/masks/whole
                - Results/metrics
            - For mini-MIAS:
                - InputImages/PreprocessImages/`Dataset the model was train on`/crop
                - InputImages/PreprocessImages/`Dataset the model was train on`/filtered
                - InputImages/PreprocessImages/`Dataset the model was train on`/grayscale
                - Results/coordinates
                - Results/`Dataset the model was train on`/joinedMasks
                - Results/masks/`Dataset the model was train on`/crop
                - Results/masks/`Dataset the model was train on`/whole
    - Run the pipeline (make sure the BCIenv is activated):
        ```bash
        $ cd Pipeline/Code/
        $ python pipeline.py
        ```
        - Wait until it finishes running
2. Configure and Run `joinMasks.py`
    - Open the joinMasks.py file located in the Pipeline/Code folder
    - Modify the script as follows:
        - Path Configuration: Update the path in line 45 to your computer's path.
        - Dataset Selection (depending on what dataset the Pipeline was tested on):
            - INbreast dataset: Uncomment lines 52, 65, and 66. Comment out everything else in these sections.
            - CBIS-DDSM dataset: Uncomment lines 55, 65, and 66. Comment out everything else in these sections.
            - mini-MIAS dataset: Uncomment lines 58, 69, and 70. Depending on the model that you use:
                - For INbreast-trained model: Uncomment line 60. Comment out everything else in these sections..
                - For CBIS-DDSM-trained model: Uncomment line 62. Comment out everything else in these sections.
    - Run the file (make sure the BCIenv is activated):
        ```bash
        $ cd Pipeline/Code/
        $ python joinMasks.py
        ```
        - Wait until it finishes running
3. Configure and Run `metrics.py` (for INbreast and CBIS-DDSM datasets only)
    - Open the metrics.py file located in the Pipeline/Code folder
     - Modify the script as follows:
        - Path Configuration: Update the path in line 71 to your computer's path.
        - Dataset Selection (depending on what dataset the Pipeline was tested on):
            - INbreast dataset: Uncomment lines 78 to 79. Comment out everything else.
            - CBIS-DDSM dataset: Uncomment lines 82 to 83. Comment out everything else.
    - Run the file (make sure the BCIenv is activated):
        ```bash
        $ cd Pipeline/Code/
        $ python metrics.py
        ```
        - Wait until it finishes running
        - If this warning appears: "I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`." Before running the script again run this command (Windows-Specific Command):
          ```bash
          $ set TF_ENABLE_ONEDNN_OPTS=0
          ```
4. Run `resultAnalysis.ipynb` (for INbreast and CBIS-DDSM datasets only)
   - Open the resultAnalysis.ipynb file located in the Pipeline/Code folder
   - Modify the cell as follows:
       - Path Configuration: Update the path in line 16 to your computer's path.
       - Dataset Selection (depending on what dataset the Pipeline was tested on):
           - INbreast dataset: Uncomment lines 23 to 24. Comment out everything else.
           - CBIS-DDSM dataset: Uncomment lines 27 to 28. Comment out everything else.
    - Run the cell (make sure the BCIenv is selected as the kernel)
        - Wait until it finishes running

## Files used and obtained from Google Collab
INbreast YOLO dataset:  
https://drive.google.com/drive/folders/146_U3MwxMTfqRdaUJPODgkIXH6vBrWWd?usp=sharing  

CBIS-DDSM YOLO dataset:  
https://drive.google.com/drive/folders/1OPgNmDeiO-PavkdBkhe1hf7xfgFFiQ4v?usp=sharing  

Results YOLO folder:   
https://drive.google.com/drive/folders/1ydnx5jLabXlkf3drCXtF9-UrS6Zv0l_s?usp=sharing  

## Train YOLO models on Google Collab
### YOLOv8 model
1. Create a folder named YOLOV8BreastCancer on your Google Drive's My Drive section.
2. Add to that folder the Google Colab Files zip located in DetectionModels/Google Colab Files
3. Open `yolo_config.py`
    - Choose dataset: uncomment the dataset you want to use, comment out the other dataset.
    - Choose model size: add to the brackets in lines 13 and 27 the name of the model size you want to train. 
        - Example: ['yolov8n.pt','yolov8s.pt'] or ['yolov8n.pt','yolov8s.pt','yolov8m.pt']
4. Download and add the INbreast and CBIS-DDSM YOLO datasets to your Google Drive My Drive section.
5. In the Colab Notebooks folder in your Google Drive's My Drive section, add the `YOLOV8BreastCancer.ipynb` file. 
6. Open the `YOLOV8BreastCancer.ipynb` file and conect to the T4 GPU on Google Colab.
7. Run all the cells of the notebook. 

## Original Datasets:
The original databases used for this project:
- [INbreast](#https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset?resource=download)
- [CBIS-DDMS](#https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)
- [mini-MIAS](#https://www.kaggle.com/datasets/kmader/mias-mammography)

## Important:
Since this project is part of a research initiative, the code for creating the experimental databases or their corresponding masks used to calculate the metrics is not included.
