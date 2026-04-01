# MassSeg-Framework: A Breast Mass Detection and Segmentation Framework Based on Deep Learning and Active Contour Model

## Project Objective:
Implement MassSeg, a new automatic two-step breast mass segmentation method that combines the YOLOv11 architecture with a Chan-Vesse active contour model to maximize the lesion’s detection and enhance the lesion’s segmentation on mammography images.

## Table of Contents:
- [Installation](#installation)
    - [MATLAB Setup](#matlab-setup)
    - [Setup Running Environment](#setup-running-environment)
- [Project Structure](#project-structure)
- [Running the Pipeline](#running-pipeline)
    - [Main Pipeline](#main-pipeline)
    - [Mask Post-processing](#mask-post-processing)
    - [Metrics Computation](#metrics-computation)
    - [Final Results](#final-results)
- [Datasets and Results YOLO Models](#datasets-and-results-YOLO-models)
- [YOLO Models (Google Colab)](#yolo-models-google-colab)
    - [Hyperparameter Search](#hyperparameter-search)
    - [Training](#training)
- [Original Datasets](#original-datasets)
- [Disclaimer](#disclaimer)

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
│   ├── ComparisonStadistic/        # Results of the statistics tests
│   │   ├── INbreast/  
│   │   └── CBIS-DDSCM/               
│   ├── GoogleColabFiles/
│   │   ├── HyperparameterSearch/   # Script and files to run the hyperparameter search
│   │   └── Training/               # Script and files to run the training        
│   ├── lossGraphs/                 # Script to crop the detected mass
│   ├── runs/                       # Results of the train models (not included in the git files)
│   │   ├── INbreast/
│   │   └── CBIS-DDSCM/ 
│   ├── mannWhitneyTest.ipynb       # Script to calculate the statistics tests
│   └── resultAnalysis.py           # Script to obtain the loss graphs and performance metrics
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
│   │   ├── activeCountours.py      # Active Countour functions for segmentation        
│   │   ├── cropImage.py            # Script to crop the detected mass
│   │   ├── joinMasks.py            # Script to join the segmented mass masks of the same mammogram image 
│   │   ├── metrics.py              # Script to obtain segmentation metrics (DICE, IOU, HD95)
│   │   ├── morphsnakes_v1.py       
│   │   ├── morphsnakes.py  
│   │   ├── pipeline.py              # Main script to run the pipeline
│   │   ├── resultAnalysis.ipynb     # Script to obtain detection metrics and average segmentation metrics (DICE, IOU, HD95)
│   │   ├── setup.py                 # Default setup to run the active countour
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
│   │   ├── cropToOriginalMaskCV.m  # Script to place the segmented mask in its original position within the full-size image
│   │   └── filtered_oneImage.m     # Script to apply the median filter
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
## Running the Pipeline
- Download the Results of training folder in the [Datasets and Results YOLO Models](#datasets-and-results-YOLO-models) section.
- Add the folder to the project at MassSeg/DetectionModels/.
### Main Pipeline
1. Open the pipeline.py file located in the Pipeline/Code folder.
2. Modify the script as follows:
    - Path Configuration: Update the path in line 21 to your computer's path.
    - Model Selection:
        - Model train with INbreast dataset: Uncomment lines 30 and 29; comment lines 33 and 34.
        - Model train with CBIS-DDSM dataset: No changes needed (already set).
    - Dataset Selection:
        - INbreast dataset: Uncomment lines 38 and 39; comment lines 41 and 42.
        - CBIS-DDSM dataset: No changes needed (already set).
        - mini-MIAS dataset: Uncomment line 44; comment lines 41 and 42. Depending on the model selected above:
            - For INbreast-trained model: Uncomment line 46.
            - For CBIS-DDSM-trained model: Uncomment line 47.
    - Dataset Path Selection:
        - If using INbreast or CBIS-DDSM datasets: No changes needed (already set).
        - If using mini-MIAS dataset:
            - Comment lines 51 to 57.
            - Uncomment lines 61 to 67.
    - Clean the following folders in the dataset you plan to use:  
   **NOTA NOEL: No sé si es necesario o si es mejor que agreguen algo extra al nombre de los archivos para diferenciar resultados.**
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
### Mask Post-processing
1. Open the joinMasks.py file located in the Pipeline/Code folder
2. Modify the script as follows:
    - Path Configuration: Update the path in line 45 to your computer's path.
    - Dataset Selection (depending on what dataset the Pipeline was tested on):
        - INbreast dataset: Uncomment lines 52 and comment line 55.
        - CBIS-DDSM dataset: No changes needed (already set).
        - mini-MIAS dataset: Uncomment lines 58, 69, and 70, and comment lines 55, 65, and 66. Depending on the model that you used:
          - For INbreast-trained model: Uncomment line 60. 
          - For CBIS-DDSM-trained model: Uncomment line 62.
    - Run the file (make sure the BCIenv is activated):
        ```bash
        $ cd Pipeline/Code/
        $ python joinMasks.py
        ```
## Metrics Computation 
> *For INbreast and CBIS-DDSM datasets only*
### Segmentation
1. Open the metrics.py file located in the Pipeline/Code folder
2. Modify the script as follows:
    - Path Configuration: Update the path in line 71 to your computer's path.
    - Dataset Selection (depending on what dataset the Pipeline was tested on):
        - INbreast dataset: Uncomment lines 120 and 121, comment lines 124 and 125.
        - CBIS-DDSM dataset: No changes needed (already set).
    - Run the file (make sure the BCIenv is activated):
        ```bash
        $ cd Pipeline/Code/
        $ python metrics.py
        ```
    > *If this warning appears: "tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`." Before running the script again, run this command (Windows-Specific Command):*
    >
    > ```bash
    > $ set TF_ENABLE_ONEDNN_OPTS=0
    > ```
### Detection
1. Open the resultAnalysis.ipynb file located in the Pipeline/Code folder
2. Go to the first section: Obtaining Detection Metrics
3. Modify the cell as follows:
    - Path Configuration: Update the path in line 7 to your computer's path.
    - Dataset Selection (depending on what dataset the Pipeline was tested on):
        - INbreast dataset: Uncomment lines 12 through 14, comment lines 16 through 18.
        - CBIS-DDSM dataset: No changes needed (already set).
    - Run the cell (make sure the BCIenv is selected as the kernel)

## Final Results
> *For INbreast and CBIS-DDSM datasets only*
### Segmentation
1. Open the resultAnalysis.ipynb file located in the Pipeline/Code folder
2. Go to the third section: Average Segmentation Metrics
3. Modify the cell as follows:
    - Path Configuration: Update the path in line 18 to your computer's path.
    - Dataset Selection (depending on what dataset the Pipeline was tested on):
        - INbreast dataset: Uncomment lines 25 and 26, comment lines 29 and 30.
        - CBIS-DDSM dataset: No changes needed (already set).
    - Run the cell (make sure the BCIenv is selected as the kernel)
### Detection
1. Open the resultAnalysis.ipynb file located in the Pipeline/Code folder
2. Go to the second section: Average Detection Metrics
3. Modify the cell as follows:
    - Path Configuration: Update the path in line 6 to your computer's path.
    - Dataset Selection (depending on what dataset the Pipeline was tested on):
        - INbreast dataset: Uncomment line 11, comment line 12.
        - CBIS-DDSM dataset: No changes needed (already set).
    - Run the cell (make sure the BCIenv is selected as the kernel)

## Datasets and Results YOLO Models
- [INbreast YOLO dataset](https://drive.google.com/drive/folders/146_U3MwxMTfqRdaUJPODgkIXH6vBrWWd?usp=sharing)
- [CBIS-DDSM YOLO dataset](https://drive.google.com/drive/folders/1OPgNmDeiO-PavkdBkhe1hf7xfgFFiQ4v?usp=sharing)
- [Results of hyperparameter search](https://drive.google.com/drive/folders/181yjJ5Qf_ARlSe3YNdFMwzuvcsqkCSwJ?usp=sharing)
- [Results of training](https://drive.google.com/drive/folders/1ydnx5jLabXlkf3drCXtF9-UrS6Zv0l_s?usp=sharing)

## YOLO Models (Google Colab)
- Download and add the INbreast and CBIS-DDSM YOLO datasets to your Google Drive My Drive section.
### Hyperparameter Search
1. Create a folder named YOLOV11HyperparameterSearch in your Google Drive's My Drive section.
2. Download and add to the YOLOV11HyperparameterSearch folder the files in .zip located in DetectionModels/GoogleColabFiles/HyperparameterSearch.
3. Download and add to the Colab Notebooks folder in your Google Drive's My Drive section, the `YOLOV11HypS.ipynb` file located in DetectionModels/GoogleColabFiles/HyperparameterSearch.
4. Create an account on https://wandb.ai to obtain the API key. 
5. Open `yolo_config.py` located in YOLOV11HyperparameterSearch folder
    - Choose dataset: uncomment the dataset you want to use, comment out the other dataset.
6. Open the `YOLOV11HypS.ipynb` file and connect to the T4 GPU on Google Colab.
7. Run all the cells of the notebook. 
### Training
1. Create a folder named YOLOV11BreastCancer in your Google Drive's My Drive section.
2. Download and add to the YOLOV11BreastCancer folder the files in .zip located in DetectionModels/GoogleColabFiles/Training.
3. Download and add to the Colab Notebooks folder in your Google Drive's My Drive section, the `YOLOV11BreastCancer.ipynb` file located in DetectionModels/GoogleColabFiles/Training.
4. Open `yolo_config.py` located in YOLOV11BreastCancer folder
    - Choose dataset: uncomment the dataset you want to use, comment out the other dataset.
    - Choose model size: add to the brackets in lines 13 and 27 the name of the model size you want to train. 
        - Example: ['yolov11n.pt'] or ['yolov11n.pt','yolov11s.pt','yolov11m.pt']
5. Open the `YOLOV8BreastCancer.ipynb` file and connect to the T4 GPU on Google Colab.
6. Run all the cells of the notebook. 

## Original Datasets:
The original databases used for this project:
- [INbreast](https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset?resource=download)
- [CBIS-DDMS](https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset)
- [mini-MIAS](https://www.kaggle.com/datasets/kmader/mias-mammography)

## Disclaimer:
Since this project is part of a research initiative, the code for creating the experimental databases or their corresponding masks used to calculate the metrics is not included.
