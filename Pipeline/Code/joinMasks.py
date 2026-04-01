from PIL import Image
from os import listdir
from pathlib import Path
import numpy as np

def binarize_array(numpy_array):
    """Binarize a numpy array."""
    for i in range(len(numpy_array)):
        for j in range(len(numpy_array[0])):
            if numpy_array[i][j] > 1:
                numpy_array[i][j] = 255
            else:
                numpy_array[i][j] = 0
    return numpy_array


def joinMasks(wholeMaskPath, outputPath):
    name_dict = {}

    for name_mask in listdir(wholeMaskPath):
        nameSplit = name_mask.split("_")
        true_name = '_'.join(nameSplit[:-2])
        if true_name in name_dict.keys():
            name_dict[true_name].append(name_mask)
        else:
            name_dict[true_name] = [name_mask]

    for name in name_dict:
        img = Image.open(wholeMaskPath + f'{name_dict[name][0]}')
        if len(name_dict[name]) == 1:
            img.save(outputPath + f'{name}_final.jpg')
        else:
            for i in range(1, len(name_dict[name])):
                img_2 = Image.open(wholeMaskPath + f'{name_dict[name][i]}')
                img_2 = img_2.convert("L")
                img = img.convert("L")
                img = Image.blend(img, img_2, 0.5)
            n_image = np.array(img)
            n_image = binarize_array(n_image)
            img = Image.fromarray(n_image)
            img.save(outputPath + f'{name}_final.jpg')

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''#
##Change according to were you are running the code
investigation_fles_path = 'C:/Users/camiz/'
running_path = investigation_fles_path + 'MassSegFramework/Pipeline/'

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''#
##Choose a dataset to join the masks:

##INbreastDataset
#dataset = 'INbreastDataset'

##CBIS-DDSMDataset
dataset = 'CBIS-DDSMDataset'

##miniMIASDataset
#dataset = 'miniMIASDataset'
##For miniMIAS chose an option:
#datasetYolo = 'INbreast'
#datasetYolo = 'CBIS-DDSM'

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''#
##For joining INbreast and CBIS-DDSM Pipeline Masks
wholeMaskPath = running_path + dataset + '/Results/masks/whole/'
outputPath = running_path + dataset + '/Results/joinedMasks/'

##For joining miniMIAS Pipeline Masks
#wholeMaskPath = running_path + dataset + '/Results/masks/' + datasetYolo + '/whole/'
#outputPath = running_path + dataset + '/Results/joinedMasks/' + datasetYolo + '/'

#'''''''''''''''''''''''''''''''''''''''''''''''''''''''#
##For joining Pipeline Masks
joinMasks(wholeMaskPath, outputPath)