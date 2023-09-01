import os
import sys
import copy
import numpy as np
import pandas as pd
#from PIL import Image
#import cv2

sys.path.insert(0, './code/')
sys.path.insert(0, './lungmask-master/')
from utils import readCsv
from lungmask import mask
from LNDData_classes import Scan
from readNoduleList import joinNodules, nodEqDiam
from match_nod import text2nodules
from ut2match import nod2finalList, substitute2Fleishcner

# Define a function to find the lobe based on a given point within a segmentation
def getOutLobe(segmentation, point):
    coords = np.array(np.where(segmentation > 0))
    difs = np.zeros(np.shape(coords))
    
    for i in range(np.size(coords, 1)):
        difs[:, i] = (coords[:, i] - point) ** 2
        
    op = np.sum(difs, axis=0)
    where = np.argmin(op)
    lobe = segmentation[coords[0, where], coords[1, where], coords[2, where]]
    
    return lobe

# Define a function to save data to a CSV file
def saveFile(ar, name2save):
    go = pd.DataFrame(ar)
    go.columns = go.iloc[0]
    go = go[1:]
    go.to_csv(name2save, index=False)

# Read the main CSV file
df = pd.read_csv('./reports/report.csv')

# Define file and folder paths
nodule_folder = './nodules/'
cts_csvs = './cts_csvs/'
chars = './chars/'
cts_dicom = './LNDETECTOR Database'
#cts_dicom = '/media/cferreira/LNDb/LNDETECTOR/LNDETECTOR Database'
lobe_segmentations = './lobe_segmentations/'
#lobe_segmentations = '/media/cferreira/LNDb/Carlos/segmentos_pulmao/save_segmentations/'

# List of characteristics related to nodules
n_chars = ['calcification', 'internalStructure', 'lobulation', 'malignancy', 'margin', 'sphericity', 'spiculation', 'subtlety']

# Create a Scan object
scan = Scan()

### NODULES ###
first = 0
test = True
train = True
list_nodules = []

# Iterate through files in the nodule folder to read and accumulate nodule data
for file in os.listdir(nodule_folder):
    list_csv = readCsv(os.path.abspath(nodule_folder + file), os.path.abspath(chars + 'chars_' + file), n_chars)
    if file[:4] == "test" and test == True:
        list_nodules = list_nodules + list_csv[first:]
        first = 1
    elif file[:5] == "train" and train == True:
        list_nodules = list_nodules + list_csv[first:]
        first = 1

# If nodules are found, process and join them
if len(list_nodules) > 0:
    nodules = joinNodules(list_nodules, name_chars=n_chars)
    nodules_copy = copy.deepcopy(nodules)
    what_lobe = np.zeros(len(nodules)-1)

    # Get indices for x, y, and z coordinates in the nodules data
    x_id = nodules[0].index('x')
    y_id = nodules[0].index('y')
    z_id = nodules[0].index('z')

### CTS ###
list_cts = []
first = True

# Iterate through files in the cts_csvs folder to read CT scan data
for file in os.listdir(cts_csvs):
    if (file[:4] == "test" and test == True) or (file[:5] == "train" and train == True):
        list_csv = pd.read_csv(cts_csvs + file)
        if first == True:
            list_cts = list_csv['LNDbID']
            first = False
        else:
            list_cts = list_cts.append(list_csv['LNDbID'])

# If CT scan data is found, remove duplicates
if len(list_cts) > 0:
    list_cts = np.unique(list_cts)

### Lobe Segmentation ###
for LNDid in list_cts:
    pat = 'LNDETECTOR-{:04d}'.format(LNDid)
    datapathlist = os.listdir(os.path.join(cts_dicom, pat))
    
    # Create a folder for storing lobe segmentations if it doesn't exist
    if datapathlist:
        if not os.path.exists(lobe_segmentations + "/" + pat):
            os.mkdir(lobe_segmentations + "/" + pat)
            
        # Load the CT scan
        scan.NewScanFromDcm(os.path.join(cts_dicom, pat))

        print(LNDid)
        segpathlist = os.path.join(lobe_segmentations + '/' + pat, 'LNDETECTOR-{:04d}_{}.npy'.format(LNDid,'fill_new'))
        
        # If the segmentation doesn't exist, create and save it
        if not os.path.exists(segpathlist):
            ct_scan = scan.Scan
            orientation = scan.Orientation
            flippedXaxis = scan.flippedXaxis
            flippedYaxis = scan.flippedYaxis
            
            if (orientation[-1] == -1):
                ct_scan = ct_scan[::-1]
    
            if flippedXaxis:
                ct_scan = np.flip(ct_scan, 1)
    
            if flippedYaxis:
                ct_scan = np.flip(ct_scan, 2)
        
            segmentation = mask.apply_fused(ct_scan)
            np.save(segpathlist, segmentation)
        else:
            segmentation = np.load(segpathlist)

        for j in range(len(nodules[1:])):
            if nodules[j+1][0] == LNDid:
                nodules[1+j][x_id], nodules[1+j][y_id], nodules[1+j][z_id] = scan.WorldToVoxel([nodules[1+j][x_id], nodules[1+j][y_id], nodules[1+j][z_id]])
                nodules[1+j][9] = nodEqDiam(nodules[1+j][9])
                point = np.array([nodules[1+j][x_id], nodules[1+j][y_id], nodules[1+j][z_id]])
                
                ### Get Lobe ###
                what_lobe[j] = segmentation[nodules[1+j][x_id], nodules[1+j][y_id], nodules[1+j][z_id]]
                if what_lobe[j] == 0:
                    what_lobe[j] = getOutLobe(segmentation, point)
                nodules[1+j].append(int(what_lobe[j]))
                nodules[1+j].append(False)

if len(list_cts) > 0:  
    # Count nodules and segmentations per lobe and CT scan
    count_nod = np.zeros([len(list_cts), 6])
    count_all = np.zeros([len(list_cts)])
    nod_proc = len(list_cts) * [None]
    
    # Iterate through CT scans to analyze nodules and segmentations
    for i in range(len(list_cts)):
        count_nod[i, 0] = list_cts[i]
        for j in range(len(nodules[1:])):
            if nodules[j+1][0] == list_cts[i]:
                count_all[i] += 1
                count_nod[i, int(what_lobe[j])] += 1
                    
    for i in range(len(list_cts)):
        it = 0
        nod_proc[i] = int(count_all[i]) * [None]
        for j in range(len(nodules[1:])):
            if nodules[j+1][0] == list_cts[i]:
                nod_proc[i][it] = nodules[j+1]
                it += 1
                
    nod_proc2, not_df, not_ct, a1, a2, dicts = text2nodules(df, lobe_segmentations, nod_proc, count_nod)
    nod_proc3 = nod2finalList(nod_proc2, df, nodules_copy)
    text2Fleischner = substitute2Fleishcner(nod_proc3, 'TextReport')
    rad2Fleischner = substitute2Fleishcner(nod_proc3, 'RadAnnotation')

    # Save processed data to CSV files
    saveFile(nod_proc3, 'allNods.csv')
    saveFile(text2Fleischner, 'text2Fleischner.csv')
    saveFile(rad2Fleischner, 'rad2Fleischner.csv')