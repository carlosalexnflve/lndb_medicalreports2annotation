#import LNDThreadFuncDlg

import SimpleITK as sitk
import numpy as np
import csv
import os
from PIL import Image
import matplotlib.pyplot as plt
from scipy.ndimage.interpolation import zoom
import random
from scipy.ndimage.measurements import label
import cv2
import pydicom
import itertools
import nibabel as nib


def dcm2numpy(PathDicom, HU=True,return_index = False, worker = None):
    # Reads .dcm files inside a folder and returns the respective numpy array,
    # rows coordinates, columns coordinates and depth coordinates, all in (mm)
    # https://pyscience.wordpress.com/2014/09/08/
    # dicom-in-python-importing-medical-image-data-into-numpy-with-pydicom-and-vtk/
    
    #if threaded update master on progress done
    #if worker:
    #    if LNDThreadFuncDlg.cancelled:
    #        return []
    #    worker.progressV.emit(worker.proglim[0]) # update progress bar    
    
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(PathDicom):
        for filename in fileList:
            if ".dcm" in filename.lower() or '.' not in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName, filename))
        break
    
    if not lstFilesDCM:
        return ['DICOM files not found']
    
    # Get ref file
    RefDs = []
    for fileDCM in lstFilesDCM:
        try:
            RefDs = pydicom.read_file(fileDCM)  # reads 1st file
            if hasattr(RefDs,'Rows') and hasattr(RefDs,'Columns') and hasattr(RefDs,'PixelSpacing') and hasattr(RefDs,'PatientID'):
                break
            else:
                RefDs = []
        except:
            pass
    
    if not RefDs:
        return ['Could not open DICOM files']
    
    # use metadata entries to retrieve shape information
    # Load dimensions based on the number of rows,
    # columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
    
    # Load spacing values (in mm)
    ConstPixelSpacing = (float(RefDs.PixelSpacing[0]),float(RefDs.PixelSpacing[1]),0)
    
    x = np.zeros(len(lstFilesDCM))
    y = np.zeros(len(lstFilesDCM))
    z = np.zeros(len(lstFilesDCM))
    # The array is sized based on 'ConstPixelDims'
    ArrayDicom = np.zeros(ConstPixelDims)#, dtype=RefDs.pixel_array.dtype)
    # loop through all the DICOM files
    indDCM = 0
    for ind,filenameDCM in enumerate(lstFilesDCM):
        
        #if threaded update master on progress done
        #if worker:
        #    if LNDThreadFuncDlg.cancelled:
        #        return []
        #    worker.progressV.emit(worker.proglim[0]+ind/len(lstFilesDCM)*worker.proglim[1]) # update progress bar
        
        try:
            # read the file
            ds = pydicom.read_file(filenameDCM,force = True)
            
            if HU == True:
                ArrayDicom[:, :, indDCM] = ds.pixel_array*ds.RescaleSlope+ds.RescaleIntercept
            else:
                ArrayDicom[:, :, indDCM] = ds.pixel_array
            
            x[indDCM] = np.float(ds.ImagePositionPatient[0])
            y[indDCM] = np.float(ds.ImagePositionPatient[1])
            z[indDCM] = np.float(ds.ImagePositionPatient[2])
            indDCM += 1
        except:
            pass
    
    z = z[0:indDCM]
    ArrayDicom = ArrayDicom[:,:,0:indDCM]
    
    z_sort = np.argsort(z)[::-1]  # sort by descending order
    x = x[z_sort]
    y = y[z_sort]
    z = z[z_sort]
    ArrayDicom = ArrayDicom[:, :, z_sort]
    ArrayDicom = np.transpose(ArrayDicom,(2,0,1))
    
    if return_index:
        return [ArrayDicom, RefDs, x, y, z]
    else:
        return [ArrayDicom, RefDs]
    
def normalizePlanes(npzarray,maxHU=400.,minHU=-1000.):
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

def returnStacks(volume,mid_slice,stack_depth=0,mip_bool=False):
    if mip_bool:
        z = mid_slice
        prev = z-np.int(stack_size/2)-stack_depth;
        
        s1 = volume[(z-prev*3,z-prev*2,z-prev),...]
        s1 = np.max(s1,axis=0)
        
        s2 = volume[z:z+1,...]
        s2 = np.max(s2,axis=0)
        
        s3 = volume[(z+prev,z+2*prev,z+3*prev),...]
        s3 = np.max(s3,axis=0)
        
        slic = np.stack((s1,s2,s3),axis=0)
        slic = np.transpose(slic,(1,2,0))  
        return slic
    else:
        #return volume[(mid_slice-stack_depth,mid_slice,mid_slice+1+stack_depth),...]
        return volume[(mid_slice-stack_depth,mid_slice,mid_slice+stack_depth),...]

def voxelToWorldCoord(voxel_coord, position, spacing, orientation,n):

    A = np.zeros((4,4))
    
    A[0:3,0] += orientation[:,1]*spacing[1]
    A[0:3,1] += orientation[:,0]*spacing[2]
    A[0:3,3] += position[0]
    A[3,3] = 1
    A[0:3,2] += (position[0]-position[1])/(1-n)
    
    temp = np.ones((4))
    
    temp[:3] = voxel_coord
    
    voxel_coord = np.dot(A,temp)  
    
    return voxel_coord[:3]
    
    
def save_nii_files(savefilename, volume,  origin, spacing):

    #originally written by: Patrick Sousa
    #adapted by: Guilherme Aresta
    #log:
    #-25/03/2019: changed input to "toe to head"
    

    #volume: toe to head
    #spacing: [x spacing, y spacing, z spacing]
    #origin: [x,y,z]

    aff1 = np.array(([1, 0, 0, origin[0]], [0, 1, 0, origin[1]], [0, 0, 1, origin[2]], [0, 0, 0, 1])) * np.array(([spacing[0], spacing[1], spacing[2], 1]))
    array_img = nib.Nifti1Image(volume, affine=aff1)
    array_img.header['regular'] = b'r'
    array_img.header['pixdim'] = [1.0, spacing[0], spacing[1], spacing[2], 0.0, 0.0, 0.0, 0.0]
    array_img.header['xyzt_units'] = 2
    array_img.header['qform_code'] = 2
    array_img.header['sform_code'] = 1
    
    nib.save(array_img, savefilename)

