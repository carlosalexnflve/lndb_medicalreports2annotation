#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 15:23:46 2023

@author: cferreira
"""

import nibabel as nib
import numpy as np
import scanwise_utils as SU

class Scan:
    def __init__(self):
        # Constant scan settings
        self.Scan = []
        self.Voxelsize = [1, 1, 1] # [Z,X,Y] or [Z,Y,X]??
        self.ScanindexX = []
        self.ScanindexY = []
        self.ScanindexZ = []
        self.SeriesInstanceUID = ''
        self.StudyInstanceUID = ''
        self.Scaninfo = ''
        self.PatientID = ''
        
        # Scan display settings
        self.image = []
        self.indexX = []
        self.indexY = []
        self.indexZ = []        
        self.worldDim = [1, 1, 1]
        self.Scansize = [0, 0, 0]
        
        #self.HUmin_def = [LNDsettings.win0_min,LNDsettings.win1_min,LNDsettings.win2_min]
        #self.HUmax_def = [LNDsettings.win0_max,LNDsettings.win1_max,LNDsettings.win2_max]        
        self.hstgrm_lim = [0,0,0,0]
        self.hstgrm_n = []
        self.hstgrm_bins = []
        
    def NewScanFromNiiGz(self,image_path,worker = None):
        # Read Image and chars from file
        #if worker:
        #    if LNDThreadFuncDlg.cancelled:
        #        return []
        #    worker.progressV.emit(worker.proglim[0]) # update progress bar
        
        img = nib.load(image_path)
        
        #if worker:
        #    if LNDThreadFuncDlg.cancelled:
        #        return []
        #    worker.progressV.emit(worker.proglim[0]+.01*worker.proglim[1]) # update progress bar
        
        for r in range(2):
            for c in range(4):
                img.affine[r,c] = -img.affine[r,c]
        M = img.affine[:3, :3]
        abc = img.affine[:3, 3]        
        
        scan = img.get_data()
        
        #if worker:
        #    if LNDThreadFuncDlg.cancelled:
        #        return []
        #    worker.progressV.emit(worker.proglim[0]+.33*worker.proglim[1]) # update progress bar
        
        # Flip scan if needed
        [x0,y0,z0]=M.dot([0,0,0]) + abc
        [xe,ye,ze]=M.dot([l-1 for l in list(scan.shape)]) + abc
        if z0<ze:
            scan = np.flip(scan,2)
        
        # Compute voxelsize and orientation vector
        voxelsize = [0,0,0]
        orientation = [0,0,0,0,0,0]
        for d in range(3):
            p = [0,0,0]
            p[d] = 1
            [x1,y1,z1]=M.dot(p) + abc
            voxelsize[d] = ((x1-x0)**2+(y1-y0)**2+(z1-z0)**2)**.5
            if d<2:
                ori = [x1-x0,y1-y0,z0-z1]
                ori = [o/voxelsize[d] for o in ori]
                if ori[d]<0:
                    ori = [-o for o in ori]
                ori[2] = -ori[2]
                if d == 0:
                    orientation[:3] = ori
                else:
                    orientation[3:] = ori
        
        # Get (0,0) world coordinates for every slice
        ix = []
        iy = []
        iz = []
        for l in range(list(scan.shape)[2]):
            [xt,yt,zt] = img.affine.dot([0,0,l,1])[:3]
            ix.append(xt)
            iy.append(yt)
            iz.append(zt)        
        
        # Transpose Z
        scan = np.transpose(scan,(2,1,0))
        scan = scan.astype(np.uint8)          
        
        # Scan settings
        self.Scan = scan
        self.Voxelsize = [voxelsize[d] for d in [2,0,1]]
        self.ScanindexX = ix[::-1]
        self.ScanindexY = iy[::-1]
        self.ScanindexZ = iz[::-1]
        self.SeriesInstanceUID = ''
        self.StudyInstanceUID = ''
        self.Scansize = list(self.Scan.shape)
        
        self.Origin = [xe,ye,ze]
        self.Orientation = np.array([np.float64(o) for o in orientation])
        self.getTransfMatrix()
        
        self.Scaninfo = ''
        
        #if worker:
        #    if LNDThreadFuncDlg.cancelled:
        #        return []
        #    worker.progressV.emit(worker.proglim[0]+.98*worker.proglim[1]) # update progress bar
        
        # Image display settings
        #self.HUmin = self.HUmin_def[1]
        #self.HUmax = self.HUmax_def[1]
        self.hstgrm_lim = [np.amin(self.Scan),np.amax(self.Scan),0,1]
        
        #if worker:
        #    if LNDThreadFuncDlg.cancelled:
        #        return []
        #    worker.progressV.emit(worker.proglim[0]+.99*worker.proglim[1]) # update progress bar        
        
        n,bins = np.histogram(np.reshape(self.Scan,[np.prod(self.Scansize)]),100)
        n = [b/np.amax(n) for b in n]
        self.hstgrm_n = n
        self.hstgrm_bins = bins[0:-1]
        self.image = self.Scan
        
    def NewScanFromDcm(self,image_path,worker = None):
        # Read Image and chars from file
        if worker:
            proglim = copy.copy(worker.proglim)
            worker.proglim[1] = .97*worker.proglim[1]
        [scan,info,ix,iy,iz] = SU.dcm2numpy(image_path,True,True,worker)
        
        if worker:
            worker.proglim = copy.copy(proglim)
        
        dx = [abs(a-b) for a,b in zip(ix[0:-1],ix[1:])]
        dx = max(set(dx), key=dx.count)
        dy = [abs(a-b) for a,b in zip(iy[0:-1],iy[1:])]
        dy = max(set(dy), key=dy.count)
        dz = [abs(a-b) for a,b in zip(iz[0:-1],iz[1:])]
        dz = max(set(dz), key=dz.count)
        
        voxelsize = [(dx**2+dy**2+dz**2)**.5]
        pixelsize = [np.float32(x) for x in info.PixelSpacing]
        voxelsize += pixelsize
        
        # Scan settings
        self.Scan = scan
        self.Voxelsize = voxelsize
        self.ScanindexX = ix
        self.ScanindexY = iy
        self.ScanindexZ = iz
        self.SeriesInstanceUID = str(info.SeriesInstanceUID)
        self.StudyInstanceUID = str(info.StudyInstanceUID)
        self.StudyDate = info.StudyDate
        self.Scansize = list(self.Scan.shape)
        
        self.Origin = [info.ImagePositionPatient[0],info.ImagePositionPatient[1],self.ScanindexZ[-1]]
        self.Orientation = np.array([np.float64(o) for o in info.ImageOrientationPatient])
        self.getTransfMatrix()
        
        pid = ''
        pname = ''
        psep = ''
        pdate = ''
        if info.PatientID:
            pid = str(info.PatientID)
            pid.replace(' ','')
        if info.PatientName:
            pname = str(info.PatientName)
            pname.replace(' ','')
        if pid and pname:
            psep = ' '
        if hasattr(info, 'AcquisitionDate'):
            if info.AcquisitionDate:
                if len(info.AcquisitionDate)==8:
                    pdate = info.AcquisitionDate[-2:]+'/'+info.AcquisitionDate[4:6]+'/'+info.AcquisitionDate[0:4]
                else:
                    pdate = info.AcquisitionDate
        
        if pid or pname:
            self.PatientID = pid
            self.Scaninfo = pid + psep + pname + ', ' + pdate
        else:
            self.Scaninfo = pdate
    
        #if worker:
        #    if LNDThreadFuncDlg.cancelled:
        #        return []
        #    worker.progressV.emit(worker.proglim[0]+.98*worker.proglim[1]) # update progress bar
        
        # Image display settings
        #self.HUmin = self.HUmin_def[1]
        #self.HUmax = self.HUmax_def[1]
        self.hstgrm_lim = [np.amin(self.Scan),np.amax(self.Scan),0,1]
        
        
        #if worker:
        #    if LNDThreadFuncDlg.cancelled:
        #        return []
        #    worker.progressV.emit(worker.proglim[0]+.99*worker.proglim[1]) # update progress bar        
        
        n,bins = np.histogram(np.reshape(self.Scan,[np.prod(self.Scansize)]),100)
        n = [b/np.amax(n) for b in n]
        self.hstgrm_n = n
        self.hstgrm_bins = bins[0:-1]
        self.image = self.Scan        
        
    def getTransfMatrix(self):
        self.flippedXaxis = False
        self.flippedYaxis = False
        orientationZ = np.cross(self.Orientation[3:],self.Orientation[:3])
        if orientationZ[-1]>0:
            orientationZ = np.asarray([-o for o in orientationZ],np.float64)
        
        self.VoxelToWorldMat = np.zeros((4,4))
        self.VoxelToWorldMat[0:3,0] = self.Orientation[3:]*self.Voxelsize[1]
        self.VoxelToWorldMat[0:3,1] = self.Orientation[:3]*self.Voxelsize[2]
        self.VoxelToWorldMat[0:3,2] = orientationZ*self.Voxelsize[0]
        self.VoxelToWorldMat[0:3,3] = [self.ScanindexX[0],self.ScanindexY[0],self.ScanindexZ[0]]
        self.VoxelToWorldMat[3,3] = 1
        
        if self.Orientation[0]<0 or self.Orientation[4]<0 :
            if self.Orientation[0]<0:
                self.Orientation[:3] = -self.Orientation[:3]
                self.Scan = np.flip(self.Scan,1)
                xind = self.Scansize[1]-1
                self.flippedXaxis = True
            else:
                xind = 0            
            if self.Orientation[4]<0:
                self.Orientation[3:] = -self.Orientation[3:]
                self.Scan = np.flip(self.Scan,2)
                yind = self.Scansize[2]-1
                self.flippedYaxis = True
            else:
                yind = 0
            
            Scanindex = [self.VoxelToWorld([z,yind,xind]) for z in range(self.Scansize[0])]
            self.ScanindexX = [si[0] for si in Scanindex]
            self.ScanindexY = [si[1] for si in Scanindex]
            self.ScanindexZ = [si[2] for si in Scanindex]
            self.VoxelToWorldMat[0:3,1] = self.Orientation[:3]*self.Voxelsize[2]
            self.VoxelToWorldMat[0:3,0] = self.Orientation[3:]*self.Voxelsize[1]
            self.VoxelToWorldMat[0:3,3] = [self.ScanindexX[0],self.ScanindexY[0],self.ScanindexZ[0]]
        
        self.WorldToVoxelMat = np.linalg.inv(self.VoxelToWorldMat[0:3,0:3])
        
    def VoxelToWorld(self,zxy_v):
        temp = np.ones((4))
        temp[:3] = [zxy_v[s] for s in [1,2,0]]
        xyz_w = np.dot(self.VoxelToWorldMat,temp)
        return list(xyz_w[:3])
    
    def WorldToVoxel(self,xyz_w):
        xyz_w = [p-o for p,o in zip(xyz_w,[self.ScanindexX[0],self.ScanindexY[0],self.ScanindexZ[0]])]
        zxy_v = np.dot(self.WorldToVoxelMat,xyz_w)
        return [int(zxy_v[s]) for s in [2,0,1]]
    
    def VoxelToNodule(self,zxy_v):
        xyz_w = self.VoxelToWorld(zxy_v)
        xyz_n = [zxy_v[2], zxy_v[1], xyz_w[2]]
        return xyz_n
    
    def NoduleToVoxel(self,xyz_n):
        if self.flippedXaxis:
            xyz_n[0] = self.Scansize[1]-1 - xyz_n[0]
        if self.flippedYaxis:
            xyz_n[1] = self.Scansize[2]-1 - xyz_n[1]
        z_w = [self.VoxelToWorld([z, int(xyz_n[1]), int(xyz_n[0])])[2] for z in range(self.Scansize[0])]
        zxy_v = [np.argmin(np.absolute(np.subtract(z_w,xyz_n[2]))), int(xyz_n[1]), int(xyz_n[0])]
        return zxy_v