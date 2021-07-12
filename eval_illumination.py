import os
import cv2
import numpy as np

import skimage
from skimage import io
from skimage import transform as tf
from skimage.filters.rank import mean_bilateral
from skimage import morphology

from PIL import Image
from PIL import ImageEnhance

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from collections import OrderedDict

import argparse

from model_illNet import illNet


def preProcess(img):
    
    img[:,:,0] = mean_bilateral(img[:,:,0], morphology.disk(20), s0=10, s1=10)
    img[:,:,1] = mean_bilateral(img[:,:,1], morphology.disk(20), s0=10, s1=10)
    img[:,:,2] = mean_bilateral(img[:,:,2], morphology.disk(20), s0=10, s1=10)
    
    return img
    

def padCropImg(img):
    
    H = img.shape[0]
    W = img.shape[1]

    patchRes = 128
    pH = patchRes
    pW = patchRes
    ovlp = int(patchRes * 0.125)

    padH = (int((H - patchRes)/(patchRes - ovlp) + 1) * (patchRes - ovlp) + patchRes) - H
    padW = (int((W - patchRes)/(patchRes - ovlp) + 1) * (patchRes - ovlp) + patchRes) - W

    padImg = cv2.copyMakeBorder(img, 0, padH, 0, padW, cv2.BORDER_REPLICATE)

    ynum = int((padImg.shape[0] - pH)/(pH - ovlp)) + 1
    xnum = int((padImg.shape[1] - pW)/(pW - ovlp)) + 1

    totalPatch = np.zeros((ynum, xnum, patchRes, patchRes, 3), dtype=np.uint8)

    for j in range(0, ynum):
        for i in range(0, xnum):

            x = int(i * (pW - ovlp))
            y = int(j * (pH - ovlp))

            totalPatch[j, i] = padImg[y:int(y + patchRes), x:int(x + patchRes)]

    return totalPatch


def load_model(modelPath):
    model = illNet()

    if torch.cuda.is_available():
        model = model.cuda()
        
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(modelPath))
    else:
        state_dict = torch.load(modelPath, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)  
        
    model.eval()
    return model

def illCorrection(model, totalPatch):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    totalPatch = totalPatch.astype(np.float32) / 255.0

    ynum = totalPatch.shape[0]
    xnum = totalPatch.shape[1]
    scal = totalPatch.shape[2]

    totalResults = np.zeros((ynum, xnum, 128, 128, 3), dtype = np.float32)

    for j in range(0, ynum):
        for i in range(0, xnum):

            patchImg = totalPatch[j, i]
            patchImg = transform(patchImg)

            if torch.cuda.is_available():
                patchImg = patchImg.cuda()

            patchImg = patchImg.view(1,3,128,128)
            patchImg = Variable(patchImg)

            output = model(patchImg)
            output = output.permute(0, 2, 3, 1).data.cpu().numpy()[0]

            output[output>1] = 1
            output[output<0] = 0
            output = output*255.0
            output = output.astype(np.uint8)

            totalResults[j,i] = output
            
    return totalResults

def composePatch(totalResults):

    ynum = totalResults.shape[0]
    xnum = totalResults.shape[1]
    patchRes = totalResults.shape[2]
    
    ovlp = int(patchRes * 0.125)
    step = patchRes - ovlp
    
    resImg = np.zeros((patchRes + (ynum - 1) * step, patchRes + (xnum - 1) * step, 3), np.uint8)
    
    for j in range(0, ynum):
        for i in range(0, xnum):
            
            sy = int(j*step)
            sx = int(i*step)

            resImg[sy:(sy + patchRes), sx:(sx + patchRes)] = totalResults[j, i]
            
    return resImg

def postProcess(img, H, W):
    img = Image.fromarray(img)
    enhancer = ImageEnhance.Contrast(img)
    factor = 2.0
    img = enhancer.enhance(factor)

    return img.crop((0, 0, W, H))


# For parsing commandline arguments
def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inPath", type=str, default='./test')
    parser.add_argument("--outPath", type=str, default='./output')
    parser.add_argument("--modelPath", type=str, default='./model_illNet.pkl')
    args = parser.parse_args()

    return args


args = init_args()
folderPath = args.inPath
outPath = args.outPath
modelPath = args.modelPath

model = load_model(modelPath)

img_lst = os.listdir(folderPath)
img_lst.sort()
for file in img_lst:
    imgPath = os.path.join(folderPath, file)
    img = io.imread(imgPath)
    H, W = img.shape[:2]

    img = preProcess(img)
    totalPatch = padCropImg(img)
    totalResults = illCorrection(model, totalPatch)
    resImg = composePatch(totalResults)
    resImg = postProcess(resImg, H, W)

    resImg = resImg.convert('L') # convert to Gray img
    savePath = os.path.join(outPath, file)

    resImg.save(savePath)