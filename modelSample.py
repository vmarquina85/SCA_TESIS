from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from PIL import Image
import cv2
import os
import time

load_data = torch.load('modelResnet.pt') 

print (load_data)