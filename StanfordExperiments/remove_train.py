import scipy.io
import os
import glob
import shutil
import random
from PIL import Image
from distutils.dir_util import copy_tree
import pathlib
import csv

one = None
with open("file.csv", 'r') as csv_file:
    file = csv.reader(csv_file)
    for k in file:
        one = k

for x in one:
    try:
        shutil.rmtree("/common/netthinker/ele3/ILSVRC/Data/CLS-LOC/train/" + x)
    except:
        print("Error deleting directory")
        
for x in one:
    try:
        shutil.rmtree("/common/netthinker/ele3/ILSVRC/Data/CLS-LOC/val/" + x)
    except:
        print("Error deleting directory for val")