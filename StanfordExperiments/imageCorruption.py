import PIL
import os
import pathlib # required to access the image folders

# Path of the local data directory
train_dir='/common/netthinker/ele3/dog_removed/train'
test_dir = '/common/netthinker/ele3/dog_removed/val'

#train_dir='/work/netthinker/shared/StanfordDataset/StanfordOriginal/train'
#test_dir = '/work/netthinker/shared/StanfordDataset/StanfordOriginal/test'
'''
Approach for Loading the Images and their Labels:
- Create a list tf Dataset containing the path to the image files
- From this list tf dataset, get the (image, label) pairs
'''


'''
For creating image file list Dataset (for training & testing images) 
- First, we need to get the image file names and paths by accessig the image folders. 

We use Path and glob methods in the pathlib module for accessing the image folder. 
'''

train_data_dir = pathlib.Path(train_dir)
train_file_paths = list(train_data_dir.glob('*/*')) # Get the list of all training image file paths
print("\nNumber of training files: ", len(train_file_paths))

test_data_dir = pathlib.Path(test_dir)
test_file_paths = list(test_data_dir.glob('*/*')) # Get the list of all test image file paths
print("Number of test files: ", len(test_file_paths))

print("\nPath of the first training image: ", str(train_file_paths[0]))
PIL.Image.open(str(train_file_paths[0]))



'''
There are two options to create the image file list Dataset.

- Option 1: Create the file list Dataset by using the Dataset.from_tensor_slices() method
- Option 2: Create the file list Dataset by using the Dataset.list_files() method

Note that, before using Option 2, if the filenames have already been globbed, 
then re-globbing every filename with the list_files() method may result 
in poor performance with remote storage systems.

Since, we have already globbed the file names & paths in the first step, we do not use Option 2.
However, the code for option 2 is provided (commented out).
'''


#_______________________Option 1: "from_tensor_slices" method____________________________

train_fnames=[]
for fname in train_file_paths:
    train_fnames.append(str(fname))

print(len(train_fnames))

test_fnames=[]
for fname in test_file_paths:
    test_fnames.append(str(fname))

print(len(test_fnames))

train_corrupt = []
test_corrupt = []
for filename in train_fnames:
    try:
        im = PIL.Image.load(filename)
        im.verify() #I perform also verify, don't know if he sees other types o defects
        im.close() #reload is necessary in my case
        im = PIL.Image.load(filename) 
        im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        im.close()
    except: 
        train_corrupt.append(filename)
        
for filename in test_fnames:
    try:
        im = PIL.Image.load(filename)
        im.verify() #I perform also verify, don't know if he sees other types o defects
        im.close() #reload is necessary in my case
        im = PIL.Image.load(filename) 
        im.transpose(PIL.Image.FLIP_LEFT_RIGHT)
        im.close()
    except: 
        test_corrupt.append(filename)

print("Train Corrupted Images")
print(train_corrupt)

print("Test Corrupted Images")
print(test_corrput)