import pickle

# This file will retreive the mask data from running the deeplab-pytorch.
# Then it will take that data, and spit merge the depth-mapping information.

# data is a dict of with the key being the label of a class, and the
# value being the numpy array of the mask for the image.
data = dict()
dictPath = './deeplab-pytorch/test_images/test.txt'

with open(dictPath, 'rb') as file:
    data = pickle.load(file, fix_imports=True, encoding="ASCII", errors="strict")

