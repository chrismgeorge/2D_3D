import numpy

# This file will retreive the mask data from running the deeplab-pytorch.
# Then it will take that data, and spit merge the depth-mapping information.

# segmentedData is a dict of with the key being the label of a class, and the
# value being the numpy array of the mask for the image.
segmentedData = dict()
dictPath = './all_image_data/school/school_segmentation_array.npy'

with open(dictPath, 'rb') as file:
    segmentedData = numpy.load(file, fix_imports=True, encoding="ASCII")

# depth is a numpy array of the depth of the image, the size of the image
depthData = None
dictPath = './all_image_data/school/school_depth_array.npy'

with open(dictPath, 'rb') as file:
    depthData = numpy.load(file, fix_imports=True, encoding="ASCII")

# print(segmentedData)
# print(depthData)

print("depth rows", len(depthData.tolist()))
print("depth cols", len(depthData.tolist()[0]))
segmentedData = dict(numpy.ndenumerate(segmentedData))

# because of weird numpy translation
for fakeKey in segmentedData:
    trueSegmentedData = segmentedData[fakeKey]
    for realKey in trueSegmentedData:
        print("segmented cols", len(trueSegmentedData[realKey].tolist()))
        print("segmented rows", len(trueSegmentedData[realKey].tolist()[0]))
        break
    break


