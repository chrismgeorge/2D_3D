import numpy
import cv2

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

segmentedData = dict(numpy.ndenumerate(segmentedData))


pic = cv2.imread('./all_image_data/school/school.jpg', cv2.IMREAD_COLOR)

# because of weird numpy translation
for fakeKey in segmentedData:
    trueSegmentedData = segmentedData[fakeKey]
    objects = dict()
    for realKey in trueSegmentedData:
        depthOfObject = depthData * trueSegmentedData[realKey]
        totalDepth = numpy.sum(depthOfObject)
        totalPoints = numpy.sum(trueSegmentedData[realKey])

        averageWidth = 0
        averageHeight = 0
        averageX = 0
        averageY = 0
        averageZ = totalDepth / totalPoints

        # get the average rgb value
        rgb = [0, 0, 0]
        for row in range(len(trueSegmentedData[realKey].tolist())):
            for col in range(len(trueSegmentedData[realKey][0].tolist())):
                if (numpy.sum(trueSegmentedData[realKey][row] != 0)):
                    if (numpy.sum(trueSegmentedData[realKey][row, col] != 0)):
                        # pic gets gbr
                        rgb[0] += pic[row, col, 2]
                        rgb[1] += pic[row, col, 1]
                        rgb[2] += pic[row, col, 0]

        # average out the colors
        rgb[0] = rgb[0] / totalPoints
        rgb[1] = rgb[1] / totalPoints
        rgb[2] = rgb[2] / totalPoints
        print(rgb)
        objects[realKey] = [averageWidth, averageHeight, averageX, averageY, averageZ, rgb]


