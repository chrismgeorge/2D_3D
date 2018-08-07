import numpy
import cv2

# This file will retreive the mask data from running the deeplab-pytorch.
# Then it will take that data, and spit merge the depth-mapping information.

# segmentedData is a dict of with the key being the label of a class, and the
# value being the numpy array of the mask for the image.
segmentedData = dict()
dictPath = './all_image_data/nyc/nyc_segmentation_array.npy'

with open(dictPath, 'rb') as file:
    segmentedData = numpy.load(file, fix_imports=True, encoding="ASCII")

# depth is a numpy array of the depth of the image, the size of the image
depthData = None
dictPath = './all_image_data/nyc/nyc_depth_array.npy'

with open(dictPath, 'rb') as file:
    depthData = numpy.load(file, fix_imports=True, encoding="ASCII")

segmentedData = dict(numpy.ndenumerate(segmentedData))


pic = cv2.imread('./all_image_data/nyc/nyc.jpg', cv2.IMREAD_COLOR)

# because of weird numpy translation
for fakeKey in segmentedData:
    trueSegmentedData = segmentedData[fakeKey]
    objects = dict()
    for realKey in trueSegmentedData:
        depthOfObject = depthData * trueSegmentedData[realKey]
        totalDepth = numpy.sum(depthOfObject)
        totalPoints = numpy.sum(trueSegmentedData[realKey])

        averageZ = totalDepth / totalPoints

        # get the average rgb value
        rgb = [0, 0, 0]
        # x = cols
        # y = rows
        smallestX = -1
        smallestY = -1
        largestX = -1
        largestY = -1
        for row in range(len(trueSegmentedData[realKey].tolist())):
            for col in range(len(trueSegmentedData[realKey][0].tolist())):
                if (numpy.sum(trueSegmentedData[realKey][row] != 0)):
                    val = trueSegmentedData[realKey][row, col]
                    if (val != 0):
                        # pic gets gbr
                        rgb[0] += pic[row, col, 2]
                        rgb[1] += pic[row, col, 1]
                        rgb[2] += pic[row, col, 0]
                        if (smallestX == -1):
                            smallestX = col
                            largestX = col
                            smallestY = row
                            largestY = row
                        else:
                            smallestX = min(smallestX, col)
                            largestX = max(largestX, col)
                            smallestY = min(smallestY, row)
                            largestY = max(largestY, row)

        centerX = (largestX + smallestX) / 2
        centerY = (largestY + smallestY) / 2
        totalWidth = largestX - smallestX
        totalHeight = largestY - smallestY

        # average out the colors
        rgb[0] = rgb[0] / totalPoints
        rgb[1] = rgb[1] / totalPoints
        rgb[2] = rgb[2] / totalPoints

        objects[realKey] = ['Cube', totalWidth, totalHeight, centerX, centerY, averageZ, rgb]

    # write new dictionary to the file with maya information
    with open('./all_image_data/nyc/nyc_maya_input.txt', 'w') as file:
        file.write(str(objects))
        file.truncate()


