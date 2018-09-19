import cv2
import numpy


def getSegmentedDepths(depthPath, segmentedPath):
    # segmentedData is a dict of with the key being the label of a class, and the
    # value being the numpy array of the mask for the image.
    segmentedData = dict()
    with open(segmentedPath, 'rb') as file:
        segmentedData = numpy.load(file, fix_imports=True, encoding="ASCII")
    segmentedData = dict(numpy.ndenumerate(segmentedData))

    # depth is a numpy array of the depth of the image, the size of the image
    depthData = None
    with open(depthPath, 'rb') as file:
        depthData = numpy.load(file, fix_imports=True, encoding="ASCII")

    # because of weird numpy translation
    for fakeKey in segmentedData:
        trueSegmentedData = segmentedData[fakeKey]
        objects = dict()

        # For each segmented object
        for realKey in trueSegmentedData:

            # numpy array of masked element
            element = trueSegmentedData[realKey]

            mappingRows =  len(depthData) / len(element)
            mappingCols = len(depthData[0]) / len(element[0])

            totalDepth = 0
            for row in range(len(element)):
                for col in range(len(element[0])):
                    depthRow = min(len(depthData) - 1, int(row*mappingRows))
                    depthCol = min(len(depthData[0]) - 1, int(col*mappingCols))
                    totalDepth += depthData[depthRow, depthCol]

            # calculate the depth values of the specific element
            #totalDepth = numpy.sum(depthData * element)
            totalPixels = numpy.sum(trueSegmentedData[realKey])

            averageDepth = totalDepth / max(1, totalPixels)

            objects[realKey] = [averageDepth, element]
    return objects

# add the blurred picture for each segmented element to the dictionary
def getBlurredPictures(objectMaps, originalStyleCopy):
    newMap = dict()
    for key in objectMaps:
        info = objectMaps[key]
        depth = int(5 - min(5, info[0]*100))
        depth = max(1, depth)
        print(depth)
        newImage = cv2.blur(originalStyleCopy, (depth, depth))
        newMap[key] = [depth, info[1], newImage]
    return newMap


def main(depthPath, segmentedPath, styleImagePath):
    with open(depthPath, 'rb') as file:
        depthData = numpy.load(file, fix_imports=True, encoding="ASCII")

    # dictionary set up like the following:
    # {name: [averageDepthValue, numpyArray of corresponding pixels]}
    objectMaps = getSegmentedDepths(depthPath, segmentedPath)

    # bat.jpg is the batman image.
    styleImage = cv2.imread(styleImagePath)
    cv2.imwrite('styleCopy.jpg', styleImage)
    styleCopy = cv2.imread('styleCopy.jpg', cv2.IMREAD_COLOR)

    # dictionary set up like the following:
    # {name: [averageDepthValue, numpyArray of corresponding pixels,
    #          blurredPictureCorresponding to depth value]}
    objectMaps = getBlurredPictures(objectMaps, styleCopy)

    #return
    # Blur by depth
    for row in range(len(styleCopy)):
        for col in range(len(styleCopy[row])):
            for key in objectMaps:
                maskedImage = objectMaps[key][1]
                factorRow = len(maskedImage) / len(styleCopy)
                factorCol = len(maskedImage[0]) / len(styleCopy[0])
                maskedRow = int(min(len(maskedImage) - 1, factorRow * row))
                maskedCol = int(min(len(maskedImage[0]) - 1, factorCol * col))
                if (maskedImage[maskedRow , maskedCol] == 1):
                    break

            pixel = objectMaps[key][2][row, col]
            styleCopy[row, col] = pixel

    cv2.imwrite('styleCopy2.jpg', styleCopy)


main("../all_image_data/tower/tower_depth_array.npy",
    "../all_image_data/tower/tower_segmentation_array.npy",
     "../all_image_data/tower/tower_style.jpg")



# arrs = []
#     for key in objectMaps:
#         arrs.append(objectMaps[key][1])

#     arrsOnes = {0:0, 1:0, 2:0, 3:0, 4:0}
#     index = 0
#     for row in range(len(arrs[0])):
#         for col in range(len(arrs[0][0])):
#             index = 0
#             for key in objectMaps:
#                 maskedImage = objectMaps[key][1]
#                 if (int(maskedImage[row, col]) == 1):
#                     arrsOnes[index] += 1
#                 index += 1
#     print(arrsOnes)
#     return


