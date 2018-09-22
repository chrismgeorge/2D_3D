import cv2
import numpy
import math


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
        depth = int(6 - min(6, info[0]*100))
        depth = max(1, depth)
        print(depth)
        # Add something here to get a baseline and then decide how to blur
        # the photos from there.
        newImage = cv2.blur(originalStyleCopy, (depth, depth))
        newMap[key] = [depth, info[1], newImage]
    return newMap


def main(depthPath, segmentedPath, styleImagePath, originalImagePath, name):
    with open(depthPath, 'rb') as file:
        depthData = numpy.load(file, fix_imports=True, encoding="ASCII")

    # dictionary set up like the following:
    # {name: [averageDepthValue, numpyArray of corresponding pixels]}
    objectMaps = getSegmentedDepths(depthPath, segmentedPath)

    # original image
    originalImage = cv2.imread(originalImagePath)

    # Style image
    styleImage = cv2.imread(styleImagePath)
    cv2.imwrite(name+'StyleCopy.jpg', styleImage)
    styleCopy = cv2.imread(name+'StyleCopy.jpg', cv2.IMREAD_COLOR)
    styleCopyBrightness = cv2.imread(name+'StyleCopy.jpg', cv2.IMREAD_COLOR)

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

            # pixel = (Blue, Green, Red)
            pixel = objectMaps[key][2][row, col]
            newLuminance = (0.2126*pixel[2] + 0.7152*pixel[1] + 0.0722*pixel[0])

            # not modifying row and col because original image and style image
            # are the same size
            fRow = len(originalImage) / len(styleCopy)
            fCol = len(originalImage[0]) / len(styleCopy[0])
            ogRow = int(min(len(originalImage) - 1, fRow * row))
            ogCol = int(min(len(originalImage[0]) - 1, fCol * col))
            ogImagePixel = originalImage[ogRow, ogCol]
            oldLuminance = (0.2126*ogImagePixel[2] + 0.7152*ogImagePixel[1] + 0.0722*ogImagePixel[0])

            # a number < 1 means that old pixel is brighter than the new one
            # so we need to multiple the new pixel by the ratio to brighten it
            ratio = (oldLuminance / newLuminance)
            if (ratio > 1):
                ratio = 1 + math.log(ratio, 10)
            else:
                # scale to .8 -> 1
                ratio = (ratio/5) + .8


            brightPixel = pixel*ratio

            # keep the values between 0, 255
            maxNewValue = max(brightPixel)
            if maxNewValue > 255:
                diff = maxNewValue / 255
                brightPixel = brightPixel / diff

            styleCopy[row, col] = pixel
            styleCopyBrightness[row, col] = brightPixel

    cv2.imwrite(name+'StyleCopyModified.jpg', styleCopy)
    cv2.imwrite(name+'StyleCopyModifiedBright.jpg', styleCopyBrightness)

names = ["tower"]
for name in names:
    main("../all_image_data/"+name+"/"+name+"_depth_array.npy",
         "../all_image_data/"+name+"/"+name+"_segmentation_array.npy",
         "../all_image_data/"+name+"/"+name+"_style.jpg",
         "../all_image_data/"+name+"/"+name+".jpg",
         name)

