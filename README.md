## Image segmentation
Repo (https://github.com/kazuto1011/deeplab-pytorch)

This repo provides the ability to semantically segment a single image using a pre-trained model. It returns n images that each represent a specific class, with a mask over the specified location of a class. This is done by taking the masks found, and exporting the numpy array of the mask to another file, where a dictionary of the numpy array. The dictionary is marked by the labels of each mapping. A downside to using this image segmentation is that unique buildings are not inherently marked somehow, but rather every building would be marked under buildings.

## Depth Mapping
Repo (https://github.com/mrharicot/monodepth)

This repo provides the ability to get the depth mappings of a single image using a pre-trained model.

## Combining _Image Segmentation & Depth Mapping_

The depth map produces a new image of the same width and height of the original image, but the pixels are representative of the depth at a particular pixel. Using that information, we could theoretically simply recreate a 3D image pixel by pixel in Maya, but that wouldn’t look right in any perspective, and create an ungodly amount of shapes within a scene. 
To combat this, we can use image segmentation to pull out specific objects in a scene. With image segmentation, we can get the width, and height of whatever object is being represented, as well as the the average x, and y location of the object. We can also get the average rgb value by averaging every color for each segmented block. Then using the depth map on the same image, we can get it’s average depth. Thus we are able to recreate the average, x, y, z,  width, height, and color The only import attribute that is missing is depth of the created shape. This issue has yet to be addressed. 
For each, object found in the image, we can recursively perform this same operation to produce to define the object even more. The deeper the recursion, the more defined your scene will be. 
In the current format, I will just be saying that every object is a rectangle, but in the future it will be cool to find or create some kind of object → shape network to more closely recreate the scene.
