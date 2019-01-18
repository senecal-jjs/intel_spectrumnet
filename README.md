## The SpectrumNet architecture for multi-spectral imagery

This project defines two small and efficient convolutional neural network architectures for use with multi-spectral imagery. The networks combine aspects of MobileNets and SqueezeNets (which were originally developed with embedded systems in mind) with adaptations to work well with high-dimensional imagery containing many spectral bands.

This architecture has been tested on the EuroSat dataset, which contains a collection of images from the European Space Agency's Sentinel 2A and 2B satellites, which carry 13 band multi-spectral instruments. The architecture has also been tested on a small scale hyper-spectral dataset which contains classes of produce at different stages of aging. This data has been collected as part of my Master's thesis project, and this code is a subset of that work presented for use on other multi-spectral and hyper-spectral problems.

## Using the architecture

The network can be used with version 1.0 which uses conventional convolutional layers, or version 1.1 which uses depthwise separable convolutions. The network architecture currently expects images to be of size (64,64), and the data loader in GeoTiffDataset.py will resize input images to conform to this requirement. This can be updated by changing the resize parameters on line 100 of the GeoTiffDataset.py file, and updating the parameter value in the final nn.AvgPool2D layer to match the output filter size of the final convolutional layer. 

### Example network declaration

net = SpectrumNet(num_bands=num_bands, version=version)

Where num_bands is the number of spectral bands that the input images will contain, and version should be either 1.0 or 1.1.

### Transfer learning 

To use this network in a transfer learning capacity the final convolutional layer should be replaced. The "get_pretrained_network" function in the main.py file provides an example of how to do this. The SpectrumNet architecture does not contain any fully connected layers to reduce the number of network parameters, necessitating the replacement of the final convolutional layer for transfer learning. 

### Training a model

The main.py file provides an example of how to train a model on a dataset consisting of a train, validation, and test set. Certain variables are hard-coded to my personal file paths and naming conventions and will need to be updated for other users. For example, filepaths to pre-trained models, and datasets are hard coded, as well as the means and standard deviations of the spectral channels in my produce dataset which are used for standardization. 

The dataset folder is expected to contain a "train", "val", and "test" folder each of which contain subfolders corresponding to each class in your dataset. 

Example usage: python main.py -n 1 2 3 

This would select bands 1, 2, and 3 from your input images to be used for training. The package rasterio is being used to do band selection and begins indexing at 1 rather than 0. 

Example usage: python main.py -n 2 7 8 10 13

This would select spectral bands 2, 7, 8, 10, and 13 to be used for training. 

## Utilities

The utilities package contains functions for converting ".bil" hyperspectral images into a geotiff format, as well as functions for converting images which are reported in terms of digital number to physically meaningfull reflectance values. The function "calibrate" performs the conversion, and relies on a numpy array of reflectance values measured from a spectralon panel to peform a valid flat field correction from digital number to reflectance. To use this conversion you must use spectralon reflectance data taken from the environment in which you captured each of your images. 

### Disclaimer
The documentation for this code is not exhaustive and is merely meant to be a guide for anyone who encounters this repository. I will link my thesis to this repo when it is complete.  



