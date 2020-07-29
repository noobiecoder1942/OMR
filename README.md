# OMR
Optical Music Recognition



This repository is a short project demonstrating the use of template matching as a crude way of object detection.

Techniques used:

1. Convolution/Separable Convolution
2. Hamming Distance
3. Edge detection using Sobel Filter
4. Hough Transform

Limitations:

1. Assumes that staves are parallel to the horizontal axis - eliminating this constraint might allow for more generalisation for the Hough Space.
2. All threshold values are based on experimentation - hence they work well for some input images and not for others.
3. Denoising the input image and/or the templates might help - currently only Gaussian Blurring is used.

Next steps:

1. Symbol recognition: Currently only symbols are detected. Recognising these symbols is a challenge in case of blurry or not-so-straight images.
2. Generalize Hough Transform to adapt to any orientation of the image.
3. Minimalize overallping bounding boxes using Non-Max Suppression.

Usage:
```
python3 file_name.png
```
