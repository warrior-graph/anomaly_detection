----------------------
Citation Details
----------------------
  
Please cite the following paper when using this source code:

  V. Reddy, C. Sanderson, B.C. Lovell.
  Improved Anomaly Detection in Crowded Scenes via Cell-based Analysis of Foreground Speed, Size and Texture.
  IEEE Conf. Computer Vision and Pattern Recognition Workshops (CVPRW), 2011.

  http://dx.doi.org/10.1109/CVPRW.2011.5981799



----------------------
License
----------------------
  
The source code is provided without any warranty of fitness for any purpose.
You can redistribute it and/or modify it under the terms of the
GNU General Public License (GPL) as published by the Free Software Foundation,
either version 3 of the License or (at your option) any later version.
A copy of the GPL license is provided in the "GPL.txt" file.



----------------------
Instructions and Notes
----------------------

To run the code the following libraries must be installed:
1. OpenCV 2.4 (later versions should also work)
2. Armadillo 3.920 - http://arma.sourceforge.net

To compile the code use the following command (all on one line):
g++ -o AnomalyDetection code/main.cpp code/ParameterInfo.cpp code/DisplayImage.cpp code/AnomalyDetection.cpp -O2 -fopenmp -I/usr/include/opencv -I/usr/local/include/opencv -L/usr/lib -L/usr/local/lib -larmadillo -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video

You may need to adapt the paths for libraries and includes to suit your environment.
The above command line has been tested on Fedora 19, using Armadillo 3.920.2 and OpenCV 2.4.6

NOTE: _do not_ change the current directory to "code" when compiling.

After successful compilation, the program can be executed using:
./AnomalyDetection  


There are sample test and train images along with their foreground masks to perform the execution.


Points to note:

1.
The paths for various input files (images & their corresponding foreground masks)
are listed in the file "input/ad_input_fread.txt"

2.
Initially, the algorithm uses the frames listed in the file "input/training_images.txt"
to build a model of what is considered as normal activities before detecting
the anomalies for each frame. The corresponding foreground masks for the training images
must be listed in the file "input/training_masks.txt"

3.
A few temporary files & folders are created and stored in the "output" folder during execution.
The folders "output/ad_masks" and "output/ad_model_prms" are hard-coded and must not be deleted.
The folder "output/ad_masks" contains the output images.

4.
To save the masks, WRITE_OUTPUT must be defined in main.hpp (by default, this is defined).
An output folder is automatically created to store the detections in each image.
The output masks are stored as png images. 

5.
Internally, the code sorts the input image files of a given folder in ascending order.
Hence, the file names must contain a constant number of digits in their suffixes
(eg. test_0001, test_0002, test_0100, test_1000, ...)

6.
This research code is currently not optimised for speed.



