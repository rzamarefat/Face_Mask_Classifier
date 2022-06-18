# Face Mask Detector
## A real-time package to detect faces and their masked-unmasked state

### Demo
![ezgif com-gif-maker](https://user-images.githubusercontent.com/79300456/173221443-bd2f719d-18b4-47ea-ad67-e00ad917d638.gif)

### How to use
1. First make a conda env using the following command
```
conda craete --name mask_detector
```
2. Install all the dependencies
3. cd into the FaceBoxes package and run the following command
```
sh build_cpu_nms.sh 
```
4. I am not very fond of argparse so I don't use it that much. Please adjust the config file to your machine. If you want to train your own mask classifier you need to provide a dataset and set the the related path. Otherwise, it is not required to do so.

### TODO
1. Compeletion of the instructions in README.md
2. Train a better classifier for mask/unmask with more data and hyperparameter tuning



