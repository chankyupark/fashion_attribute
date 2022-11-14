# Fashion Multi-attribute Classification Module

This is an implementation of Fashion Multi-attribute Classification Model in [Cloud Robot Project](https://github.com/aai4r/aai4r-master).
The module has two main parts, an human ROI detector and a fashion multi-attribute classifier.
Through each sample image, object human detector detects all human ROIs in the image and transfer all ROIs to multi-attribute classifier.

### Major Environment
* python 3.8
* pytorch 1.7.0
* pytorchvision 0.8.1

### Installation
Setup the environment
```bash
    conda create -n torch17 python=3.8 
    conda activate torch17
    pip install -r requirement.txt
```
## Download model weights

1.   Download yolo-v3 model from [here](https://drive.google.com/file/d/1yCz6pc6qHJD2Zcz8ldDmJ3NzE8wjaiT6/view?usp=sharing) and put in 'fashion_attribute/cfg directory'.  
2.   Downoad fashion multi-attribute classification model from [here](https://drive.google.com/file/d/1oM-KrNIklN4uK14XkxGqlYcSLKKYqv7C/view?usp=sharing) and put in 'fashion_attribute/cfg directory'.
 
   
## Demo
(off-line)
Run the demo code with the sample images.
   ```bash
   python file_demo_outer.py
```
(on-line)   
On-line demo access [here](https://fashion-classifier-demo.herokuapp.com) 
