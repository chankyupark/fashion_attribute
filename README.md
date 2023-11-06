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

1.   Download yolo-v3 model from [here](https://drive.google.com/file/d/1kD12GEZw6nRYaqO9-1m8dairpX89z5VE/view?usp=sharing) and put in 'fashion_attribute/cfg directory'.  
2.   Downoad fashion multi-attribute classification model from [here](https://drive.google.com/file/d/1hu3F7Ly1rEbk8L8OZCeVgZba-IYrPKXB/view?usp=sharing) and put in 'fashion_attribute/cfg directory'.
 
   
## Demo
(off-line)
Run the demo code with the sample images.
   ```bash
   python file_demo_outer.py
```
## Test Dataset
You can get Test Dataset of fashion attribute that includes 5,000 images and Ground Truth labels
Download Test dataset from [here](https://drive.google.com/file/d/1JGNKF9vusQcZ6Did7SyNc3nQexPoLJ70/view?usp=sharing) 

(on-line)   
On-line demo access [here](https://fashion-classifier-demo.herokuapp.com) 
