# Fashion Multi-attribute Classification Module

This is an implementation of Fashion Multi-attribute Classification Model in [Cloud Robot Project](https://github.com/aai4r/aai4r-master).
The module has two main parts, an human ROI detector and a fashion multi-attribute classifier.
Through each sample image, object human detector detects all human ROIs in the image and transfer all ROIs to multi-attribute classifier.

### Environment
* python 3.8
* pytorch 1.7.0
* pytorchvision 0.8.1

### Installation
0. Setup the environment
    conda create -n torch17 python=3.8 
    conda activate torch17
    pip install -r requirements.txt

1. Clone this repository.
    ```bash
    git clone https://github.com/aai4r/aai4r-ServiceContextUnderstanding
    cd aai4r-ServiceContextUnderstanding
    ```


3. Make output folder and download [all weight files (detection and classification)](https://drive.google.com/drive/folders/1K4BJ0HryAPMJSsRM4e8NcVmwq_lWSn-j?usp=sharing) and move them to output folder.
    ```bash
    mkdir output
    ```
    ```bash
    ├── output
    │   ├── class_info_Food101.pkl
    │   ├── class_info_FoodX251.pkl
    │   ├── class_info_Kfood.pkl
    │   ├── faster_rcnn_1_7_9999.pth
    │   └── model_best.pth.tar
    ```
 
4. Download [nanumgothic.ttf](https://fonts.google.com/download?family=Nanum%20Gothic) and install (unzip, mv the folder to /usr/share/fonts/, then fc-cache -f -v)
   ```
   Copyright (c) 2010, NAVER Corporation (https://www.navercorp.com/),

   with Reserved Font Name Nanum, Naver Nanum, NanumGothic, Naver NanumGothic, NanumMyeongjo, 
   Naver NanumMyeongjo, NanumBrush, Naver NanumBrush, NanumPen, Naver NanumPen, Naver NanumGothicEco, 
   NanumGothicEco, Naver NanumMyeongjoEco, NanumMyeongjoEco, Naver NanumGothicLight, NanumGothicLight, 
   NanumBarunGothic, Naver NanumBarunGothic, NanumSquareRound, NanumBarunPen, MaruBuri

   This Font Software is licensed under the SIL Open Font License, Version 1.1.
   This license is copied below, and is also available with a FAQ at: http://scripts.sil.org/OFL

   SIL OPEN FONT LICENSE
   Version 1.1 - 26 February 2007 
   ```
   
### Run
#### Webcamera demo

Run the demo code with the webcam number (we used 0).
   ```bash
   python my_demo_det_multi.py --webcam_num 0
   ```
   
#### Image demo

Run the demo code with the sample images.
   ```bash
   python my_demo_det_multi.py --image_dir sample_images
   ```
   
#### CLOi demo

The code will be updated.
