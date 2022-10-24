# -*- coding: utf-8 -*-
import torch
import numpy as np 
import argparse
import pickle 
import os
from os import listdir, getcwd
import os.path as osp
import glob
import torchvision
from torchvision import transforms 
import torch.backends.cudnn as cudnn
from darknet import Darknet
from PIL import Image
from util import *
import cv2
import pickle as pkl
import random
from preprocess import prep_image
from preprocess import automatic_brightness_and_contrast
import natsort

import resnet50 as model_n 
import resnet50_attn as model_attn
import resnet50_fc as model_fc
import matplotlib.pyplot as plt
import sys

#file based demo progarm
#funtion : upper + lower , upper only, lower only

transform_test = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
#  "winter scarf", "cane", "bag", "shoes", "hat", "face"]
#attribute categories = #6
colors_a =   ["----", "white", "black", "gray", "pink", "red", "green", "blue", "brown", "navy", "beige", \
    "yellow", "purple", "orange", "mixed-color", "other-color"]
pattern_a =  ["----", "plain", "checker", "dotted", "floral", "striped", "mixed", "stripe-horizon", "stripe-vertical", \
        "letter", "diamond", "character", "leopard", "lace", "others"]
gender_a =   ["----", "man", "woman"]
season_a =   ["----", "spring", "summer", "fall", "winter"]
top_type_a =    ["----", "shirt", "jumper", "jacket", "vest", "parka", "coat", "dress", "sweater", "t-shirt", "top", \
             "blouse", "blazer", "cardigon"]
sleeves_a =  ["----", "short-sleeves", "long-sleeves", "no-sleeves"]
texture_a =  ["normal", "normal", "fur", "denim", "leathers", "shiny", "wool", "knit"]
button_a =   ["none-button", "none-button", "zipper", "button", "open", "belt"]
length_a =   ["----", "short", "medium", "long"]
fit_a =      ["----", "normal", "slim", "loose"]
collar_a =   ["----", "none", "v-neck", "square-neck", "round-neck", "turtle", "v-shape", "round-shirt", "notched", \
        "off-shoulder", "hood", "band"]

    #"winter scarf", "cane", "bag", "shoes", "hat", "face"]

#Bottom => 5 attributes are shared by top attributes items: colors, pattern, gender, season, length

bottom_type_a = ["----", "pants", "skirt", "jeans", "tights", "hot-pants", "suit", "capri", "leggings"]
leg_pose_a = ["----", "standing", "sitting", "lying"]

#Acceary -> 4 attributes : color, gender, season shared
acc_type_a = ["----", "scarf/muffler", "cane", "bag", "shose", "hat", "sandles", "boots", "heels"]

#face attributes : face gender is shared

glasses_a =  ["----", "none-glasses", "glasses", "sun-glasses"]

# style
style_a = ["----", "none", "rocker", "casual", "comfortable", "basic", "eclectic", "trendy", "classic", "chic", "urban", "romantic", \
           "elegant", "bohemian", "sexy", "preppy", "denim", "military", "school", "sport", "hiking", "uniform", "suit"]


attribute_pool = [colors_a, pattern_a, gender_a, season_a, top_type_a, sleeves_a, \
                  texture_a, button_a, length_a, fit_a, collar_a , \
                      colors_a, pattern_a, gender_a, season_a, length_a, bottom_type_a, leg_pose_a, \
                          gender_a, glasses_a, style_a]

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
 #   inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def main(args):

    #1. Yolov3 : for Human ROI detection
    num_classes = 80
    yolov3  = Darknet(args.cfg_file)
    yolov3.load_weights(args.weights_file)
    yolov3.net_info["height"] = args.reso    
    inp_dim = int(yolov3.net_info["height"])
    assert inp_dim % 32 == 0 
    assert inp_dim > 32
    yolov3.to(device)
    yolov3.eval()

    print("yolo-v3 network successfully loaded")

    attribute_dim = [16, 15, 3, 5, 14, 4, 8, 6, 4, 4, 12, 16, 15, 3, 5, 4, 9, 4, 3, 4, 23] #21개

    #2. listing image files from sample directory
    try:
        list_dir = os.listdir(args.test)
     #   list_dir.sort(key=lambda f: int(filter(str.isdigit, f)))
     #  list_dir.sort(key=lambda x: int(x[:-4]))
        list_dir = natsort.natsorted(list_dir, reverse=False)
        imlist = [osp.join(osp.realpath('.'), args.test, img) for img in list_dir if os.path.splitext(img)[1] =='.jpg'  or os.path.splitext(img)[1] == '.jpeg' or os.path.splitext(img)[1] =='.JPG' or os.path.splitext(img)[1] =='.png']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), args.test))
        print('Not a directory error')
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(args.test))
        exit()

     #3. loading model
    check_point = torch.load('cfg/'+ args.model+ '_best.pth.tar')
    state_dict = check_point['state_dict']
    if args.arch == 'resnet50':
        model = model_n.__dict__['resnet50'](pretrained=True, num_classes=len(attribute_dim), attribute_dim=attribute_dim)
    
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    print(args.model)
    
    #4. demo routine for each sample file
    with torch.no_grad():
        for inx, image in enumerate(imlist):

            print('\n'+ list_dir[inx])
            orig_pil_image = Image.open(image)
            image, orig_img, im_dim, orig  = prep_image(image, inp_dim)
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)
            
            image_tensor = image.to(device)
            im_dim = im_dim.to(device)
            
            #4-1. detect candidates for human ROI
            detections = yolov3(image_tensor, device, True) # prediction mode for yolo-v3
            detections = write_results(detections, args.confidence, num_classes, device, nms=True, nms_conf=args.nms_thresh)
            # original image dimension --> im_dim

            os.system('clear')
            if type(detections) != int: 
                if detections.shape[0]:
                    
                    im_dim = im_dim.repeat(detections.shape[0], 1)
                    scaling_factor = torch.min(inp_dim/im_dim, 1)[0].view(-1, 1)
                    
                    detections[:, [1, 3]] -= (inp_dim - scaling_factor*im_dim[:, 0].view(-1, 1))/2
                    detections[:, [2, 4]] -= (inp_dim - scaling_factor*im_dim[:, 1].view(-1, 1))/2

                    detections[:, 1:5] /= scaling_factor

                    small_object_ratio = torch.FloatTensor(detections.shape[0])

                    for i in range(detections.shape[0]):
                        detections[i, [1, 3]] = torch.clamp(detections[i, [1, 3]], 0.0, im_dim[i, 0])
                        detections[i, [2, 4]] = torch.clamp(detections[i, [2, 4]], 0.0, im_dim[i, 1])

                        object_area = (detections[i, 3] - detections[i, 1])*(detections[i, 4] - detections[i, 2])
                        orig_img_area = im_dim[i, 0]*im_dim[i, 1]
                        small_object_ratio[i] = object_area/orig_img_area
                    
                    #4-2. remove small human ROI
                    detections = detections[small_object_ratio > 0.05]
                    im_dim = im_dim[small_object_ratio > 0.05] 
                    bboxs = detections[:, 1:5].clone()
                    
                    if detections.size(0) > 0:

                        Roi = detections.cpu().numpy().astype(int)
                        #4-3 space margin for accessory
                        rois = []
                        for i in range(detections.shape[0]):
                            #roi = orig_img[Roi[i][2]:Roi[i][4], Roi[i][1]:Roi[i][3]]
                            roi = orig_pil_image.crop([Roi[i][1], Roi[i][2], Roi[i][3], Roi[i][4]])
                        #    roi.save(str(i)+ list_dir[inx])
                            roi = transform_test(roi).unsqueeze(0)
                            rois.append(roi)
                        
                        rois = torch.cat(rois, 0).cuda()
                        outputs = model(rois)
                                                
                        #4.3 ouput multi-attributre results for fahion clothing
                        for i in range(detections.shape[0]):
                            sampled_caption = []
                            dress = False
                            sampled_caption.append(' top       :')

                            for j in range(len(outputs)):
                                #temp = outputs[j][i].data
                                max_index = torch.max(outputs[j][i].data, 0)[1]
                                word = attribute_pool[j][max_index]

                                if j == 1 : # pattern
                                    sampled_caption.append(word + '-pattern')
                                elif j == 6 : # texture
                                    sampled_caption.append(word + '-texture')
                                #elif j == 7 : # button
                                #    sampled_caption.append(word + '-button')
                                elif j == 8 : # length
                                    sampled_caption.append(word + '-length')
                                elif j == 9 : # fit
                                    sampled_caption.append(word + '-fit')
                                elif j == 10 : # collar
                                    sampled_caption.append(word + '-collar \n bottom    :')
                                elif j == 12 : # pattern
                                    sampled_caption.append(word + '-pattern')
                                elif j == 17 : 
                                    sampled_caption.append('\n face      :')
                                elif j == 19 : 
                                    sampled_caption.append(word + ' \n style     :')
                                else:
                                    sampled_caption.append(word)

                            sentence = ' '.join(sampled_caption)
                    
                            print ('\n'+ str(i+1) + ') ' + '\n' + sentence)
                            write(Roi[i], orig_img, sentence, i+1, coco_classes, colors)
            
            cv2.imshow("frame", orig_img)
            key = cv2.waitKey(0)
            os.system('clear')
            if key & 0xFF == ord('q'): 
                break
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()  
    
    
    parser.add_argument('--model', type=str, default='41k_outer', help='path for trained encoder')
    parser.add_argument('--arch', type=str, default='resnet50', help='arch for main model')
    parser.add_argument('--test', type=str, default='image', help='path for vocabulary wrapper')
   
    # Encoder - Yolo-v3 parameters 
    parser.add_argument('--confidence', type=float, default = 0.5, help = 'Object Confidence to filter predictions')
    parser.add_argument('--nms_thresh', type=float , default = 0.4, help = 'NMS Threshhold')
    parser.add_argument('--cfg_file', type = str, default = 'cfg/yolov3.cfg', help ='Config file')
    parser.add_argument('--weights_file', type = str, default = 'cfg/yolov3.weights', help = 'weightsfile')
    parser.add_argument('--reso', type=str, default = '416', help = 'Input resolution of the network. Increase to increase accuracy. Decrease to increase speed')
    parser.add_argument('--scales', type=str, default = '1,2,3', help =  'Scales to use for detection')

    args = parser.parse_args()
     
    coco_classes = load_classes('cfg/coco.names')
    colors = pkl.load(open("cfg/pallete2", "rb"))
    
    main(args)
  

    
