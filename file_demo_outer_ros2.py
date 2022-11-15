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
from preprocess import prep_image0
from preprocess import automatic_brightness_and_contrast
import natsort
from collections import OrderedDict

import resnet50 as model_n 

path = os.environ['ROS2_SRC'] + '/fashion_att_pkg/fashion_attribute/'


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


transform_test = transforms.Compose([
        transforms.Resize(size=(256, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
path_t = osp.abspath(os.getcwd())

def convert_image_np(inp):
    """Convert a Tensor to numpy image."""
 #   inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp

def make_json(detections, outputs):

    clothings = []
    for i in range(detections.shape[0]):
        
        clothing = OrderedDict()        
        #temp = outputs[j][i].data
        #사람 id : 사진속의 사람이 여러명일 경우 이 id로 구분
        clothing["id"] = i
        #상의는 top 의 t_ 사용, 하의는 bottom 의 b_ 사용
        #상의 속성 11가지
        clothing["t_color"] = attribute_pool[0][torch.max(outputs[0][i].data, 0)[1]]
        clothing["t_pattern"] = attribute_pool[1][torch.max(outputs[1][i].data, 0)[1]]
        clothing["t_gender"] = attribute_pool[2][torch.max(outputs[2][i].data, 0)[1]]
        clothing["t_season"] = attribute_pool[3][torch.max(outputs[3][i].data, 0)[1]]
        clothing["t_type"] = attribute_pool[4][torch.max(outputs[4][i].data, 0)[1]]
        clothing["t_sleeves"] = attribute_pool[5][torch.max(outputs[5][i].data, 0)[1]]
        clothing["t_texture"] = attribute_pool[6][torch.max(outputs[6][i].data, 0)[1]]
        clothing["t_button"] = attribute_pool[7][torch.max(outputs[7][i].data, 0)[1]]
        clothing["t_length"] = attribute_pool[8][torch.max(outputs[8][i].data, 0)[1]]
        clothing["t_fit"] = attribute_pool[9][torch.max(outputs[9][i].data, 0)[1]]
        clothing["t_collar"] = attribute_pool[10][torch.max(outputs[10][i].data, 0)[1]]
        #하의 속성 7가지
        clothing["b_color"] = attribute_pool[11][torch.max(outputs[11][i].data, 0)[1]]
        clothing["b_pattern"] = attribute_pool[12][torch.max(outputs[12][i].data, 0)[1]]
        clothing["b_gender"] = attribute_pool[13][torch.max(outputs[13][i].data, 0)[1]]
        clothing["b_season"] = attribute_pool[14][torch.max(outputs[14][i].data, 0)[1]]
        clothing["b_length"] = attribute_pool[15][torch.max(outputs[15][i].data, 0)[1]]
        clothing["b_type"] = attribute_pool[16][torch.max(outputs[16][i].data, 0)[1]]
        clothing["b_legpose"] = attribute_pool[17][torch.max(outputs[17][i].data, 0)[1]]
        #사진속의 사람의 성별 속성
        clothing["gender"] = attribute_pool[18][torch.max(outputs[18][i].data, 0)[1]]
        #사진속의 사람이 안경/선글라스 착용여부
        clothing["glasses"] = attribute_pool[19][torch.max(outputs[19][i].data, 0)[1]]
        #사진속의 사람이 착용한 옷의 스타일
        clothing["style"] = attribute_pool[20][torch.max(outputs[20][i].data, 0)[1]]
        
        clothings.append(clothing)

    return clothings

class FashionAttributeDetection(object):

    def __init__(self) -> None:
        super().__init__()
    
    def load_models(self):
        self.confidence = 0.5
        self.nms_thresh = 0.4
        self.num_classes = 80
        self.test = 'image'
        
        self.yolov3  = Darknet(path + 'cfg/yolov3.cfg')
        self.yolov3.load_weights(path + 'cfg/yolov3.weights')
        self.yolov3.net_info["height"] = 416    
        self.inp_dim = int(self.yolov3.net_info["height"])
        assert self.inp_dim % 32 == 0 
        assert self.inp_dim > 32
        self.yolov3.to(device)
        self.yolov3.eval()

        print("yolo-v3 network successfully loaded")

        self.attribute_dim = [16, 15, 3, 5, 14, 4, 8, 6, 4, 4, 12, 16, 15, 3, 5, 4, 9, 4, 3, 4, 23] #21개

        check_point = torch.load(path + 'cfg/41k_outer_best.pth.tar')
        state_dict = check_point['state_dict']
        
        self.model = model_n.__dict__['resnet50'](pretrained=True, num_classes=len(self.attribute_dim), attribute_dim=self.attribute_dim)
        
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
        self.model.to(device)
        self.model.eval()

        self.coco_classes = load_classes(path + 'cfg/coco.names')
        self.colors = pkl.load(open(path + 'cfg/pallete2', "rb"))

    def predict(self, image):

        with torch.no_grad():
            orig_pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))    
            #orig_pil_image = Image.open(image)
            
            image, orig_img, im_dim, orig  = prep_image0(image, self.inp_dim)
            im_dim = torch.FloatTensor(im_dim).repeat(1, 2)
            
            image_tensor = image.to(device)
            im_dim = im_dim.to(device)
            
            #4-1. detect candidates for human ROI
            detections = self.yolov3(image_tensor, device, True) # prediction mode for yolo-v3
            detections = write_results(detections, self.confidence, self.num_classes, device, nms=True, nms_conf=self.nms_thresh)
            # original image dimension --> im_dim

            
            os.system('clear')
            sentence_final = []
            
            #if type(detections) != int: 
            if detections != [] :
                if detections.shape[0]:
                    
                    im_dim = im_dim.repeat(detections.shape[0], 1)
                    scaling_factor = torch.min(self.inp_dim/im_dim, 1)[0].view(-1, 1)
                    
                    detections[:, [1, 3]] -= (self.inp_dim - scaling_factor*im_dim[:, 0].view(-1, 1))/2
                    detections[:, [2, 4]] -= (self.inp_dim - scaling_factor*im_dim[:, 1].view(-1, 1))/2

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
                        outputs = self.model(rois)

                        results = make_json(detections, outputs)                        
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
                            sentence_final.append(sentence)
                            write(Roi[i], orig_img, sentence, i+1, self.coco_classes, self.colors)

                        return orig_img, results  
                    else:
                        return [], []     
            else:
                return [], []           

def main():

    test_path = 'image'

    try:
        list_dir = os.listdir(test_path)
     #   list_dir.sort(key=lambda f: int(filter(str.isdigit, f)))
     #  list_dir.sort(key=lambda x: int(x[:-4]))
        list_dir = natsort.natsorted(list_dir, reverse=False)
        imlist = [osp.join(osp.realpath('.'), test_path, img) for img in list_dir if os.path.splitext(img)[1] =='.jpg'  or os.path.splitext(img)[1] == '.jpeg' or os.path.splitext(img)[1] =='.JPG' or os.path.splitext(img)[1] =='.png']
    except NotADirectoryError:
        imlist = []
        imlist.append(osp.join(osp.realpath('.'), test_path))
        print('Not a directory error')
    except FileNotFoundError:
        print ("No file or directory with the name {}".format(test_path))
        exit()
    
    fashion = FashionAttributeDetection()
    fashion.load_models()

    #4. demo routine for each sample file
    with torch.no_grad():
        for inx, image in enumerate(imlist):

            print('\n'+ list_dir[inx])

            orig_img, sentence_results = fashion.predict(image)

            print (sentence_results)
            
            cv2.imshow("frame", orig_img)
            key = cv2.waitKey(0)
            os.system('clear')
            if key & 0xFF == ord('q'): 
                break
    
if __name__ == '__main__':
     
    
    coco_classes = load_classes(path + 'cfg/coco.names')
    colors = pkl.load(open(path + 'cfg/pallete2', "rb"))
    
    ###
    main()
  

    

