import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision


__all__ = ['resnet50']

def resnet50(pretrained=True, debug=False, **kwargs):
    model = Resnet50(**kwargs)
    
    return model

class ChannelAttn(nn.Module):
    def __init__(self, in_channels, reduction_rate=16):
        super(ChannelAttn, self).__init__()
        assert in_channels%reduction_rate == 0
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction_rate, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels // reduction_rate, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # squeeze operation (global average pooling)
        x = F.avg_pool2d(x, x.size()[2:])
        # excitation operation (2 conv layers)
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return torch.sigmoid(x)




class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class Resnet50(nn.Module):
    def __init__(self, num_classes=13, attribute_dim=None):
        super(Resnet50, self).__init__()
        self.num_classes = num_classes
        self.main_branch = torchvision.models.resnet50(pretrained=True)
        self.global_pool = nn.AvgPool2d((8,4), stride=1, padding=0, ceil_mode=True, count_include_pad=True)
        self.finalfc = nn.Linear(2048, num_classes)

        
        # Lateral layers

        self.latlayer_1 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer_2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer_3 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
        self.latlayer_4 = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)

        self.new_fc = nn.Sequential(
            nn.Conv2d(2048, 256, 1, stride=1,
                    padding=0, bias=False),
            nn.BatchNorm2d(256, eps=1e-3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.BatchNorm2d(256, eps=1e-3),
            nn.ReLU(),
            nn.Dropout(0.5),
            Reshape(-1, 256),
            nn.Linear(256, num_classes))

        self.gap_list = nn.ModuleList()
        self.fc_list = nn.ModuleList()
        self.pooling_size = 8
        self.channels = 2048

        #self.module_list = nn.ModuleList([self.conv_bn(2048, 256, 1, 256, att_size) for att_size in attribute_dim])
        
        for i in range(self.num_classes):
            self.gap_list.append(nn.AvgPool2d((self.pooling_size, self.pooling_size//2), stride=1, padding=0, ceil_mode=True, count_include_pad=True))
            self.fc_list.append(nn.Linear(self.channels, attribute_dim[i]))


        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def get_gradients(self):
        return self.gradients


    def conv_bn(self, in_planes, out_planes, kernel_size, embed_size, att_size, stride=1, padding=0, bias=False):
    #"convolution with batchnorm, relu"
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 1, stride=stride,
                    padding=padding, bias=False),
            nn.BatchNorm2d(out_planes, eps=1e-3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.BatchNorm2d(out_planes, eps=1e-3),
            nn.ReLU(),
            nn.Dropout(0.5),
            Reshape(-1, embed_size),
            nn.Linear(embed_size, att_size)
        )

    def _upsample_add(self, x, y):
        _,_,H,W = y.size()
        up_feat = F.interpolate(x, (H, W), mode='bilinear', align_corners=False)
        return torch.cat([up_feat,y], 1)

    def forward(self, input):
        bs = input.size(0)
        #feat_1, feat_2, feat_3, feat_4 = self.main_branch(input)
        inter_layers = torchvision.models._utils.IntermediateLayerGetter(self.main_branch, 
            {'layer1': 'feat_1', 'layer2': 'feat_2', 'layer3':'feat_3', 'layer4':'feat_4'})
        out = inter_layers(input)

        feat_4 = out['feat_4'] 
     #   feat_4.register_hook(self.save_gradient)
        

        pred_list = []  
        for i in range(self.num_classes): 
            pred = self.gap_list[i](feat_4).view(bs, -1) 
            pred = self.fc_list[i](pred)
            pred_list.append(pred)
        main_preds = pred_list

        return  main_preds



