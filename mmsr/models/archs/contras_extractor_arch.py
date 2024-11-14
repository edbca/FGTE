from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.vgg as vgg


class ContrasExtractorLayer(nn.Module):

    def __init__(self):
        super(ContrasExtractorLayer, self).__init__()

        vgg16_layers = [
            'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
            'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
            'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'pool3', 'conv4_1',
            'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3', 'relu4_3', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3',
            'pool5'
        ]
        conv3_1_idx = vgg16_layers.index('conv3_1')
        features = getattr(vgg,'vgg16')(pretrained=True).features[:conv3_1_idx + 1]

        modified_net = OrderedDict()
        for k, v in zip(vgg16_layers, features):
            modified_net[k] = v

        self.model = nn.Sequential(modified_net)
        # the mean is for image with range [0, 1]
        self.register_buffer(
            'mean',
            torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        # the std is for image with range [0, 1]
        self.register_buffer(
            'std',
            torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, batch):
        batch = (batch - self.mean) / self.std #[4,3,160,160]
        output = self.model(batch) #[4,256,40,40]
        return output


class ContrasExtractorSep(nn.Module):

    def __init__(self):
        super(ContrasExtractorSep, self).__init__()

        self.feature_extraction_image1 = ContrasExtractorLayer()
        self.feature_extraction_image2 = ContrasExtractorLayer()

    def forward(self, image1, image2):

        _, _, h, w = image2.size()
        dense_features1 = self.feature_extraction_image1(image1)
        #ref same scale
        dense_features2 = self.feature_extraction_image2(image2)

        #ref down scale
        image2_down = torch.zeros_like(image2)
        image2_down1 = F.interpolate(image2, scale_factor=0.75, mode='bicubic', align_corners=False)
        image2_down2 = F.interpolate(image2, scale_factor=0.5, mode='bicubic', align_corners=False)
        image2_down3 = F.interpolate(image2, scale_factor=0.25, mode='bicubic', align_corners=False)
        _, _, n_h1, n_w1 = image2_down1.size()
        _, _, n_h2, n_w2 = image2_down2.size()
        _, _, n_h3, n_w3 = image2_down3.size()
        image2_down[:, :, 0:n_h1, 0:n_w1]=image2_down1
        image2_down[:, :, 0:n_h3, n_w1:n_w1+n_w3]=image2_down3

        image2_down[:, :, n_h1:n_h1+n_h2//2, 0:n_w2] = image2_down2[:,:,0:n_h2//2, 0:n_w2]
        image2_down[:, :, n_h1:n_h1+n_h2//2, n_w2:n_w2+n_w2] = image2_down2[:,:,n_h2//2:n_h2//2+n_h2//2, 0:n_w2]
        image2_down[:, :, n_h3:n_h3+n_h2, n_w1:n_w1+n_w2//2] = image2_down2[:, :, 0:n_h2, n_w2//4:n_w2//4+n_w2//2]
        dense_features2_down = self.feature_extraction_image2(image2_down)
        #ref up scale
        image2_up = F.interpolate(image2, scale_factor=1.25, mode='bicubic', align_corners=False)
        _, _, n_h, n_w = image2_up.size()
        image2_up = image2_up[:, :, (n_h-h)//2:(n_h-h)//2 + h, (n_w-w)//2:(n_w-w)//2 + w]
        dense_features2_up = self.feature_extraction_image2(image2_up)

        return {
            'dense_features1': dense_features1,
            'dense_features2': dense_features2,
            'dense_features2_down': dense_features2_down,
            'dense_features2_up': dense_features2_up,
        }






