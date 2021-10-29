##############################
#        Evaluator
##############################

import torch
import torch.nn as nn
import torch.nn.functional as F
class PatchEvaluator(nn.Module):
    def __init__(self, in_channels=5):
        super(PatchEvaluator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 16, normalization=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(1,-1),
            nn.Linear(128, 1)
#             nn.ZeroPad2d((1, 0, 1, 0)),
#             nn.Conv2d(128, 1, 4, padding=1, bias=False)
        )

    def forward(self, img, img_A, img_B, include_image):
        # Concatenate image and condition image by channels to produce input
        if include_image:
            if not img.size(2)==img_A.size(2):
                img = F.interpolate(img, size=(img_A.size(2), img_A.size(3)), mode='bicubic', align_corners=True)
            img_input = torch.cat((img, img_A, img_B), 1)
        else:
            img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
    

class Evaluator(nn.Module):
    def __init__(self, in_channels=5):
        super(Evaluator, self).__init__()
        def discriminator_block(in_filters, out_filters, normalization=True):
            layers = [nn.Conv2d(in_filters, out_filters, 3, stride=1, padding=1),
                      nn.InstanceNorm2d(out_filters),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.MaxPool2d(2, stride=2)]
            return layers
        def head(in_filters):
            layers = [nn.Conv2d(in_filters, in_filters, 3, stride=1, padding=1),
                      nn.InstanceNorm2d(in_filters),
                      nn.LeakyReLU(0.2, inplace=True),
                      nn.AdaptiveAvgPool2d((1,1)),
                      nn.Flatten(1,-1),
                      nn.Linear(in_filters, 1)
                     ]
            return layers
        self.model = nn.Sequential(
            *discriminator_block(in_channels, 32, normalization=False),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *head(256)
        )
    def forward(self, img, img_A, img_B, include_image):
        # Concatenate image and condition image by channels to produce input
        if include_image:
            if not img.size(2)==img_A.size(2):
                img = F.interpolate(img, size=(img_A.size(2), img_A.size(3)), mode='bicubic', align_corners=True)
            img_input = torch.cat((img, img_A, img_B), 1)
        else:
            img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)
        
if __name__ == '__main__':
    import torch
    D = Evaluator()
    img_A = torch.rand((1,1,128,128))
    img_B = torch.rand((1,1,128,128))
    o = D(img_A, img_B)
    print(o.size())