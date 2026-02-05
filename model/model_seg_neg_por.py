import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from . import backbone as encoder
from . import decoder
"""
Borrow from https://github.com/facebookresearch/dino
"""
class CTCHead(nn.Module):
    def __init__(self, in_dim, out_dim=4096, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        # pdb.set_trace()
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            # trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):  #4*768
        x = self.mlp(x)  #4*256
        x = nn.functional.normalize(x, dim=-1, p=2)   #4*256
        x = self.last_layer(x)   #4*1024
        return x


class network(nn.Module):
    def __init__(self, backbone, num_classes=None, pretrained=None, init_momentum=None, aux_layer=None):
        super().__init__()
        self.num_classes = num_classes
        self.init_momentum = init_momentum

        self.encoder = getattr(encoder, backbone)(pretrained=pretrained, aux_layer=aux_layer)

        self.proj_head = CTCHead(in_dim=self.encoder.embed_dim, out_dim=1024)
        self.proj_head_t = CTCHead(in_dim=self.encoder.embed_dim, out_dim=1024,)

        for param, param_t in zip(self.proj_head.parameters(), self.proj_head_t.parameters()):
            param_t.data.copy_(param.data)  # initialize teacher with student
            param_t.requires_grad = False  # do not update by gradient

        self.in_channels = [self.encoder.embed_dim] * 4 if hasattr(self.encoder, "embed_dim") else [self.encoder.embed_dims[-1]] * 4 

        self.pooling = F.adaptive_max_pool2d

        self.decoder = decoder.LargeFOV(in_planes=self.in_channels[-1], out_planes=self.num_classes,)

        self.classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes-1, kernel_size=1, bias=False,)
        self.aux_classifier = nn.Conv2d(in_channels=self.in_channels[-1], out_channels=self.num_classes-1, kernel_size=1, bias=False,)

        self.classifier_bg = nn.Conv2d(768, 1, 1, bias=False)

    @torch.no_grad()
    def _EMA_update_encoder_teacher(self, n_iter=None):
        ## no scheduler here
        momentum = self.init_momentum
        for param, param_t in zip(self.proj_head.parameters(), self.proj_head_t.parameters()):
            param_t.data = momentum * param_t.data + (1. - momentum) * param.data

    def get_param_groups(self):

        param_groups = [[], [], [], []] # backbone; backbone_norm; cls_head; seg_head;

        for name, param in list(self.encoder.named_parameters()):

            if "norm" in name:
                param_groups[1].append(param)
            else:
                param_groups[0].append(param)

        param_groups[2].append(self.classifier.weight)
        param_groups[2].append(self.aux_classifier.weight)

        for param in list(self.proj_head.parameters()):
            param_groups[2].append(param)

        for param in list(self.decoder.parameters()):
            param_groups[3].append(param)

        return param_groups

    def to_2D(self, x, h, w):  # 4*784*768
        n, hw, c = x.shape
        x = x.transpose(1, 2).reshape(n, c, h, w)  # 4*768*28*28
        return x

    def forward(self, x, cam_only=False):

        _, _x, x_aux = self.encoder.forward_features(x)
        #4*768   4*784*768   4*784*768

        h, w = x.shape[-2] // self.encoder.patch_size, x.shape[-1] // self.encoder.patch_size

        _x4 = self.to_2D(_x, h, w) #4*768*28*28
        _x_aux = self.to_2D(x_aux, h, w)#4*768*28*28

        seg = self.decoder(_x4)  #4*21*28*28

        if cam_only:

            cam = F.conv2d(_x4, self.classifier.weight).detach()   #4*20*28*28
            cam_aux = F.conv2d(_x_aux, self.aux_classifier.weight).detach()  #4*20*28*28

            return cam_aux, cam


        cls_aux = self.pooling(_x_aux, (1, 1))  # 2*768*1*1
        cls_aux = self.aux_classifier(cls_aux)  # 2*20*1*1

        cls_x4 = self.pooling(_x4, (1, 1))
        cls_x4 = self.classifier(cls_x4)  # 2*20*1*1

        cls_x4 = cls_x4.view(-1, self.num_classes - 1)  # 2*20
        cls_aux = cls_aux.view(-1, self.num_classes - 1)  # 2*20

        return cls_x4, seg, _x4, cls_aux


    def forward_feat(self, x, cam_only=False, crops=None, n_iter=None):

        cls_token, _x, x_aux = self.encoder.forward_features(x)
        # 2*768   2*784*768   2*784*768

        h, w = x.shape[-2] // self.encoder.patch_size, x.shape[-1] // self.encoder.patch_size

        _x = self.to_2D(_x, h, w)  # 2*768*28*28

        # cam = F.conv2d(_x4, self.classifier.weight).detach()  # 2*20*28*28
        cam = F.conv2d(_x, self.aux_classifier.weight).detach()  # 2*20*28*28
        cam_bg = self.classifier_bg(_x)

        cls = self.pooling(_x, (1, 1))  # 2*768*1*1
        cls = self.aux_classifier(cls)  # 2*20*1*1


        logits = cls.view(-1, self.num_classes - 1)  # 2*20

        return torch.cat([cam, cam_bg],dim=1), _x, logits

