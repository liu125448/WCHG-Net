import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50



from modules.WAGP import AGP_W_M  as AGPWM
from modules.HGNN import CMGR



import torch.cuda.amp as amp


class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out


class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=2):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        #self.inter_channels = reduc_ratio//reduc_ratio
        self.inter_channels = in_channels//reduc_ratio
        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                      padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                               kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.zeros_(m.bias.data)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.01)
        init.zeros_(m.bias.data)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0, 0.001)
        if m.bias:
            init.zeros_(m.bias.data)


class visible_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_module, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v
        #self.conv1 = nn.Conv2d(64, 64, kernel_size=7,padding=3,bias=False)

    def forward(self, x):
        x = self.visible.conv1(x)
        #x = self.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_module(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_module, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t
        #self.conv1 = nn.Conv2d(64, 64, kernel_size=7,padding=3,bias=False)

    def forward(self, x):
        x = self.thermal.conv1(x)
        #x = self.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class visible_moduleA(nn.Module):
    def __init__(self, arch='resnet50'):
        super(visible_moduleA, self).__init__()

        model_v = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.visible = model_v
        #self.conv1 = nn.Conv2d(64, 64, kernel_size=7,padding=3,bias=False)

    def forward(self, x):
        x = self.visible.conv1(x)
        #x = self.conv1(x)
        x = self.visible.bn1(x)
        x = self.visible.relu(x)
        x = self.visible.maxpool(x)
        return x


class thermal_moduleA(nn.Module):
    def __init__(self, arch='resnet50'):
        super(thermal_moduleA, self).__init__()

        model_t = resnet50(pretrained=True,
                           last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        self.thermal = model_t
        #self.conv1 = nn.Conv2d(64, 64, kernel_size=7,padding=3,bias=False)

    def forward(self, x):
        x = self.thermal.conv1(x)
        #x = self.conv1(x)
        x = self.thermal.bn1(x)
        x = self.thermal.relu(x)
        x = self.thermal.maxpool(x)
        return x


class base_resnet(nn.Module):
    def __init__(self, arch='resnet50'):
        super(base_resnet, self).__init__()

        model_base = resnet50(pretrained=True,
                              last_conv_stride=1, last_conv_dilation=1)
        # avg pooling to global pooling
        model_base.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.base = model_base

    def forward(self, x):
        x = self.base.layer1(x)
        x = self.base.layer2(x)
        x = self.base.layer3(x)
        x = self.base.layer4(x)
        return x


class embed_net(nn.Module):
    # def __init__(self,  class_num, no_local='on', gm_pool='on', arch='resnet50', wave_bands=3, wave_qs=None):
    def __init__(self, class_num, no_local='on', gm_pool='on', arch='resnet50',
                 wave_bands=3, lambda_her=2e-4, lambda_hor=1e-4,
                 lambda_ent=None, lambda_orth=None):
        super(embed_net, self).__init__()
        if lambda_ent is not None:
            lambda_her = lambda_ent
        if lambda_orth is not None:
            lambda_hor = lambda_orth
        self.lambda_her = float(lambda_her)
        self.lambda_hor = float(lambda_hor)
        self.wave_bands = int(wave_bands)

        self.thermal_module = thermal_module(arch=arch)
        self.visible_module = visible_module(arch=arch)
        self.thermal_moduleA = thermal_moduleA(arch=arch)
        self.visible_moduleA = visible_moduleA(arch=arch)
        self.base_resnet = base_resnet(arch=arch)
        self.non_local = no_local
        if self.non_local == 'on':
            layers = [3, 4, 6, 3]
            non_layers = [0, 2, 3, 0]
            self.NL_1 = nn.ModuleList(
                [Non_local(256) for i in range(non_layers[0])])
            self.NL_1_idx = sorted([layers[0] - (i + 1)
                                   for i in range(non_layers[0])])
            self.NL_2 = nn.ModuleList(
                [Non_local(512) for i in range(non_layers[1])])
            self.NL_2_idx = sorted([layers[1] - (i + 1)
                                   for i in range(non_layers[1])])
            self.NL_3 = nn.ModuleList(
                [Non_local(1024) for i in range(non_layers[2])])
            self.NL_3_idx = sorted([layers[2] - (i + 1)
                                   for i in range(non_layers[2])])
            self.NL_4 = nn.ModuleList(
                [Non_local(2048) for i in range(non_layers[3])])
            self.NL_4_idx = sorted([layers[3] - (i + 1)
                                   for i in range(non_layers[3])])

        pool_dim = 2048
        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.classifier = nn.Linear(pool_dim, class_num, bias=False)

        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.gm_pool = gm_pool
        dropout = 0
        self.dropout = dropout
        self.drop = nn.Dropout(self.dropout)
        out_planes = 64

        self.local_conv = nn.Conv2d(
            out_planes, 128, kernel_size=1, padding=0, bias=False)
        init.kaiming_normal(self.local_conv.weight, mode='fan_out')
        # may not be used, not working on caffe
        self.feat_bn2d = nn.BatchNorm2d(128)
        # initialize BN, may not be used
        init.constant(self.feat_bn2d.weight, 1)
        # iniitialize BN, may not be used
        init.constant(self.feat_bn2d.bias, 0)
        self.max_pool_PCB = nn.AdaptiveMaxPool2d((1, 1))

        self.agp2 = AGPWM(channels=512, split_ratios=[1, 1], rerank_method='learnable')
        self.agp = AGPWM(channels=256, split_ratios=[1, 1], rerank_method='learnable')

        self.cmgr = CMGR(
            in_channels=1024,
            out_channels=1024,
            cheb_order=6,
            bands=self.wave_bands,
            fuse='concat',
            graph_hidden_dims=(768, 512, 256),
            graph_readout='mean',
            lambda_her=self.lambda_her,
            lambda_hor=self.lambda_hor,
        )




    def forward(self, x1_1, x1_2,x2_1,x2_2, modal=0):
        if modal == 0:
            x1_1 = self.visible_module(x1_1)
            x2_1 = self.thermal_module(x2_1)
            x1_2 = self.visible_moduleA(x1_2)
            x2_2 = self.thermal_moduleA(x2_2)
            x = torch.cat((x1_1, x1_2,x2_1,x2_2), 0)
        elif modal == 1:
            x1_1 = self.visible_module(x1_1)       
            x1_2 = self.visible_moduleA(x1_2)
            x_mix=(x1_1+x1_2)/2
            x=x_mix

            #x = torch.cat((x1_1, x1_1), 0)
            #x = torch.cat((x1_2, x1_2), 0)
        elif modal == 2:
            x2_1 = self.thermal_module(x2_1)
            x2_2 = self.thermal_moduleA(x2_2)
            x_mix=(x2_1+x2_2)/2
            x=x_mix

            #x = torch.cat((x2_1, x2_1), 0)
            #x = torch.cat((x2_2, x2_2), 0)
            



        # shared block
        loss_dict = {}
        if self.non_local == 'on':
            NL1_counter = 0
            if len(self.NL_1_idx) == 0:
                self.NL_1_idx = [-1]
            for i in range(len(self.base_resnet.base.layer1)):
                x = self.base_resnet.base.layer1[i](x)
                if i == self.NL_1_idx[NL1_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_1[NL1_counter](x)
                    NL1_counter += 1
            # Layer 2
            # #([16, 256, 108, 54])
            # x= self.agp1(x)
            with amp.autocast(enabled=False):
                x= self.agp(x.float())
            NL2_counter = 0
            if len(self.NL_2_idx) == 0:
                self.NL_2_idx = [-1]
            for i in range(len(self.base_resnet.base.layer2)):
                x = self.base_resnet.base.layer2[i](x)
                if i == self.NL_2_idx[NL2_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_2[NL2_counter](x)
                    NL2_counter += 1

            with amp.autocast(enabled=False):
                x = self.agp2(x.float())

            NL3_counter = 0
            if len(self.NL_3_idx) == 0:
                self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet.base.layer3)):
                x = self.base_resnet.base.layer3[i](x)
                if i == self.NL_3_idx[NL3_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1

            with amp.autocast(enabled=False):
                x, loss_dict = self.cmgr(x.float())


            NL4_counter = 0
            if len(self.NL_4_idx) == 0:
                self.NL_4_idx = [-1]
            for i in range(len(self.base_resnet.base.layer4)):
                x = self.base_resnet.base.layer4[i](x)
                if i == self.NL_4_idx[NL4_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_4[NL4_counter](x)
                    NL4_counter += 1

        else:
           
            x = self.base_resnet.base.layer1(x) 
            x = self.base_resnet.base.layer2(x)
            x = self.base_resnet.base.layer3(x)           
            x = self.base_resnet.base.layer4(x)
            

        x=x.float()
  
        
        if self.gm_pool == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        feat = self.bottleneck(x_pool)

        if self.training:
            return x_pool, self.classifier(feat), loss_dict
            
        else:
            return self.l2norm(x_pool), self.l2norm(feat)
