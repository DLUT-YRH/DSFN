import torch
import torch.nn as nn
import torch.nn.functional as F



def build_model(net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor):
    
    # transform for deployment
    for m in net.modules():
        if isinstance(m, RepConvN):
            m.fuse_convs()
            m.forward = m.forward_fuse  # update forward

    out  = net(warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)

    learned_mask1 = (mask1_tensor - mask1_tensor*mask2_tensor) + mask1_tensor*mask2_tensor*out
    learned_mask2 = (mask2_tensor - mask1_tensor*mask2_tensor) + mask1_tensor*mask2_tensor*(1-out)
    stitched_image = (warp1_tensor+1.) * learned_mask1 + (warp2_tensor+1.)*learned_mask2 - 1.

    out_dict = {}
    out_dict.update(learned_mask1=learned_mask1, learned_mask2=learned_mask2, stitched_image = stitched_image)


    return out_dict


def build_depth_output_model(net, warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor, depth_warp1_tensor, depth_warp2_tensor):

    # transform for deployment
    for m in net.modules():
        if isinstance(m, RepConvN):
            m.fuse_convs()
            m.forward = m.forward_fuse  # update forward
            
    out  = net(warp1_tensor, warp2_tensor, mask1_tensor, mask2_tensor)

    learned_mask1 = (mask1_tensor - mask1_tensor*mask2_tensor) + mask1_tensor*mask2_tensor*out
    learned_mask2 = (mask2_tensor - mask1_tensor*mask2_tensor) + mask1_tensor*mask2_tensor*(1-out)
    stitched_image = (warp1_tensor+1.) * learned_mask1 + (warp2_tensor+1.)*learned_mask2 - 1.
    stitched_depth_iamge = depth_warp1_tensor * learned_mask1 + depth_warp1_tensor * learned_mask2

    out_dict = {}
    out_dict.update(learned_mask1=learned_mask1, learned_mask2=learned_mask2, stitched_image = stitched_image, stitched_depth_iamge = stitched_depth_iamge)


    return out_dict




class DownBlock(nn.Module):
    def __init__(self, inchannels, outchannels, dilation, pool=True):
        super(DownBlock, self).__init__()
        blk = []
        if pool:
            blk.append(nn.MaxPool2d(kernel_size=2, stride=2))
        blk.append(nn.Conv2d(inchannels, outchannels, kernel_size=3, padding=1, dilation = dilation))
        blk.append(nn.ReLU(inplace=True))
        blk.append(nn.Conv2d(outchannels, outchannels, kernel_size=3, padding=1, dilation = dilation))
        blk.append(nn.ReLU(inplace=True))
        self.layer = nn.Sequential(*blk)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        return self.layer(x)

class RepConvN(nn.Module):
    default_act = nn.ReLU(inplace=True)  # default activation

    def __init__(self, inchannels, outchannels, kernel_size=3, stride=1, padding=1, groups=1, dilation=1,
                 bias=False, act=False):
        super().__init__()
        assert (kernel_size == 3 or kernel_size == (3, 3)) and padding == 1
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

        self.conv1 = nn.Conv2d(inchannels, outchannels, kernel_size, stride, 1, groups=groups, bias=bias)
        self.conv2 = nn.Conv2d(inchannels, outchannels, 1, stride, 0, groups=groups, bias=bias)

    def forward_fuse(self, x):
        return self.act((self.conv(x)))

    def forward(self, x):
        return self.act((self.conv1(x) + self.conv2(x)))

    def get_equivalent_kernel_bias(self):
        """Returns equivalent kernel and bias by adding 3x3 kernel, 1x1 kernel and identity kernel with their biases."""
        kernel3x3 = self.conv1.weight
        kernel1x1 = self.conv2.weight
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1)

    def _pad_1x1_to_3x3_tensor(self, kernel1x1):
        """Pads a 1x1 tensor to a 3x3 tensor."""
        if kernel1x1 is None:
            return 0
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])  # (left, right, top, bottom)

    def fuse_convs(self):
        """Combines two convolution layers into a single layer and removes unused attributes from the class."""
        if hasattr(self, 'conv'):
            return
        kernel = self.get_equivalent_kernel_bias()
        self.conv = nn.Conv2d(in_channels=self.conv1.in_channels,
                              out_channels=self.conv1.out_channels,
                              kernel_size=self.conv1.kernel_size,
                              stride=self.conv1.stride,
                              padding=self.conv1.padding,
                              dilation=self.conv1.dilation,
                              groups=self.conv1.groups,
                              bias=False).requires_grad_(False)
        self.conv.weight.data = kernel
        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv1')
        self.__delattr__('conv2')
        if hasattr(self, 'nm'):
            self.__delattr__('nm')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

class UpBlock(nn.Module):
    def __init__(self, inchannels, outchannels, dilation):
        super(UpBlock, self).__init__()
        #self.convt = nn.ConvTranspose2d(inchannels, outchannels, kernel_size=2, stride=2)
        self.halfChanelConv = nn.Sequential(
            RepConvN(inchannels, outchannels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
            )


        self.conv = nn.Sequential(
            RepConvN(inchannels, outchannels, kernel_size=3, padding=1, dilation = dilation),
            nn.ReLU(inplace=True),
            RepConvN(outchannels, outchannels, kernel_size=3, padding=1, dilation = dilation),
            nn.ReLU(inplace=True)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x1, x2):

        x1 = F.interpolate(x1, size = (x2.size()[2], x2.size()[3]), mode='nearest')
        x1 = self.halfChanelConv(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

# predict the composition mask of img1
class Network(nn.Module):
    def __init__(self, nclasses=1):
        super(Network, self).__init__()


        self.down1 = DownBlock(3, 32, 1, pool=False)
        self.down2 = DownBlock(32, 64, 2)
        self.down3 = DownBlock(64, 128,3)
        self.down4 = DownBlock(128, 256, 4)
        self.down5 = DownBlock(256, 512, 5)
        self.up1 = UpBlock(512, 256, 4)
        self.up2 = UpBlock(256, 128, 3)
        self.up3 = UpBlock(128, 64, 2)
        self.up4 = UpBlock(64, 32, 1)


        self.out = nn.Sequential(
            nn.Conv2d(32, nclasses, kernel_size=1),
            nn.Sigmoid()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x, y, m1, m2):


        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)

        y1 = self.down1(y)
        y2 = self.down2(y1)
        y3 = self.down3(y2)
        y4 = self.down4(y3)
        y5 = self.down5(y4)

        res = self.up1(x5-y5, x4-y4)
        res = self.up2(res, x3-y3)
        res = self.up3(res, x2-y2)
        res = self.up4(res, x1-y1)
        res = self.out(res)

        return res


