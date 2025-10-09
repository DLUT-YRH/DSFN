import torch
import torch.nn as nn
import torch.nn.functional as F



# def get_vgg19_FeatureMap(vgg_model, input_255, layer_index):

#     vgg_mean = torch.tensor([123.6800, 116.7790, 103.9390]).reshape((1,3,1,1))
#     if torch.cuda.is_available():
#         vgg_mean = vgg_mean.cuda()
#     vgg_input = input_255-vgg_mean
#     #x = vgg_model.features[0](vgg_input)
#     #FeatureMap_list.append(x)


#     for i in range(0,layer_index+1):
#         if i == 0:
#             x = vgg_model.features[0](vgg_input)
#         else:
#             x = vgg_model.features[i](x)

#     return x



def l_num_loss(img1, img2, l_num=1):
    return torch.mean(torch.abs((img1 - img2)**l_num))


def boundary_extraction(mask):

    ones = torch.ones_like(mask)
    zeros = torch.zeros_like(mask)
    #define kernel
    in_channel = 1
    out_channel = 1
    kernel = [[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]]
    kernel = torch.FloatTensor(kernel).expand(out_channel,in_channel,3,3)
    if torch.cuda.is_available():
        kernel = kernel.cuda()
        ones = ones.cuda()
        zeros = zeros.cuda()
    weight = nn.Parameter(data=kernel, requires_grad=False)

    #dilation
    x = F.conv2d(1-mask,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)
    x = F.conv2d(x,weight,stride=1,padding=1)
    x = torch.where(x < 1, zeros, ones)

    return x*mask

def cal_boundary_term(inpu1_tesnor, inpu2_tesnor, mask1_tesnor, mask2_tesnor, stitched_image):
    boundary_mask1 = mask1_tesnor * boundary_extraction(mask2_tesnor)
    boundary_mask2 = mask2_tesnor * boundary_extraction(mask1_tesnor)

    loss1 = l_num_loss(inpu1_tesnor*boundary_mask1, stitched_image*boundary_mask1, 1)
    loss2 = l_num_loss(inpu2_tesnor*boundary_mask2, stitched_image*boundary_mask2, 1)

    return loss1+loss2, boundary_mask1

# not use
def cal_smooth_term_stitch(stitched_image, learned_mask1):


    delta = 1
    dh_mask = torch.abs(learned_mask1[:,:,0:-1*delta,:] - learned_mask1[:,:,delta:,:])
    dw_mask = torch.abs(learned_mask1[:,:,:,0:-1*delta] - learned_mask1[:,:,:,delta:])
    dh_diff_img = torch.abs(stitched_image[:,:,0:-1*delta,:] - stitched_image[:,:,delta:,:])
    dw_diff_img = torch.abs(stitched_image[:,:,:,0:-1*delta] - stitched_image[:,:,:,delta:])

    dh_pixel = dh_mask * dh_diff_img
    dw_pixel = dw_mask * dw_diff_img

    loss = torch.mean(dh_pixel) + torch.mean(dw_pixel)

    return loss

def cal_smooth_term_stitch_new(stitched_image, learned_mask1, learned_mask2):

    seam_route = learned_mask1 * learned_mask2
    delta = 1
    
    dh_mask = torch.abs(seam_route[:,:,0:-1*delta,:] - seam_route[:,:,delta:,:])
    dw_mask = torch.abs(seam_route[:,:,:,0:-1*delta] - seam_route[:,:,:,delta:])
    dh_diff_img = torch.abs(stitched_image[:,:,0:-1*delta,:] - stitched_image[:,:,delta:,:])
    dw_diff_img = torch.abs(stitched_image[:,:,:,0:-1*delta] - stitched_image[:,:,:,delta:])

    dh_pixel = dh_mask * dh_diff_img
    dw_pixel = dw_mask * dw_diff_img

    loss = torch.mean(dh_pixel) + torch.mean(dw_pixel)

    return loss

def cal_smooth_term_diff_new(img1, img2, learned_mask1, learned_mask2, overlap):

    diff_feature = torch.abs(img1-img2)**2 * overlap
    seam_route = learned_mask1 * learned_mask2
    delta = 1
    seam_route_h = torch.abs(seam_route[:,:,0:-1*delta,:] - seam_route[:,:,delta:,:])
    seam_route_w = torch.abs(seam_route[:,:,:,0:-1*delta] - seam_route[:,:,:,delta:])
    # dh_mask = torch.abs(learned_mask1[:,:,0:-1*delta,:] - learned_mask1[:,:,delta:,:])
    # dw_mask = torch.abs(learned_mask1[:,:,:,0:-1*delta] - learned_mask1[:,:,:,delta:])
    dh_diff_img = torch.abs(diff_feature[:,:,0:-1*delta,:] + diff_feature[:,:,delta:,:])
    dw_diff_img = torch.abs(diff_feature[:,:,:,0:-1*delta] + diff_feature[:,:,:,delta:])

    dh_pixel = seam_route_h * dh_diff_img
    dw_pixel = seam_route_w * dw_diff_img

    loss = torch.mean(dh_pixel) + torch.mean(dw_pixel)
    # loss = torch.sum(dh_pixel) + torch.sum(dw_pixel)
    return loss


def compute_gradient(image, mask):
    """
    计算图像的梯度，并应用掩膜选择特定区域的梯度。
    
    参数:
    - image: 输入图像，形状为 (B, C, H, W) 的 torch.Tensor。
    - mask: 掩膜，形状为 (B, 1, H, W) 的 torch.Tensor，值为0或1。
    
    返回:
    - gradient: 计算得到的梯度图，形状为 (B, C, H, W) 的 torch.Tensor。
    """
    # 定义Sobel算子
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(image.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(image.device)
    
    # 将图像和掩膜扩展到相同的通道数
    if image.shape[1] > 1:
        sobel_x = sobel_x.repeat(1, image.shape[1], 1, 1)
        sobel_y = sobel_y.repeat(1, image.shape[1], 1, 1)
    
    # 计算x方向和y方向的梯度
    grad_x = F.conv2d(image, sobel_x, padding=1)
    grad_y = F.conv2d(image, sobel_y, padding=1)
    
    # 计算梯度的幅度
    gradient = torch.sqrt(grad_x**2 + grad_y**2)
    
    # 应用掩膜
    gradient = gradient * mask
    
    return gradient




def cal_seam_area_loss(warp1_tensor, warp2_tensor, learned_mask1, learned_mask2, stitched_image):
    seam_area = learned_mask1 * learned_mask2
    seam_area_mask = (seam_area != 0).float()
    gradient_warp1 = compute_gradient(warp1_tensor, seam_area_mask)
    gradient_warp2 = compute_gradient(warp2_tensor, seam_area_mask)
    gradient_stitched = compute_gradient(stitched_image, seam_area_mask)
    gradient_loss_s1 = torch.abs(gradient_stitched - gradient_warp1)
    gradient_loss_s2 = torch.abs(gradient_stitched - gradient_warp2)
    gradient_loss = torch.mean(gradient_loss_s1) + torch.mean(gradient_loss_s2)
    return gradient_loss


def cal_depth_diff_loss(depth_warp1_tensor, depth_warp2_tensor, learned_mask1, learned_mask2):
    
    seam_area = learned_mask1 * learned_mask2
    seam_area_mask = (seam_area != 0).float()
    
    mask_overlap = depth_warp1_tensor * depth_warp2_tensor
    depth_mask_overlap = torch.where(mask_overlap > 0, torch.ones_like(mask_overlap), torch.zeros_like(mask_overlap))
    masked_depth_warp1 = depth_warp1_tensor * depth_mask_overlap
    masked_depth_warp2 = depth_warp2_tensor * depth_mask_overlap
    
    depth_mean = (torch.mean(masked_depth_warp1[depth_mask_overlap == 1]) + torch.mean(masked_depth_warp2[depth_mask_overlap == 1])) / 2
    depth_std = (torch.std(masked_depth_warp1[depth_mask_overlap == 1]) + torch.std(masked_depth_warp2[depth_mask_overlap == 1])) / 2
    nor_depth_warp1 = (masked_depth_warp1 - depth_mean) / depth_std
    nor_depth_warp2 = (masked_depth_warp2 - depth_mean) / depth_std
    final_nor_depth_1 = nor_depth_warp1 * depth_mask_overlap
    final_nor_depth_2 = nor_depth_warp2 * depth_mask_overlap
    stitched_depth = final_nor_depth_1 * learned_mask1 + final_nor_depth_2*learned_mask2
    #stitched_depth_masked = stitched_depth * seam_area_mask
    seam_route = learned_mask1 * learned_mask2
    delta = 1
    
    dh_mask = torch.abs(seam_route[:,:,0:-1*delta,:] - seam_route[:,:,delta:,:])
    dw_mask = torch.abs(seam_route[:,:,:,0:-1*delta] - seam_route[:,:,:,delta:])
    stitched_depth_diff_h = torch.abs(stitched_depth[:,:,0:-1,:] + stitched_depth[:,:,1:,:]) 
    stitched_depth_diff_w = torch.abs(stitched_depth[:,:,:,0:-1] + stitched_depth[:,:,:,1:]) 

#     seam_route = learned_mask1 * learned_mask2
#     delta = 1
#     seam_area_mask_h = torch.abs(seam_area[:,:,0:-1,:] - seam_area[:,:,1:,:])
#     seam_area_mask_w = torch.abs(seam_area[:,:,:,0:-1] - seam_area[:,:,:,1:])
    #depth_warp1_seam_area_tensor = final_nor_depth_1 * seam_area_mask
    depth_warp1_diff_h = torch.abs(final_nor_depth_1[:,:,0:-1,:] + final_nor_depth_1[:,:,1:,:]) 
    depth_warp1_diff_w = torch.abs(final_nor_depth_1[:,:,:,0:-1] + final_nor_depth_1[:,:,:,1:]) 
    depth_warp1_diff_h_masked = depth_warp1_diff_h * dh_mask
    depth_warp1_diff_w_masked = depth_warp1_diff_w * dw_mask
    stitched_depth_diff_h_masked = stitched_depth_diff_h * dh_mask
    stitched_depth_diff_w_masked = stitched_depth_diff_w * dw_mask
#     depth_warp1_diff_h_masked = depth_warp1_diff_h * seam_area_mask_h
#     depth_warp1_diff_w_masked = depth_warp1_diff_w * seam_area_mask_w
#     stitched_depth_diff_h_masked = stitched_depth_diff_h * seam_area_mask_h
#     stitched_depth_diff_w_masked = stitched_depth_diff_w * seam_area_mask_w
    
    diff_h = torch.abs(stitched_depth_diff_h_masked - depth_warp1_diff_h_masked)
    diff_w = torch.abs(stitched_depth_diff_w_masked - depth_warp1_diff_w_masked)
    seam_route_one = torch.where(seam_route > 0, torch.ones_like(seam_route), torch.zeros_like(seam_route))
    stitched_depth_masked = stitched_depth * seam_route_one
    final_nor_depth_1_masked = final_nor_depth_1 * seam_route_one
    depth_diff_s = stitched_depth_masked - final_nor_depth_1_masked
    #loss = torch.mean(depth_diff_s)
    loss = torch.mean(diff_h) + torch.mean(diff_w)
    return loss






def cal_smooth_term_diff(img1, img2, learned_mask1, overlap):

    diff_feature = torch.abs(img1-img2)**2 * overlap

    delta = 1
    dh_mask = torch.abs(learned_mask1[:,:,0:-1*delta,:] - learned_mask1[:,:,delta:,:])
    dw_mask = torch.abs(learned_mask1[:,:,:,0:-1*delta] - learned_mask1[:,:,:,delta:])
    dh_diff_img = torch.abs(diff_feature[:,:,0:-1*delta,:] + diff_feature[:,:,delta:,:])
    dw_diff_img = torch.abs(diff_feature[:,:,:,0:-1*delta] + diff_feature[:,:,:,delta:])

    dh_pixel = dh_mask * dh_diff_img
    dw_pixel = dw_mask * dw_diff_img

    loss = torch.mean(dh_pixel) + torch.mean(dw_pixel)

    return loss

    # dh_zeros = torch.zeros_like(dh_pixel)
    # dw_zeros = torch.zeros_like(dw_pixel)
    # if torch.cuda.is_available():
    #     dh_zeros = dh_zeros.cuda()
    #     dw_zeros = dw_zeros.cuda()


    # loss = l_num_loss(dh_pixel, dh_zeros, 1) + l_num_loss(dw_pixel, dw_zeros, 1)


    # return  loss, dh_pixel