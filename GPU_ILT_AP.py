import lithosim.lithosim_cuda as litho
from utils.ilt_utils import compute_common_term, bit_mask_to_two_value_mask, sigmoid_ilt_mask
import torch
import torchvision
import os
import time
from PIL import Image
import numpy as np
import cv2
import torch.nn as nn
import argparse


def calc_ilt_gradient(mask, target, kernels, kernels_ct, kernels_def, kernels_def_ct, weight, weight_def,
                      gamma=4.0, theta_z=50, theta_m=4, alpha=0.8, beta=0.2):

    # nominal
    common_term = compute_common_term(
        mask, target, kernels, weight, dose=1.0, gamma=gamma, theta_z=theta_z, theta_m=theta_m)

    mask_convolve_kernel_ct_output = litho.convolve_kernel(
        mask, kernels_ct, weight, dose=1.0)  # [1 * H * W * 2]
    mask_convolve_kernel_ct_output = mask_convolve_kernel_ct_output * common_term

    kernels_flip = torch.flip(kernels, [1, 2])
    kernels_ct_flip = torch.flip(kernels_ct, [1, 2])
    gradient_right_term = litho.convolve_kernel(
        mask_convolve_kernel_ct_output, kernels_flip, weight, dose=1.0).real  # [1 * H * W], take the real part

    mask_convolve_kernel_output = litho.convolve_kernel(
        mask, kernels, weight, dose=1.0)  # [1 * H * W]
    mask_convolve_kernel_output = mask_convolve_kernel_output * common_term
    gradient_left_term = litho.convolve_kernel(
        mask_convolve_kernel_output, kernels_ct_flip, weight, dose=1.0).real  # [1 * H * W], take the real part

    gradient_temp_ilt = (gradient_right_term + gradient_left_term)

    # inner
    common_term = compute_common_term(
        mask, target, kernels_def, weight_def, dose=0.98, gamma=gamma, theta_z=theta_z, theta_m=theta_m)

    mask_convolve_kernel_ct_output = litho.convolve_kernel(
        mask, kernels_def, weight_def, dose=0.98)  # [1 * H * W * 2]
    mask_convolve_kernel_ct_output = mask_convolve_kernel_ct_output * common_term

    kernels_def_flip = torch.flip(kernels_def, [1, 2])
    kernels_def_ct_flip = torch.flip(kernels_def_ct, [1, 2])
    gradient_right_term = litho.convolve_kernel(
        mask_convolve_kernel_ct_output, kernels_def_flip, weight_def, dose=0.98).real  # [1 * H * W], take the real part

    mask_convolve_kernel_output = litho.convolve_kernel(
        mask, kernels_def, weight_def, dose=0.98)  # [1 * H * W * 2]
    mask_convolve_kernel_output = mask_convolve_kernel_output * common_term
    gradient_left_term = litho.convolve_kernel(
        mask_convolve_kernel_output, kernels_def_ct_flip, weight_def, dose=0.98).real  # [1 * H * W], take the real part

    gradient_temp_pvb = (gradient_right_term + gradient_left_term)

    # outer
    common_term = compute_common_term(
        mask, target, kernels, weight, dose=1.02, gamma=gamma, theta_z=theta_z, theta_m=theta_m)

    mask_convolve_kernel_ct_output = litho.convolve_kernel(
        mask, kernels_ct, weight, dose=1.02)  # [1 * H * W * 2]
    mask_convolve_kernel_ct_output = mask_convolve_kernel_ct_output * common_term
    gradient_right_term = litho.convolve_kernel(
        mask_convolve_kernel_ct_output, kernels_flip, weight, dose=1.02).real  # [1 * H * W], take the real part

    mask_convolve_kernel_output = litho.convolve_kernel(
        mask, kernels, weight, dose=1.02)  # [1 * H * W * 2]
    mask_convolve_kernel_output = mask_convolve_kernel_output * common_term
    gradient_left_term = litho.convolve_kernel(
        mask_convolve_kernel_output, kernels_ct_flip, weight, dose=1.02).real  # [1 * H * W], take the real part

    gradient_temp_pvb += (gradient_right_term + gradient_left_term)
    gradient_temp = alpha * gradient_temp_ilt + beta * gradient_temp_pvb

    constant = gamma * theta_z * theta_m
    discrete_penalty_mask = 0.025 * (-8 * mask + 4)
    gradient = (constant * gradient_temp + theta_m *
                discrete_penalty_mask) * mask * (1 - mask)

    return gradient


def FillHole(mask):
    contours, hierarchy = cv2.findContours(
        mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    len_contour = len(contours)
    contour_list = []
    for i in range(len_contour):
        drawing = np.zeros_like(mask, np.uint8)
        img_contour = cv2.drawContours(
            drawing, contours, i, (255, 255, 255), -1)
        contour_list.append(img_contour)

    out = sum(contour_list)
    return out/255


def gpu_ilt_ap(target_path, attn_kernel_selection, kernels, kernels_ct, kernels_def, kernels_def_ct, weight, weight_def, device,
               ilt_iter=50, out_root='output/', scale_factor=4, save_optimized=False):

    target_img = Image.open(target_path)

    target_img_resized = target_img.resize(
        size=(2048//scale_factor, 2048//scale_factor), resample=Image.NEAREST)

    design_name = target_path.split("/")[-1].split("_")[0]

    gray_scale_img_loader = torchvision.transforms.Compose([
        torchvision.transforms.Grayscale(num_output_channels=1),
        torchvision.transforms.ToTensor(),
    ])

    target_origin = gray_scale_img_loader(target_img).to(device)

    target = gray_scale_img_loader(target_img).to(device)
    target_resized = gray_scale_img_loader(target_img_resized).to(device)

    kernel_size0 = attn_kernel_selection[0]
    kernel_size1 = attn_kernel_selection[1]
    kernel_size2 = attn_kernel_selection[2]
    kernel_size3 = attn_kernel_selection[3]

    kernel0 = np.ones((kernel_size0, kernel_size0), np.uint8)
    attn_map0 = cv2.erode(torch.squeeze(
        target_resized, 0).cpu().numpy(), kernel0)
    attn_map0 = torch.from_numpy(attn_map0*1).to(device)

    kernel1 = np.ones((kernel_size1, kernel_size1), np.uint8)
    attn_map1 = cv2.dilate(torch.squeeze(
        target_resized, 0).cpu().numpy(), kernel1)
    attn_map1 = torch.from_numpy(attn_map1*0.5).to(device)

    kernel2 = np.ones((kernel_size2, kernel_size2), np.uint8)
    attn_map2 = cv2.dilate(torch.squeeze(
        target_resized, 0).cpu().numpy(), kernel2)
    attn_map2 = torch.from_numpy(attn_map2*0.3).to(device)

    kernel3 = np.ones((kernel_size3, kernel_size3), np.uint8)
    attn_map3 = cv2.dilate(torch.squeeze(
        target_resized, 0).cpu().numpy(), kernel3)
    attn_map3 = torch.from_numpy(attn_map3*0.2).to(device)

    attn_map = attn_map1 + attn_map2 + attn_map3 - attn_map0

    mask = target.clone()
    mask = bit_mask_to_two_value_mask(mask)
    params_mask = torch.clone(mask)
    mask = sigmoid_ilt_mask(mask, theta_m=4)

    m_mask = torch.clone(mask)

    m_best_mask = None
    learning_rate = 1.5

    best_loss = 1e15

    down_sample = nn.AvgPool2d(scale_factor)
    up_sample = torch.nn.Upsample(scale_factor=scale_factor,
                                  mode='nearest', align_corners=None)

    for i in range(ilt_iter):
        gradient = calc_ilt_gradient(
            m_mask, target, kernels, kernels_ct, kernels_def, kernels_def_ct, weight, weight_def)

        gradient_down = down_sample(gradient)

        params_mask_down = down_sample(params_mask)

        params_mask_down = params_mask_down - learning_rate * gradient_down * attn_map

        params_mask = torch.squeeze(
            up_sample(torch.unsqueeze(params_mask_down, 0)), 0)

        m_mask = sigmoid_ilt_mask(params_mask)       
        m_best_mask = m_mask

    bin_mask = (m_best_mask > 0.5)

    bin_mask = FillHole(bin_mask.permute(1, 2, 0).cpu().numpy().astype(np.uint8))
    bin_mask = torch.from_numpy(bin_mask).permute(2, 0, 1).to(device).type(torch.cuda.FloatTensor)

    result_nominal, _ = litho.lithosim(bin_mask, 0.225, kernels, weight,
                                    None, False, return_binary_wafer=False)
    wafer_img = (result_nominal >= 0.225).type(torch.cuda.FloatTensor)

    result_inner, _ = litho.lithosim(bin_mask, 0.225, kernels_def, weight_def,
                                    None, False, return_binary_wafer=False, dose=0.98)
    wafer_img_inner = (result_inner >= 0.225).type(
        torch.cuda.FloatTensor)

    result_outer, _ = litho.lithosim(bin_mask, 0.225, kernels, weight,
                                    None, False, return_binary_wafer=False, dose=1.02)
    wafer_img_outer = (result_outer >= 0.225).type(
        torch.cuda.FloatTensor)
    
    if save_optimized:
        # SAVE mask
        mask_out_path = os.path.join(out_root, "%s_final_mask.png" % design_name)
        if not os.path.exists(os.path.dirname(mask_out_path)):
            os.makedirs(os.path.dirname(mask_out_path))
        torchvision.utils.save_image(bin_mask, mask_out_path)

        # SAVE wafer
        wafer_out_path = os.path.join(out_root, "%s_final_wafer.png" % design_name)
        if not os.path.exists(os.path.dirname(wafer_out_path)):
            os.makedirs(os.path.dirname(wafer_out_path))
        torchvision.utils.save_image(wafer_img, wafer_out_path)

        wafer_inner_out_path = os.path.join(out_root, "%s_final_wafer_inner.png" % design_name)
        if not os.path.exists(os.path.dirname(wafer_inner_out_path)):
            os.makedirs(os.path.dirname(wafer_inner_out_path))
        torchvision.utils.save_image(wafer_img_inner, wafer_inner_out_path)

        wafer_outer_out_path = os.path.join(out_root, "%s_final_wafer_outer.png" % design_name)
        if not os.path.exists(os.path.dirname(wafer_outer_out_path)):
            os.makedirs(os.path.dirname(wafer_outer_out_path))
        torchvision.utils.save_image(wafer_img_outer, wafer_outer_out_path)

    L2 = torch.logical_xor(wafer_img, target_origin.type(
        torch.cuda.BoolTensor)).sum().cpu().numpy()
    PVB = torch.logical_xor(
        wafer_img_inner, wafer_img_outer).sum().cpu().numpy()

    print("L2 =", L2, "pvb =", PVB)
    return L2, PVB


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch_data_path = 'lithosim/lithosim_kernels/torch_tensor'
    kernels_path = os.path.join(torch_data_path, 'kernel_focus_tensor.pt')
    kernels_ct_path = os.path.join(
        torch_data_path, 'kernel_ct_focus_tensor.pt')
    kernels_def_path = os.path.join(
        torch_data_path, 'kernel_defocus_tensor.pt')
    kernels_def_ct_path = os.path.join(
        torch_data_path, 'kernel_ct_defocus_tensor.pt')
    weight_path = os.path.join(torch_data_path, 'weight_focus_tensor.pt')
    weight_def_path = os.path.join(torch_data_path, 'weight_defocus_tensor.pt')

    kernels = torch.load(kernels_path, map_location=device)
    kernels_ct = torch.load(kernels_ct_path, map_location=device)
    kernels_def = torch.load(kernels_def_path, map_location=device)
    kernels_def_ct = torch.load(kernels_def_ct_path, map_location=device)
    weight = torch.load(weight_path, map_location=device)
    weight_def = torch.load(weight_def_path, map_location=device)

    parser = argparse.ArgumentParser()
    parser.add_argument('--layout_list', type=str,
                        help='path of layout list', default='./dataset/ibm_opc_test_list.txt')
    parser.add_argument('--ref_attn_kernel', nargs='+', type=int,
                        help='manually selected reference attention kernel sizes', default=[5, 10, 20, 30])                    
    parser.add_argument('--create_RL_data', action='store_true',
                        help='create RL data or not')
    parser.add_argument('--RL_reward_log', type=str,
                        help='if create RL data, the file to record reward', default='./dataset/RL_train_ref_reward.txt')
    parser.add_argument('--save_optimized', action='store_true',
                        help='save optimized mask and resulting wafer image or not')
            
    args = parser.parse_args()

    avg_l2 = 0
    avg_pvb = 0

    layout_files = open(args.layout_list, 'r').readlines()
    num_layout = len(layout_files)

    print('num of layout:', num_layout)

    for i in range(0, num_layout, 1):
        layout_path = layout_files[i].strip()
        print('procesing:', layout_path)
        l2, pvb = gpu_ilt_ap(layout_path, args.ref_attn_kernel, kernels, kernels_ct, kernels_def,
                                 kernels_def_ct, weight, weight_def, device, ilt_iter=30, save_optimized=args.save_optimized)
        
        avg_l2 += l2
        avg_pvb += pvb

        if args.create_RL_data:
            reward = -(l2+pvb)
            with open(args.RL_reward_log, 'a') as f:
                f.write(str(reward)+'\n')

    print('average L2: ', avg_l2/num_layout)
    print('average pvb: ', avg_pvb/num_layout)

