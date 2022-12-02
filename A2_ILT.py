import os
import torch
from PPO_agent import PPO
from PPO_agent import Env_Reward_Update
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import GPU_ILT_AP as gpuiltap
import argparse

class MYDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        return data


def main_worker(layoutlist, ref_attn_kernel, log, ckpt, save_optimized, LOAD_FROM_CONFIG):

    env_name = "A2ILT"

    ################ PPO hyperparameters ################

    K_epochs = 80           # update policy for K epochs in one PPO update

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)

    action_dim = 50

    checkpoint_index = ckpt      # change here for checkpoint selection

    directory = "PPO_model"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + \
        "PPO_{}_{}_{}.pth".format(env_name, random_seed, checkpoint_index)

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

    test_data = []

    if LOAD_FROM_CONFIG:
        attn_kernels = []
        attn_kernel_configs = open('./configs/attn_kernels.txt').readlines()
        for i in range(len(attn_kernel_configs)):
            attn_kernels.append([int(i) for i in attn_kernel_configs[i].strip().split(',')[:]])

        print('attn_kernels:', attn_kernels)
    
    else:
        ppo_agent = PPO(action_dim, lr_actor, lr_critic, gamma,
                    K_epochs, eps_clip, device)

        ppo_agent.load(checkpoint_path)

    with open(layoutlist, "r") as split_list:
        for line in split_list.readlines():
            line = line.split("\n")[0]
            test_data.append(line)

    test_dataset = MYDataset(test_data)
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=4)

    avg_l2 = 0
    avg_pvb = 0

    with open(log, "a") as f:
        print('results saving to:', log)
        print('loading ckpt:', ckpt)
        
        f.write('loading ckpt: ' + str(ckpt) + '\n')

        layout_num = 0

        for idx, data in enumerate(test_dataloader):
            print('processing layout:', idx)
            layout_num += 1

            input_layout_path = data

            input_layout = Image.open(input_layout_path[0])

            if input_layout.getbbox():

                if not LOAD_FROM_CONFIG:

                    input_layout = input_layout.resize(
                        size=(512, 512), resample=Image.NEAREST)

                    gray_scale_img_loader = torchvision.transforms.Compose([
                        torchvision.transforms.Grayscale(num_output_channels=1),
                        torchvision.transforms.ToTensor(),
                    ])

                    input_layout = gray_scale_img_loader(
                        input_layout).to(device)

                    state = input_layout

                    attn_kernel_selections = []

                    for step in range(1, 5, 1):
                        action = ppo_agent.select_action(state, mode='test')
                        state, attn_kernel_selections = Env_Reward_Update(input_layout_path[0], state, step, action, attn_kernel_selections, kernels, kernels_ct, kernels_def,
                                                                        kernels_def_ct, weight, weight_def, device, ilt_iter=30, mode='test', ref_attn_kernel=ref_attn_kernel)

                    print('Layout %d attn_kernel_selections:' %
                        idx, attn_kernel_selections)
                    
                else:
                    attn_kernel_selections = attn_kernels[idx]

                    print('Layout %d attn_kernel_selections:' %
                        idx, attn_kernel_selections)

                l2, pvb = gpuiltap.gpu_ilt_ap(input_layout_path[0], attn_kernel_selections, kernels, kernels_ct,
                                              kernels_def, kernels_def_ct, weight, weight_def, device, ilt_iter=30, save_optimized=save_optimized)

                f.write(input_layout_path[0].split('/')[-1])
                f.write(' ')
                f.write(str(l2))
                f.write(' ')
                f.write(str(pvb))
                f.write('\n')

            else:

                f.write(input_layout_path[0].split('/')[-1])
                f.write(' ')
                f.write('ERROR')
                f.write('\n')

            avg_l2 += l2
            avg_pvb += pvb

    print('average L2: ', avg_l2/layout_num)
    print('average PVB: ', avg_pvb/layout_num)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--layout_list', type=str,
                        help='path of layout list', default='./dataset/ibm_opc_test_list.txt')
    parser.add_argument('--ref_attn_kernel', nargs='+', type=int,
                        help='manually selected reference attention kernel sizes', default=[5, 10, 20, 30])     
    parser.add_argument('--log', type=str,
                        help='path of log file', default='./result/ibm_opc_test_l2_pvb.txt')
    parser.add_argument('--ckpt', type=str,
                        help='ppo checkpoint', default='best')
    parser.add_argument('--save_optimized', action='store_true',
                        help='save optimized mask and resulting wafer image or not')
    parser.add_argument('--load_from_configs', action='store_true',
                        help='load from configs or not')
    args = parser.parse_args()

    main_worker(args.layout_list, args.ref_attn_kernel, args.log, args.ckpt, args.save_optimized, args.load_from_configs)
