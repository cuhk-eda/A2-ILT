import os
import time
from datetime import datetime
from PIL.Image import NONE
import torch
import numpy as np
from PPO_agent import PPO
from PPO_agent import Env_Reward_Update
from torch.utils.data import Dataset, DataLoader
import torchvision
from PIL import Image
import csv
import argparse

class MYDataset(Dataset):
    def __init__(self, data, ref):
        self.data = data
        self.ref = ref

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        ref = self.ref[index]
        return data, ref


def write_into_csv(data):
    with open("epoch_loss.csv", "a+") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([data])


################################### Training ###################################
def train(layoutlist, ref_attn_kernel, refreward, resume_ckpt):

    print("============================================================================================")

    env_name = "A2ILT"
    print("training environment name : " + env_name)

    K_epochs = 80           # update policy for K epochs in one PPO update
    num_epochs = 1000

    eps_clip = 0.2          # clip parameter for PPO
    gamma = 0.99            # discount factor

    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network

    random_seed = 0         # set random seed if required (0 = no random seed)

    action_dim = 50
    
    if resume_ckpt != 'None':
        checkpoint_index = int(resume_ckpt.split('.')[0].split('_')[-1]) + 1
    else:
        # change this to prevent overwriting weights in same env_name folder
        checkpoint_index = 0

    directory = "PPO_model"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + env_name + '/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    checkpoint_path = directory + \
        "PPO_{}_{}_{}.pth".format(env_name, random_seed, checkpoint_index)
    print("save checkpoint path : " + directory)

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")

    print("action space dimension : ", action_dim)

    print("Initializing a discrete action space policy")

    print("PPO K epochs : ", K_epochs)
    print("PPO epsilon clip : ", eps_clip)
    print("discount factor (gamma) : ", gamma)

    print("optimizer learning rate actor : ", lr_actor)
    print("optimizer learning rate critic : ", lr_critic)

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

    train_data = []
    train_reward_ref = []
    with open(layoutlist, "r") as split_list:
        for line in split_list.readlines():
            line = line.strip("\n")
            layout_path = line
            train_data.append(layout_path)
    with open(refreward, "r") as split_list:
        for line in split_list.readlines():
            line = line.strip("\n")
            ref_reward = line
            train_reward_ref.append(float(ref_reward))

    train_dataset = MYDataset(train_data, train_reward_ref)
    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4)

    print("============================================================================================")

    ################# training procedure ################
    # initialize PPO agent
    ppo_agent = PPO(action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device=device)

    print(ppo_agent.policy)
  
    if resume_ckpt != 'None':
        print('resume from:', resume_ckpt)
        ppo_agent.load(resume_ckpt)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")

    # training loop
    update_step = 0

    for epoch in range(checkpoint_index, num_epochs):
        print('----- Epoch %d/%d -----' % (epoch, num_epochs - 1))
        since = time.time()
        iter_since = time.time()

        current_reward_sum = 0
        epoch_loss = 0
        
        # adjust training configurations
        for p in ppo_agent.optimizer.param_groups:
            p['lr'] = max(p['lr'] / (2**int(epoch/100)), p['lr'] / 4)
        ppo_agent.K_epochs = max(ppo_agent.K_epochs // (2**int(epoch/100)), 20)

        for idx, data in enumerate(train_dataloader):
            dataloader_len = train_dataset.__len__()
            input_layout_path, ref_reward_ = data
            input_layout = Image.open(input_layout_path[0])

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

            cur_action = []
            for step in range(1, 5, 1):
                
                action = ppo_agent.select_action(state, mode='train')
                cur_action.append(action)
                state, reward, attn_kernel_selections, done = Env_Reward_Update(input_layout_path[0], input_layout, step, action, attn_kernel_selections, kernels, kernels_ct, kernels_def,
                                                                                kernels_def_ct, weight, weight_def, device, ilt_iter=30, mode='train', ref_attn_kernel=ref_attn_kernel)
                if step == 4:
                    ppo_agent.buffer.rewards.append(
                        (reward-ref_reward_.item())/abs(ref_reward_.item()))

                else:
                    ppo_agent.buffer.rewards.append(reward)

                ppo_agent.buffer.is_terminals.append(done)

            current_reward_sum += (reward-ref_reward_.item()
                                   )/abs(ref_reward_.item())

            epoch_loss += -reward

            update_step += 1

            if (idx+1) % 20 == 0:
                print("time: %.2fs \tepoch: [%d/%d] \titer: [%d/%d] \taverage_reward: %.4f" %
                      ((time.time() - iter_since), epoch, (num_epochs - 1), idx+1,
                       dataloader_len, current_reward_sum / 100))
                iter_since = time.time()
                current_reward_sum = 0

            if update_step % 100 == 0:
                print("---------------------Updating policy------------------------")
                ppo_agent.update()
                update_step = 0


        write_into_csv(epoch_loss)

        print("-----------------------------------------------------------------------------------------")
        print("epoch: [%d/%d] \tepoch_sum_loss: %.4f" %
              (epoch, (num_epochs - 1), epoch_loss))

        if True:
            print(
                "--------------------------------------------------------------------------------------------")
            checkpoint_path = directory + \
                "PPO_{}_{}_{}.pth".format(
                    env_name, random_seed, checkpoint_index)
            print("saving model at : " + checkpoint_path)
            ppo_agent.save(checkpoint_path)
            print("model saved")
            time_elapsed = time.time() - since
            print('Epoch %d\tTotal Time: %.0fm %.2fs\n' %
                  (epoch, time_elapsed // 60, time_elapsed % 60))
            print(
                "--------------------------------------------------------------------------------------------")
            checkpoint_index += 1

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--layout_list', type=str,
                        help='path of layout list', default='./dataset/RL_train_layout_name.txt')
    parser.add_argument('--ref_attn_kernel', nargs='+', type=int,
                        help='manually selected reference attention kernel sizes', default=[5, 10, 20, 30])     
    parser.add_argument('--ref_reward', type=str,
                        help='path of reward file', default='./dataset/RL_train_ref_reward.txt')
    parser.add_argument('--resume_ckpt', type=str,
                        help='path of resume checkpoint', default="None")
    args = parser.parse_args()

    train(args.layout_list, args.ref_attn_kernel, args.ref_reward, args.resume_ckpt)
