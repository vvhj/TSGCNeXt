import argparse
import email
import pickle
import os

import numpy as np
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA','ntuhrnet/xsub', 'ntuhrnet/xview', 'ntu120hrnet/xsub', 'ntu120hrnet/xset'},
                        help='the work folder for storing results')
    parser.add_argument('--alpha',
                        default=1.1,
                        help='weighted summation',
                        type=float)

    parser.add_argument('--joint-dir',
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint-motion-dir', default=None)
    parser.add_argument('--bone-motion-dir', default=None)

    parser.add_argument('--ema',
                        default=False,
                        help='EMA',
                        type=bool)
    parser.add_argument('--s2',
                        default=False,
                        help='s2',
                        type=bool)
    arg = parser.parse_args()

    dataset = arg.dataset
    if 'UCLA' in arg.dataset:
        label = []
        with open('./data/' + 'NW-UCLA/' + '/val_label.pkl', 'rb') as f:
            data_info = pickle.load(f)
            for index in range(len(data_info)):
                info = data_info[index]
                label.append(int(info['label']) - 1)
    elif 'ntu120' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSub.npz')
            if 'hrnet' in arg.dataset:
                npz_data = np.load('./data/' + 'HRNet/' + 'NTU120_CSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu120/' + 'NTU120_CSet.npz')
            if 'hrnet' in arg.dataset:
                npz_data = np.load('./data/' + 'HRNet/' + 'NTU120_CSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in arg.dataset:
        if 'xsub' in arg.dataset:
            npz_data = np.load('./data/' + 'ntu/' + 'NTU60_CS.npz')
            if 'hrnet' in arg.dataset:
                npz_data = np.load('./data/' + 'HRNet/' + 'NTU60_CS.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
            
        elif 'xview' in arg.dataset:
            npz_data = np.load('./data/' + 'newntu/'+"NTU60_CV.npz")
            if 'hrnet' in arg.dataset:
                npz_data = np.load('./data/' + 'HRNet/' + 'NTU60_CV.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    else:
        raise NotImplementedError
    mp = 'epoch1_test_score.pkl'
    emp = 'epoch1_test_ema_score.pkl'
    if arg.ema:
        pp = emp
    else:
        pp = mp
    with open(os.path.join(arg.joint_dir, pp), 'rb') as r1:
        r1 = list(pickle.load(r1).items())

    with open(os.path.join(arg.bone_dir, pp), 'rb') as r2:
        r2 = list(pickle.load(r2).items())
    if arg.s2 == False:
        if arg.joint_motion_dir is not None:
            with open(os.path.join(arg.joint_motion_dir, pp), 'rb') as r3:
                r3 = list(pickle.load(r3).items())
        if arg.bone_motion_dir is not None:
            with open(os.path.join(arg.bone_motion_dir, pp), 'rb') as r4:
                r4 = list(pickle.load(r4).items())
    else:
        if arg.joint_dir is not None:
            with open(os.path.join(arg.joint_dir, emp), 'rb') as r3:
                r3 = list(pickle.load(r3).items())
        if arg.bone_dir is not None:
            with open(os.path.join(arg.bone_dir, emp), 'rb') as r4:
                r4 = list(pickle.load(r4).items())
    right_num = total_num = right_num_5 = 0

    if (arg.joint_motion_dir is not None and arg.bone_motion_dir is not None) or arg.s2:
        print("1")
        arg.alpha = [0.95,0,0,1]
        for i in tqdm(range(len(label))):
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            _, r44 = r4[i]
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2] + r44 * arg.alpha[3]
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
    elif arg.joint_motion_dir is not None and arg.bone_motion_dir is None:
        arg.alpha = [0.6, 0.6, 0.4]
        for i in tqdm(range(len(label))):
            l = label[:, i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            _, r33 = r3[i]
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1] + r33 * arg.alpha[2]
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num
    else:
        for i in tqdm(range(len(label))):
            arg.alpha = [1, 0.9]
            l = label[i]
            _, r11 = r1[i]
            _, r22 = r2[i]
            r = r11 * arg.alpha[0] + r22 * arg.alpha[1]
            rank_5 = r.argsort()[-5:]
            right_num_5 += int(int(l) in rank_5)
            r = np.argmax(r)
            right_num += int(r == int(l))
            total_num += 1
        acc = right_num / total_num
        acc5 = right_num_5 / total_num

    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
    right_num = total_num = right_num_5 = 0
    for i in tqdm(range(len(label))):
        l = label[i]
        _, r11 = r1[i]
        _, r22 = r2[i]
        r = r11 + r22*0.5 #* arg.alpha
        rank_5 = r.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        r = np.argmax(r)
        right_num += int(r == int(l))
        total_num += 1
    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print('Top1 Acc: {:.4f}%'.format(acc * 100))
    print('Top5 Acc: {:.4f}%'.format(acc5 * 100))
