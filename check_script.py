from util.io import read_pfm
import torch
import numpy as np
import argparse
import os
import cv2


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt)).numpy()
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()
    #
    # rmse = (gt - pred) ** 2
    # rmse = np.sqrt(rmse.mean())
    #
    # rmse_log = (np.log(gt) - np.log(pred)) ** 2
    # rmse_log = np.sqrt(rmse_log.mean())
    #
    # abs_rel = np.mean(np.abs(gt - pred) / gt)
    # sq_rel = np.mean(((gt - pred) ** 2) / gt)
    #
    # err = np.log(pred) - np.log(gt)
    # silog = np.sqrt(np.mean(err ** 2) - np.mean(err) ** 2) * 100
    #
    # err = np.abs(np.log10(pred) - np.log10(gt))
    # log10 = np.mean(err)

    return d1, d2, d3


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    # A needs to be a positive definite matrix.
    valid = det > 0

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def get_scale_and_shift(gt, preds, mask):
    target_disparity = torch.zeros_like(gt)
    target_disparity[mask == 1] = 1.0 / gt[mask == 1]

    # t_gt, t_d, t_m = (torch.unsqueeze(item, 0) for item in [target_disparity, preds, mask])
    scale, shift = compute_scale_and_shift(preds, target_disparity, mask)
    return scale, shift


def _main(gt, preds, mask):
    t_gt, t_d, t_m = (torch.unsqueeze(torch.from_numpy(item.copy()), 0) for item in [gt, preds, mask])
    scale, shift = get_scale_and_shift(t_gt, t_d, t_m)
    print(scale, shift)
    new_preds = t_d * scale + shift
    return compute_errors(t_gt[t_m], (1 / new_preds)[t_m])


def convert_arg_line_to_args(arg_line):
    for arg in arg_line.split():
        if not arg.strip():
            continue
        yield arg


parser = argparse.ArgumentParser(description='BTS TensorFlow implementation.', fromfile_prefix_chars='@')
parser.convert_arg_line_to_args = convert_arg_line_to_args

parser.add_argument('--pred_path', type=str, help='path to the prediction results in png', required=True)
parser.add_argument('--gt_path', type=str, help='root path to the groundtruth data', required=False)
parser.add_argument('--dataset', type=str, help='dataset to test on, nyu or kitti', default='nyu')
parser.add_argument('--eigen_crop', help='if set, crops according to Eigen NIPS14', action='store_true')
parser.add_argument('--garg_crop', help='if set, crops according to Garg  ECCV16', action='store_true')
parser.add_argument('--min_depth_eval', type=float, help='minimum depth for evaluation', default=1e-3)
parser.add_argument('--max_depth_eval', type=float, help='maximum depth for evaluation', default=80)
parser.add_argument('--do_kb_crop', help='if set, crop input images as kitti benchmark images', action='store_true')

args = parser.parse_args()


def read_data():
    names_ls = [item.split('.')[0] for item in os.listdir(args.pred_path) if item.endswith('.pfm')]
    preds_ls = [f'{args.pred_path}/{name}.pfm' for name in names_ls]
    ls_a, ls_b, ls_c = [], [], []

    def fn(st):
        dirc, name = st.rsplit('_', 1)
        return f'{args.gt_path}/{dirc}/proj_depth/groundtruth/image_02/{name}.png'

    gt_ls = [fn(name) for name in names_ls]
    for p_gt, p_p in zip(gt_ls, preds_ls):
        pred, _ = read_pfm(p_p)
        depth = cv2.imread(p_gt, -1)
        if depth is None:
            print('Missing: %s ' % p_gt)
            continue
        depth = depth.astype(np.float32) / 256.0
        mask = depth > 0
        a, b, c = _main(depth, pred, mask)
        ls_a.append(a)
        ls_b.append(b)
        ls_c.append(c)
    print(np.mean(ls_a), np.mean(ls_b), np.mean(ls_c))


if __name__ == "__main__":
    read_data()

