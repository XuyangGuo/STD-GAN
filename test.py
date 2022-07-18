import sys
sys.path.append('.')

from global_setting import MODEL_ROOT_DIR
import os

import torch
import torchlib
import torchvision
import config
import model
import torchlib


def get_test_img_from_celeba(att_name, target_path, num=20):
    """
    generate test images from celeba dataset
    :param att_name: 'bangs', 'eyeglasses', 'smiling', 'hair_color'
    :param target_path:
    :param num: the num of images
    """
    test_start = 190000

    import data
    img_size = cfg.img_size
    test_dataset = data.get_dataset_celeba(img_dir=cfg.img_dir, att_file=cfg.att_file, use_atts=cfg.use_atts,
                                           well_cropped=cfg.well_cropped, size=cfg.img_size, split='test',
                                           pair=True,
                                           test_start=test_start)

    att_idx = cfg.use_atts.index(att_name)
    target_path = os.path.join(target_path, att_name)

    for p in [os.path.join(target_path, '0'), os.path.join(target_path, '1')]:
        if not os.path.exists(p):
            os.makedirs(p)

    for id in range(num):
        results = test_dataset.get_by_index(id, att_idx)
        torchvision.utils.save_image(results[0][0] / 2 + 0.5, os.path.join(target_path, '0', '%02d.jpg' % id))
        torchvision.utils.save_image(results[1][0] / 2 + 0.5, os.path.join(target_path, '1', '%02d.jpg' % id))


def test_transfer(whole_model, x1, x0, att_idx):
    """
    :param x1: input source image
    :param x0: input target image
    :param att_idx: The list of transferred attribute
    :param result_path:
    :return:
    """
    if isinstance(att_idx, int):
        att_idx = [att_idx]

    sample_y_fake = torch.zeros(x0.size(0), len(cfg.use_atts)).type_as(x0)
    for a in att_idx:
        sample_y_fake[:, a] = 1
    if x1 is not None:
        sample_z = whole_model(None, None, x1, None, mask='embedding')
        model.mask_z(cfg, sample_y_fake, sample_z)
    else:
        sample_z = model.generate_z(cfg, sample_y_fake)
    res = whole_model(sample_y_fake, sample_z, x0, None, mask='test')
    return res


def transfer(whole_model, x1_path, x0_path, att_names, result_path):
    device = torch.device('cuda' if use_gpu else 'cpu')
    if x1_path:
        x1 = torchlib.get_img_from_file(x1_path, target_device=device, transform=False)
        x1 = x1.unsqueeze(0)
    else:
        x1 = None
    x0 = torchlib.get_img_from_file(x0_path, target_device=device, transform=False)
    att_idx = []
    for att_name in att_names:
        att_idx.append(cfg.use_atts.index(att_name))
    x0 = x0.unsqueeze(0)
    res = test_transfer(whole_model, x1, x0, att_idx)[0]
    torchvision.utils.save_image(res / 2 + 0.5, result_path, nrow=1)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--attribute', type=str, help='Specify attribute name')
    parser.add_argument('-s', '--source', type=str, help='Specify source image path', default=None)
    parser.add_argument('-t', '--target', type=str, help='Specify target image path')
    parser.add_argument('-r', '--result', type=str, help='Specify result image path')
    parser.add_argument('-c', '--config', type=str, help='Specify config number', default='001')

    args = parser.parse_args()
    cfg = config.get_config(args.config)

    out_dir = os.path.join(MODEL_ROOT_DIR, cfg.experiment_name)
    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu
    use_gpu = torch.cuda.is_available()
    device = torch.device("cuda" if use_gpu else "cpu")

    att_name = args.attribute.split(',')
    for a in att_name:
        assert a in cfg.use_atts
    source_path = args.source
    target_path = args.target
    result_path = args.result

    # load model
    whole_model = model.WholeModel(cfg)
    whole_model.to(device)
    # load checkpoint
    ckpt_dir = out_dir + '/checkpoints'
    ckpt = torchlib.load_checkpoint(ckpt_dir)
    whole_model.G.load_state_dict(ckpt['Model_G'], False)
    whole_model.D.load_state_dict(ckpt['Model_D'], True)

    result_dir = os.path.split(result_path)[0]
    if len(result_dir) > 0 and not os.path.exists(result_dir):
        os.makedirs(result_dir)


    transfer(whole_model, source_path, target_path, att_name, result_path=result_path)
