import sys
sys.path.append('.')

from global_setting import MODEL_ROOT_DIR
import os
import tensorboardX
import torch
import torchlib
import torchvision
import tqdm
import config
import data
import model
import pylib
import utils
from torchlib import get_img_from_file
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, help='Specify config number', default='001')
args = parser.parse_args()

# ==============================================================================
# =                                   setting                                  =
# ==============================================================================

# ======================================
# =             hyperparameters        =
# ======================================
cfg = config.get_config(args.config)
model_dir = os.path.join(MODEL_ROOT_DIR, cfg.experiment_name)
pylib.mkdir(model_dir)

# save setting
pylib.save_json(model_dir + '/setting.json', cfg, indent=4, separators=(',', ': '))

# device
os.environ['CUDA_VISIBLE_DEVICES'] = cfg.gpu
use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")

# ======================================
# =                data                =
# ======================================
train_dataset = data.get_dataset_celeba(img_dir=cfg.img_dir, att_file=cfg.att_file, use_atts=cfg.use_atts,
                                        well_cropped=cfg.well_cropped, size=cfg.img_size, split='train',
                                        pair_crop=True)

val_dataset = data.get_dataset_celeba(img_dir=cfg.img_dir, att_file=cfg.att_file, use_atts=cfg.use_atts,
                                      well_cropped=cfg.well_cropped, size=cfg.img_size, split='val')

# ======================================
# =          module & optimizer        =
# ======================================

whole_model = model.WholeModel(cfg)
whole_model.to(device)

D_optimizer = torch.optim.Adam(whole_model.D.parameters(),
                               lr=cfg.lr_d, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay_d)
G_optimizer = torch.optim.Adam(whole_model.G.parameters(),
                               lr=cfg.lr_g, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay_g)

Dz_optimizer = [torch.optim.Adam(dz.parameters(), lr=cfg.lr_dz, betas=(cfg.beta1, cfg.beta2))
                for dz in whole_model.Dz]

# ==============================================================================
# =                                run training                                =
# ==============================================================================

# load checkpoint
ckpt_dir = model_dir + '/checkpoints'
pylib.mkdir(ckpt_dir)

try:
    ckpt = torchlib.load_checkpoint(ckpt_dir)
    start_step = ckpt['step'] + 1
    whole_model.D.load_state_dict(ckpt['Model_D'], False)
    whole_model.G.load_state_dict(ckpt['Model_G'], False)
    D_optimizer.load_state_dict(ckpt['D_optimizer'])
    G_optimizer.load_state_dict(ckpt['G_optimizer'])
    whole_model.Dz.load_state_dict(ckpt['Model_Dz'], False)
    for i in range(len(cfg.use_atts)):
        Dz_optimizer[i].load_state_dict(ckpt['Dz_optimizer_%d' % i])
except:
    print(' [*] No checkpoint!')
    start_step = 0

D_scheduler = utils.get_scheduler(D_optimizer, cfg, start_step - 1)
G_scheduler = utils.get_scheduler(G_optimizer, cfg, start_step - 1)
Dz_scheduler = [utils.get_scheduler(Dz_optimizer[i], cfg, start_step - 1) for i in range(len(cfg.use_atts))]

# writer
writer = tensorboardX.SummaryWriter(model_dir + '/summaries')

# start training
sample_x_all_att = val_dataset.get_batch(cfg.display_batch_size)
sample_x_all_att = [[ss.to(device) for ss in s] for s in sample_x_all_att]


def test_multi_different_with_my_input(atts, x0_files=None, x1_files=None, str_prefix='', x1_batch_size=-1,
                                       x0_batch_size=-1, batch_num=10, editing_model=None):
    """
    used in validation
    :param atts:
    :param x0_files:
    :param x1_files:
    :param str_prefix:
    :param x1_batch_size:
    :param x0_batch_size:
    :param batch_num:
    :param editing_model:
    :return:
    """
    if editing_model is None:
        editing_model = whole_model

    batch_size = 10

    if x1_batch_size < 0:
        x1_batch_size = batch_size
    if x0_batch_size < 0:
        x0_batch_size = batch_size

    for i in range(0, batch_num):
        x1_batch = i
        sample_x1 = torch.cat([get_img_from_file(j, target_device=device, transform=True).unsqueeze(0) for j in
                               x1_files[x1_batch * x1_batch_size: (x1_batch + 1) * x1_batch_size]],
                              dim=0).to(device)

        sample_y_fake = torch.zeros(sample_x1.size(0), len(cfg.use_atts)).to(device)
        for att in atts:
            sample_y_fake[:, cfg.use_atts.index(att)] = 1

        sample_x0 = [get_img_from_file(j, target_device=device, transform=True).unsqueeze(0) for j in
                     x0_files[i * x0_batch_size: (i + 1) * x0_batch_size]]
        sample_x0 = torch.cat(sample_x0, dim=0)
        sample_x0 = sample_x0.to(device)

        sample_x_s2 = [torch.cat([2 * torch.ones(1, sample_x1.size(1), sample_x1.size(2), sample_x1.size(3)) - 1.0,
                                  sample_x1.cpu()], dim=0)]
        sample_z = editing_model(None, None, sample_x1, None, mask='embedding')

        model.mask_z(cfg, sample_y_fake, sample_z)

        for cur_x in sample_x0:
            cur_x = cur_x.unsqueeze(0)
            sample_x_ = editing_model(sample_y_fake, sample_z, cur_x.repeat(sample_z.size(0), 1, 1, 1),
                                      None, mask='test')
            sample_x_s2.append(torch.cat([cur_x.cpu(), sample_x_.detach().cpu()], dim=0))
            del sample_x_

        # sample_x_s2 = (torch.cat(sample_x_s2, dim=-1) + 1) / 2.0
        sample_x_s2 = (torch.cat(sample_x_s2, dim=-2) + 1) / 2.0
        sample_x_s2 = sample_x_s2.permute([1, 2, 0, 3])
        sample_x_s2 = sample_x_s2.contiguous().view(3, sample_x_s2.shape[1], -1)

        save_dir = model_dir + '/sample_training_multi_diff_x0' + '/%s' % str(atts)
        pylib.mkdir(save_dir)
        torchvision.utils.save_image(sample_x_s2, '%s/%s%03d.jpg' % (save_dir, str_prefix, i * x0_batch_size), nrow=1)


def print_val_save_model(step, cl):
    """
    validation
    :param step:
    :return:
    """
    if step > 0 and step % cfg.display_frequency == 0:
        for i in range(len(cfg.use_atts)):
            for j in range(i + 1, len(cfg.use_atts)):
                x0_dict = {cfg.use_atts[i]: -1, cfg.use_atts[j]: -1}
                x1_dict = {cfg.use_atts[i]: 1, cfg.use_atts[j]: 1}
                if 'Black_Hair' in x0_dict:
                    x0_dict['Black_Hair'] = 1
                    x1_dict['Black_Hair'] = -1

                multi_x0 = cl.filter(x0_dict)[:20]
                multi_x1 = cl.filter(x1_dict)[:20]
                test_multi_different_with_my_input({cfg.use_atts[i], cfg.use_atts[j]},
                                                   multi_x0, multi_x1, '%07d_' % step, batch_num=1,
                                                   editing_model=whole_model)

        save_dir = model_dir + '/sample_training'
        pylib.mkdir(save_dir)
        whole_model.eval()
        for att_index, att in enumerate(cfg.use_atts):
            # get sample x
            sample_x = torch.cat(sample_x_all_att[att_index], dim=0)
            sample_x0 = sample_x_all_att[att_index][0]
            sample_x1 = sample_x_all_att[att_index][1]

            sample_x_s = [sample_x.cpu()]
            sample_x_0_and_10 = sample_x.clone()

            # get sample y
            y_edit0 = torch.zeros(1, cfg.display_batch_size * 2, len(cfg.use_atts))
            y_edit0[:, :, att_index] = -1
            y_edit1 = torch.zeros(cfg.display_style_num, cfg.display_batch_size * 2, len(cfg.use_atts))
            y_edit1[:, :, att_index] = 1
            sample_ys = torch.cat([y_edit0, y_edit1]).to(device)

            # get sample z
            sample_zs = model.generate_z(cfg, y_edit1[:, 0, :])
            sample_zs = torch.cat([torch.zeros([1, 1, cfg.z_dim * len(cfg.use_atts)]), sample_zs.unsqueeze(1)], dim=0)
            sample_zs = sample_zs.repeat(1, cfg.display_batch_size * 2, 1).to(device)

            for k, (sample_z, sample_y) in enumerate(zip(sample_zs, sample_ys)):
                if k % cfg.display_style_num == 1:  # black line
                    if k == 1:
                        sample_x_0_and_10[sample_x.shape[0] // 2:] = sample_x_s[1][sample_x.shape[0] // 2:]
                    sample_x_s.append(torch.zeros(sample_x.size(0), sample_x.size(1), sample_x.size(2), 10)
                                      .type_as(sample_x).cpu() - 1.0)

                sample_x_ = whole_model(sample_y, sample_z, sample_x_0_and_10, None, mask="test").detach()
                sample_x_s.append(sample_x_)
                del sample_x_  # free the memory

            sample_x_s = (torch.cat(sample_x_s, dim=-1) + 1) / 2.0

            torchvision.utils.save_image(sample_x_s, '%s/%07d_%s.jpg' % (save_dir, step, att), nrow=1)
            del sample_x_s

            sample_x0_ = whole_model(sample_ys[0], sample_zs[0], sample_x1, None, mask='test').detach()

            # exchange
            sample_x_s2 = [torch.cat([torch.zeros(1, sample_x1.size(1), sample_x1.size(2), sample_x1.size(3)) - 1.0,
                                      sample_x1.cpu()], dim=0)]
            sample_z = whole_model(None, None, sample_x1, None, mask='embedding').detach()

            sample_y_fake = torch.zeros(cfg.display_batch_size, len(cfg.use_atts)).type_as(sample_z)
            sample_y_fake[:, att_index] = 1

            model.mask_z(cfg, sample_y_fake, sample_z)

            if "du_rec_x" in cfg or "id_rec_x" not in cfg:
                sample_x_ = whole_model(sample_y_fake, sample_z, sample_x0_.cuda(), None, mask='test')
            else:
                sample_x_ = whole_model(sample_y_fake, sample_z, sample_x1, None, mask='test')
            sample_x_s2.append(torch.cat([torch.zeros(1, sample_x1.size(1), sample_x1.size(2), sample_x1.size(3)) - 1.0,
                                          sample_x_.detach().cpu()], dim=0))
            del sample_x_

            for cur_x in sample_x0:
                cur_x = cur_x.unsqueeze(0)
                sample_x_ = whole_model(sample_y_fake, sample_z, cur_x.repeat(sample_z.size(0), 1, 1, 1),
                                        None, mask='test')
                sample_x_s2.append(torch.cat([cur_x.cpu(), sample_x_.detach().cpu()], dim=0))
                del sample_x_, cur_x

            sample_x_s2 = (torch.cat(sample_x_s2, dim=-1) + 1) / 2.0
            torchvision.utils.save_image(sample_x_s2, '%s/%07d_%s_exchange.jpg' % (save_dir, step, att), nrow=1)
            del sample_x_s2

            # exchange attention
            if 'sa_id' in cfg or cfg.generation_type == 'sa':
                sample_x_s2 = [torch.cat([torch.zeros(1, sample_x1.size(1), sample_x1.size(2), sample_x1.size(3)) - 1.0,
                                          sample_x1.cpu()], dim=0)]

                if "du_rec_x" in cfg or "id_rec_x" not in cfg:
                    sample_x_ = whole_model(sample_y_fake, sample_z, sample_x0_.cuda(), None,
                                            mask='test_attention').repeat(1, 3, 1, 1)
                else:
                    sample_x_ = whole_model(sample_y_fake, sample_z, sample_x1, None, mask='test_attention').repeat(1,
                                                                                                                    3,
                                                                                                                    1,
                                                                                                                    1)
                sample_x_s2.append(
                    torch.cat([torch.zeros(1, sample_x1.size(1), sample_x1.size(2), sample_x1.size(3)) - 1.0,
                               sample_x_.detach().cpu()], dim=0))
                del sample_x_

                for cur_x in sample_x0:
                    cur_x = cur_x.unsqueeze(0)
                    sample_x_ = whole_model(sample_y_fake, sample_z, cur_x.repeat(sample_z.size(0), 1, 1, 1),
                                            None, mask='test_attention').repeat(1, 3, 1, 1)
                    sample_x_s2.append(torch.cat([cur_x.cpu(), sample_x_.detach().cpu()], dim=0))
                    del sample_x_

                sample_x_s2 = (torch.cat(sample_x_s2, dim=-1) + 1) / 2.0
                torchvision.utils.save_image(sample_x_s2, '%s/%07d_%s_exchange_attention.jpg' % (save_dir, step, att),
                                             nrow=1)
                del sample_x_s2

        whole_model.train()

    if step > 0 and step % cfg.save_frequency == 0:
        save_dic = {'step': step,
                    'Model_G': whole_model.G.state_dict(), 'Model_D': whole_model.D.state_dict(),
                    'D_optimizer': D_optimizer.state_dict(), 'G_optimizer': G_optimizer.state_dict(),
                    'Model_Dz': whole_model.Dz.state_dict()}
        for i in range(len(cfg.use_atts)):
            save_dic['Dz_optimizer_%d' % i] = Dz_optimizer[i].state_dict()
        torchlib.save_checkpoint(save_dic, '%s/%07d.ckpt' % (ckpt_dir, step), max_keep=100)


def train(*args, mask, optimizers=None, loss_func=None, att_name):
    if not loss_func:
        loss_func = whole_model
    loss_total, loss_dict = loss_func(*args, mask, step=step, att_name=att_name)

    for o in optimizers:
        o.zero_grad()
    loss_total.mean().backward()
    for o in optimizers:
        o.step()

    # summary
    if (step // (cfg.D_per_G + 1)) % 20 == 0:
        for k, v in loss_dict.items():
            writer.add_scalar('%s/%s/%s' % (mask, k, att_name), loss_dict[k].data.mean().cpu().numpy(),
                              global_step=step)
        writer.add_scalar('%s/%s/%s' % (mask, 'total', att_name), loss_total.data.mean().cpu().numpy(),
                          global_step=step)


if __name__ == '__main__':
    cl = data.Celeba_labels(img_dir=cfg.img_dir, att_file=cfg.att_file)
    for step in tqdm.tqdm(range(start_step, cfg.step), total=cfg.step, initial=start_step, desc='step'):
        if step > 2:
            print_val_save_model(step, cl)
        G_scheduler.step()
        D_scheduler.step()

        for att in range(len(cfg.use_atts)):
            for i in range(cfg.D_per_G + 1 + cfg.D_per_G):
                x0s, x0ls, x1s, x1ls = train_dataset.get_batch_randomly_with_att_index(cfg.batch_size, att)
                x = torch.cat(x0s + x1s, dim=0).to(device)

                if step < cfg.get('multi_training', 9999999):
                    y_edit0 = torch.zeros(cfg.batch_size, len(cfg.use_atts))
                    y_edit0[:, att] = 1
                    y_edit1 = torch.zeros(cfg.batch_size, len(cfg.use_atts))
                    y_edit1[:, att] = -1
                    y_edit = torch.cat((y_edit0, y_edit1), dim=0).to(device)

                    y_real0 = torch.zeros(cfg.batch_size, len(cfg.use_atts))
                    # y_real0[:, att] = 0
                    y_real1 = torch.zeros(cfg.batch_size, len(cfg.use_atts))
                    y_real1[:, att] = 1
                    y_real = torch.cat((y_real0, y_real1), dim=0).to(device)
                else:
                    x0ls = torch.cat(x0ls, dim=0)
                    x1ls = torch.cat(x1ls, dim=0)
                    y_real = torch.cat([x0ls, x1ls], dim=0).type_as(x)
                    y_edit = torch.cat([x1ls - x0ls, x0ls - x1ls], dim=0).type_as(x)

                if i == 0:
                    train(y_edit, None, x, y_real, mask='G', optimizers=[G_optimizer], att_name=cfg.use_atts[att])
                elif 0 < i < cfg.D_per_G + 1:
                    train(y_edit, None, x, y_real, mask='D', optimizers=[D_optimizer], att_name=cfg.use_atts[att])
                else:  # use_Dz:
                    Dz_scheduler[att].step()
                    train(y_edit, None, x, y_real, mask='Dz', optimizers=[Dz_optimizer[att]],
                          att_name=cfg.use_atts[att])
