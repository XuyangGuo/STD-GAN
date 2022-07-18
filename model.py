from torch.autograd import Variable
from modules import *


def generate_z(cfg, y):
    z = torch.zeros(y.size(0), cfg.z_dim * len(cfg.use_atts))
    for i in range(y.size(0)):
        for j in range(y.size(1)):
            if y[i, j] == 1:
                if cfg.z_distribution == 'unif_-11':
                    z[i, j * cfg.z_dim: (j + 1) * cfg.z_dim] = \
                        torch.rand(cfg.z_dim) * 2. - 1.
    return z.type_as(y)


def mask_z(cfg, y, z):
    for i in range(y.size(0)):
        for j in range(y.size(1)):
            if y[i, j] != 1:
                z[i, j * cfg.z_dim: (j + 1) * cfg.z_dim] = 0


# ==============================================================================
# =                                  networks                                  =
# ==============================================================================
class WholeModel(nn.Module):
    def __init__(self, cfg):
        # network
        super().__init__()
        self.cfg = cfg

        self.G = Gen(cfg)
        self.D = MultiDis(cfg)
        self.use_Dz = "lr_dz" in cfg
        self.Dz = nn.ModuleList()
        for _ in range(len(cfg.use_atts)):
            self.Dz.append(MLP(cfg.z_dim, 1, dim=128, n_blk=4, activ='lrelu'))

    def forward(self, y_edit, z, x, y_real, mask=None, get_z_real_x_fake_=False, step=-1, att_name=None):
        if mask == 'embedding':
            z_real_ = self.D(x)[1][0]
            return z_real_.detach()

        # how to generate zp
        output_real_ = self.D(x)
        z_real_ = output_real_[1][0]
        # z_feature_real_ = output_real_[3][0]
        if z is None:
            z = generate_z(self.cfg, y_edit)

        if mask.startswith('test'):
            part_size = 2
            x_, x_res, x_attention = [], [], []
            for data_part in range(0, x.size(0), part_size):
                part_left = data_part
                part_right = min(data_part + part_size, x.size(0))
                g_results_part = self.G(x[part_left:part_right], y_edit[part_left:part_right], z[part_left:part_right])

                x_.append(g_results_part[0].detach())

                x_res.append(g_results_part[1].detach())
            x_ = torch.cat(x_, dim=0)
            x_res = torch.cat(x_res, dim=0)
            if x_attention:
                x_attention = torch.cat(x_attention, dim=0)
        elif mask is not "Dz":

            g_results = self.G(x, y_edit, z)
            x_ = g_results[0]
            x_res = g_results[1]

        if mask is "test":
            return torch.clamp(x_, -1.0, 1.0).detach().cpu()

        if mask is "test_res":
            # if self.cfg.generation_type != 'sa':
            return x_res.detach().cpu()

        loss_dict = ({})

        att_idx = self.cfg.use_atts.index(att_name)
        y_real_1 = y_real[:, att_idx] == 1
        y_real_0 = y_real[:, att_idx] == 0

        # temp for no two stage
        mask_z(self.cfg, -y_edit, z_real_)

        if mask is "G":
            if "du_rec_x" in self.cfg and step > self.cfg.rec_after and y_real_1.sum() > 0:
                # temp no two stage
                if 'no_two_stage' in self.cfg:
                    x_du_rec_101 = self.G(x_.detach()[y_real_1], y_real[y_real_1], z_real_[y_real_1])[0]
                else:
                    x_du_rec_101 = self.G(x_.detach()[y_real_1], -y_edit[y_real_1], z_real_[y_real_1])[0]
                loss_dict["du_rec_x"] = (x[y_real_1] - x_du_rec_101).abs().mean()

            if "du_rec_x_010" in self.cfg and step > self.cfg.rec_after and y_real_0.sum() > 0:
                if 'no_two_stage' in self.cfg:
                    x_du_rec_010 = self.G(x_.detach()[y_real_0], y_real[y_real_0] - 1, z_real_[y_real_0])[0]
                else:
                    x_du_rec_010 = self.G(x_.detach()[y_real_0], -y_edit[y_real_0], z_real_[y_real_0])[0]
                loss_dict["du_rec_x_010"] = (x[y_real_0] - x_du_rec_010).abs().mean()

        if mask is "D":
            if "du_rec_x_D" in self.cfg and step > self.cfg.rec_after and y_real_1.sum() > 0:
                if 'no_two_stage' in self.cfg:
                    x_du_rec_101 = self.G(x_.detach()[y_real_1], y_real[y_real_1], z_real_[y_real_1])[0]
                else:
                    x_du_rec_101 = self.G(x_.detach()[y_real_1], -y_edit[y_real_1], z_real_[y_real_1])[0]
                loss_dict["du_rec_x_D"] = (x[y_real_1] - x_du_rec_101).abs().mean()

            if "du_rec_x_010" in self.cfg and step > self.cfg.rec_after and y_real_1.sum() > 0:
                if 'no_two_stage' in self.cfg:
                    x_du_rec_010 = self.G(x_.detach()[y_real_0], y_real[y_real_0] - 1, z_real_[y_real_0])[0]
                else:
                    x_du_rec_010 = self.G(x_.detach()[y_real_0], -y_edit[y_real_0], z_real_[y_real_0])[0]
                loss_dict["du_rec_x_010"] = (x[y_real_0] - x_du_rec_010).abs().mean()
        # adv info
        assert y_real[:, att_idx][:4].sum() == 0
        assert y_real[:, att_idx][4:].mean() == 1
        z_real_att = z_real_.view(y_real.size(0), y_real.size(1), -1)[:, att_idx, :]
        z_real_filtered = z_real_att[y_real[:, att_idx] == 1]

        if mask is "D":
            loss_dict["dis_w"] = calc_dis_loss(self.D, x_.detach(), x, z, y_edit, y_real, loss_dict, x_res.detach(),
                                               step)
            loss_dict['dis_w_Dz'] = -torch.mean(self.Dz[att_idx](z_real_filtered))


        elif mask is "G":
            loss_dict["dis_w"] = calc_gen_loss(self.D, x_, z, y_edit, loss_dict, x_res, step)
        elif mask is "Dz":
            z = generate_z(self.cfg, y_real)

            z_unif_att = z.view(y_real.size(0), y_real.size(1), -1)[:, att_idx, :]
            z_unif_filtered = z_unif_att[y_real[:, att_idx] == 1]
            loss_dict["dis_w_Dz"] = calc_Dz_dis_loss(self.Dz[att_idx],
                                                     z_real_filtered.detach(), z_unif_filtered, loss_dict)

        loss_total = 0

        for k, v in loss_dict.items():
            if k not in self.cfg:  # skip rgs_zp
                continue

            loss_total += v * self.cfg[k]

        # just for log and debug
        if mask is "Dz":
            loss_dict['test_dis'] = (z_real_filtered - z_unif_filtered).abs().mean()
            loss_dict['test_pass_1'] = (z_real_filtered.abs() - 1).clamp(0).mean()
            loss_dict['test_diff'] = (z_real_filtered - z_real_filtered[-1, :]).abs().mean()

        if get_z_real_x_fake_:
            return loss_total, loss_dict, z_real_, x_
        else:
            return loss_total, loss_dict


# ======================================
# =            discriminator           =
# ======================================
def zp_loss(z_distribution, z_rgs, z, y_edit):
    y_add = (y_edit == 1)

    z_rgs = z_rgs.view(*y_add.size(), -1)
    z = z.view(*y_add.size(), -1)
    return F.mse_loss(z_rgs[y_add], z[y_add])


def get_zp_rgs_loss(loss_name, result_0, D, loss_dict, z, y_edit):
    if (y_edit == 1).sum() > 0:
        loss_dict[loss_name] = 0

        for z_rgs in result_0[1]:
            loss_dict[loss_name] += zp_loss(D.z_distribution, z_rgs, z, y_edit)


def calc_dis_loss(D, input_fake, input_real, z, y_edit, y_real, loss_dict, input_fake_res, step):
    z_without_101 = z
    y_fake_without_101 = y_edit

    # calculate the loss to train D
    dis_result_0 = D.forward(input_fake)
    dis_result_1 = D.forward(input_real)
    loss_dis = 0

    for dis0, dis1 in zip(dis_result_0[0], dis_result_1[0]):
        if D.gan_type == 'lsgan':
            loss_dis += torch.mean((dis0 - 0) ** 2) + torch.mean((dis1 - 1) ** 2)
        elif D.gan_type == 'nsgan':
            all0 = Variable(torch.zeros_like(dis0.data).cuda(), requires_grad=False)
            all1 = Variable(torch.ones_like(dis1.data).cuda(), requires_grad=False)
            loss_dis += torch.mean(F.binary_cross_entropy(torch.sigmoid(dis0), all0) +
                                   F.binary_cross_entropy(torch.sigmoid(dis1), all1))
        elif D.gan_type == 'wgan_gp':
            loss_dis += torch.mean(dis0) - torch.mean(dis1)
        else:
            assert 0, "Unsupported GAN type: {}".format(D.gan_type)

    if D.gan_type == 'wgan_gp':
        loss_gp = 0

        alpha = torch.rand(input_real.size(0), 1, 1, 1).type_as(input_real)
        x_hat = (alpha * input_real + (1 - alpha) * input_fake).requires_grad_(True)
        dis_result_hat = D.forward(x_hat)
        for it, out_hat in enumerate(dis_result_hat[0]):
            # gradient penalty
            weight = torch.ones(out_hat.size()).type_as(out_hat)
            dydx = torch.autograd.grad(outputs=out_hat, inputs=x_hat, grad_outputs=weight, retain_graph=True,
                                       create_graph=True, only_inputs=True)[0]

            dydx = dydx.contiguous().view(dydx.size(0), -1)
            dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
            loss_gp += torch.mean((dydx_l2norm - 1) ** 2)
        loss_dict['lambda_gp'] = loss_gp

    dis_result_0_without_101 = dis_result_0
    get_zp_rgs_loss('rgs_zp_D', dis_result_0_without_101, D, loss_dict, z_without_101, y_fake_without_101)
    dis_result_1_without_101 = dis_result_1

    loss_dict["cls_y_D"] = 0
    y_edit_mask = (y_edit != 0)
    for y_cls in dis_result_1_without_101[2]:
        loss_dict["cls_y_D"] += F.binary_cross_entropy_with_logits(y_cls[y_edit_mask], y_real[y_edit_mask])

    return loss_dis


def calc_gen_loss(D, input_fake, z, y_edit, loss_dict, input_fake_res, step):
    # calculate the loss to train G
    dis_result = D.forward(input_fake)

    loss_dis = 0
    for out0 in dis_result[0]:
        if D.gan_type == 'lsgan':
            loss_dis += torch.mean((out0 - 1) ** 2)
        elif D.gan_type == 'nsgan':
            all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
            loss_dis += torch.mean(F.binary_cross_entropy(torch.sigmoid(out0), all1))
        elif D.gan_type == 'wgan_gp':
            loss_dis -= torch.mean(out0)
        else:
            assert 0, "Unsupported GAN type: {}".format(D.gan_type)
    get_zp_rgs_loss('rgs_zp', dis_result, D, loss_dict, z, y_edit)

    loss_dict["cls_y"] = 0
    y_edit_mask = (y_edit != 0)
    for y_cls in dis_result[2]:
        loss_dict["cls_y"] += F.binary_cross_entropy_with_logits(y_cls[y_edit_mask], (y_edit[y_edit_mask] + 1) / 2)

    return loss_dis


def calc_Dz_dis_loss(Dz, z_fake, z_real, loss_dict):
    out0 = Dz(z_fake)
    out1 = Dz(z_real)

    loss_dis = torch.mean(out0) - torch.mean(out1)
    alpha = torch.rand(z_real.size(0), 1).type_as(z_real)
    x_hat = (alpha * z_real + (1 - alpha) * z_fake.detach()).requires_grad_(True)
    out_hat = Dz.forward(x_hat)
    weight = torch.ones(out_hat.size()).type_as(out_hat)
    dydx = torch.autograd.grad(outputs=out_hat, inputs=x_hat, grad_outputs=weight, retain_graph=True,
                               create_graph=True, only_inputs=True)[0]
    dydx = dydx.contiguous().view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx ** 2, dim=1))
    loss_gp = torch.mean((dydx_l2norm - 1) ** 2)
    loss_dict['lambda_gp_Dz'] = loss_gp
    return loss_dis


class MultiDis(nn.Module):
    # Multi-scale discriminator architecture
    def __init__(self, cfg):
        super(MultiDis, self).__init__()
        P = cfg.discriminator
        self.cfg = cfg
        self.gan_type = cfg.gan_type
        self.conditional_generation = cfg.conditional_generation
        self.z_distribution = cfg.z_distribution
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
        self.cnns = nn.ModuleList()

        input_size = cfg.img_size
        for _ in range(P.num_scales):
            self.cnns.append(
                SingleDis(P, cfg.img_channels, input_size, cfg.z_dim, len(cfg.use_atts), cfg.conditional_generation))
            input_size //= 2

    def forward(self, x, label=None):
        dis_results = [[], [], [], []]
        for model in self.cnns:
            result = model(x)
            for i, r in enumerate(result):
                dis_results[i].append(r)
            x = self.downsample(x)
        return dis_results


class SingleDis(nn.Module):
    def __init__(self, P, input_dim, input_size, z_dim, y_dim, conditional_generation):
        super(SingleDis, self).__init__()
        self.conditional_generation = conditional_generation
        self.y_dim = y_dim

        extra_dim = y_dim if self.conditional_generation == 'cgan' else 0
        self.cnn_x, dim, final_size = self._make_net(P, input_dim, input_size, P.dim, extra_dim)
        self.dis_layer = nn.Linear(final_size * final_size * (dim + extra_dim), 1)

        if P.norm == 'in' or P.norm == 'myin':
            self.zp_rgs_layer = MLP(final_size * final_size * (dim + extra_dim), (z_dim + 1) * y_dim - extra_dim, 128,
                                    n_blk=2, norm='none', activ=P.activ)
        else:
            self.zp_rgs_layer = MLP(final_size * final_size * (dim + extra_dim), (z_dim + 1) * y_dim - extra_dim, 128,
                                    n_blk=2, norm=P.norm, activ=P.activ)

    def _make_net(self, P, input_dim, input_size, dim, extra_dim):
        cnn_x = [Conv2dBlock(input_dim + extra_dim, dim, 4, 2, 1, norm='none', activation=P.activ, pad_type=P.pad_type)]
        for i in range(P.n_layer - 1):
            if dim < 2048:
                cnn_x += [
                    Conv2dBlock(dim + extra_dim, dim * 2, 4, 2, 1, norm=P.norm, activation=P.activ,
                                pad_type=P.pad_type)]
                dim *= 2
            else:
                cnn_x += [
                    Conv2dBlock(dim + extra_dim, dim, 4, 2, 1, norm=P.norm, activation=P.activ,
                                pad_type=P.pad_type)]

        cnn_x = nn.Sequential(*cnn_x)
        return cnn_x, dim, input_size // (2 ** P.n_layer)

    def forward(self, x, label=None):
        x = self.cnn_x(x)
        x_f = x.view(x.size(0), -1)
        res_zp_feature = self.zp_rgs_layer.model[0](x_f)
        res_zp = self.zp_rgs_layer.model[1](res_zp_feature)
        return self.dis_layer(x_f), res_zp[:, self.y_dim:], res_zp[:, 0:self.y_dim], res_zp_feature


# ======================================
# =              generator             =
# ======================================

class Gen(nn.Module):

    def __init__(self, cfg):
        super(Gen, self).__init__()
        # P = cfg.generator
        self.cfg = cfg
        self.y_dim = len(cfg.use_atts)
        P = cfg.generator
        self.cnn_e = Encoder(P, cfg.img_channels)
        dims = self.cnn_e.get_out_dim()
        res_dim = dims[-1]
        self.cnn_r = ResBlocks(P.n_res, res_dim + cfg.z_dim * self.y_dim + self.y_dim, norm=P.norm,
                               activation=P.activ, pad_type=P.pad_type)
        self.cnn_d = Decoder(P, res_dim + cfg.z_dim * self.y_dim + self.y_dim, dims, output_dim=cfg.img_channels,
                             last_activ=P.res_last_activ, skip_connect=cfg.skip_connect)

    def forward(self, x, y, z):
        y = y.view(y.size(0), y.size(1), 1, 1)
        z = z.view(z.size(0), z.size(1), 1, 1)
        x_zp_cat = x
        x_e = self.cnn_e(x_zp_cat)
        x_e_zp_cat = torch.cat([x_e[-1], y.repeat(1, 1, x_e[-1].size(2), x_e[-1].size(3)),
                                z.repeat(1, 1, x_e[-1].size(2), x_e[-1].size(3))], dim=1)
        x_r = self.cnn_r(x_e_zp_cat)
        x_d = self.cnn_d(x_r, x_e)

        if self.cfg.generation_type == 'default':
            if self.cfg.generator.res_last_activ == 'tanh':
                return x_d * 2 + x, x_d
            return x_d + x, x_d
        elif self.cfg.generation_type == 'vanilla':
            return x_d, x_d - x


###################################
# Basic Module                    #
###################################

class Encoder(nn.Module):
    def __init__(self, P, input_dim, first_layer_norm=True):
        super(Encoder, self).__init__()
        self.dim = P.dim
        self.dims = []

        # self.dim = input_dim * 2
        if first_layer_norm:
            cnn = [Conv2dBlock(input_dim, self.dim, 7, 1, 3, norm=P.norm, activation=P.activ, pad_type=P.pad_type)]
        else:
            cnn = [Conv2dBlock(input_dim, self.dim, 7, 1, 3, norm='none', activation=P.activ, pad_type=P.pad_type)]

        self.dims.append(self.dim)

        for i in range(P.n_sample):
            self.next_dim = self.dim * 2

            self.dims.append(self.next_dim)
            cnn.append(
                Conv2dBlock(self.dim, self.next_dim, 4, 2, 1, norm=P.norm, activation=P.activ, pad_type=P.pad_type))
            self.dim = self.next_dim

        self.cnn = nn.Sequential(*cnn)

    def get_out_dim(self):
        return self.dims

    def forward(self, x):
        zs = []
        for layer in self.cnn:
            x = layer(x)
            zs.append(x)
        return zs


class Decoder(nn.Module):
    def __init__(self, P, input_dim, dims, output_dim=-1, last_activ='tanh', last_norm='none', skip_connect=False):
        super(Decoder, self).__init__()
        self.skip_connect = skip_connect

        if P.up_sample_method == 'conv_transpose':
            cnn = [Conv2dBlock(input_dim, dims[-1], 4, 2, 1, norm=P.norm, activation=P.activ, pad_type=P.pad_type,
                               transpose=True)]
        else:
            cnn = [nn.Sequential(nn.Upsample(scale_factor=2),
                                 Conv2dBlock(input_dim, dims[-1], 5, 1, 2, norm=P.norm, activation=P.activ,
                                             pad_type=P.pad_type))]

        input_dim_ratio = 1.5 if skip_connect else 1

        for i in range(P.n_sample - 1):
            dim = dims[-i - 2] * 2
            if P.up_sample_method == 'conv_transpose':
                cnn.append(Conv2dBlock(int(dim * input_dim_ratio),
                                       dim // 2, 4, 2, 1, norm=P.norm, activation=P.activ, pad_type=P.pad_type,
                                       transpose=True))
            else:
                cnn.append(nn.Sequential(nn.Upsample(scale_factor=2), Conv2dBlock(int(
                    dim * input_dim_ratio), dim // 2, 5, 1, 2, norm=P.norm, activation=P.activ, pad_type=P.pad_type)))

        dim = dims[0] * 2
        cnn.append(Conv2dBlock(int(dim * input_dim_ratio),  # remove transpose
                               output_dim, 7, 1, 3, norm=last_norm, activation=last_activ, pad_type=P.pad_type))

        self.cnn = nn.Sequential(*cnn)

    def forward(self, x, zs=None):
        if self.skip_connect:
            x = self.cnn[0](x)
            for i, layer in enumerate(self.cnn[1:]):
                x = torch.cat([x, zs[-i - 2]], dim=1)
                x = layer(x)
            return x
        else:
            return self.cnn(x)
