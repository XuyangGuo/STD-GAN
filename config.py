import addict  # nesting dict

configs = [
    addict.Dict({
        "experiment_name": "001_default",
        "gpu": "0",

        # data
        "img_dir": "your/dataset/dir/celeba_data/img_align_celeba",
        "att_file": "your/dataset/dir/celeba_data/Anno/list_attr_celeba.txt",
        "use_atts": ['Bangs', 'Eyeglasses', 'Black_Hair', 'Smiling'],  # Black_Hair means change black to others
        "img_size": 256,
        "img_channels": 3,
        "well_cropped": False,

        # training
        "step": 60100,
        "batch_size": 4,
        # loss
        "gan_type": "wgan_gp",
        "lambda_gp": 10.0,  # wgan_gp
        "dis_w": 1.0,
        "rgs_zp": 5.0,
        "rgs_zp_D": 5.0,
        "cls_y": 5.0,
        "cls_y_D": 5.0,
        "du_rec_x": 30.0,
        "du_rec_x_D": 30.0,
        "du_rec_x_010": 30.0,
        "rec_after": 40000,  # begin reconstruction

        "dis_w_Dz": 0.05,
        "lambda_gp_Dz": 0.5,  # wgan_gp
        "lr_dz": 0.005,

        "D_per_G": 1,
        "num_workers": 8,
        "display_batch_size": 30,
        "display_style_num": 10,
        "display_frequency": 5000,
        "save_frequency": 20000,

        # opimizer
        "lr_d": 0.0001,
        "lr_g": 0.0001,
        "beta1": 0.5,
        "beta2": 0.999,
        "weight_decay_d": 0.000,
        "weight_decay_g": 0.0,
        "lr_policy": "step",
        "step_size": 40000,
        "gamma": 0.1,

        "multi_training": 50000,

        # model
        "z_dim": 16,
        "z_distribution": "unif_-11",
        "generation_type": "default",
        "skip_connect": True,
        "discriminator": {
            "activ": "lrelu",
            # "leaky_slope": 0.2,
            "dim": 64,
            "n_layer": 5,
            "norm": "none",
            "num_scales": 1,
            "pad_type": "zero",
            # "weight_init": "gaussian"
        },
        "generator": {
            "activ": "relu",
            "dim": 32,
            "n_sample": 3,
            "n_res": 6,
            "norm": "ln",
            "pad_type": "zero",
            # "weight_init": "gaussian",
            "up_sample_method": "conv_transpose",
            "res_last_activ": "tanh",
        },
    })
]


def get_config(id):
    for c in configs:
        if c.experiment_name.startswith(id):
            config = c
            return config
