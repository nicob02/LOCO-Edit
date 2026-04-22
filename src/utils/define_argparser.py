import argparse
import traceback
import logging
import yaml
import sys
import os
import torch
import numpy as np
import random
import shutil

from configs.params import X_SPACE_GUIDANCE_SCALE_DICT, X_SPACE_EDIT_STEP_SIZE_DICT

def parse_args():
    parser = argparse.ArgumentParser()

    # default setting 
    parser.add_argument('--sh_file_name',   type=str,   default='',      required=False, help="for logging")
    parser.add_argument('--device',         type=str,   default='',      required=False, help="'cuda', 'cpu'")
    parser.add_argument('--dtype',          type=str,   default='fp16',  required=False, help="'fp32', 'fp16'")
    parser.add_argument('--seed',           type=int,   default=0,       required=False, help='Random seed')
    parser.add_argument('--result_folder',  type=str,   default='./runs/', help='Path for saving running related data.')
    parser.add_argument('--cache_folder',  type=str,   default='/nfs/turbo/coe-qingqu1/huijie/exp/controllable_generation', help='Path for saving stable diffusion.')
    parser.add_argument('--dataset_root',  type=str,   default='/scratch/qingqu_root/qingqu1/shared_data/celebA-HQ-mask/CelebAMask-HQ', help='Path for saving running related data.')
    
    # model, dataset setting
    parser.add_argument('--model_name',     type=str,   default='',     required=False)
    parser.add_argument('--dataset_name',   type=str,   default='',     required=False)
    parser.add_argument('--num_imgs',       type=int,   default=100,    required=False)
    parser.add_argument('--image_size',     type=int,   default=256,    required=False)
    parser.add_argument('--c_in',           type=int,   default=3,      required=False)
    parser.add_argument('--sample_idx',     type=int,   default=0,      required=False)

    # args (prompt)
    parser.add_argument('--for_prompt',     type=str,   default='',     required=False)
    parser.add_argument('--inv_prompt',     type=str,   default='',     required=False)
    parser.add_argument('--neg_prompt',     type=str,   default='',     required=False)
    
    # args (diffusion schedule)
    parser.add_argument('--for_steps',      type=int,   default=100,    required=False)
    parser.add_argument('--inv_steps',      type=int,   default=100,    required=False)
    parser.add_argument('--performance_boosting_t',     type=float,     default=0.0,    required=False)
    parser.add_argument('--use_yh_custom_scheduler',    type=str2bool,  default='True', required=False, help='Use custom scheduler for better inversion quality')

    # args (guidance)
    parser.add_argument('--guidance_scale', type=float, default=0,  required=False)
    parser.add_argument('--guidance_scale_edit', type=float, default=4.0,  required=False)
    
    # args (h space edit)
    parser.add_argument('--edit_prompt',     type=str,  default='',      required=False)
    parser.add_argument('--original_prompt',     type=str,  default='',      required=False)

    parser.add_argument('--edit_xt',        type=str,   default='default',      required=False, help="'parallel-x' or 'parallel-h'")

    parser.add_argument('--use_x_space_guidance',                       type=str2bool,  default='False',    required=False)
    parser.add_argument('--x_space_guidance_direct',                    type=str2bool,  default='False',    required=False)
    parser.add_argument('--x_space_guidance_edit_step',                 type=float,     default=1,          required=False)
    parser.add_argument('--x_space_guidance_scale',                     type=float,     default=0,          required=False)
    parser.add_argument('--x_space_guidance_num_step',                  type=int,       default=0,          required=False)
    parser.add_argument('--x_space_guidance_use_edit_prompt',           type=str2bool,  default='True',     required=False)
    parser.add_argument('--pca_rank_null',                  type=int,       default=5,          required=False)
    parser.add_argument('--pca_rank',                  type=int,       default=5,          required=False)

    parser.add_argument('--h_t',            type=float, default=0.8,            required=False)
    parser.add_argument('--edit_t',         type=float, default=1.0,            required=False, help="after no_edit_t_idx, do not apply edit")
    parser.add_argument('--no_edit_t',      type=float, default=0.5,            required=False, help="after no_edit_t_idx, do not apply edit")
    parser.add_argument('--h_edit_step_size', type=float, default=0,            required=False)
    parser.add_argument('--x_edit_step_size', type=float, default=0,            required=False)

    
    # memory
    parser.add_argument('--pca_device',     type=str,   default='cpu',      required=False)
    parser.add_argument('--buffer_device',  type=str,   default='cpu',      required=False)
    parser.add_argument('--save_result_as', type=str,   default='image',    required=False, help='image or tensor')

    # exp setting    
    parser.add_argument('--note',                                           type=str,       required=False)
    parser.add_argument('--run_cfg_forward',                                type=str2bool,  default='False', required=False)
    parser.add_argument('--run_mcg_forward',                                type=str2bool,  default='False', required=False)
    parser.add_argument('--run_pfg_forward',                                type=str2bool,  default='False', required=False)
    parser.add_argument('--run_ddim_forward',                               type=str2bool,  default='False', required=False)
    parser.add_argument('--run_ddim_inversion',                             type=str2bool,  default='False', required=False)

    parser.add_argument('--run_edit_local_encoder_pullback_zt',             type=str2bool,  default='False', required=False)
    parser.add_argument('--run_edit_local_decoder_pullback_zt',             type=str2bool,  default='False', required=False)
    parser.add_argument('--run_edit_local_encoder_decoder_pullback_zt',     type=str2bool,  default='False', required=False)
    parser.add_argument('--encoder_decoder_by_et',                          type=str2bool,  default='False', required=False)
    parser.add_argument('--use_mask',                          type=str2bool,  default='True', required=False)
    parser.add_argument('--run_edit_local_x0_decoder_pullback_zt',          type=str2bool,  default='False', required=False)
    parser.add_argument('--run_edit_local_pca_zt',                          type=str2bool,  default='False', required=False)
    parser.add_argument('--run_edit_null_space_projection',             type=str2bool,  default='False', required=False)
    parser.add_argument('--run_edit_null_space_projection_zt',             type=str2bool,  default='False', required=False)
    parser.add_argument('--run_edit_null_space_projection_zt_semantic',     type=str2bool,  default='False', required=False)
    parser.add_argument('--run_edit_null_space_projection_xt',             type=str2bool,  default='False', required=False)
    parser.add_argument('--run_edit_null_space_projection_xt_semantic',     type=str2bool,  default='False', required=False)
    parser.add_argument('--group_edit_null_space_projection',             type=str2bool,  default='False', required=False)    
    


    parser.add_argument('--vis_num',      type=int,   default=4,    required=False)
    parser.add_argument('--choose_sem',                         type=str,  default='hair', required=False)
    parser.add_argument('--null_space_projection',                          type=str2bool,  default='False', required=False)
    
    # mode
    parser.add_argument('--debug_mode',                                     type=str2bool,  default='False', required=False)
    parser.add_argument('--sampling_mode',                                  type=str2bool,  default='False', required=False)
    parser.add_argument('--non_semantic',                                  type=str2bool,  default='False', required=False)


    ## mask segmentation
    parser.add_argument('--mask_model_name',                    type=str,  default="facebook/sam-vit-large", required=False)
    parser.add_argument('--filter_mask',                        type=int,  default=100, required=False)
    parser.add_argument('--mask_index',                         type=int,  default=0, required=False)
    parser.add_argument('--mask_type',                         type=str,  default="SAM", required=False, choices=["SAM", "diffedit"])
    parser.add_argument('--ablation_method',                   type=str, required=False, choices=["null-space-proj", "sega", "diffedit"])
    parser.add_argument('--tilda_v_score_type',                 type=str, choices=["proj_null[for-null](edit-null)-direct", "(for-edit)-direct", "(edit-null)-direct", "null+(for-null)+(edit-null)", "null+(for-null)", "null+(edit-null)","(for-edit)", "edit-proj[for](edit)", "null+for+edit-proj[for](edit)"],  required=False)
    parser.add_argument('--vT_path',                            type=str, default="", required=False)
    parser.add_argument('--vT1_path',                            type=str, default="", required=False)
    parser.add_argument('--jacobian',                           type=str2bool,  default='False', required=False)
    parser.add_argument('--use_sega',                           type=str2bool, default='False')
    parser.add_argument('--edit_t_idx',                  type=int,       default=1,          required=False)
    parser.add_argument('--num_inference_steps',         type=int,       default=3,          required=False)

    parser.add_argument('--random_edit',                                type=str2bool,  default='False', required=False)

    # ----- Phase 2 / Attack A (direction instability via PGD on x_t) -----
    parser.add_argument('--run_attack_a',     type=str2bool, default='False', required=False,
                        help='Phase 2: run PGD attack on x_t to rotate the discovered editing direction.')
    parser.add_argument('--attack_eps',       type=float,    default=0.02,    required=False,
                        help='Per-coordinate (linf) or total (l2) radius of the perturbation ball.')
    parser.add_argument('--attack_alpha',     type=float,    default=0.005,   required=False,
                        help='PGD step size.')
    parser.add_argument('--attack_steps',     type=int,      default=40,      required=False,
                        help='Number of PGD iterations.')
    parser.add_argument('--attack_norm',      type=str,      default='linf',  required=False,
                        choices=['linf', 'l2'],
                        help='Threat-model ball geometry.')
    parser.add_argument('--attack_init',      type=str,      default='zero',  required=False,
                        choices=['zero', 'rand'],
                        help='Initialisation of delta inside the eps-ball.')
    parser.add_argument('--attack_eps_sweep', type=str,      default='',      required=False,
                        help='Optional comma-separated list of eps values, e.g. "0.005,0.01,0.02".')
    parser.add_argument('--attack_render_edit', type=str2bool, default='True', required=False,
                        help='After the attack, run the full LOCO edit at x_t + delta_adv to render a comparison strip.')
    parser.add_argument('--attack_basis_src',   type=str,      default='',      required=False,
                        help='Optional path to an existing `basis/local_basis-*` folder (or any parent '
                             'directory thereof) produced by Phase 1. If set, the script reuses the '
                             'cached vT-modify / vT-null tensors from there instead of recomputing them.')

    # ----- Phase 2 / Attack B (image-space PGD on x_0) -----
    parser.add_argument('--run_attack_b',           type=str2bool, default='False', required=False,
                        help='Phase 2: run PGD attack on the *input image* x_0 to break LOCO '
                             'after DDIM inversion + forward. Threat model: attacker controls '
                             'uploaded pixels with ||delta_img||_inf <= eps_img.')
    parser.add_argument('--attack_b_eps_img',       type=float,    default=0.031,   required=False,
                        help='L_inf radius of the image-space perturbation in the native [-1, 1] '
                             'pixel range. Default 0.031 = 8/255 in [-1,1] units, which is 4/255 '
                             'in the usual [0,1] convention - the canonical adversarial budget.')
    parser.add_argument('--attack_b_alpha_img',     type=float,    default=0.005,   required=False,
                        help='PGD step size in image-space units.')
    parser.add_argument('--attack_b_steps',         type=int,      default=40,      required=False,
                        help='Number of image-space PGD iterations.')
    parser.add_argument('--attack_b_eps_sweep',     type=str,      default='',      required=False,
                        help='Optional comma-separated sweep over eps_img, e.g. "0.004,0.008,0.016,0.031,0.063".')
    parser.add_argument('--attack_b_target_sem',    type=str,      default='',      required=False,
                        help='If set (e.g. "hair"), enables TARGETED HIJACK: the attacker also tries '
                             'to steer v_adv toward the top singular direction of the target semantic '
                             'masked GPM. Requires a pre-computed Phase 1 basis for that semantic on '
                             'the same sample_idx.')
    parser.add_argument('--attack_b_target_beta',   type=float,    default=1.0,     required=False,
                        help='Weight of the hijack term in the PGD objective. 0 = pure destruction.')

    # ----- Phase 2 post-hoc tools (locality / transfer) -----
    parser.add_argument('--attack_type',  type=str, default='B', required=False,
                        choices=['A', 'B'],
                        help='Which attack family to post-process (A=latent, B=image).')
    parser.add_argument('--sweep_dir',    type=str, default='', required=False,
                        help='Directory containing attack run subfolders. Auto-discovered if empty.')
    parser.add_argument('--locality_lambda', type=float, default=None, required=False,
                        help='Override edit magnitude for locality (defaults to LOCO max lambda).')
    # Transfer-attack args: perturb is taken from --attack_b_result; targets listed here.
    parser.add_argument('--attack_b_result', type=str, default='', required=False,
                        help='Path to an attackB_result.pt produced on the SOURCE sample.')
    parser.add_argument('--transfer_targets', type=str, default='', required=False,
                        help='Comma-separated list of target sample indices, e.g. "1000,2000,3000".')

    # ----- Phase 3 / Defenses -----
    # D1: randomized smoothing.
    parser.add_argument('--defense_sigmas', type=str, default='0.01,0.02,0.05', required=False,
                        help='Comma-separated sigmas for D1 randomized smoothing of x_t.')
    parser.add_argument('--defense_n_samples', type=str, default='5,10', required=False,
                        help='Comma-separated number-of-samples for D1.')
    # D2: input purification.
    parser.add_argument('--purify_plan', type=str,
                        default='jpeg:75,90 blur:0.5,1.0 bits:4,6',
                        required=False,
                        help='Space-separated list of method:param-csv tokens, '
                             'e.g. "jpeg:75,90 blur:0.5,1.0 bits:4,6".')

    args = parser.parse_args()
    return args

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def preset(args):
    # reproducatibility
    if args.seed == 0:
        args.seed = int(torch.randint(2**32, ()).item())
    seed_everything(args.seed)
    ###############
    # config file #
    ###############
    # parse config file (pretrained model)
    if 'stable-diffusion' in args.model_name:
        args.is_stable_diffusion = True
        args.is_DeepFloyd_IF_diffusion = False
        args.is_LCM = False
        # save path
        args.exp = f'Stable_Diffusion-{args.dataset_name}-{args.note}'
        args.exp_folder = os.path.join(args.result_folder, args.exp)
    elif 'DeepFloyd' in args.model_name:
        args.is_stable_diffusion = False
        args.is_DeepFloyd_IF_diffusion = True
        args.is_LCM = False
        args.exp = f'DeepFloyd-IF-{args.dataset_name}-{args.note}'
        args.exp_folder = os.path.join(args.result_folder, args.exp)  
    elif 'LCM' in args.model_name:
        args.is_stable_diffusion = False
        args.is_DeepFloyd_IF_diffusion = False
        args.is_LCM = True
        args.exp = f'LCM-{args.dataset_name}-{args.note}'
        args.exp_folder = os.path.join(args.result_folder, args.exp)        
    else:
        args.is_stable_diffusion = False
        args.is_DeepFloyd_IF_diffusion = False
        args.is_LCM = False
        if args.model_name == 'CelebA_HQ':
            raise NotImplementedError('Model weight deprecated...')
        elif args.model_name in ["FFHQ_P2", "AFHQ_P2", "Flower_P2", "Cub_P2", "Metface_P2"]:
            pass
        elif args.model_name in ['LSUN_bedroom', 'LSUN_cat', 'LSUN_horse']:
            raise NotImplementedError('Please download P2 weight from https://github.com/jychoi118/P2-weighting')
        elif args.model_name in ['CelebA_HQ_HF', 'LSUN_church_HF', 'LSUN_bedroom_HF', 'FFHQ_HF']:
            pass
        else:
            raise ValueError('model_name choice: [CelebA_HQ_HF, LSUN_church_HF, FFHQ_HF]')

        # save path
        direct = 'i'
        if args.x_space_guidance_direct:
            direct = 'd'
        # NOTE: include `--note` in exp folder so different runs don't clobber
        # each other (upstream dropped it on this branch, which silently merged
        # all runs into one folder).
        args.exp = f'{args.model_name}-{args.dataset_name}'
        if getattr(args, 'note', None):
            args.exp = f'{args.exp}-{args.note}'
        args.exp_folder = os.path.join(args.result_folder, args.exp)

    ##########
    # folder #    
    ##########
    os.makedirs(args.exp_folder, exist_ok=True)

    # Provenance copies: archive the launch script + key source files into the
    # run folder. Treat missing files as a warning, not a crash, and search a
    # few likely locations so nested script layouts (e.g. scripts/nibi/...) work.
    def _safe_copy(candidates, dst_dir):
        for src in candidates:
            if os.path.isfile(src):
                dst = os.path.join(dst_dir, os.path.basename(src))
                try:
                    shutil.copy(src, dst)
                except Exception as exc:
                    print(f"[preset] warn: could not copy {src!r} -> {dst!r}: {exc}")
                return
        print(f"[preset] warn: none of the provenance candidates exist: {candidates}")

    _safe_copy(
        [
            os.path.join('scripts', args.sh_file_name),
            os.path.join('scripts', 'nibi', args.sh_file_name),
            args.sh_file_name,
        ],
        args.exp_folder,
    )
    _safe_copy([os.path.join('utils', 'define_argparser.py')], args.exp_folder)
    _safe_copy(['main.py'], args.exp_folder)

    args.obs_folder    = os.path.join(args.exp_folder, 'obs')
    args.result_folder = os.path.join(args.exp_folder, 'results')
    
    os.makedirs(args.obs_folder, exist_ok=True)
    os.makedirs(args.result_folder, exist_ok=True)

    ##################
    # dependent args #
    ##################
    args.device = torch.device(args.device)
    args.dtype = torch.float32 if args.dtype == 'fp32' else torch.float16
    print(f'device : {args.device}, dtype : {args.dtype}')

    # edit scale
    if args.use_x_space_guidance:
        if args.is_stable_diffusion:
            args.x_space_guidance_scale = X_SPACE_GUIDANCE_SCALE_DICT['stable-diffusion'][args.h_t]
        else:
            args.x_space_guidance_scale = X_SPACE_GUIDANCE_SCALE_DICT['uncond'][args.h_t]

    # input size, memory bound to avoid OOM
    if args.is_stable_diffusion:
        args.c_in = 4
        args.image_size = 64
        args.memory_bound = 5
    elif 'CIFAR10' in args.model_name:
        args.c_in = 3
        args.memory_bound = 50
        args.image_size = 32
    elif args.is_DeepFloyd_IF_diffusion:
        args.c_in = 3
        args.image_size = 64
        args.memory_bound = 5
    else:
        args.c_in = 3
        args.image_size = 256
        args.memory_bound = 50
        args.noise_schedule = 'linear'

    ##########
    # assert #
    ##########
    if args.is_stable_diffusion or args.is_DeepFloyd_IF_diffusion:
        assert args.use_yh_custom_scheduler
        # assert args.for_steps == 100
        assert args.performance_boosting_t <= 0
    elif args.is_LCM:
        pass
    else:
        assert args.use_yh_custom_scheduler
        assert args.for_steps == 100
        assert args.performance_boosting_t == 0.2

    return args

def seed_everything(seed):
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True