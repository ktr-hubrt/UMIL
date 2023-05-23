import os
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import argparse
import datetime
import shutil
from pathlib import Path
from utils.config import get_config
from utils.optimizer import build_optimizer, build_scheduler
from utils.tools import AverageMeter, reduce_tensor, epoch_saving, load_checkpoint, generate_text, auto_resume_helper, evaluate_result
from utils.cluster import ClusterLoss, Normalize, BCE, NCLMemory, PairEnum
from datasets.build import build_dataloader
from utils.logger import create_logger
import time
import numpy as np
import random
import mmcv
from apex import amp
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from datasets.blending import CutmixMixupBlending
from utils.config import get_config
from models import xclip
from einops import rearrange
import torch.nn.functional as F

def parse_option():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-cfg', required=True, type=str, default='configs/k400/32_8.yaml')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )
    parser.add_argument('--output', type=str, default="exp")
    parser.add_argument('--resume', type=str)
    parser.add_argument('--pretrained', type=str)
    parser.add_argument('--only_test', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--accumulation-steps', type=int)
    # model parameters
    parser.add_argument("--local_rank", type=int, default=-1, help='local rank for DistributedDataParallel')
    parser.add_argument('--w-smooth', default=0.01, type=float, help='weight of smooth loss')
    parser.add_argument('--w-sparse', default=0.001, type=float, help='weight of sparse loss')

    args = parser.parse_args()

    config = get_config(args)

    return args, config


def main(config):
    train_data, val_data, test_data, train_loader, val_loader, test_loader, val_loader_train,_ = build_dataloader(logger, config)
    model, _ = xclip.load(config.MODEL.PRETRAINED, config.MODEL.ARCH, 
                         device="cpu", jit=False, 
                         T=config.DATA.NUM_FRAMES,
                         droppath=config.MODEL.DROP_PATH_RATE, 
                         use_checkpoint=config.TRAIN.USE_CHECKPOINT, 
                         use_cache=config.MODEL.FIX_TEXT,
                         logger=logger,
                        )
    model = model.cuda()


    optimizer, _ = build_optimizer(config, model)
    lr_scheduler = build_scheduler(config, optimizer, len(train_loader))
    if config.TRAIN.OPT_LEVEL != 'O0':
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, opt_level=config.TRAIN.OPT_LEVEL)

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False, find_unused_parameters=False)

    start_epoch, best_epoch, max_auc = 0, 0, 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        start_epoch, max_accuracy = load_checkpoint(config, model.module, optimizer, lr_scheduler, logger)

    text_labels = generate_text(train_data)
    
    if config.TEST.ONLY_TEST:
        if not os.path.isdir(config.MODEL.PRETRAINED):
            #evaluate on val set
            out_path = config.MODEL.PRETRAINED.replace('pth','pkl')
            if os.path.exists(out_path):
                scores_dict = mmcv.load(out_path)
            else:
                scores_dict = validate(test_loader, text_labels, model, config, out_path)

            tmp_dict = {}
            for v_name in scores_dict["prd"].keys():
                p_scores = np.array(scores_dict["prd"][v_name]).copy()
                if p_scores.shape[0] == 1:
                    # 1,32,2
                    tmp_dict[v_name] = [p_scores[0, :, 1]]
                else:
                    # T,1,2
                    tmp_dict[v_name] = [p_scores[:, 0, 1]]

            auc_all, auc_ano = evaluate_result(tmp_dict, config.DATA.VAL_FILE)

            logger.info(f"AUC@all/ano of version {out_path.split('/')[-2]} on epoch {out_path.split('/')[-1].split('_')[-1][:-4]} : {auc_all:.4f}({auc_ano:.4f})")
            return
        else:
            for epoch in range(config.TRAIN.EPOCHS):
                out_path = os.path.join(config.MODEL.PRETRAINED, 'ckpt_epoch_' + str(epoch) + '.pkl')
                scores_dict = validate(test_loader, text_labels, model, config, out_path)
                tmp_dict = {}
                for v_name in scores_dict["cls"].keys():
                    tmp_dict[v_name] = [np.array(scores_dict["prd"][v_name])[:,0]]  # 1,32,2  +
                auc_all, auc_ano = evaluate_result(tmp_dict, config.DATA.VAL_FILE)
                is_best = auc_all > max_auc
                if is_best:
                    best_epoch = epoch
                max_auc = max(max_auc, auc_all)
                logger.info(f"Auc on epoch {epoch}: {auc_all:.4f}({auc_ano:.4f})")
                logger.info(f'Max AUC@all {best_epoch}/{epoch} : {max_auc:.4f}')

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_loader.sampler.set_epoch(epoch)
        train_one_epoch(epoch, model, optimizer, lr_scheduler, train_loader, text_labels, config)

        if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
            epoch_saving(config, epoch, model.module, max_auc, optimizer, lr_scheduler,_,_, logger, config.OUTPUT, is_best)

def train_one_epoch(epoch, model, optimizer, lr_scheduler, train_loader, text_labels, config, data_dict=None):
    model.train()

    optimizer.zero_grad()
    
    num_steps = len(train_loader)
    batch_time = AverageMeter()
    tot_loss_meter = AverageMeter()
    mil_loss_meter = AverageMeter()
    sm_loss_meter = AverageMeter()
    sp_loss_meter = AverageMeter()

    start = time.time()
    end = time.time()

    texts = text_labels.cuda(non_blocking=True)
    
    for idx, batch_data in enumerate(train_loader):
        images = batch_data["imgs"].cuda(non_blocking=True)[:,:1]
        label_id = batch_data["label"].cuda(non_blocking=True)[:,:1]
        label_id = label_id.reshape(-1)
        bz = images.shape[0]
        a_aug = images.shape[1]
        n_clips = images.shape[2]

        images = rearrange(images, 'b a k c t h w -> (b a k) t c h w')# bz*num_aug*num_clips,num_frames,h,w

        if texts.shape[0] == 1:
            texts = texts.view(1, -1)

        output = model(images, texts)
        # mil loss on max scores among bags, view instance of max scores as labeled data
        logits = rearrange(output['y'], '(b a k) c -> (b a) k c', b=bz, a=a_aug,)

        scores = F.softmax(logits, dim=-1)
        scores_ano = scores[:,:,1]
        scores_nor = scores[:,:,0]
        max_prob_ano, max_ind = torch.max(scores_ano, dim=-1)
        max_prob_nor, _ = torch.max(scores_nor, dim=-1)

        logits_video = torch.gather(logits, 1, max_ind[:, None, None].repeat((1, 1, 2))).squeeze(1)
        max_prob_video, _ = torch.max(torch.gather(scores, 1, max_ind[:, None, None].repeat((1, 1, 2))).squeeze(1),
                                      dim=-1)
        labels_binary = label_id > 0

        # MIL loss
        loss_mil = F.cross_entropy(logits_video, labels_binary.long(), reduction='none')
        loss_mil = loss_mil * max_prob_video
        loss_mil = loss_mil.mean()

        scores_all = scores
        smoothed_scores = (scores_all[:,1:,1] - scores_all[:,:-1,1])
        smoothed_loss = smoothed_scores.pow(2).sum(dim=-1).mean()

        sparsity_loss = scores_all[:,:,1].sum(dim=-1).mean()

        w_smooth = args.w_smooth
        w_sparse = args.w_sparse

        total_loss = loss_mil + smoothed_loss * w_smooth + sparsity_loss * w_sparse

        total_loss = total_loss / config.TRAIN.ACCUMULATION_STEPS

        if config.TRAIN.ACCUMULATION_STEPS == 1:
            optimizer.zero_grad()
        if config.TRAIN.OPT_LEVEL != 'O0':
            with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            total_loss.backward()
        if config.TRAIN.ACCUMULATION_STEPS > 1:
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                optimizer.step()
                optimizer.zero_grad()
                lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            optimizer.step()
            lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()
        
        tot_loss_meter.update(total_loss.item(), len(label_id))
        mil_loss_meter.update(loss_mil.item(), len(label_id))
        sm_loss_meter.update((smoothed_loss * w_smooth).item(), len(label_id))
        sp_loss_meter.update((sparsity_loss * w_sparse).item(), len(label_id))
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.9f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'tot {tot_loss_meter.val:.4f} ({tot_loss_meter.avg:.4f})\t'
                f'mil {mil_loss_meter.val:.4f} ({mil_loss_meter.avg:.4f})\t'
                f'sm {sm_loss_meter.val:.4f} ({sm_loss_meter.avg:.4f})\t'
                f'sp {sp_loss_meter.val:.4f} ({sp_loss_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')

    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(data_loader, text_labels, model, config, out_path):
    model.eval()
    vid_list = []

    anno_file = config.DATA.VAL_FILE

    with open(anno_file, 'r') as fin:
        for line in fin:
            line_split = line.strip().split()
            filename = line_split[0].split('/')[-1]
            vid_list.append(filename)

    with torch.no_grad():
        text_inputs = text_labels.cuda()
        logger.info(f"{config.TEST.NUM_CLIP * config.TEST.NUM_CROP} views inference")
        scores_dict = dict()
        scores_dict['prd'] = dict()
        for idx, batch_data in enumerate(data_loader):
            _image = batch_data["imgs"]
            label_id = batch_data["label"]
            label_id = label_id.reshape(-1)
            b, n, c, t, h, w = _image.size()
            _image = rearrange(_image, 'b n c t h w -> (b n) t c h w')
            output = model(_image, text_inputs)

            scores_prd = F.softmax(output['y'], dim=-1)
            scores_prd = rearrange(scores_prd, '(b n) c -> b n c', b=b)
            scores_np_prd = scores_prd.cpu().data.numpy()

            for ind in range(scores_np_prd.shape[0]):
                v_name = vid_list[batch_data["vid"][ind]]
                if v_name not in scores_dict['prd']:
                    scores_dict['prd'][v_name] = []
                scores_dict['prd'][v_name].append(scores_np_prd[ind])
            if idx % 100 == 0 and len(data_loader) >= 100:
                logger.info(
                    f'Test: [{idx}/{len(data_loader)}]\t'
                )
    tmp_dict = {}
    for v_name in scores_dict["prd"].keys():
        p_scores = np.array(scores_dict["prd"][v_name]).copy()
        if p_scores.shape[0] == 1:
            # 1,T,2
            tmp_dict[v_name] = [p_scores[0, :, 1]]
        else:
            # T,1,2
            tmp_dict[v_name] = [p_scores[:, 0, 1]]

    auc_all_p, auc_ano_p = evaluate_result(tmp_dict, config.DATA.VAL_FILE)

    logger.info(
        f'AUC: [{auc_all_p:.3f}/{auc_ano_p:.3f}]\t'
    )
    logger.info(f'writing results to {out_path}')
    mmcv.dump(scores_dict, out_path)
    return scores_dict


if __name__ == '__main__':
    # prepare config
    args, config = parse_option()

    # init_distributed
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier(device_ids=[args.local_rank])

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # create working_dir
    Path(config.OUTPUT).mkdir(parents=True, exist_ok=True)
    
    # logger
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.ARCH}")
    logger.info(f"working dir: {config.OUTPUT}")
    
    # save config 
    if dist.get_rank() == 0:
        logger.info(config)
        shutil.copy(args.config, config.OUTPUT)

    main(config)
