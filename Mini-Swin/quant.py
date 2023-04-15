import os
import time
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import quant_util

import warnings
warnings.filterwarnings(action="ignore", category=UserWarning)

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy
from timm.models import create_model
from my_meter import AverageMeter

from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, parse_option

from models.swin_transformer_distill import SwinTransformerDISTILL


def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    loss_batch = torch.sum(- targets_prob * student_likelihood, dim=-1)
    return loss_batch.mean()

def cal_relation_loss(student_attn_list, teacher_attn_list, Ar):
    layer_num = len(student_attn_list)
    relation_loss = 0.
    for student_att, teacher_att in zip(student_attn_list, teacher_attn_list):
        B, N, Cs = student_att[0].shape
        _, _, Ct = teacher_att[0].shape
        for i in range(3):
            for j in range(3):
                # (B, Ar, N, Cs // Ar) @ (B, Ar, Cs // Ar, N)
                # (B, Ar) + (N, N)
                matrix_i = student_att[i].view(B, N, Ar, Cs//Ar).transpose(1, 2) / (Cs/Ar)**0.5
                matrix_j = student_att[j].view(B, N, Ar, Cs//Ar).permute(0, 2, 3, 1)
                As_ij = (matrix_i @ matrix_j) 

                matrix_i = teacher_att[i].view(B, N, Ar, Ct//Ar).transpose(1, 2) / (Ct/Ar)**0.5
                matrix_j = teacher_att[j].view(B, N, Ar, Ct//Ar).permute(0, 2, 3, 1)
                At_ij = (matrix_i @ matrix_j)
                relation_loss += soft_cross_entropy(As_ij, At_ij)
    return relation_loss/(9. * layer_num)

def cal_hidden_loss(student_hidden_list, teacher_hidden_list):
    layer_num = len(student_hidden_list)
    hidden_loss = 0.
    for student_hidden, teacher_hidden in zip(student_hidden_list, teacher_hidden_list):
        hidden_loss +=  torch.nn.MSELoss()(student_hidden, teacher_hidden)
    return hidden_loss/layer_num

def cal_hidden_relation_loss(student_hidden_list, teacher_hidden_list):
    layer_num = len(student_hidden_list)
    B, N, Cs = student_hidden_list[0].shape
    _, _, Ct = teacher_hidden_list[0].shape
    hidden_loss = 0.
    for student_hidden, teacher_hidden in zip(student_hidden_list, teacher_hidden_list):
        student_hidden = torch.nn.functional.normalize(student_hidden, dim=-1)
        teacher_hidden = torch.nn.functional.normalize(teacher_hidden, dim=-1)
        student_relation = student_hidden @ student_hidden.transpose(-1, -2)
        teacher_relation = teacher_hidden @ teacher_hidden.transpose(-1, -2)
        hidden_loss += torch.mean((student_relation - teacher_relation)**2) * 49 #Window size x Window size
    return hidden_loss/layer_num

def load_teacher_model(type='large'):
    if type == 'large':
        embed_dim = 192
        depths = [ 2, 2, 18, 2 ]
        num_heads = [ 6, 12, 24, 48 ]
        window_size = 7
    elif type == 'base':
        embed_dim = 128
        depths = [ 2, 2, 18, 2 ]
        num_heads = [ 4, 8, 16, 32 ]
        window_size = 7
    else:
        raise ValueError('Unsupported type: %s'%type)
    model = SwinTransformerDISTILL(img_size=224,
                                patch_size=4,
                                in_chans=3,
                                num_classes=1000,
                                embed_dim=embed_dim,
                                depths=depths,
                                num_heads=num_heads,
                                window_size=window_size,
                                mlp_ratio=4.0,
                                qkv_bias=True,
                                qk_scale=None,
                                drop_rate=0.0,
                                drop_path_rate=0.1,
                                ape=False,
                                patch_norm=True,
                                use_checkpoint=False,
                                # distillation
                                is_student=False)
    return model

def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    if config.DISTILL.DO_DISTILL:
        logger.info(f"Loading teacher model:{config.MODEL.TYPE}/{config.DISTILL.TEACHER}")
        model_checkpoint_name = os.path.basename(config.DISTILL.TEACHER)
        if 'regnety_160' in model_checkpoint_name:
            model_teacher = create_model(
            'regnety_160',
            pretrained=False,
            num_classes=config.MODEL.NUM_CLASSES,
            global_pool='avg',
            )
            if config.DISTILL.TEACHER.startswith('https'):
                checkpoint = torch.hub.load_state_dict_from_url(
                    config.DISTILL.TEACHER, map_location='cpu', check_hash=True)
            else:
                checkpoint = torch.load(config.DISTILL.TEACHER, map_location='cpu')
            model_teacher.load_state_dict(checkpoint['model'])
            model_teacher.cuda()
            model_teacher.eval()
            del checkpoint
            torch.cuda.empty_cache()
        else:
            if 'base' in model_checkpoint_name:
                teacher_type = 'base'
            elif 'large' in model_checkpoint_name:
                teacher_type = 'large'
            else:
                teacher_type = None
            model_teacher = load_teacher_model(type=teacher_type)
            model_teacher.cuda()
            checkpoint = torch.load(config.DISTILL.TEACHER, map_location='cpu')
            msg = model_teacher.module.load_state_dict(checkpoint['model'], strict=False)
            logger.info(msg)
            del checkpoint
            torch.cuda.empty_cache()

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))

    optimizer = build_optimizer(config, model)

    model_without_ddp = model

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model_without_ddp, 'flops'):
        flops = model_without_ddp.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    
    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    criterion_soft = soft_cross_entropy
    criterion_attn = cal_relation_loss
    criterion_hidden = cal_hidden_relation_loss if config.DISTILL.HIDDEN_RELATION else cal_hidden_loss


    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion_truth = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion_truth = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion_truth = torch.nn.CrossEntropyLoss()
    
    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.DISTILL.RESUME_WEIGHT_ONLY = False
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger)
        acc1, acc5, loss = validate(config, data_loader_val, model, logger)
        logger.info(f"Accuracy of the network on the {len(dataset_val)} test images: {acc1:.1f}%")
        if config.EVAL_MODE:
            return

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return


@torch.no_grad()
def validate(config, data_loader, model, logger):

    quant_util.activation_bw = 16


    model.to('cpu')
    bit_width = 8
    for name, param in model.named_parameters():

        min = torch.min(param.data).item()
        max = torch.max(param.data).item()
        scaling_fac = (max-min)/((2**bit_width)-1)
        quantized_tensor = torch.round(param.data/scaling_fac)
        param.data = quantized_tensor*scaling_fac
    model.to("cuda:0")
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()

    end = time.time()
    idx_count = 0
    avg_1 = 0
    avg_5 =0
    for idx, (images, target) in enumerate(data_loader):
        idx_count += 1
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        output = model(images)

        # measure accuracy and record loss
        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
            avg_1 = acc1_meter.avg
            avg_5 = acc5_meter.avg

    # loss_meter.sync()
    # acc1_meter.sync()
    # acc5_meter.sync()

    logger.info(f' * Acc@1 {(avg_1):.3f} Acc@5 {(avg_5):.3f}')
    return (avg_1), (avg_5), loss_meter.avg


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return


if __name__ == '__main__':
    _, config = parse_option()

    quant_util.init()

    cudnn.benchmark = True


    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=0, name=f"{config.MODEL.NAME}")

    if True:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)