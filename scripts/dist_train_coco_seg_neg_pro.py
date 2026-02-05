import argparse
import datetime
import logging
import os
import random
import sys

sys.path.append(".")

import numpy as np
import torch
# import torch.distributed as dist
import torch.nn.functional as F
from datasets import coco as coco
from model.losses import CTCLoss_neg, get_masked_ptc_loss, get_seg_loss, DenseEnergyLoss, get_energy_loss
from model.model_seg_neg_por import network
# from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from model.PAR import PAR
from utils import evaluate, imutils, optimizer
from utils.camutils import cam_to_label, cam_to_roi_mask2, crop_from_roi_neg, multi_scale_cam2, label_to_aff_mask, refine_cams_with_bkg_v2
from utils.pyutils import AverageMeter, cal_eta, format_tabs, setup_logger

from step.cluster_utils import get_regional_cluster
import importlib

torch.hub.set_dir("./pretrained")
parser = argparse.ArgumentParser()

parser.add_argument("--backbone", default='vit_base_patch16_224', type=str, help="vit_base_patch16_224")
parser.add_argument("--pooling", default='gmp', type=str, help="pooling choice for patch tokens")
parser.add_argument("--pretrained", default=True, type=bool, help="use imagenet pretrained weights")

parser.add_argument("--img_folder", default='../datasets/coco/MSCOCO/coco2014', type=str, help="dataset folder")
parser.add_argument("--label_folder", default='../datasets/coco/MSCOCO/SegmentationClass', type=str, help="dataset folder")
parser.add_argument("--list_folder", default='../datasets/coco', type=str, help="train/val/test list file")
parser.add_argument("--num_classes", default=81, type=int, help="number of classes")
parser.add_argument("--crop_size", default=448, type=int, help="crop_size in training")
parser.add_argument("--local_crop_size", default=96, type=int, help="crop_size for local view")
parser.add_argument("--ignore_index", default=255, type=int, help="random index")

parser.add_argument("--work_dir", default="work_dir_coco_wseg", type=str, help="work_dir_coco_wseg")

parser.add_argument("--train_set", default="train", type=str, help="training split")
parser.add_argument("--val_set", default="val_part", type=str, help="validation split")
parser.add_argument("--spg", default=4, type=int, help="samples_per_gpu")
parser.add_argument("--scales", default=(0.5, 2), help="random rescale in training")

parser.add_argument("--optimizer", default='PolyWarmupAdamW', type=str, help="optimizer")
parser.add_argument("--lr", default=6e-5, type=float, help="learning rate")
parser.add_argument("--warmup_lr", default=1e-6, type=float, help="warmup_lr")
parser.add_argument("--wt_decay", default=1e-2, type=float, help="weights decay")
parser.add_argument("--betas", default=(0.9, 0.999), help="betas for Adam")
parser.add_argument("--power", default=0.9, type=float, help="poweer factor for poly scheduler")

parser.add_argument("--max_iters", default=80000, type=int, help="max training iters")
parser.add_argument("--log_iters", default=200, type=int, help=" logging iters")
parser.add_argument("--eval_iters", default=4000, type=int, help="validation iters")
parser.add_argument("--warmup_iters", default=1500, type=int, help="warmup_iters")

# parser.add_argument("--max_iters", default=1, type=int, help="max training iters")
# parser.add_argument("--log_iters", default=1, type=int, help=" logging iters")
# parser.add_argument("--eval_iters", default=1, type=int, help="validation iters")
# parser.add_argument("--warmup_iters", default=1, type=int, help="warmup_iters")

parser.add_argument("--high_thre", default=0.65, type=float, help="high_bkg_score")
parser.add_argument("--low_thre", default=0.25, type=float, help="low_bkg_score")
parser.add_argument("--bkg_thre", default=0.45, type=float, help="bkg_score")
parser.add_argument("--cam_scales", default=(1.0, 0.5, 1.5), help="multi_scales for cam")

parser.add_argument("--w_reg", default=0.05, type=float, help="w_reg")
parser.add_argument("--temp", default=0.5, type=float, help="temp")
parser.add_argument("--momentum", default=0.9, type=float, help="temp")

parser.add_argument("--seed", default=0, type=int, help="fix random seed")
# parser.add_argument("--save_ckpt", action="store_true", help="save_ckpt")
parser.add_argument("--save_ckpt", default=True, help="save_ckpt")

parser.add_argument("--local_rank", default=0, type=int, help="local_rank")
parser.add_argument("--num_workers", default=1, type=int, help="num_workers")
parser.add_argument('--backend', default='gloo')

parser.add_argument("--num_cluster", default=10, type=int)
parser.add_argument("--mask_thresh", default=0.1, type=float)
parser.add_argument("--cam_num_epoches", default=5, type=int)
parser.add_argument("--exp_name", default='exp_fpr', type=str)
parser.add_argument("--cam_network", default="model.backbone.resnet50_cam", type=str)
parser.add_argument("--RC_weight", default=12e-2, type=float)
parser.add_argument("--cam_weight_path", default='../step/res50_cam_orig.pth', type=str)
parser.add_argument("--PR_weight", default=15e-5, type=float)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def exp_similarity(a, b, temperature):
    """
    calculate the distance of a, b. return the exp{-L1(a,b)}
    """
    if len(b) == 0 or len(a) == 0:  # no clusters
        return torch.Tensor([0.5]).to(a.device)

    dis = ((a - b) ** 2 + 1e-4).mean(1)
    dis = torch.sqrt(dis)
    dis = dis / temperature + 0.1  # prevent too large gradient
    return torch.exp(-dis)

def hard_triplet_dig(anchor, positive, negative):
    """

    Args:
        anchor: [length, c]
        positive: [nums_x, c]
        negative: [nums_y, c]
    Returns: triplet loss
    """
    for i in range(anchor.shape[0]):
        edu_distance_pos = F.pairwise_distance(F.normalize(anchor[i], p=2, dim=-1),
                                                 F.normalize(positive, p=2, dim=-1))
        edu_distance_neg = F.pairwise_distance(F.normalize(anchor[i], p=2, dim=-1), torch.mean(F.normalize(negative, p=2, dim=-1), dim=0, keepdim=True))
        neg_val, _ = edu_distance_neg.sort()
        pos_val, _ = edu_distance_pos.sort()

        triplet_loss = max(0, 0.5 + pos_val[-1] - neg_val[0])
    return triplet_loss

def closest_dis(a, b):
    """
    Args:
        a: with shape of [1, C, HW]
        b: with shape of [num_clusters, C, 1]
    Return:
        dis: with shape of [HW]
    """
    if len(b) == 0 or len(a) == 0:  # no clusters
        return torch.Tensor([123456]).to(a.device)
    dis = ((a - b) ** 2).mean(1)
    return dis.min(0)[0]



def validate(model=None, data_loader=None, args=None):

    preds, gts, cams, cams_aux = [], [], [], []
    model.eval()
    avg_meter = AverageMeter()
    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ncols=100, ascii=" >="):

            name, inputs, labels, cls_label = data
            inputs = inputs.cuda()
            labels = labels.cuda()
            cls_label = cls_label.cuda()

            inputs  = F.interpolate(inputs, size=[args.crop_size, args.crop_size], mode='bilinear', align_corners=False)

            cls, segs, _, _ = model(inputs,)

            cls_pred = (cls>0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            _cams, _cams_aux = multi_scale_cam2(model, inputs, args.cam_scales)
            resized_cam = F.interpolate(_cams, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label = cam_to_label(resized_cam, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            resized_cam_aux = F.interpolate(_cams_aux, size=labels.shape[1:], mode='bilinear', align_corners=False)
            cam_label_aux = cam_to_label(resized_cam_aux, cls_label, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)

            cls_pred = (cls > 0).type(torch.int16)
            _f1 = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])
            avg_meter.add({"cls_score": _f1})

            resized_segs = F.interpolate(segs, size=labels.shape[1:], mode='bilinear', align_corners=False)

            preds += list(torch.argmax(resized_segs, dim=1).cpu().numpy().astype(np.int16))
            cams += list(cam_label.cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))
            cams_aux += list(cam_label_aux.cpu().numpy().astype(np.int16))

            # valid_label = torch.nonzero(cls_label[0])[:, 0]
            # out_cam = torch.squeeze(resized_cam)[valid_label]
            # np.save(os.path.join(cfg.work_dir.pred_dir, name[0]+'.npy'), {"keys":valid_label.cpu().numpy(), "cam":out_cam.cpu().numpy()})

    cls_score = avg_meter.pop('cls_score')
    seg_score = evaluate.scores(gts, preds, num_classes=args.num_classes)
    cam_score = evaluate.scores(gts, cams, num_classes=args.num_classes)
    cam_aux_score = evaluate.scores(gts, cams_aux, num_classes=args.num_classes)
    model.train()

    tab_results = format_tabs([cam_score, cam_aux_score, seg_score], name_list=["CAM", "aux_CAM", "Seg_Pred"], cat_list=coco.class_list)

    return cls_score, tab_results

def train(args=None):

    # torch.cuda.set_device(args.local_rank)
    # dist.init_process_group(backend=args.backend, )
    # logging.info("Total gpus: %d, samples per gpu: %d..."%(dist.get_world_size(), args.spg))

    time0 = datetime.datetime.now()
    time0 = time0.replace(microsecond=0)

    net = getattr(importlib.import_module(args.cam_network), 'CAM')()
    # pth_path = args.cam_weight_path
    # net.load_state_dict(torch.load(pth_path), strict=False)
    net = net.cuda()

    train_dataset = coco.CocoClsDataset(
        img_dir=args.img_folder,
        label_dir=args.label_folder,
        name_list_dir=args.list_folder,
        split=args.train_set,
        stage='train',
        aug=True,
        # resize_range=cfg.dataset.resize_range,
        rescale_range=args.scales,
        crop_size=args.crop_size,
        img_fliplr=True,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    val_dataset = coco.CocoSegDataset(
        img_dir=args.img_folder,
        label_dir=args.label_folder,
        name_list_dir=args.list_folder,
        split=args.val_set,
        stage='val',
        aug=False,
        ignore_index=args.ignore_index,
        num_classes=args.num_classes,
    )

    # train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.spg,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        # sampler=train_sampler,
        prefetch_factor=4)

    val_loader = DataLoader(val_dataset,
                            batch_size=1,
                            shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=False,
                            drop_last=False)

    device = torch.device('cuda')

    model = network(
        backbone=args.backbone,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        init_momentum=args.momentum,
        aux_layer=9
    )
    para = sum(p.numel() for p in model.parameters() if p.requires_grad)

    param_groups = model.get_param_groups()
    model.to(device)

    # cfg.optimizer.learning_rate *= 2
    optim = getattr(optimizer, args.optimizer)(
        params=[
            {
                "params": param_groups[0],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[1],
                "lr": args.lr,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[2],
                "lr": args.lr * 10,
                "weight_decay": args.wt_decay,
            },
            {
                "params": param_groups[3],
                "lr": args.lr * 10,
                "weight_decay": args.wt_decay,
            },
        ],
        lr=args.lr,
        weight_decay=args.wt_decay,
        betas=args.betas,
        warmup_iter=args.warmup_iters,
        max_iter=args.max_iters,
        warmup_ratio=args.warmup_lr,
        power=args.power)

    logging.info('\nOptimizer: \n%s' % optim)
    # model = DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)

    # train_sampler.set_epoch(np.random.randint(args.max_iters))
    train_loader_iter = iter(train_loader)
    avg_meter = AverageMeter()


    loss_layer = DenseEnergyLoss(weight=1e-7, sigma_rgb=15, sigma_xy=100, scale_factor=0.5)
    ncrops = 10
    CTC_loss = CTCLoss_neg(ncrops=ncrops, temp=1.0).cuda()

    par = PAR(num_iter=10, dilations=[1,2,4,8,12,24]).cuda()

    # 1. online cluster
    print('generating positive clusters and negative clusters ...')
    vis_path = os.path.join(args.exp_name, '{}_imgs_for_cluster'.format(0))
    pos_feats, neg_feats, cur_neg_feats_unshared = get_regional_cluster(vis_path, net, num_cluster=args.num_cluster, region_thresh=args.mask_thresh)
    print('Done...')

    for n_iter in range(args.max_iters):

        try:
            img_name, inputs, cls_label, img_box, crops = next(train_loader_iter)

        except:
            # train_sampler.set_epoch(np.random.randint(args.max_iters))
            train_loader_iter = iter(train_loader)
            img_name, inputs, cls_label, img_box, crops = next(train_loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        inputs_denorm = imutils.denormalize_img2(inputs.clone())
        cls_label = cls_label.to(device, non_blocking=True)

        cam1, x4, logits = model.forward_feat(inputs)  # 2*21*28*28 2*756*28*28 2*20
        norm_cams = F.relu(cam1) / (F.adaptive_max_pool2d(cam1, 1) + 1e-4)  # 2*21*28*28
        norm_cams = (norm_cams > args.mask_thresh).float().detach().flatten(2)  # 2*21*784
        out_features = x4.flatten(2)  # 2*768*784

        pos_loss = []
        neg_loss = []
        triplet_loss = []

        for logit, feat, norm_cam, cam, gt in zip(logits, out_features, norm_cams, cam1, cls_label):
            for idx, is_exist in enumerate(gt):

                if cam[idx].max() <= 0 or len(pos_feats[idx]) == 0 or len(neg_feats[idx]) == 0:
                    continue

                cls_norm_cam = norm_cam[idx]  # 1024
                region_feat = (feat * cls_norm_cam[None]).sum(1) / (cls_norm_cam.sum() + 1e-5)  # 2048  fc

                if is_exist:
                    # distance to pos clusters
                    pos_feat = pos_feats[idx].mean(0, keepdim=True)  # 1*2048
                    pos_prob = exp_similarity(region_feat[None], pos_feat, temperature=13)
                    loss_pos = -torch.log(pos_prob)
                    pos_loss.append(loss_pos.squeeze())

                    # distance to neg clusters
                    neg_feat = neg_feats[idx]  # 6*2048
                    neg_prob = exp_similarity(region_feat[None], neg_feat, temperature=13).max()
                    loss_neg = -torch.log(1 - neg_prob)
                    neg_loss.append(loss_neg)

        loss_pos = torch.stack(pos_loss).mean() if len(pos_loss) > 0 else 0
        loss_neg = torch.stack(neg_loss).mean() if len(neg_loss) > 0 else 0

        loss_RC = (loss_pos + loss_neg) * args.RC_weight

        bs, num_cls = cls_label.shape
        cam_aux2 = cam1.flatten(2)
        probs = []
        for bs_id in range(bs):
            for cls_id in range(num_cls):
                if cls_label[bs_id][cls_id]:
                    dis_pos = closest_dis(out_features[bs_id][None], pos_feats[cls_id][..., None])
                    dis_neg = closest_dis(out_features[bs_id][None], cur_neg_feats_unshared[cls_id][..., None])

                    fp_location = (norm_cams[bs_id][cls_id] > args.mask_thresh) * (dis_pos > dis_neg)
                    fp_pixels = cam_aux2[bs_id][cls_id][fp_location]
                    if len(fp_pixels) > 0:
                        probs.append(fp_pixels)

        # loss_PC
        if len(probs) > 0:
            probs = torch.cat(probs)
            loss_revise = probs.mean()
        else:
            loss_revise = 0

        loss_PR = loss_revise * args.PR_weight

        cams, cams_aux = multi_scale_cam2(model, inputs=inputs, scales=args.cam_scales)

        cls, segs, fmap, cls_aux = model(inputs)

        # cls loss & aux cls loss
        cls_loss = F.multilabel_soft_margin_loss(cls, cls_label)
        cls_loss_aux = F.multilabel_soft_margin_loss(cls_aux, cls_label)

        
        valid_cam, _ = cam_to_label(
            cams.detach(), 
            cls_label=cls_label, 
            img_box=img_box, ignore_mid=True, 
            bkg_thre=args.bkg_thre, 
            high_thre=args.high_thre, 
            low_thre=args.low_thre, 
            ignore_index=args.ignore_index)
        valid_cam_aux, _ = cam_to_label(
            cams_aux.detach(), 
            cls_label=cls_label, 
            img_box=img_box, 
            ignore_mid=True, 
            bkg_thre=args.bkg_thre, 
            high_thre=args.high_thre, 
            low_thre=args.low_thre, 
            ignore_index=args.ignore_index)

        if n_iter <= 12000:
            refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam_aux, cls_labels=cls_label, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )
        else:
            refined_pseudo_label = refine_cams_with_bkg_v2(par, inputs_denorm, cams=valid_cam, cls_labels=cls_label, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index, img_box=img_box, )
        
        segs = F.interpolate(segs, size=refined_pseudo_label.shape[1:], mode='bilinear', align_corners=False)
        seg_loss = get_seg_loss(segs, refined_pseudo_label.type(torch.long), ignore_index=args.ignore_index)

        reg_loss = get_energy_loss(img=inputs, logit=segs, label=refined_pseudo_label, img_box=img_box, loss_layer=loss_layer)

        resized_cams_aux = F.interpolate(cams_aux, size=fmap.shape[2:], mode="bilinear", align_corners=False)
        _, pseudo_label_aux = cam_to_label(resized_cams_aux.detach(), cls_label=cls_label, img_box=img_box, ignore_mid=True, bkg_thre=args.bkg_thre, high_thre=args.high_thre, low_thre=args.low_thre, ignore_index=args.ignore_index)
        aff_mask = label_to_aff_mask(pseudo_label_aux)
        ptc_loss = get_masked_ptc_loss(fmap, aff_mask)
        # ptc_loss = get_ptc_loss(fmap, low_fmap)

        if n_iter <= 8000:
            loss = loss_RC + loss_PR + 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.0 * ptc_loss + 0.0 * seg_loss + 0.0 * reg_loss
        elif n_iter <= 12000:
            loss = loss_RC + loss_PR + 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.0 * ptc_loss + 0.1 * seg_loss + args.w_reg * reg_loss
        else:
            loss = loss_RC + loss_PR + 1.0 * cls_loss + 1.0 * cls_loss_aux + 0.2 * ptc_loss + 0.1 * seg_loss + args.w_reg * reg_loss

        cls_pred = (cls > 0).type(torch.int16)
        cls_score = evaluate.multilabel_score(cls_label.cpu().numpy()[0], cls_pred.cpu().numpy()[0])

        avg_meter.add({
            'RC_loss': loss_RC,
            'PR_loss': loss_PR,
            'cls_loss': cls_loss.item(),
            'ptc_loss': ptc_loss.item(),
            'cls_loss_aux': cls_loss_aux.item(),
            'seg_loss': seg_loss.item(),
            'cls_score': cls_score.item(),
        })

        optim.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if (n_iter + 1) % args.log_iters == 0:

            delta, eta = cal_eta(time0, n_iter + 1, args.max_iters)
            cur_lr = optim.param_groups[0]['lr']

            if args.local_rank == 0:
                logging.info(
                    "Iter: %d; Elasped: %s; ETA: %s; LR: %.3e; RC_loss: %.4f, PR_loss: %.4f, cls_loss: %.4f, cls_loss_aux: %.4f, ptc_loss: %.4f, seg_loss: %.4f..." % (
                    n_iter + 1, delta, eta, cur_lr, avg_meter.pop('RC_loss'), avg_meter.pop('PR_loss'),
                    avg_meter.pop('cls_loss'), avg_meter.pop('cls_loss_aux'), avg_meter.pop('ptc_loss'),
                    avg_meter.pop('seg_loss')))

        if (n_iter + 1) % args.eval_iters == 0:
            ckpt_name = os.path.join(args.ckpt_dir, "model_iter_%d.pth" % (n_iter + 1))
            if args.local_rank == 0:
                logging.info('Validating...')
                if args.save_ckpt:
                    torch.save(model.state_dict(), ckpt_name)
            val_cls_score, tab_results = validate(model=model, data_loader=val_loader, args=args)
            if args.local_rank == 0:
                logging.info("val cls score: %.6f" % (val_cls_score))
                logging.info("\n"+tab_results)
    # val_cls_score, tab_results = validate(model=model, data_loader=val_loader, args=args)
    # if args.local_rank == 0:
    #     logging.info("val cls score: %.6f" % (val_cls_score))
    #     logging.info("\n"+tab_results)
    return True


if __name__ == "__main__":

    args = parser.parse_args()

    timestamp = "{0:%Y-%m-%d-%H-%M-%S-%f}".format(datetime.datetime.now())
    args.work_dir = os.path.join(args.work_dir, timestamp)
    args.ckpt_dir = os.path.join(args.work_dir, "checkpoints")
    args.pred_dir = os.path.join(args.work_dir, "predictions")

    if args.local_rank == 0:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        os.makedirs(args.pred_dir, exist_ok=True)

        setup_logger(filename=os.path.join(args.work_dir, 'train_vit_coco.log'))
        logging.info('Pytorch version: %s' % torch.__version__)
        logging.info("GPU type: %s"%(torch.cuda.get_device_name(0)))
        logging.info('\nargs: %s' % args)

    ## fix random seed
    setup_seed(args.seed)
    train(args=args)
