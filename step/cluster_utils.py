import random

import torch
import torch.multiprocessing
from torch.multiprocessing import Manager
from torch.utils.data import DataLoader
# import voc12.dataloader
import datasets.dataloader_coco
import torch.nn.functional as F
import cv2
from sklearn.cluster import KMeans
import os



cls_names = CAT_LIST = ['0_person', '1_bicycle', '2_car', '3_motorcycle', '4_airplane', '5_bus', '6_train', '7_truck', '8_boat',
                        '9_traffic light', '10_fire hydrant', '11_stop sign', '12_parking meter', '13_bench',
                        '14_bird', '15_cat', '16_dog', '17_horse', '18_sheep', '19_cow', '20_elephant', '21_bear', '22_zebra', '23_giraffe',
                        '24_backpack', '25_umbrella', '26_handbag', '27_tie', '28_suitcase',
                        '29_frisbee', '30_skis', '31_snowboard', '32_sports ball','33_kite', '34_baseball bat', '35_baseball glove', '36_skateboard', '37_surfboard', '38_tennis racket',
                        '39_bottle', '40_wine glass', '41_cup', '42_fork', '43_knife', '44_spoon', '45_bowl',
                        '46_banana', '47_apple', '48_sandwich', '49_orange', '50_broccoli', '51_carrot', '52_hot dog', '53_pizza', '54_donut', '55_cake',
                        '56_chair', '57_couch', '58_potted plant', '59_bed', '60_dining table', '61_toilet',
                        '62_tv', '63_laptop', '64_mouse', '65_remote', '66_keyboard', '67_cell phone',
                        '68_microwave', '69_oven', '70_toaster', '71_sink', '72_refrigerator',
                        '73_book', '74_clock', '75_vase', '76_scissors', '77_teddy bear', '78_hair drier', '79_toothbrush']

super_class = ['PERSON', 'VEHICLE', 'FACILITIES', 'ANIMAL', 'DAILY', 'SPORT', 'TABLEWARE', 'FRUIT', 'INDOOR', 'ELETRONICS', 'APPLIANCES', 'PRODUCTS']
num_sub_class = {super_class[0]: 1, super_class[1]: 8, super_class[2]: 5, super_class[3]: 10, super_class[4]: 5, super_class[5]: 10,
                 super_class[6]: 7, super_class[7]: 10, super_class[8]: 6, super_class[9]: 6, super_class[10]: 5, super_class[11]: 7}
super_class_map = {0: super_class[0], 1: super_class[1], 2: super_class[1], 3: super_class[1], 4: super_class[1],
                   5: super_class[1], 6: super_class[1], 7: super_class[1], 8: super_class[1], 9: super_class[2],
                   10: super_class[2], 11: super_class[2], 12: super_class[2], 13: super_class[2], 14: super_class[3],
                   15: super_class[3], 16: super_class[3], 17: super_class[3], 18: super_class[3], 19: super_class[3],
                   20: super_class[3], 21: super_class[3], 22: super_class[3], 23: super_class[3], 24: super_class[4],
                   25: super_class[4], 26: super_class[4], 27: super_class[4], 28: super_class[4], 29: super_class[5],
                   30: super_class[5], 31: super_class[5], 32: super_class[5], 33: super_class[5], 34: super_class[5],
                   35: super_class[5], 36: super_class[5], 37: super_class[5], 38: super_class[5], 39: super_class[6],
                   40: super_class[6], 41: super_class[6], 42: super_class[6], 43: super_class[6], 44: super_class[6],
                   45: super_class[6], 46: super_class[7], 47: super_class[7], 48: super_class[7], 49: super_class[7],
                   50: super_class[7], 51: super_class[7], 52: super_class[7], 53: super_class[7], 54: super_class[7],
                   55: super_class[7], 56: super_class[8], 57: super_class[8], 58: super_class[8], 59: super_class[8],
                   60: super_class[8], 61: super_class[8], 62: super_class[9], 63: super_class[9], 64: super_class[9],
                   65: super_class[9], 66: super_class[9], 67: super_class[9], 68: super_class[10], 69: super_class[10],
                   70: super_class[10], 71: super_class[10], 72: super_class[10], 73: super_class[11], 74: super_class[11],
                   75: super_class[11], 76: super_class[11], 77: super_class[11], 78: super_class[11], 79: super_class[11],
                   }


def without_shared_feats(cls_idx, tgt):
    num_cls = len(tgt)
    # TODO
    return super_class_map[cls_idx] not in [super_class_map[i] for i in range(num_cls) if tgt[i]] and num_sub_class[super_class_map[cls_idx]]>1


def list2tensor(feature_list):
    if len(feature_list) > 0:
        return torch.stack(feature_list)
    else:
        return torch.Tensor([])

# TODO SS/MS
def _get_cluster(model, num_cls, region_thresh, vis_path, num_cluster, single_stage=False, mask_img=False, sampling_ratio=1.):

    trainmsf_dataset = datasets.dataloader_coco.VOC12ClassificationDatasetMSF("../datasets/coco/train_1.txt",
                                                                              voc12_root="../datasets/coco/MSCOCO/coco2014/train2014",
                                                                              scales=(1.0, 0.5, 1.5, 2.0))
    trainmsf_dataloader = DataLoader(trainmsf_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True,
                                     drop_last=True)

    pos_clusters = [[] for _ in range(num_cls)]
    neg_info = [[] for _ in range(num_cls)]
    neg_clusters = [[] for _ in range(num_cls)]
    neg_clusters_unshared = [[] for _ in range(num_cls)]

    with torch.no_grad():
        model = model.cuda()
        model.eval()
        # 1. collect feature embeddings
        for img_idx, pack in enumerate(trainmsf_dataloader):
            #if random.random() > sampling_ratio:
            #    continue

            if img_idx % 1000 == 0:
                print('processing regions features in img of {}/{}'.format(img_idx, len(trainmsf_dataloader)))
            ms_imgs = pack['img']
            # 1*2*3*281*500 1*2*3*140*250 1*2*3*422*750 1*2*3*562*1000
            tgt = pack['label'][0].cuda()
            name = pack['name'][0]

            # 1-1. ss/ms inferring
            cams_list, feats_list = [], []
            for idx, img in enumerate(ms_imgs):
                _, feat, cam, _ = model.forward_feat(img[0].cuda(non_blocking=True)) #2*2048*18*32 2*21*18*32
                cam = cam[:, :num_cls] #2*20*18*32
                if idx == 0:
                    size = cam.shape[-2:]
                else:
                    feat = F.interpolate(feat, size, mode='bilinear', align_corners=True)
                    cam = F.interpolate(cam, size, mode='bilinear', align_corners=True)
                feat = (feat[0] + feat[1].flip(-1)) / 2 #2048*18*32
                cam = (cam[0] + cam[1].flip(-1)) / 2 #20*18*32
                cams_list.append(cam) #20*18*32(4个)
                feats_list.append(feat)  #2048*18*32(4个)
                if single_stage:
                    break
            ms_cam = torch.stack(cams_list).mean(0) # [num_cls, H, W]20*18*32
            ms_feat = torch.stack(feats_list).mean(0)  # [C, H, W]2048*18*32
            _, h, w = ms_cam.shape #18 32

            # 1-2. normalize
            pred_prob = ms_cam.flatten(1).mean(1)  #20
            norm_ms_cam = F.relu(ms_cam) / (F.adaptive_max_pool2d(F.relu(ms_cam), (1, 1)) + 1e-5)  #20*18*32

            # 1-3. collect regional features
            orig_img = cv2.imread('../datasets/coco/MSCOCO/coco2014/train2014/{}.jpg'.format(name))  # 281*500*3
            for cls_idx, is_exist in enumerate(tgt[:num_cls]):
                if is_exist:
                    region_feat = (ms_feat[:, norm_ms_cam[cls_idx] > region_thresh])  #2048*77只保留了那些归一化CAM值大于阈值的特征  fc
                    if region_feat.shape[-1] > 0:
                        pos_clusters[cls_idx].append(region_feat.mean(1))
                        # # vis
                        # if cls_idx == 18:
                        #     if_activate = (norm_ms_cam[cls_idx] > region_thresh).reshape(h, w)[None, None].float()
                        #     if_activate = F.interpolate(if_activate, orig_img.shape[:2], mode='nearest')
                        #     if_activate = if_activate.squeeze().cpu().numpy()
                        #     if_activate = orig_img * 0.5 + if_activate[..., None] * 255 * 0.5
                        #     cv2.imwrite('{}/{}_pos_{}.png'.format(vis_path, cls_idx, name), if_activate)
                        #     # vis the pixel-level rectification
                        #     def closest_dis(a, b):
                        #         return ((a[None]-b[...,None,None])**2).mean(1).min(0)[0]
                        #     cluster = torch.load('/home/liyi/proj/w-ood/cluster10_disloss0.10_temper13_revise5e-5/0_imgs_for_cluster/clusters.pth')
                        #     pos_cluster = cluster['pos'][18]
                        #     neg_cluster = cluster['neg_unshared'][18]
                        #     if_closed = closest_dis(ms_feat, pos_cluster) > closest_dis(ms_feat, neg_cluster)
                        #     if_closed = (if_closed * (norm_ms_cam[18] > region_thresh)).float()
                        #     if_closed = F.interpolate(if_closed[None,None], orig_img.shape[:2], mode='nearest')
                        #     if_closed = if_closed.squeeze().cpu().numpy()
                        #     red_cloed = orig_img * 0.5
                        #     red_cloed[..., 2] = red_cloed[..., 2] + if_closed * 250 * 0.5
                        #     cv2.imwrite('{}/{}_pos_{}_rectification.png'.format(vis_path, cls_idx, name), red_cloed)

                if not is_exist:
                    cam_mask = norm_ms_cam[cls_idx] > region_thresh  #18*32
                    if cam_mask.sum() > 0:
                        info = [pred_prob[cls_idx], img_idx, cam_mask, tgt[:num_cls]]
                        neg_info[cls_idx].append(info)
                        # if True and cls_idx == 18:
                        #     # # vis
                        #     vis_high_cam = F.interpolate(F.relu(norm_ms_cam[cls_idx][None,None]), orig_img.shape[:2], mode='bilinear', align_corners=False)
                        #     vis_high_cam = vis_high_cam.squeeze().cpu()
                        #     show_cam_on_image(orig_img, vis_high_cam, '{}/{}_neg_{}.png'.format(vis_path, cls_idx, name))

        # 2. get clusters from collected features
        for cls_idx in range(num_cls):
            # 2-1. positive cluster
            if len(pos_clusters[cls_idx]) > 0:
                pos_feats_np = torch.stack(pos_clusters[cls_idx]).cpu().numpy() #3*2048
                num_k = min(num_cluster, len(pos_feats_np))
                centers = KMeans(n_clusters=num_k, random_state=0, max_iter=10).fit(pos_feats_np).cluster_centers_ #3*2048
                pos_clusters[cls_idx] = torch.from_numpy(centers).cuda()
            else:
                pos_clusters[cls_idx] = torch.Tensor([]).cuda()

            # 2-2. negative cluster
            probs = torch.stack([item[0] for item in neg_info[cls_idx]])  #6
            num_k = min(num_cluster, len(neg_info[cls_idx]))
            top_prob_idx = torch.topk(probs, num_k)[1]
            for item_idx in top_prob_idx:
                prob, img_idx, cam_mask, tgt = neg_info[cls_idx][item_idx]
                pack = trainmsf_dataset.__getitem__(img_idx)
                ms_imgs, name = pack['img'], pack['name'] #2*3*375*500 2*3*188*250  2*3*562*750 2*3*750*1000
                feats_list = []
                for idx, img in enumerate(ms_imgs):
                    img_mask = F.interpolate(cam_mask[None, None].float(), img.shape[-2:], mode='nearest')  #1*1*375*500
                    img_mask_flip = torch.cat([img_mask, img_mask.flip(-1)])   #2*1*375*500
                    masked_img = torch.from_numpy(img.copy()).cuda() * img_mask_flip  #2*3*375*500
                    if idx == 0:
                        alpha = 0.6
                        orig_img = cv2.imread('../datasets/coco/MSCOCO/coco2014/train2014/{}.jpg'.format(pack['name']))
                        cv2.imwrite('{}/{}_{}_orig.png'.format(vis_path, cls_names[cls_idx], name), orig_img)
                        fp_masked_img = img_mask[0].permute(1, 2, 0).cpu().numpy() * 255 * alpha + orig_img* (1-alpha)
                        cv2.imwrite('{}/{}_{}_prob{:.2f}.png'.format(vis_path, cls_names[cls_idx], name, prob),
                                    fp_masked_img)

                    if mask_img:
                        feat = model.forward_feat(masked_img)[1]
                    else:
                        unmask_img = torch.from_numpy(img.copy()).cuda() #2*3*375*500
                        feat = model.forward_feat(unmask_img)[1]  #2*2048*24*32

                    feat = F.interpolate(feat, cam_mask.shape, mode='bilinear', align_corners=True) #2*2048*24*32
                    feat = (feat[0] + feat[1].flip(-1)) / 2 #2048*24*32
                    feats_list.append(feat)  #2048*24*32(4)
                    if single_stage:
                        break
                ms_feat = torch.stack(feats_list).mean(0)  # [C, H, W]#2048*24*32
                ms_feat = ms_feat[:, cam_mask]  #2048*19
                neg_clusters[cls_idx].append(ms_feat.mean(1)) #2048
                if without_shared_feats(cls_idx, tgt):
                    neg_clusters_unshared[cls_idx].append(ms_feat.mean(1))

            neg_clusters[cls_idx] = list2tensor(neg_clusters[cls_idx]).cuda()
            neg_clusters_unshared[cls_idx] = list2tensor(neg_clusters_unshared[cls_idx])

    return pos_clusters, neg_clusters, neg_clusters_unshared


def get_regional_cluster(vis_path, model, num_cls=80, num_cluster=10, region_thresh=0.1, sampling_ratio=1.):
    """
    Args:
        model: the training model
        num_cls: the number of classes
        num_pos_k: the number of positive cluster for each class
        num_neg_k: the number of negative cluster for each class
        region_thresh: the threshold for getting reliable regions
    Return:
        clustered_fp_feats: a tensor with shape [num_cls, num_k, C]
    """

    if not os.path.exists(vis_path):
        os.makedirs(vis_path)

    pos_clusters, neg_clusters, neg_clusters_unshared = _get_cluster(model, num_cls, region_thresh, vis_path,
                                                                     num_cluster, sampling_ratio=sampling_ratio)

    torch.save({'pos': pos_clusters, 'neg': neg_clusters, 'neg_unshared': neg_clusters_unshared},
               '{}/clusters.pth'.format(vis_path))

    return pos_clusters, neg_clusters, neg_clusters_unshared
