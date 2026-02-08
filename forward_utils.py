import numpy as np
import cv2
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from kornia.filters import gaussian_blur2d
import ipdb
from dataset.constants import CLASS_NAMES, REAL_NAMES, PROMPTS
from model.tokenizer import tokenize
from sklearn.metrics import roc_auc_score, average_precision_score
import pandas as pd
from dataset.constants import DATA_PATH
from utils import cos_sim

# ================================================================================================
# The following code is used to get criterion for training


class FocalLoss(nn.Module):
    """
    copy from: https://github.com/Hsuxu/Loss_ToolBox-PyTorch/blob/master/FocalLoss/FocalLoss.py
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
        Focal_Loss= -1*alpha*(1-pt)*log(pt)
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss for well-classified examples (p>0.5) putting more
                    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index, should be specific when alpha is float
    :param size_average: (bool, optional) By default, the losses are averaged over each loss element in the batch.
    """

    def __init__(
        self,
        apply_nonlin=None,
        alpha=None,
        gamma=2,
        balance_index=0,
        smooth=1e-5,
        size_average=True,
    ):
        super(FocalLoss, self).__init__()
        self.apply_nonlin = apply_nonlin
        self.alpha = alpha
        self.gamma = gamma
        self.balance_index = balance_index
        self.smooth = smooth
        self.size_average = size_average

        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError("smooth value should be in [0,1]")

    def forward(self, logit, target):
        if self.apply_nonlin is not None:
            logit = self.apply_nonlin(logit)
        num_class = logit.shape[1]

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = torch.squeeze(target, 1)
        target = target.view(-1, 1)
        alpha = self.alpha

        if alpha is None:
            alpha = torch.ones(num_class, 1)
        elif isinstance(alpha, (list, np.ndarray)):
            assert len(alpha) == num_class
            alpha = torch.FloatTensor(alpha).view(num_class, 1)
            alpha = alpha / alpha.sum()
        elif isinstance(alpha, float):
            alpha = torch.ones(num_class, 1)
            alpha = alpha * (1 - self.alpha)
            alpha[self.balance_index] = self.alpha

        else:
            raise TypeError("Not support alpha type")

        if alpha.device != logit.device:
            alpha = alpha.to(logit.device)

        idx = target.cpu().long()

        one_hot_key = torch.FloatTensor(target.size(0), num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth / (num_class - 1), 1.0 - self.smooth
            )
        pt = (one_hot_key * logit).sum(1) + self.smooth
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        alpha = torch.squeeze(alpha)
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        return loss


class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()

    def forward(self, input, targets):
        N = targets.size()[0]
        smooth = 1
        input_flat = input.view(N, -1)
        targets_flat = targets.view(N, -1)
        intersection = input_flat * targets_flat
        N_dice_eff = (2 * intersection.sum(1) + smooth) / (
            input_flat.sum(1) + targets_flat.sum(1) + smooth
        )
        loss = 1 - N_dice_eff.sum() / N
        return loss


# ================================================================================================
# The following code is used to get adapted text embeddings
prompt = PROMPTS
prompt_normal = prompt["prompt_normal"]
prompt_abnormal = prompt["prompt_abnormal"]
prompt_state = [prompt_normal, prompt_abnormal]
prompt_templates = prompt["prompt_templates"]


def get_adapted_single_class_text_embedding(model, dataset_name, class_name, device):
    if class_name == "object":
        real_name = class_name
    else:
        assert class_name in CLASS_NAMES[dataset_name], (
            f"class_name {class_name} not found; available class_names: {CLASS_NAMES[dataset_name]}"
        )
        real_name = REAL_NAMES[dataset_name][class_name]
    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(real_name) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence).to(device)
        class_embeddings = model.encode_text(prompted_sentence)
        class_embeddings = class_embeddings / class_embeddings.norm(
            dim=-1, keepdim=True
        )
        class_embedding = class_embeddings.mean(dim=0)
        class_embedding = class_embedding / class_embedding.norm()
        text_features.append(class_embedding)
    text_features = torch.stack(text_features, dim=1).to(device)
    return text_features

# =========================
# 1) 域词表（你可以按任务增删）
# =========================
DOMAIN_VOCAB = [
    "low light", "underexposed", "overexposed",
    "noisy", "grainy", "sensor noise",
    "blurred", "motion blur", "out of focus",
    "strong reflection", "specular highlight",
    "uneven illumination", "shadow",
    "low contrast", "high contrast",
    "scratched texture", "rough texture", "smooth surface",
]
DOMAIN_VOCAB_PROMPTS = [f"a photo under {w} conditions." for w in DOMAIN_VOCAB]

_DOMAIN_TEXT_FEATS_CACHE = {}  # key: str(device) -> [V, D] normalized


@torch.no_grad()
def prepare_domain_vocab_text_features(model, device):
    """
    预计算域词表 text features（缓存）
    返回: [V, D]，已归一化
    """
    key = str(device)
    if key in _DOMAIN_TEXT_FEATS_CACHE:
        return _DOMAIN_TEXT_FEATS_CACHE[key]

    # 这里 tokenize 需要你工程里已有（你原函数里就用了 tokenize）
    tokens = torch.cat([tokenize(t) for t in DOMAIN_VOCAB_PROMPTS], dim=0).to(device)
    feats = model.encode_text(tokens).float()
    feats = F.normalize(feats, dim=-1)
    _DOMAIN_TEXT_FEATS_CACHE[key] = feats
    return feats


@torch.no_grad()
def retrieve_domain_phrase(diff_feat: torch.Tensor, domain_text_feats: torch.Tensor, topk: int = 3):
    """
    diff_feat: [B,D] or [1,D] or [D]
    domain_text_feats: [V,D]
    返回: "noisy, low contrast, uneven illumination"
    """
    if diff_feat.dim() == 1:
        diff_feat = diff_feat.unsqueeze(0)
    diff_vec = diff_feat.mean(dim=0, keepdim=True).float()
    diff_vec = F.normalize(diff_vec, dim=-1)
    sims = diff_vec @ domain_text_feats.t()           # [1,V]
    idx = sims.topk(k=topk, dim=-1).indices[0].tolist()
    words = [DOMAIN_VOCAB[i] for i in idx]
    return ", ".join(words)


def _extract_image_from_batch(batch, device):
    """
    兼容你 train/test 两种 dataloader 返回：
    - dict: batch["image"]
    - tuple/list: batch[0]
    """
    if isinstance(batch, dict):
        return batch["image"].to(device)
    return batch[0].to(device)


@torch.no_grad()
def get_support_global_proto(model, support_loader, device):
    """
    用 support set 得到全局特征原型：proto = mean(normalized(det_feature))
    依赖 model(image)->(patch_features, det_feature)
    返回: [1,D]
    """
    model.eval()
    feats = []
    for batch in support_loader:
        image = _extract_image_from_batch(batch, device)
        _, det_feature = model(image)            # [B,D]
        det_feature = F.normalize(det_feature, dim=-1)
        feats.append(det_feature)
    proto = torch.cat(feats, dim=0).mean(dim=0, keepdim=True)  # [1,D]
    proto = F.normalize(proto, dim=-1)
    return proto


# ==========================================================
# 2) 改造你的 text embedding：加入 diff_feat -> 自适应域选择
# ==========================================================
def get_adapted_single_class_text_embedding(
    model,
    dataset_name,
    class_name,
    device,
    diff_feat: torch.Tensor = None,   # NEW: query-support residual
    topk: int = 3,                    # NEW: 选择 top-k 域词
):
    # 你原逻辑：确定 real_name
    if class_name == "object":
        real_name = class_name
    else:
        assert class_name in CLASS_NAMES[dataset_name], (
            f"class_name {class_name} not found; available class_names: {CLASS_NAMES[dataset_name]}"
        )
        real_name = REAL_NAMES[dataset_name][class_name]

    # NEW: 自适应域选择（domain phrase）
    domain_phrase = None
    if diff_feat is not None:
        domain_text_feats = prepare_domain_vocab_text_features(model, device)
        domain_phrase = retrieve_domain_phrase(diff_feat, domain_text_feats, topk=topk)

    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(real_name) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                sent = template.format(s)
                # NEW: 注入域词（保持原模板不变，最稳）
                if domain_phrase is not None:
                    # 例如："... ." -> "... under xxx conditions."
                    if sent.endswith("."):
                        sent = sent[:-1] + f" under {domain_phrase} conditions."
                    else:
                        sent = sent + f" under {domain_phrase} conditions."
                prompted_sentence.append(sent)


        prompted_sentence = tokenize(prompted_sentence).to(device)
        class_embeddings = model.encode_text(prompted_sentence)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)

        class_embedding = class_embeddings.mean(dim=0)
        class_embedding = class_embedding / class_embedding.norm()
        text_features.append(class_embedding)

    # 维持你原输出： [D, 2]（假设 prompt_state 长度为2：normal/anomaly）
    text_features = torch.stack(text_features, dim=1).to(device)
    return text_features
@torch.no_grad()
def get_dynamic_class_text_embeddings(
    model,
    dataset_name,
    class_name,
    device,
    support_proto: torch.Tensor,   # [1,D]
    det_feature: torch.Tensor,     # [B,D]
    topk: int = 1,
):
    det_feature_n = F.normalize(det_feature, dim=-1)
    diff_feat = det_feature_n.mean(dim=0, keepdim=True) - support_proto  # [1,D]
    diff_feat = F.normalize(diff_feat, dim=-1)
    return get_adapted_single_class_text_embedding(
        model=model,
        dataset_name=dataset_name,
        class_name=class_name,
        device=device,
        diff_feat=diff_feat,
        topk=topk,
    )
@torch.no_grad()
def get_support_global_proto(model, support_loader, device):
    feats = []
    model.eval()
    for batch in support_loader:
        # 兼容两种dataset返回
        if isinstance(batch, dict):
            image = batch["image"].to(device)
        else:
            image = batch[0].to(device)
        _, det_feature = model(image)            # det_feature: [B, D]
        det_feature = F.normalize(det_feature, dim=-1)
        feats.append(det_feature)
    proto = torch.cat(feats, dim=0).mean(dim=0, keepdim=True)  # [1, D]
    proto = F.normalize(proto, dim=-1)
    return proto
def get_adapted_single_sentence_text_embedding(model, dataset_name, class_name, device):
    assert class_name in CLASS_NAMES[dataset_name], (
        f"class_name {class_name} not found; available class_names: {CLASS_NAMES[dataset_name]}"
    )
    real_name = REAL_NAMES[dataset_name][class_name]
    text_features = []
    for i in range(len(prompt_state)):
        prompted_state = [state.format(real_name) for state in prompt_state[i]]
        prompted_sentence = []
        for s in prompted_state:
            for template in prompt_templates:
                prompted_sentence.append(template.format(s))
        prompted_sentence = tokenize(prompted_sentence).to(device)
        class_embeddings = model.encode_text(prompted_sentence)
        class_embeddings = F.normalize(class_embeddings, dim=-1)
        text_features.append(class_embeddings)
    text_features = torch.cat(text_features, dim=0).to(device)
    return text_features


def get_adapted_text_embedding(model, dataset_name, device):
    ret_dict = {}
    for class_name in CLASS_NAMES[dataset_name]:
        text_features = get_adapted_single_class_text_embedding(
            model, dataset_name, class_name, device
        )
        ret_dict[class_name] = text_features
    return ret_dict[class_name]
def get_dynamic_adapted_text_embedding(support_proto, model, dataset_name,image_dataloader,det_feature, device):
    ret_dict = {}
    for class_name in CLASS_NAMES[dataset_name]:
        text_features = get_dynamic_class_text_embeddings(
            model,
            dataset_name,
            class_name,
            device,
            support_proto,  # [1,D]
            det_feature,  # det_feature 会在内部计算
            1,
        )
        ret_dict[class_name] = text_features
    return ret_dict[class_name]


# ================================================================================================
def calculate_similarity_map(
    patch_features, epoch_text_feature, img_size, test=False, domain="Medical"
):
    patch_anomaly_scores = 100.0 * torch.matmul(patch_features, epoch_text_feature)

    B, L, C = patch_anomaly_scores.shape
    H = int(np.sqrt(L))
    patch_pred = patch_anomaly_scores.permute(0, 2, 1).view(B, C, H, H)
    if test:
        assert C == 2
        sigma = 1 if domain == "Industrial" else 1.5
        kernel_size = 7 if domain == "Industrial" else 9
        patch_pred = (patch_pred[:, 1] + 1 - patch_pred[:, 0]) / 2
        patch_pred = gaussian_blur2d(
            patch_pred.unsqueeze(1), (kernel_size, kernel_size), (sigma, sigma)
        )
    patch_preds = F.interpolate(
        patch_pred, size=img_size, mode="bilinear", align_corners=True
    )
    if not test and C > 1:
        patch_preds = torch.softmax(patch_preds, dim=1)
    return patch_preds


focal_loss = FocalLoss()
dice_loss = BinaryDiceLoss()


def calculate_seg_loss(patch_preds, mask):
    loss = focal_loss(patch_preds, mask)
    loss += dice_loss(patch_preds[:, 0, :, :], 1 - mask)
    loss += dice_loss(patch_preds[:, 1, :, :], mask)
    return loss

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
)
try:
    from scipy.ndimage import label as cc_label
except Exception:
    cc_label = None


def _safe_minmax_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x_min, x_max = float(np.min(x)), float(np.max(x))
    if abs(x_max - x_min) < eps:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x_min) / (x_max - x_min + eps)).astype(np.float32)


def _safe_roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    # roc_auc_score 要求 y_true 里至少包含两类
    if np.max(y_true) == np.min(y_true):
        return 0.0
    return float(roc_auc_score(y_true, y_score))


def _safe_ap(y_true: np.ndarray, y_score: np.ndarray) -> float:
    if np.max(y_true) == np.min(y_true):
        return 0.0
    return float(average_precision_score(y_true, y_score))


def _f1_max_image(gt_sp: np.ndarray, pr_sp: np.ndarray, eps: float = 1e-8) -> float:
    if np.max(gt_sp) == np.min(gt_sp):
        return 0.0
    precisions, recalls, thresholds = precision_recall_curve(gt_sp, pr_sp)
    f1 = (2 * precisions * recalls) / (precisions + recalls + eps)
    f1 = f1[np.isfinite(f1)]
    return float(np.max(f1)) if f1.size > 0 else 0.0


def _f1_max_pixel(gt_px: np.ndarray, pr_px_norm: np.ndarray, step: float = 0.05, eps: float = 1e-8) -> float:
    # 按参考代码：对阈值扫描，基于全数据总面积求 precision/recall，再取 max
    gt = gt_px.astype(np.bool_)
    best = 0.0
    for thr in np.arange(0.0, 1.0 + 1e-6, step):
        pr = pr_px_norm > thr
        inter = np.logical_and(gt, pr).sum()
        pred_area = pr.sum()
        gt_area = gt.sum()
        precision = inter / (pred_area + eps)
        recall = inter / (gt_area + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        if np.isfinite(f1) and f1 > best:
            best = float(f1)
    return best


def _cal_pro_score(
    gt_px: np.ndarray,
    pr_px: np.ndarray,
    max_step: int = 200,
    max_fpr: float = 0.3,
    eps: float = 1e-8,
) -> float:
    """
    AUPRO: 计算 PRO-Recall vs FPR 曲线，并在 FPR∈[0, max_fpr] 下积分后归一化。
    gt_px/pr_px: [N,H,W]，gt_px 为 0/1 mask，pr_px 为连续得分（未必0-1也可）
    """
    if cc_label is None:
        raise ImportError("scipy is required for P-AUPRO. Please install scipy (pip install scipy).")

    gt = (gt_px > 0).astype(np.uint8)
    pr = pr_px.astype(np.float32)

    # 负样本像素总数
    neg_total = float((gt == 0).sum())
    if neg_total < 1:
        return 0.0

    # 阈值从高到低扫（更稳定）
    pr_min, pr_max = float(pr.min()), float(pr.max())
    if abs(pr_max - pr_min) < eps:
        return 0.0

    thresholds = np.linspace(pr_max, pr_min, max_step, endpoint=True)

    fprs = []
    pros = []

    for thr in thresholds:
        bin_pred = (pr >= thr).astype(np.uint8)

        # FPR
        fp = float(((bin_pred == 1) & (gt == 0)).sum())
        fpr = fp / (neg_total + eps)

        # PRO：对每个 GT 连通域计算 overlap / area，再平均
        pro_list = []
        for i in range(gt.shape[0]):
            lab, num = cc_label(gt[i])
            if num == 0:
                continue
            for rid in range(1, num + 1):
                region = (lab == rid)
                area = float(region.sum())
                if area < 1:
                    continue
                overlap = float((bin_pred[i][region] == 1).sum())
                pro_list.append(overlap / (area + eps))

        pro = float(np.mean(pro_list)) if len(pro_list) > 0 else 0.0

        fprs.append(fpr)
        pros.append(pro)

    fprs = np.asarray(fprs, dtype=np.float32)
    pros = np.asarray(pros, dtype=np.float32)

    # 按 fpr 排序，截断到 [0, max_fpr]
    order = np.argsort(fprs)
    fprs = fprs[order]
    pros = pros[order]

    # 只取 <= max_fpr 的点；保证至少两个点可积分
    keep = fprs <= max_fpr
    if keep.sum() < 2:
        return 0.0

    fprs = fprs[keep]
    pros = pros[keep]

    # trapezoid 积分并归一化
    area = float(np.trapz(pros, fprs))
    return area / (max_fpr + eps)


def metrics_eval(
    pixel_label: np.ndarray,
    image_label: np.ndarray,
    pixel_preds: np.ndarray,
    image_preds: np.ndarray,
    class_names: str,
    domain: str,
    max_step_aupro: int = 200,
):
    """
    输出 7 个指标：
      I-AUROC, I-AP, I-F1max,
      P-AUROC, P-AP, P-F1max, P-AUPRO
    保留原始的 image score 融合策略。
    """

    # ---------- shape 兼容 ----------
    # pixel_label: [N,H,W] or [N,1,H,W]
    if pixel_label.ndim == 4:
        pixel_label_3d = pixel_label.squeeze(1)
    else:
        pixel_label_3d = pixel_label

    # pixel_preds: [N,H,W] (你这边通常已是)
    if pixel_preds.ndim == 4:
        pixel_preds_3d = pixel_preds.squeeze(1)
    else:
        pixel_preds_3d = pixel_preds

    # image_label/image_preds: [N] or [N,1]
    image_label_1d = image_label.reshape(-1)
    image_preds_1d = image_preds.reshape(-1)

    # ---------- normalization ----------
    pixel_preds_norm = _safe_minmax_norm(pixel_preds_3d)
    image_preds_norm = _safe_minmax_norm(image_preds_1d)

    # ---------- image score 融合（沿用你原逻辑） ----------
    pmax_pred = pixel_preds_norm.max(axis=(1, 2))
    if domain != "Medical":
        pr_sp = 0.5 * pmax_pred + 0.5 * image_preds_norm
    else:
        pr_sp = pmax_pred

    gt_sp = image_label_1d.astype(np.int32)

    # ---------- image-level metrics ----------
    i_auroc = _safe_roc_auc(gt_sp, pr_sp)
    i_ap = _safe_ap(gt_sp, pr_sp)
    i_f1max = _f1_max_image(gt_sp, pr_sp)

    # ---------- pixel-level metrics ----------
    gt_px_flat = pixel_label_3d.reshape(-1).astype(np.int32)
    pr_px_flat = pixel_preds_norm.reshape(-1)

    p_auroc = _safe_roc_auc(gt_px_flat, pr_px_flat)
    p_ap = _safe_ap(gt_px_flat, pr_px_flat)
    p_f1max = _f1_max_pixel(pixel_label_3d, pixel_preds_norm)

    # AUPRO（需要 scipy）
    try:
        p_aupro = _cal_pro_score(pixel_label_3d, pixel_preds_norm, max_step=max_step_aupro)
    except Exception:
        p_aupro = 0.0  # 若缺依赖或异常，返回0，避免评估中断

    # ---------- output ----------
    result = {
        "class name": class_names,
        "I-AUROC": round(i_auroc, 4) * 100,
        "I-AP": round(i_ap, 4) * 100,
        "I-F1max": round(i_f1max, 4) * 100,
        "P-AUROC": round(p_auroc, 4) * 100,
        "P-AP": round(p_ap, 4) * 100,
        "P-F1max": round(p_f1max, 4) * 100,
        "P-AUPRO": round(p_aupro, 4) * 100,
    }
    return result
# ================================================================================================


# def metrics_eval(
#     pixel_label: np.ndarray,
#     image_label: np.ndarray,
#     pixel_preds: np.ndarray,
#     image_preds: np.ndarray,
#     class_names: str,
#     domain: str,
# ):
#     if pixel_preds.max() != 1:
#         pixel_preds = (pixel_preds - pixel_preds.min()) / (
#             pixel_preds.max() - pixel_preds.min()
#         )
#     if image_preds.max() != 1:
#         image_preds = (image_preds - image_preds.min()) / (
#             image_preds.max() - image_preds.min()
#         )
#
#     pmax_pred = pixel_preds.max(axis=(1, 2))
#     if domain != "Medical":
#         image_preds = pmax_pred * 0.5 + image_preds * 0.5
#     else:
#         image_preds = pmax_pred
#     # ================================================================================================
#     # pixel level auc & ap
#     pixel_label = pixel_label.flatten()
#     pixel_preds = pixel_preds.flatten()
#
#     zero_pixel_auc = roc_auc_score(pixel_label, pixel_preds)
#     zero_pixel_ap = average_precision_score(pixel_label, pixel_preds)
#     # ================================================================================================
#     # image level auc & ap
#     if image_label.max() != image_label.min():
#         image_label = image_label.flatten()
#         agg_image_preds = image_preds.flatten()
#         agg_image_auc = roc_auc_score(image_label, agg_image_preds)
#         agg_image_ap = average_precision_score(image_label, agg_image_preds)
#     else:
#         agg_image_auc = 0
#         agg_image_ap = 0
#     # ================================================================================================
#     result = {
#         "class name": class_names,
#         "pixel AUC": round(zero_pixel_auc, 4) * 100,
#         "pixel AP": round(zero_pixel_ap, 4) * 100,
#         "image AUC": round(agg_image_auc, 4) * 100,
#         "image AP": round(agg_image_ap, 4) * 100,
#     }
#     return result


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    return (alpha * image + (1 - alpha) * scoremap).astype(np.uint8)


def visualize(
    pixel_label: np.ndarray,
    pixel_preds: np.ndarray,
    file_names: list[str],
    save_dir: str,
    dataset_name: str,
    class_name: str,
):
    if pixel_preds.max() != 1:
        pixel_preds = (pixel_preds - pixel_preds.min()) / (
            pixel_preds.max() - pixel_preds.min()
        )
        pixel_preds = (pixel_preds * 255).astype(np.uint8)
    if pixel_label.dtype != np.uint8:
        pixel_label = pixel_label != 0
        pixel_label = (pixel_label * 255).astype(np.uint8)
    # ===============================================================================================
    # save path
    save_dir = os.path.join(save_dir, "visualization", dataset_name, class_name)
    os.makedirs(save_dir, exist_ok=True)
    for idx, file in enumerate(file_names):
        image_file = os.path.join(DATA_PATH[dataset_name], file)
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, pixel_label.shape[-2:])
        save_image_list = [image]

        # if dataset_name == "MVTec":
        damage_name, image_name = file.split("/")[-2:]
        file_name = f"{damage_name}_{image_name}"
        # else:
        #     raise NotImplementedError

        save_image_list.append(cv2.cvtColor(pixel_label[idx, 0], cv2.COLOR_GRAY2RGB))
        save_image_list.append(cv2.cvtColor(pixel_preds[idx], cv2.COLOR_GRAY2RGB))
        save_image_list = save_image_list[:1] + [
            apply_ad_scoremap(image, _) for _ in save_image_list[1:]
        ]
        scoremap = np.vstack(save_image_list)
        cv2.imwrite(os.path.join(save_dir, file_name), scoremap)
