#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
import torch.nn.functional as F
from nnunet.training.loss_functions.TopK_loss import TopKLoss
from nnunet.training.loss_functions.crossentropy import RobustCrossEntropyLoss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from torch import nn
import numpy as np
from nnunet.training.loss_functions.focal_loss import FocalLoss
import random
from scipy.ndimage import distance_transform_edt

def project_along_x_axis(data, axis):
    # 沿 x 轴投影
    #projection = np.max(data, axis=axis)  # 可以选择 'max' 或 'mean' 作为投影方法
    projection = torch.max(data, dim=axis)[0]
    return projection

def abs_ln_plus_one_torch(x):
    if torch.any(x <= 0):
        raise ValueError("All input values must be greater than 0")

    result = torch.abs(torch.log(x)) + 1
    result = torch.clamp(result, max=100)

    return result

class GDL(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.,
                 square=False, square_volumes=False):
        """
        square_volumes will square the weight term. The paper recommends square_volumes=True; I don't (just an intuition)
        """
        super(GDL, self).__init__()

        self.square_volumes = square_volumes
        self.square = square
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if len(shp_x) != len(shp_y):
            y = y.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(x.shape, y.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = y
        else:
            gt = y.long()
            y_onehot = torch.zeros(shp_x)
            if x.device.type == "cuda":
                y_onehot = y_onehot.cuda(x.device.index)
            y_onehot.scatter_(1, gt, 1)

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        if not self.do_bg:
            x = x[:, 1:]
            y_onehot = y_onehot[:, 1:]

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y_onehot, axes, loss_mask, self.square)

        # GDL weight computation, we use 1/V
        volumes = sum_tensor(y_onehot, axes) + 1e-6 # add some eps to prevent div by zero

        if self.square_volumes:
            volumes = volumes ** 2

        # apply weights
        tp = tp / volumes
        fp = fp / volumes
        fn = fn / volumes

        # sum over classes
        if self.batch_dice:
            axis = 0
        else:
            axis = 1

        tp = tp.sum(axis, keepdim=False)
        fp = fp.sum(axis, keepdim=False)
        fn = fn.sum(axis, keepdim=False)

        # compute dice
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        dc = dc.mean()

        return -dc


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    """
    net_output must be (b, c, x, y(, z)))
    gt must be a label map (shape (b, 1, x, y(, z)) OR shape (b, x, y(, z))) or one hot encoding (b, c, x, y(, z))
    if mask is provided it must have shape (b, 1, x, y(, z)))
    :param net_output:
    :param gt:
    :param axes: can be (, ) = no summation
    :param mask: mask must be 1 for valid pixels and 0 for invalid pixels
    :param square: if True then fp, tp and fn will be squared before summation
    :return:
    """
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x, device=net_output.device)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)
        tn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


class SoftDiceLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        """
        super(SoftDiceLoss, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    # todo add the w para
    def forward(self, x, y, w, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)
        assert not torch.any(torch.isnan(x))
        assert not torch.any(torch.isinf(x))
        assert not torch.any(torch.isnan(x) + torch.isinf(x))
        assert not torch.any(torch.isnan(y) + torch.isinf(y))
        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)  #######################################

        
        pp = fp / (fn + fp)
        pp_n = 1 - pp
        nominator = 2 * tp + self.smooth
        #denominator = 2 * tp + 2 * pp * fp + 2 * pp_n * fn + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth
        assert not torch.any(torch.isnan(nominator) + torch.isinf(nominator))
        assert not torch.any(torch.isnan(denominator) + torch.isinf(denominator))

        dc = nominator / denominator
        assert not torch.any(torch.isnan(dc) + torch.isinf(dc))

        dc = dc * w
        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]

        dc = dc.sum(axis=1).mean()
        #assert not torch.any(torch.isnan(dc) + torch.isinf(dc))

        return -dc


    def nnunet_forward(self, x, y, loss_mask=None):
        shp_x = x.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth
        denominator = 2 * tp + fp + fn + self.smooth

        dc = nominator / (denominator + 1e-8)

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc


class MCCLoss(nn.Module):
    def __init__(self, apply_nonlin=None, batch_mcc=False, do_bg=True, smooth=0.0):
        """
        based on matthews correlation coefficient
        https://en.wikipedia.org/wiki/Matthews_correlation_coefficient

        Does not work. Really unstable. F this.
        """
        super(MCCLoss, self).__init__()

        self.smooth = smooth
        self.do_bg = do_bg
        self.batch_mcc = batch_mcc
        self.apply_nonlin = apply_nonlin

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        voxels = np.prod(shp_x[2:])

        if self.batch_mcc:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        tp, fp, fn, tn = get_tp_fp_fn_tn(x, y, axes, loss_mask, False)
        tp /= voxels
        fp /= voxels
        fn /= voxels
        tn /= voxels

        nominator = tp * tn - fp * fn + self.smooth
        denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5 + self.smooth

        mcc = nominator / denominator

        if not self.do_bg:
            if self.batch_mcc:
                mcc = mcc[1:]
            else:
                mcc = mcc[:, 1:]
        mcc = mcc.mean()

        return -mcc


class SoftDiceLossSquared(nn.Module):
    def __init__(self, apply_nonlin=None, batch_dice=False, do_bg=True, smooth=1.):
        """
        squares the terms in the denominator as proposed by Milletari et al.
        """
        super(SoftDiceLossSquared, self).__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.apply_nonlin = apply_nonlin
        self.smooth = smooth

    def forward(self, x, y, loss_mask=None):
        shp_x = x.shape
        shp_y = y.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x = self.apply_nonlin(x)

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                y = y.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(x.shape, y.shape)]):
                # if this is the case then gt is probably already a one hot encoding
                y_onehot = y
            else:
                y = y.long()
                y_onehot = torch.zeros(shp_x)
                if x.device.type == "cuda":
                    y_onehot = y_onehot.cuda(x.device.index)
                y_onehot.scatter_(1, y, 1).float()

        intersect = x * y_onehot
        # values in the denominator get smoothed
        denominator = x ** 2 + y_onehot ** 2

        # aggregation was previously done in get_tp_fp_fn, but needs to be done here now (needs to be done after
        # squaring)
        intersect = sum_tensor(intersect, axes, False) + self.smooth
        denominator = sum_tensor(denominator, axes, False) + self.smooth

        dc = 2 * intersect / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return -dc

def calculate_mse_in_nonzero_area(true_labels, predicted_labels):
    """
    计算标签和预测数据中非零标签区域的均方误差（MSE）。

    参数：
    true_labels (numpy.ndarray): 真实标签数组。
    predicted_labels (numpy.ndarray): 预测标签数组。
    outside_value (int/float): 代表背景或无关区域的值。

    返回：
    mse (float): 非零标签区域的均方误差。
    """
    # 获取非零标签区域的掩码
    true_mask = true_labels != 0
    pred_mask = predicted_labels != 0
    mask = true_mask | pred_mask
    # 获取标签和预测数据中非零区域
    true_nonzero_area = true_labels[mask]
    pred_nonzero_area = predicted_labels[mask]

    if len(true_nonzero_area) == 0:
        mse1 = 0
    else:
        mse1 = nn.MSELoss(reduction='mean')(true_nonzero_area, pred_nonzero_area)

    return mse1

def cosine_similarity_loss(true_labels, predicted_labels):
    #余玄相似度损失
    # 计算余弦相似度
    true_mask = true_labels != 0
    pred_mask = predicted_labels != 0
    mask = true_mask | pred_mask
    # 获取标签和预测数据中非零区域
    true_nonzero_area = true_labels[mask]
    pred_nonzero_area = predicted_labels[mask]

    if len(true_nonzero_area) == 0:
        cos_sim = 1
    else:
    
      cos_sim = F.cosine_similarity(true_nonzero_area, pred_nonzero_area, dim=-1)
      # 计算负余弦相似度损失
    loss = 1 - cos_sim
    return loss  # 返回平均损失

def all_loss(true_labels, predicted_labels):
    return (calculate_mse_in_nonzero_area(true_labels, predicted_labels) + cosine_similarity_loss(true_labels, predicted_labels)) / 2


def compute_distance_transform(boundary: torch.Tensor) -> torch.Tensor:
    """
    计算距离变换（距离边界的距离）。
    :param boundary: 2D 边界二值图像 (H, W)，值为 1 表示前景，0 表示背景
    :return: 距离变换结果 (H, W)，值越大表示离边界越远
    """
    h, w = boundary.shape
    # 获取边界点
    boundary_points = torch.nonzero(boundary, as_tuple=False)  # (N, 2)，N为边界点数量
    if boundary_points.size(0) == 0:  # 如果没有边界点，返回全零
        return torch.zeros_like(boundary, dtype=torch.float32)

    # 获取所有点的位置
    all_points = torch.stack(torch.meshgrid(
        torch.arange(h, device=boundary.device),
        torch.arange(w, device=boundary.device),
        indexing="ij"
    ), dim=-1).reshape(-1, 2)  # (H*W, 2)

    # 计算每个点到边界的最小距离
    distances = torch.cdist(all_points.float(), boundary_points.float())  # (H*W, N)
    min_distances, _ = distances.min(dim=1)  # 取每个点到最近边界点的距离

    # 重塑为 (H, W)
    return min_distances.reshape(h, w)


def nsd_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    计算基于2D的标准化表面距离（NSD）损失。
    :param pred: 预测的二值图像 (H, W)，值为 1 表示前景，0 表示背景，经过 argmax 后
    :param target: 真实标签的二值图像 (H, W)，值为 1 表示前景，0 表示背景，经过 argmax 后
    :return: NSD 损失值
    """
    # 计算目标和预测的距离变换
    pred_boundary = (pred == 1).float()  # 预测边界
    target_boundary = (target == 1).float()  # 真实边界

    pred_distance = compute_distance_transform(pred_boundary)  # 计算预测的距离变换
    target_distance = compute_distance_transform(target_boundary)  # 计算真实的距离变换

    # 计算表面距离
    surface_distance = torch.abs(pred_distance - target_distance)
    
    # 标准化处理（可以按最大值标准化）
    max_distance = torch.max(surface_distance)
    if max_distance > 0:
        surface_distance /= max_distance  # 标准化至 0-1 范围

    # 返回 NSD 损失值（取均值）
    return surface_distance.mean()





class DC_and_CE_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(DC_and_CE_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.ignore_label = ignore_label
        self.two = True
        self.two_d_loss = 0
        self.project_loss = nsd_loss
        if not square_dice:   #true
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)
    # todo add w para
    def forward(self, net_output, target, w, w1):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        # todo add w para

        if self.two == True:
            print("使用投影损失")
            #weight_s = np.array([1, 1.1, 1.2, 1.2, 1.3, 1.4, 1.4, 2, 2, 1.6, 1.8, 1.2, 1.3, 1.2])
            weight_s = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
            weight_s = weight_s / weight_s.sum()
            weight_s = torch.tensor(weight_s)
            
            out = net_output[0]
            out = out.softmax(0).argmax(0)
            target_out = target[0][0]
            for i in range(14):
                if i == 0:
                    continue
                else:
                    out_t = out
                    target_out_t = target_out
                    target_out_t = torch.where(target_out_t != i, torch.tensor(0).cuda(), target_out_t)
                    target_out_t = torch.where(target_out_t == i, torch.tensor(1).cuda(), target_out_t)
                    out_t = torch.where(out_t != i, torch.tensor(0).cuda(), out_t)
                    out_t = torch.where(out_t == i, torch.tensor(1).cuda(), out_t)

                    o_x = project_along_x_axis(out_t, 0)
                    o_y = project_along_x_axis(out_t, 1)
                    o_z = project_along_x_axis(out_t, 2)
                    t_x = project_along_x_axis(target_out_t, 0)
                    t_y = project_along_x_axis(target_out_t, 1)
                    t_z = project_along_x_axis(target_out_t, 2)
                    two_d_loss = self.project_loss(t_x, o_x) + self.project_loss(t_y, o_y) + self.project_loss(t_z, o_z)
                    self.two_d_loss += two_d_loss
            self.two_d_loss = self.two_d_loss / 13
            
            

        dc_loss = self.dc(net_output, target, w, loss_mask=mask) if self.weight_dice != 0 else 0
        #assert not torch.any(torch.isnan(dc_loss) + torch.isinf(dc_loss))

        if self.log_dice:   #不进入
            dc_loss = -torch.log(-dc_loss)
        ce_loss = self.ce(net_output, target[:, 0].long(), w1) if self.weight_ce != 0 else 0
        #assert not torch.any(torch.isnan(ce_loss) + torch.isinf(ce_loss))

        if self.ignore_label is not None:
            print("ignore_label",self.ignore_label)
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()
        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.two_d_loss / 3
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result


    def nnunet_forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result


class Weight_DC_and_Weight_CE_loss_and_Focal_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, focal_kwargs, aggregate="sum", square_dice=False, weight_ce=1, weight_dice=1, weight_focal =0.5,
                 log_dice=False, ignore_label=None):
        """
        CAREFUL. Weights for CE and Dice do not need to sum to one. You can set whatever you want.
        :param soft_dice_kwargs:
        :param ce_kwargs:
        :param aggregate:
        :param square_dice:
        :param weight_ce:
        :param weight_dice:
        """
        super(Weight_DC_and_Weight_CE_loss_and_Focal_loss, self).__init__()
        if ignore_label is not None:
            assert not square_dice, 'not implemented'
            ce_kwargs['reduction'] = 'none'
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_ce = weight_ce
        self.weight_focal = weight_focal
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.focal = FocalLoss(**focal_kwargs)
        self.ignore_label = ignore_label

        if not square_dice:   #true
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)
    # todo add w para
    def forward(self, net_output, target, w):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        # todo add w para

        dc_loss = self.dc(net_output, target, w, loss_mask=mask) if self.weight_dice != 0 else 0
        focal_loss = self.focal(net_output, target[:, 0].long())
        assert not torch.any(torch.isnan(dc_loss) + torch.isinf(dc_loss))

        if self.log_dice:   #不进入
            dc_loss = -torch.log(-dc_loss)
        ce_loss = self.ce(net_output, target[:, 0].long(), w) if self.weight_ce != 0 else 0
        assert not torch.any(torch.isnan(ce_loss) + torch.isinf(ce_loss))

        if self.ignore_label is not None:
            print("ignore_label",self.ignore_label)
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()
        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss + self.weight_focal * focal_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result


    def nnunet_forward(self, net_output, target):
        """
        target must be b, c, x, y(, z) with c=1
        :param net_output:
        :param target:
        :return:
        """
        if self.ignore_label is not None:
            assert target.shape[1] == 1, 'not implemented for one hot encoding'
            mask = target != self.ignore_label
            target[~mask] = 0
            mask = mask.float()
        else:
            mask = None

        dc_loss = self.dc(net_output, target, loss_mask=mask) if self.weight_dice != 0 else 0
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        ce_loss = self.ce(net_output, target[:, 0].long()) if self.weight_ce != 0 else 0
        if self.ignore_label is not None:
            ce_loss *= mask[:, 0]
            ce_loss = ce_loss.sum() / mask.sum()

        if self.aggregate == "sum":
            result = self.weight_ce * ce_loss + self.weight_dice * dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result


class DC_and_BCE_loss(nn.Module):
    def __init__(self, bce_kwargs, soft_dice_kwargs, aggregate="sum"):
        """
        DO NOT APPLY NONLINEARITY IN YOUR NETWORK!

        THIS LOSS IS INTENDED TO BE USED FOR BRATS REGIONS ONLY
        :param soft_dice_kwargs:
        :param bce_kwargs:
        :param aggregate:
        """
        super(DC_and_BCE_loss, self).__init__()

        self.aggregate = aggregate
        self.ce = nn.BCEWithLogitsLoss(**bce_kwargs)
        self.dc = SoftDiceLoss(apply_nonlin=torch.sigmoid, **soft_dice_kwargs)

    def forward(self, net_output, target):
        ce_loss = self.ce(net_output, target)
        dc_loss = self.dc(net_output, target)

        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)

        return result


class GDL_and_CE_loss(nn.Module):
    def __init__(self, gdl_dice_kwargs, ce_kwargs, aggregate="sum"):
        super(GDL_and_CE_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = RobustCrossEntropyLoss(**ce_kwargs)
        self.dc = GDL(softmax_helper, **gdl_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later)
        return result


class DC_and_topk_loss(nn.Module):
    def __init__(self, soft_dice_kwargs, ce_kwargs, aggregate="sum", square_dice=False):
        super(DC_and_topk_loss, self).__init__()
        self.aggregate = aggregate
        self.ce = TopKLoss(**ce_kwargs)
        if not square_dice:
            self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        else:
            self.dc = SoftDiceLossSquared(apply_nonlin=softmax_helper, **soft_dice_kwargs)

    def forward(self, net_output, target):
        dc_loss = self.dc(net_output, target)
        ce_loss = self.ce(net_output, target)
        if self.aggregate == "sum":
            result = ce_loss + dc_loss
        else:
            raise NotImplementedError("nah son") # reserved for other stuff (later?)
        return result

class DC_and_Focal_Loss(nn.Module):
    def __init__(self, soft_dice_kwargs, focal_kwargs, weight_focal=1, weight_dice=1, log_dice=False):
        super(DC_and_Focal_Loss, self).__init__()
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal

        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.focal = FocalLoss(**focal_kwargs)

    def forward(self, net_output, target, weight=None):
        if weight is not None and net_output.device.type == "cuda":
            weight = weight.cuda(net_output.device.index)

        dc_loss = self.dc(net_output, target, weight)
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)

        focal_loss = self.focal(net_output, target[:, 0].long())

        result = self.weight_dice * dc_loss + self.weight_focal * focal_loss
        # result = torch.softmax(self.loss_weights, dim=0) * result

        return result

class DC_and_Focal_Loss_and_BCE(nn.Module):
    def __init__(self, soft_dice_kwargs, focal_kwargs, bce_kwargs, weight_focal=1, weight_dice=1, weight_bce = 0.4, log_dice=False):
        super(DC_and_Focal_Loss_and_BCE, self).__init__()
        self.log_dice = log_dice
        self.weight_dice = weight_dice
        self.weight_focal = weight_focal
        self.weight_bce = weight_bce
        self.dc = SoftDiceLoss(apply_nonlin=softmax_helper, **soft_dice_kwargs)
        self.focal = FocalLoss(**focal_kwargs)
        self.bce = CustomBCELossWithLogits(**bce_kwargs)
    def forward(self, net_output, target, weight=None):
        if weight is not None and net_output.device.type == "cuda":
            weight = weight.cuda(net_output.device.index)

        dc_loss = self.dc(net_output, target, weight)
        if self.log_dice:
            dc_loss = -torch.log(-dc_loss)
        num_pos_samples = torch.sum(target == 1)
        num_neg_samples = torch.sum(target == 0)
        pos_weight = (num_neg_samples + 1e-4) / (num_pos_samples + 1e-4)
        pos_weight.cuda()
        bce_loss = self.bce(torch.argmax(torch.softmax(net_output, dim=1),dim=1).unsqueeze(1).float(), target, pos_weight=pos_weight)
        #bce_loss = self.bce(torch.argmax(torch.softmax(net_output, dim=1),dim=1).unsqueeze(1).float(), target)
        focal_loss = self.focal(net_output, target[:, 0].long())

        result = self.weight_dice * dc_loss + self.weight_focal * focal_loss + self.weight_bce * bce_loss
        # result = torch.softmax(self.loss_weights, dim=0) * result

        return result

class CustomBCELossWithLogits(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(CustomBCELossWithLogits, self).__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(input, target,
                                                  self.weight,
                                                  pos_weight=pos_weight,
                                                  reduction=self.reduction)
