import torch
import math
from config import *
config = Config()

# Colloss
def coll_smoothed_loss(pred_batch ,seq_start_end,  mask):

    coll_pro_szene = torch.zeros([seq_start_end.size()[0]]).to(pred_batch)
    z = torch.zeros([1]).to(pred_batch)
    y = torch.ones([1]).to(pred_batch)
    for i, (start, end) in enumerate(seq_start_end):
        start = start.item()
        end = end.item()
        pred = pred_batch #*mask # ToDo Change this if mask is needed!
        currSzene = pred[:, start:end].contiguous()
        dist_mat_pred = torch.cdist(currSzene, currSzene, p=2.0, compute_mode='donot_use_mm_for_euclid_dist')
        dist_mat_pred = dist_mat_pred * mask[:,start:end, start:end].detach() # detach mask from computational graph!
        dist_mat_pred = dist_mat_pred[dist_mat_pred !=0.]
        dist_mat_pred = torch.sigmoid((dist_mat_pred - config.collision_distance)*35.) * (z - y) + y   # cause binary tensor is not differentiable
        dist_mat_pred = dist_mat_pred.sum(dim=-1) # get number of coll for every pedestrian traj
        coll_pro_szene[i] = dist_mat_pred.sum().unsqueeze(dim=0)/(end - start)

    return coll_pro_szene.mean()


def smooth_l1_loss(input, target,loss_mask, beta=1. / 9, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter
    """
    n = torch.abs(input - target)*loss_mask
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss #.sum()

def l2_loss(pred_traj, pred_traj_gt, loss_mask, mode='average'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - loss_mask: Tensor of shape (batch, seq_len)
    - mode: Can be one of sum, average, raw
    Output:
    - loss: l2 loss depending on mode
    """
    seq_len, batch, _ = pred_traj.size()
    loss = smooth_l1_loss(pred_traj_gt.permute(1, 0, 2),
                          pred_traj.permute(1, 0, 2),loss_mask.permute(1, 0, 2),
                          beta = 0.05, size_average=False)
    if mode == 'sum':
        return torch.sum(loss)
    elif mode == 'average':
        return torch.sum(loss) / torch.numel(loss_mask.data) # numel Calculate The Number Of Elements In A PyTorch Tensor
    elif mode == 'raw':  # FOR TRAINING!
        return loss.sum(dim=2).sum(dim=1)

def GaußNLL(mu, scale, pred_traj_gt, loss_mask=None):
    # scale = torch.clamp(scale, min=-9, max=4)
    var = torch.exp(scale).pow(2)

    loss = 0.5 * math.log(2 * math.pi) + 0.5 * (torch.log(var) + (pred_traj_gt - mu) ** 2 / var)

    return loss.mean() #/ torch.numel(loss_mask.data)

def displacement_error(pred_traj, pred_traj_gt, val_mask, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_traj: Tensor of shape (seq_len, batch, 2). Predicted trajectory.
    - pred_traj_gt: Tensor of shape (seq_len, batch, 2). Ground truth
    predictions.
    - consider_ped: Tensor of shape (batch)
    - mode: Can be one of sum, raw
    Output:
    - loss: gives the eculidian displacement error
    """
    seq_len, _, _ = pred_traj.size()
    loss = (pred_traj_gt.permute(1, 0, 2) - pred_traj.permute(1, 0, 2))*val_mask.permute(1, 0, 2)
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1) * consider_ped
    else:
        loss = torch.sqrt(loss.sum(dim=2)).sum(dim=1)
    if mode == 'sum':
        return torch.sum(loss)
    if mode == 'sum':
        return torch.mean(loss)
    elif mode == 'raw':
        return loss

def final_displacement_error(pred_pos, pred_pos_gt, consider_ped=None, mode='sum'):
    """
    Input:
    - pred_pos: Tensor of shape (batch, 2). Predicted last pos.
    - pred_pos_gt: Tensor of shape (seq_len, batch, 2). Groud truth
    last pos
    - consider_ped: Tensor of shape (batch)
    Output:
    - loss: gives the eculidian displacement error
    """
    loss = pred_pos_gt - (pred_pos) # + 0.2*torch.randn_like(pred_pos)) # verbessert zumindest auf
    loss = loss**2
    if consider_ped is not None:
        loss = torch.sqrt((loss*consider_ped).sum(dim=1))
        loss = loss[loss != 0] # filter out zeros
    else:
        loss = torch.sqrt(loss.sum(dim=1))
    if mode == 'raw':
        return loss
    else:
        return torch.sum(loss)
