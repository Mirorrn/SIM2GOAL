import torch
import numpy as np
from config import *
config = Config()
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.rcParams['figure.dpi'] = 300
def int_tuple(s):
    return tuple(int(i) for i in s.split(','))


def get_dset_path(dset_name, dset_type):
    if config.trajnet:
        _dir = os.path.dirname(__file__)
        _dir = _dir.split("/")[:-2]
        _dir = "/".join(_dir) + "/Sim2Goal"
        return os.path.join(_dir, 'datasets/trajnet', dset_name, dset_type)
    else:
        _dir = os.path.dirname(__file__)
        _dir = _dir.split("/")[:-2]
        _dir = "/".join(_dir) + "/Sim2Goal"
        return os.path.join(_dir, 'datasets/ETHandUTC', dset_name, dset_type)


def relative_to_abs(rel_traj, start_pos):
    """
    Inputs:
    - rel_traj: pytorch tensor of shape (seq_len, batch, 2)
    - start_pos: pytorch tensor of shape (batch, 2)
    Outputs:
    - abs_traj: pytorch tensor of shape (seq_len, batch, 2)
    """
    # batch, seq_len, 2
    if config.nabs:
        # rel_traj = rel_traj.permute(1, 0, 2)
        # start_pos = torch.unsqueeze(start_pos, dim=1)
        # return (rel_traj + start_pos).permute(1, 0, 2)
        return rel_traj * 28.
    else:
        rel_traj = rel_traj.permute(1, 0, 2)
        displacement = torch.cumsum(rel_traj, dim=1)
        start_pos = torch.unsqueeze(start_pos, dim=1)
        abs_traj = displacement + start_pos # traj are rel, so add them up and add to the last known position! so they are rel to last position!
        return abs_traj.permute(1, 0, 2)

def fast_coll_counter(pred_batch,seq_start_end, ids, mask, sample=20):
    ids_of_col_szenes = np.zeros([seq_start_end.shape[0]])
    coll_pro_szene = 0
    stack_of_coll_indeces = []
    pred_batch = torch.stack(pred_batch)
    if not ids:
        mask = mask.unsqueeze(dim=0).repeat(20, 1, 1, 1)
        mask = mask.view(20 * 12, pred_batch.shape[2], pred_batch.shape[2])
        pred = pred_batch.view(20 * 12, pred_batch.shape[2], 2)
    for i, (start, end) in enumerate(seq_start_end):
        start = start.item()
        end = end.item()
        if ids:
            pred = pred_batch[ids[i]]
        currSzene = pred[:, start:end]
        dist_mat = torch.cdist(currSzene, currSzene, p=2.0, compute_mode='donot_use_mm_for_euclid_dist')
        dist_mat_one_triu = torch.triu(dist_mat)
        dist_mat_one_triu = dist_mat_one_triu * mask[:,start:end, start:end]
        filter_zeros = torch.logical_and(0. != dist_mat_one_triu,  dist_mat_one_triu <= config.collision_distance)
        filter_col_pos = torch.logical_and(0. != dist_mat,  dist_mat <= config.collision_distance)
        filter_zeros_sum=filter_zeros.sum().unsqueeze(dim=0).item()
        coll_pro_szene += filter_zeros_sum
        if filter_zeros_sum > 0.:
            ids_of_col_szenes[i] = 1
        stack_of_coll_indeces.append(filter_col_pos)
    count = len(seq_start_end) #* count_empty
    if not ids:
        count = count * sample
    return coll_pro_szene, count, ids_of_col_szenes, stack_of_coll_indeces


DIR = home +'/Documents/SI_pic/'
cmap = plt.cm.get_cmap('hsv')

human_colors = cmap(np.linspace(0.5, 1.0, 5))
human_colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']

def plot_multimodal(obs_traj, pred_traj_gt, batch_pred_traj_fake_rel, att_score_list_batch, ids, stack_of_coll_indeces,
                   seq_start_end, ids_of_col_szenes_fake, loss_mask, config, szene_id, samples=None):
    # samples = torch.stack(samples)
    if config.num_samples > 1:
        all_pred = torch.stack(batch_pred_traj_fake_rel)
    szene_id_batch_offset = szene_id*len(seq_start_end)
    # search_ids = [18, 38, 25]
    for k, (start, end) in enumerate(seq_start_end):
        nmp_humans = end-start

        # if ids_of_col_szenes_fake[k]:  # and nmp_humans <= 5:               # plot scenes only with collisions
        if nmp_humans <= 15:                                              # plot scenes with only less than x humans
        # if szene_id_batch_offset + k in search_ids:                       # look for scenes with a specific id
            if config.num_samples > 1:
                all_pred_szene = all_pred[:, :, start:end, :].cpu().numpy()
                # samples_scene = samples[:, : ,start:end].cpu().numpy()
            obs_traj_szene = obs_traj[:, start:end, :].cpu().numpy()

                # samples_scene = samples[:, start:end].cpu().numpy()
            loss_mask_szene = loss_mask[:, start:end, :].cpu().numpy()
            ground_truth_input_x_piccoor = (obs_traj_szene[:, :, 0].T)
            ground_truth_input_y_piccoor = (obs_traj_szene[:, :, 1].T)
            fig, ax = plt.subplots()
            ax.axis('off')
            for i in range(end-start):
                obs_traj_i = obs_traj_szene[:,i]
                loss_mask_szene_i = loss_mask_szene[:,i]
                for t, timestep in enumerate(obs_traj_i):
                    if loss_mask_szene_i[t,0]:
                        obs_pos = plt.Circle(timestep, config.collision_distance/2., fill=False, color='black',
                                             label='Traj. History', alpha=.5)
                        ax.add_artist(obs_pos)
                # for t in range(12):
                #     for s in range(2):
                #         testi = samples_scene[t,i]
                        # plt.scatter(samples_scene[s, t, i, :, 0], samples_scene[s, t, i, :, 1], s=0.1, color=human_colors[s%5])

                obs_traj_i_filtered = obs_traj_i[np.nonzero(loss_mask_szene_i[0:config.obs_len,0])[0]]
                observed_line = plt.plot(obs_traj_i_filtered[:,0],obs_traj_i_filtered[:,1],"black",linewidth=2,
                                         label="History",alpha=.5)[0]

                observed_line.axes.annotate(
                    "",
                    xytext=(
                        ground_truth_input_x_piccoor[i, -2],
                        ground_truth_input_y_piccoor[i, -2],
                    ),
                    xy=(
                        ground_truth_input_x_piccoor[i, -1],
                        ground_truth_input_y_piccoor[i, -1],
                    ),
                    arrowprops=dict(
                        arrowstyle="-|>", color=observed_line.get_color(), lw=1
                    ),
                    size=15, alpha=1
                )

                if config.num_samples > 1:
                    for s in range(2):
                        pred_traj = all_pred_szene[s]
                        pred_traj = pred_traj[:,i]
                        plt.plot(
                            np.append(
                                obs_traj_i_filtered[-1,0],
                                pred_traj[:, 0],
                            ),
                            np.append(
                                obs_traj_i_filtered[-1,1],
                                pred_traj[:,1],
                            ),
                            color=human_colors[s%5],
                            linewidth=1,
                            alpha=.5
                        )
            fig.suptitle(szene_id_batch_offset + k, fontsize=20)
            ax.set_aspect("equal")
            # plt.savefig(DIR + 'SGAN2'+ str(szene_id_batch_offset + k) +'.png', dpi=800,bbox_inches='tight')
            plt.show()

        plt.close()

def plot_best(obs_traj, pred_traj_gt, batch_pred_traj_fake_rel, att_score_list_batch, ids, stack_of_coll_indeces,
                   seq_start_end, ids_of_col_szenes_fake, loss_mask, config, szene_id, samples= None):
    #cmap = plt.cm.get_cmap('hsv', 5)
    if config.num_samples > 1:
        all_pred = torch.stack(batch_pred_traj_fake_rel)
    szene_id_batch_offset = szene_id*len(seq_start_end)
    # search_ids = [18, 38, 25]
    for k, (start, end) in enumerate(seq_start_end):
        nmp_humans = end-start
        # if ids_of_col_szenes_fake[k]:  # and nmp_humans <= 5:               # plot scenes only with collisions
        if nmp_humans <= 5:                                              # plot scenes with only less than x humans
        # if szene_id_batch_offset + k in search_ids:                       # look for scenes with a specific id
            tmp_stack_of_coll_indeces = stack_of_coll_indeces[k].cpu().numpy()
            model_output_traj_best = batch_pred_traj_fake_rel[ids[k]]
            obs_traj_szene = obs_traj[:, start:end, :].cpu().numpy()
            #if torch.is_tensor(samples):
            # samples_scene = samples[ids[k]][:, start:end].cpu().numpy()
            pred_traj_gt_szene = pred_traj_gt[:, start:end, :].cpu().numpy()
            model_output_traj_best_szene = model_output_traj_best[:, start:end, :].cpu().numpy()
            loss_mask_szene = loss_mask[:, start:end, :].cpu().numpy()
            ground_truth_input_x_piccoor = (obs_traj_szene[:, :, 0].T)
            ground_truth_input_y_piccoor = (obs_traj_szene[:, :, 1].T)
            fig, ax = plt.subplots()
            ax.axis('off')
            timesteps = np.linspace(1, 0.1, config.pred_len)

            for i in range(end-start):
                pred_traj_gt_i = pred_traj_gt_szene[:,i]
                obs_traj_i = obs_traj_szene[:,i]
                model_output_traj_best_i = model_output_traj_best_szene[:,i]
                loss_mask_szene_i = loss_mask_szene[:,i]

                for t, timestep in enumerate(model_output_traj_best_i):
                    if loss_mask_szene_i[config.obs_len + t,0]:
                        gt_pos = plt.Circle(timestep, config.collision_distance/2., fill=False, color='b',
                                            label='Prediction', alpha=timesteps[t])
                        ax.add_artist(gt_pos)
                        pred_pos = plt.Circle(pred_traj_gt_i[t], config.collision_distance/2., fill=False, color='r',
                                              label='Ground Truth', alpha=timesteps[t])
                        ax.add_artist(pred_pos)
                        # if torch.is_tensor(samples):
                            # testi = samples_scene[t,i]
                        # plt.scatter(samples_scene[t,i,:,0],samples_scene[t,i,:,1],s=0.1,color='g', alpha=timesteps[t])
                            # samples_plot = plt.Circle([samples_scene[t,i,:,0],samples_scene[t,i,:,1]], config.collision_distance / 2., fill=False, color='g',
                            #                     label='samples_plot', alpha=timesteps[t])
                            # ax.add_artist(samples_plot)
                        if True in tmp_stack_of_coll_indeces[t,i]:
                            coll_pers_pos = plt.Circle(timestep, config.collision_distance/2., fill=False, color='green',
                                                       label='Collisions')
                            ax.add_artist(coll_pers_pos)
                for t, timestep in enumerate(obs_traj_i):
                    if loss_mask_szene_i[t,0]:
                        obs_pos = plt.Circle(timestep, config.collision_distance/2., fill=False, color='black',
                                             label='Traj. History', alpha=.5)
                        ax.add_artist(obs_pos)

                obs_traj_i_filtered = obs_traj_i[np.nonzero(loss_mask_szene_i[0:config.obs_len,0])[0]]
                mask_tmp = np.nonzero(loss_mask_szene_i[config.obs_len:(config.pred_len+ config.obs_len),0])[0]
                pred_traj_gt_i_filtered = pred_traj_gt_i[mask_tmp]
                model_output_traj_best_i_filtered = model_output_traj_best_i[mask_tmp]
                observed_line = plt.plot(obs_traj_i_filtered[:,0],obs_traj_i_filtered[:,1],"black",linewidth=2,
                                         label="History",alpha=.5)[0]

                observed_line.axes.annotate(
                    "",
                    xytext=(
                        ground_truth_input_x_piccoor[i, -2],
                        ground_truth_input_y_piccoor[i, -2],
                    ),
                    xy=(
                        ground_truth_input_x_piccoor[i, -1],
                        ground_truth_input_y_piccoor[i, -1],
                    ),
                    arrowprops=dict(
                        arrowstyle="-|>", color=observed_line.get_color(), lw=1
                    ),
                    size=15, alpha=1
                )

                plt.plot(
                    np.append(
                        obs_traj_i_filtered[-1,0],
                        pred_traj_gt_i_filtered[:, 0],
                    ),
                    np.append(
                        obs_traj_i_filtered[-1,1],
                        pred_traj_gt_i_filtered[:,1],
                    ),
                    "r",
                    linewidth=2,
                    alpha=.1
                )


                plt.plot(
                    np.append(
                        obs_traj_i_filtered[-1,0],
                        model_output_traj_best_i_filtered[:, 0],
                    ),
                    np.append(
                        obs_traj_i_filtered[-1,1],
                        model_output_traj_best_i_filtered[:,1],
                    ),
                    "b",
                    # ls="--",
                    linewidth=2,
                    alpha=.1
                )
            fig.suptitle(szene_id_batch_offset + k, fontsize=20)
            ax.set_aspect("equal")
            # plt.savefig(DIR + 'SGAN2'+ str(szene_id_batch_offset + k) +'.png', dpi=800,bbox_inches='tight')
            plt.show()

        plt.close()

def plot_goals(obs_traj, pred_traj_gt, batch_pred_traj_fake_rel, seq_start_end,samples):
    all_pred = torch.stack(batch_pred_traj_fake_rel)
    # samples = torch.stack(samples)
    for k, (start, end) in enumerate(seq_start_end):
        nmp_humans = end - start
        # if ids_of_col_szenes_fake[k]:  # and nmp_humans <= 5:               # plot scenes only with collisions
        if nmp_humans <= 5:
            obs_traj_szene = obs_traj[:, start:end, :].cpu().numpy()
            pred_traj_gt_szene = pred_traj_gt[:, start:end, :].cpu().numpy()
            all_pred_szene = all_pred[:, start:end, :].cpu().numpy()
            # samples_scene = samples[:, :, start:end].cpu().numpy()
            for i in range(end - start):
                obs_traj_i = obs_traj_szene[:, i]
                plt.plot(obs_traj_i[:, 0], obs_traj_i[:, 1], "black", linewidth=2, label="History", alpha=.5)[0]
                plt.plot(pred_traj_gt_szene[:,i, 0], pred_traj_gt_szene[:, i, 1], "red", linewidth=2, label="FD", alpha=1.)[0]
                for s in range(config.num_samples):
                    plt.scatter(all_pred_szene[s, i, 0], all_pred_szene[s, i, 1], s=1, color=human_colors[i % 5])
                # for s in range(20):
                #     for j in range(20):
                #         plt.scatter(samples_scene[s, j, i, 0], samples_scene[s, j, i, 1], s=1, color=human_colors[i % 5])
            plt.show()
    plt.close()

def plot_best_goal(obs_traj, pred_traj_gt, batch_pred_traj_fake_rel, att_score_list_batch, ids, stack_of_coll_indeces,
                   seq_start_end, ids_of_col_szenes_fake, loss_mask, config, szene_id, samples, goal):
    #cmap = plt.cm.get_cmap('hsv', 5)
    if config.num_samples > 1:
        all_pred = torch.stack(batch_pred_traj_fake_rel)
    szene_id_batch_offset = szene_id*len(seq_start_end)
    # search_ids = [18, 38, 25]
    for k, (start, end) in enumerate(seq_start_end):
        nmp_humans = end-start
        # if ids_of_col_szenes_fake[k]:  # and nmp_humans <= 5:               # plot scenes only with collisions
        if nmp_humans <= 5:                                              # plot scenes with only less than x humans
        # if szene_id_batch_offset + k in search_ids:                       # look for scenes with a specific id
            tmp_stack_of_coll_indeces = stack_of_coll_indeces[k].cpu().numpy()
            model_output_traj_best = batch_pred_traj_fake_rel[ids[k]]
            goal_best = goal[ids[k]]
            obs_traj_szene = obs_traj[:, start:end, :].cpu().numpy()
            #if torch.is_tensor(samples):
            # samples_scene = samples[ids[k]][:, start:end].cpu().numpy()
            pred_traj_gt_szene = pred_traj_gt[:, start:end, :].cpu().numpy()
            model_output_traj_best_szene = model_output_traj_best[:, start:end, :].cpu().numpy()
            goal_best_best_szene = goal_best[start:end, :].cpu().numpy()
            loss_mask_szene = loss_mask[:, start:end, :].cpu().numpy()
            ground_truth_input_x_piccoor = (obs_traj_szene[:, :, 0].T)
            ground_truth_input_y_piccoor = (obs_traj_szene[:, :, 1].T)
            fig, ax = plt.subplots()
            ax.axis('off')
            timesteps = np.linspace(1, 0.1, config.pred_len)

            for i in range(end-start):
                pred_traj_gt_i = pred_traj_gt_szene[:,i]
                obs_traj_i = obs_traj_szene[:,i]
                model_output_traj_best_i = model_output_traj_best_szene[:,i]
                goal_best_best_szene_i = goal_best_best_szene[i]
                loss_mask_szene_i = loss_mask_szene[:,i]

                for t, timestep in enumerate(model_output_traj_best_i):
                    if loss_mask_szene_i[config.obs_len + t,0]:
                        gt_pos = plt.Circle(timestep, config.collision_distance/2., fill=False, color='b',
                                            label='Prediction', alpha=timesteps[t])
                        ax.add_artist(gt_pos)
                        pred_pos = plt.Circle(pred_traj_gt_i[t], config.collision_distance/2., fill=False, color='r',
                                              label='Ground Truth', alpha=timesteps[t])
                        ax.add_artist(pred_pos)
                        # if torch.is_tensor(samples):
                            # testi = samples_scene[t,i]
                        # plt.scatter(samples_scene[t,i,:,0],samples_scene[t,i,:,1],s=0.1,color='g', alpha=timesteps[t])
                            # samples_plot = plt.Circle([samples_scene[t,i,:,0],samples_scene[t,i,:,1]], config.collision_distance / 2., fill=False, color='g',
                            #                     label='samples_plot', alpha=timesteps[t])
                            # ax.add_artist(samples_plot)
                        if True in tmp_stack_of_coll_indeces[t,i]:
                            coll_pers_pos = plt.Circle(timestep, config.collision_distance/2., fill=False, color='green',
                                                       label='Collisions')
                            ax.add_artist(coll_pers_pos)
                goal_plot = plt.Circle(goal_best_best_szene_i, config.collision_distance / 2., fill=False, color='purple',
                                           label='Goal')
                ax.add_artist(goal_plot)
                for t, timestep in enumerate(obs_traj_i):
                    if loss_mask_szene_i[t,0]:
                        obs_pos = plt.Circle(timestep, config.collision_distance/2., fill=False, color='black',
                                             label='Traj. History', alpha=.5)
                        ax.add_artist(obs_pos)

                obs_traj_i_filtered = obs_traj_i[np.nonzero(loss_mask_szene_i[0:config.obs_len,0])[0]]
                mask_tmp = np.nonzero(loss_mask_szene_i[config.obs_len:(config.pred_len+ config.obs_len),0])[0]
                pred_traj_gt_i_filtered = pred_traj_gt_i[mask_tmp]
                model_output_traj_best_i_filtered = model_output_traj_best_i[mask_tmp]
                observed_line = plt.plot(obs_traj_i_filtered[:,0],obs_traj_i_filtered[:,1],"black",linewidth=2,
                                         label="History",alpha=.5)[0]

                observed_line.axes.annotate(
                    "",
                    xytext=(
                        ground_truth_input_x_piccoor[i, -2],
                        ground_truth_input_y_piccoor[i, -2],
                    ),
                    xy=(
                        ground_truth_input_x_piccoor[i, -1],
                        ground_truth_input_y_piccoor[i, -1],
                    ),
                    arrowprops=dict(
                        arrowstyle="-|>", color=observed_line.get_color(), lw=1
                    ),
                    size=15, alpha=1
                )

                plt.plot(
                    np.append(
                        obs_traj_i_filtered[-1,0],
                        pred_traj_gt_i_filtered[:, 0],
                    ),
                    np.append(
                        obs_traj_i_filtered[-1,1],
                        pred_traj_gt_i_filtered[:,1],
                    ),
                    "r",
                    linewidth=2,
                    alpha=.1
                )


                plt.plot(
                    np.append(
                        obs_traj_i_filtered[-1,0],
                        model_output_traj_best_i_filtered[:, 0],
                    ),
                    np.append(
                        obs_traj_i_filtered[-1,1],
                        model_output_traj_best_i_filtered[:,1],
                    ),
                    "b",
                    # ls="--",
                    linewidth=2,
                    alpha=.1
                )
            fig.suptitle(szene_id_batch_offset + k,)
            ax.set_aspect("equal")
            # plt.savefig(DIR + 'SGAN2'+ str(szene_id_batch_offset + k) +'.png', dpi=800,bbox_inches='tight')
            plt.show()

        plt.close()

def plot_influencer(obs_traj, pred_traj_gt, batch_pred_traj_fake_rel, id_influencer,
                    temp_batch_inx, unconditioned_resuts, goals, min_id, max_id):
    batch= obs_traj.shape[1]
    fig, ax = plt.subplots()
    ax.axis('off')
    timesteps = np.linspace(1, 0.1, config.pred_len)
    j = 0
    for i in range(batch):
        inf_color = 'blue'
        if i == id_influencer:
            inf_color = 'green'
        pred_traj_gt_i = pred_traj_gt[:, i]
        obs_traj_i = obs_traj[:, i]
        model_output_traj_best_i = batch_pred_traj_fake_rel[:, :, i]
        if temp_batch_inx[i]:
            unconditioned_resuts_j = unconditioned_resuts[:, j]
            j += 1

        for t in range(unconditioned_resuts.shape[0]):
            for k, sample in enumerate(model_output_traj_best_i):
                if k == min_id or k == max_id:
                    alpha = 0.1
                    inf_color_min_max = inf_color
                    # if i == id_influencer:
                    if k == min_id:
                        inf_color_min_max = 'orange'
                        alpha = 1.
                    elif k == max_id:
                        inf_color_min_max = 'purple'
                        alpha =1.


                    cond = plt.Circle(sample[t], config.collision_distance / 2., fill=False, color=inf_color_min_max,
                                    label='cond', alpha=alpha)
                    ax.add_artist(cond)
            if temp_batch_inx[i]:
               uncond = plt.Circle(unconditioned_resuts_j[t], config.collision_distance / 2., fill=False, color='pink',
                                label='ucond', alpha=1.)
               ax.add_artist(uncond)

            # pred_pos = plt.Circle(pred_traj_gt_i[t], config.collision_distance / 2., fill=False, color=inf_color,
            #                       label='Ground Truth', alpha=timesteps[t])
            # ax.add_artist(pred_pos)
        for t, timestep in enumerate(obs_traj_i):
            obs_pos = plt.Circle(timestep, config.collision_distance / 2., fill=False, color='black',
                                 label='Traj. History', alpha=.2)
            ax.add_artist(obs_pos)

        goal = plt.Circle(goals[i], config.collision_distance / 2., fill=False, color='red',
                            label='Goal', alpha=1.)
        ax.add_artist(goal)
        # plt.plot(pred_traj_gt_i[:, 0], pred_traj_gt_i[:, 1], "red", linewidth=2,
        #          label="GT", alpha=.2)[0]
        plt.plot(obs_traj_i[:, 0], obs_traj_i[:, 1], "black", linewidth=2,
                                 label="History", alpha=.2)[0]

        ax.annotate(str(i), xy=(obs_traj_i[-1, 0], obs_traj_i[-1, 1]), xytext=(obs_traj_i[-1, 0] +.1, obs_traj_i[-1, 1]+.1)),
                    # arrowprops=dict(facecolor='black', shrink=0.5))

        for k, sample in enumerate(model_output_traj_best_i):
            if k == min_id or k == max_id:
                plt.plot(sample[:, 0], sample[:, 1], inf_color, linewidth=2,
                                         label="Prediction Conditioned", alpha=.2)[0]
        if temp_batch_inx[i]:
            plt.plot(unconditioned_resuts_j[:, 0], unconditioned_resuts_j[:, 1], 'pink', linewidth=2,
                                 label="Prediction Unconditioned", alpha=.2)[0]

    ax.set_aspect("equal")
    # plt.savefig(DIR + 'SGAN2'+ str(szene_id_batch_offset + k) +'.png', dpi=800,bbox_inches='tight')
    plt.show()


    plt.close()

human_colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple']
def plot_influencer_for_paper(obs_traj, pred_traj_gt, conditioned_pos, id_influencer,
                              ids_no_robot, unconditioned_pos, goals, id_min_max, szene_id):

    search_ids = [54, 100, 111, 602, 1051, 1070, 1104, 1320, 1335]
    if szene_id not in search_ids:
        return 0
    batch = obs_traj.shape[1]
    fig, ax = plt.subplots()
    ax.axis('off')
    timesteps = np.linspace(0.1, 1, config.pred_len)
    j = 0
    for i in range(batch):
        inf_color_cond = 'tab:green'
        pred_traj_gt_i = pred_traj_gt[:, i]
        obs_traj_i = obs_traj[:, i]
        conditioned_pos_i = conditioned_pos[id_min_max, :, i]
        if ids_no_robot[i]:
            inf_color_cond = 'tab:blue'
            unconditioned_resuts_j = unconditioned_pos[:, j]
            j += 1

        for t in range(unconditioned_pos.shape[0]):


            cond = plt.Circle(conditioned_pos_i[t], config.collision_distance / 2., fill=False,
                              color=inf_color_cond,
                              label='cond', alpha=timesteps[t])
            ax.add_artist(cond)


            if ids_no_robot[i]:

               rect = patches.Rectangle((unconditioned_resuts_j[t, 0] - 0.1, unconditioned_resuts_j[t, 1] - 0.1),
                                        0.2, 0.2, linewidth=1, fill=False, edgecolor='tab:red', alpha=timesteps[t],
                                        facecolor=None,label='ucond')
               ax.add_patch(rect)

        for t, timestep in enumerate(obs_traj_i):
            obs_pos = plt.Circle(timestep, config.collision_distance / 2., fill=False, color='black',
                                 label='Traj. History', alpha=.1)
            ax.add_artist(obs_pos)

        plt.plot(goals[i,0],goals[i,1], marker='X',color='black')

        plt.plot(obs_traj_i[:, 0], obs_traj_i[:, 1], "black", linewidth=2,
                                 label="History", alpha=.2)[0]

        ax.annotate(str(i), xy=(obs_traj_i[-1, 0], obs_traj_i[-1, 1]), xytext=(obs_traj_i[-1, 0] +.1, obs_traj_i[-1, 1]+.1)),
                    # arrowprops=dict(facecolor='black', shrink=0.5))


        if ids_no_robot[i]:
            plt.plot(unconditioned_resuts_j[:, 0], unconditioned_resuts_j[:, 1], 'tab:red', linewidth=2,
                                 label="Prediction Unconditioned", alpha=.2)[0]

    fig.suptitle(szene_id, fontsize=20)
    ax.set_aspect("equal")
    plt.savefig(DIR + 'SI_Max'+ str(szene_id) +'.pdf', dpi=800,bbox_inches='tight')
    plt.show()
    plt.close()
