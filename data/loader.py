from torch.utils.data import DataLoader
from data.trajectories_trajnet import TrajectoryDatasetTrajnet, CollateTrajnet
from data.trajectories_full import TrajectoryDataset, Collate


def data_loader(args, path, augment = False):
    try:
        args.trajnet == True
    except:
        args.trajnet = False

    if args.trajnet:
        dset = TrajectoryDatasetTrajnet(
            path,
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            skip=args.skip,
            delim=args.delim)
        collate = CollateTrajnet(augment,args.dataset_name)
        loader = DataLoader(
            dset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.loader_num_workers,
            collate_fn=collate.seq_collate,
            pin_memory=False)
    else:
        dset = TrajectoryDataset(
            path,
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            skip=args.skip,
            delim=args.delim)
        collate = Collate(augment, args.dataset_name)
        loader = DataLoader(
            dset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.loader_num_workers,
            collate_fn=collate.seq_collate,
            pin_memory=False)

    return dset, loader
