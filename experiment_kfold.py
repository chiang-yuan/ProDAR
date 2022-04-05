import os
import json
import random
import argparse
import numpy as np

import torch
from torch.utils.data import SubsetRandomSampler

from torch_geometric.loader import DataLoader

import prodar
from prodar import ProDAR

from sklearn.model_selection import KFold


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', type=str, choices=prodar.datasets.keys())
    parser.add_argument('model', type=str, choices=prodar.models.keys())
    parser.add_argument('nfolds', type=int)

    # model architecture

    parser.add_argument('--num-layers', dest="num_layers", type=int, default=5)
    parser.add_argument('--dim-node-embeddings', dest="dim_node_embedding", type=int, default=256)
    parser.add_argument('--dim-graph-embeddings', dest="dim_graph_embedding", type=int, default=512)
    parser.add_argument('--dim-pers-embeddings', dest="dim_pers_embedding", type=int, default=512)

    parser.add_argument('--aggregation', dest="aggregation")
    parser.add_argument('--dropout', dest="dropout", type=float, default=0.1)
    
    ## GAT
    parser.add_argument('--heads', dest="heads", type=int, default=1)

    # training configuration

    parser.add_argument('--batch-size', dest="batch_size", type=int, default=64)
    parser.add_argument('--epochs', dest="epochs", type=int, default=300)

    parser.add_argument('--seed', dest="seed", type=int, default=12345)
    parser.add_argument('--device', dest="device", default='cuda')
    parser.add_argument('--num-workers', dest="num_workers", type=int, default=4)

    ## optimizer
    parser.add_argument('--optimizer', dest="optimizer", choices=prodar.optimizers.keys(), default='Adam')

    parser.add_argument('--lr', dest="lr", type=float, default=1e-5)
    parser.add_argument('--weight-decay', dest="weight_decay", type=float, default=1e-5)

    # SDG
    parser.add_argument('--momentum', dest="momentum", type=float, default=0.9)

    ## scheduler
    parser.add_argument('--scheduler', dest="scheduler", choices=prodar.schedulers.keys(), default=None)

    ### ReduceLROnPlateau
    parser.add_argument('--patience', dest="patience", type=int, default=10)
    parser.add_argument('--cooldown', dest="cooldown", type=int, default=0)

    ### CosAnnealWR
    parser.add_argument('--t0', dest="t0", type=int, default=10)
    parser.add_argument('--tmult', dest="tmult", type=int, default=2)

    ## earlystopper
    parser.add_argument('--earlystopping', dest="earlystopping", action='store_true')
    parser.add_argument('--stop-patience', dest="stop_patience", default=10)

    
    parser.add_argument('--demo', dest="demo", action='store_true')
    parser.add_argument('--demo-num', dest="demo_num", type=int, default=1000)
    
    config = parser.parse_args()

    print("Initialization...")

    print(f'\t{config}')

    device = torch.device('cuda' if torch.cuda.is_available() and config.device == 'cuda' else 'cpu')

    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)

    # torch.backends.cudnn.enabled = False
    # torch.backends.cudnn.benchmark = False
    # # torch.backends.cudnn.deterministic = True
    # torch.set_deterministic(True)
    # torch.use_deterministic_algorithms(True)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(0)


    print("Loading ProDAR...")

    model = ProDAR(
        # dataset=config.dataset, 
        # model=config.model, device=config.device, 
        # num_layers=config.num_layers, 
        # dim_node_embedding=config.dim_node_embedding, 
        # dim_graph_embedding=config.dim_graph_embedding, 
        # dim_pers_embedding=config.dim_pers_embedding,
        **vars(config)
    )

    dataset = model.dataset.shuffle()

    train_dataset = dataset[:int(len(dataset)*0.9)]

    # valid_dataset = dataset[int(len(dataset)*0.8):int(len(dataset)*0.9)]

    test_dataset = dataset[int(len(dataset)*0.9):]

    kfold = KFold(n_splits=config.nfolds, shuffle=True)

    # print(f'Initiate {nfolds:d}-fold cross validation...')

    for fold, (train_ids, valid_ids) in enumerate(kfold.split(train_dataset)):

        model = ProDAR(
            # dataset=config.dataset, 
            # model=config.model, device=config.device, 
            # num_layers=config.num_layers, 
            # dim_node_embedding=config.dim_node_embedding, 
            # dim_graph_embedding=config.dim_graph_embedding, 
            # dim_pers_embedding=config.dim_pers_embedding,
            **vars(config)
        )

        print(f'Start fold: {fold+1:d}...')
        train_subsampler = SubsetRandomSampler(train_ids)
        valid_subsampler = SubsetRandomSampler(valid_ids)

        train_loader = DataLoader(train_dataset, 
            batch_size=config.batch_size, 
            num_workers=config.num_workers,
            worker_init_fn=seed_worker,
            generator=g, 
            sampler=train_subsampler
        )
        valid_loader = DataLoader(train_dataset,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            sampler=valid_subsampler
        )

        foldername = "{}{}_K{}_N{}_G{}_P{}".format(
            config.model,
            config.aggregation,
            config.num_layers,
            config.dim_node_embedding,
            config.dim_graph_embedding,
            config.dim_pers_embedding
        )

        if config.dim_pers_embedding == 0:
            history_dir = os.path.join('history', config.dataset, foldername)
        else:
            history_dir = os.path.join('history', f'{config.dataset}-pers', foldername)
        
        if not os.path.exists(history_dir):
            os.makedirs(history_dir, exist_ok=True)


        best_valid_loss = -1
        stop_wait = 0

        rootname = 'kfold_{}-{}'.format(
            fold+1, config.nfolds
        )

        with open(f'{history_dir}/{rootname}.args', 'w') as f:
            json.dump(vars(config), f, indent=4)

        with open(f'{history_dir}/{rootname}.history', 'w') as f, open(f'{history_dir}/{rootname}.lr', 'w') as l:
            f.write(f"{'epoch':5s}\t {'train_loss':10s}\t {'valid_loss':10s}\t {'train_acc':10s}\t {'valid_acc':10s}\t {'train_ppv':10s}\t {'train_tpr':10s}\t {'train_tnr':10s}\t {'valid_ppv':10s}\t {'valid_tpr':10s}\t {'valid_tnr':10s}\n")
            l.write(f'epoch\t lr\n')
            for epoch in range(1, config.epochs+1):

                if config.scheduler is not None:
                    model.set_optimizer(**vars(config))
                    model.set_scheduler(**vars(config))
                    l.write(f'{epoch:5d}\t {model.scheduler.get_last_lr()[0]:10.5e}\n')
                    l.flush()
                else:
                    l.write(f'{epoch:5d}\t {config.lr:10.5e}\n')
                    l.flush()

                train_loss, train_ppv, train_tpr, train_tnr = model.train(
                    train_loader, epoch, **vars(config))

                valid_loss, valid_ppv, valid_tpr, valid_tnr = model.valid(
                    valid_loader, **vars(config))
                    
                train_acc = (train_tpr + train_tnr)/2
                valid_acc = (valid_tpr + valid_tnr)/2
 
                if config.scheduler is not None and config.scheduler == 'ReduceLROnPlateau':
                    model.scheduler.step(valid_loss)

                if best_valid_loss == -1:
                    best_valid_loss = valid_loss
                else:
                    if valid_loss <= best_valid_loss:
                        model.save(f'{history_dir}/{rootname}-model.pt')
                        best_valid_loss = valid_loss
                        stop_wait = 0
                    else:
                        if config.earlystopping:
                            if stop_wait > config.stop_patience:
                                break
                            stop_wait += 1

                print(f'Epoch: {epoch:03d}, Train loss: {train_loss:.4f}, Valid loss: {valid_loss:.4f}, Train acc: {train_acc:.4f}, Valid acc: {valid_acc:.4f}')
                f.write(f'{epoch:5d}\t {train_loss:10.5e}\t {valid_loss:10.5e}\t {train_acc:10.5e}\t {valid_acc:10.5e}\t {train_ppv:10.5e}\t {train_tpr:10.5e}\t {train_tnr:10.5e}\t {valid_ppv:10.5e}\t {valid_tpr:10.5e}\t {valid_tnr:10.5e}\n')
                f.flush()



