import random
import os
import sys
import argparse
import copy
import shutil
import json
import logging
import yaml
import pickle
from pprint import pprint
from datetime import datetime
from functools import partial
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from icecream import ic
from collections import defaultdict
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import torch_geometric

import tqdm

import wandb

# Change working directory to project's main directory, and add it to path - for library and config usages
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

# Project dependencies
from dataloaders import get_data_loaders
from utils.log import write_checkpoint, load_config, load_checkpoint, config_logging, save_config, print_model_summary, get_terminal_columns, center_text, make_table, numeric_runtime

class ArgDict:
    pass

DEVICE = 'cuda:0'

def parse_args():
    """
    Define and retrieve command line arguements
    :return: argparser instance
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config', default='configs/cluster.yaml')
    argparser.add_argument('-g', '--gpu', default='0', help='The gpu to run on')
    argparser.add_argument('--auto', action='store_true')
    argparser.add_argument('--save', dest='save', action='store_true', help='Whether to save all to disk')
    argparser.add_argument('--no_save', dest='save', action='store_false')
    argparser.set_defaults(save=True, debug_load=False)
    argparser.add_argument('-v', '--verbose', action='store_true')
    argparser.add_argument('--show-config', action='store_true')
    argparser.add_argument('--resume', action='store_true', default=0, help='Resume from last checkpoint')
    
    # Logging
    argparser.add_argument('--name', type=str, default=None,
            help="Run name")
    argparser.add_argument('--use_wandb', action='store_true',
                        help="use wandb project name")
    argparser.add_argument('--skip_wandb_init', action='store_true',
                        help="Skip wandb initialization (helpful if wandb was pre-initialized)")
    argparser.add_argument('--log_interval', type=int, default=25,
            help="Number of steps between logging key stats")
    argparser.add_argument('--print_interval', type=int, default=250,
            help="Number of steps between printing key stats")

    # Early Stopping
    argparser.add_argument('--early_stopping', action='store_true')
    argparser.set_defaults(early_stopping=False)
    argparser.add_argument('--early_stopping_accuracy', type=float, default=0.65)
    argparser.add_argument('--early_stopping_epoch', type=int, default=1)

    args = argparser.parse_args()

    return args

def calc_metrics(trigger, pred, accum_info):
    with torch.no_grad():
        tp = torch.sum((trigger == 1) * (pred >= 0)).item()
        tn = torch.sum((trigger == 0) * (pred < 0)).item()
        fp = torch.sum((trigger == 0) * (pred >= 0)).item()
        fn = torch.sum((trigger == 1) * (pred < 0)).item()

        accum_info['true_positives'] += tp
        accum_info['true_negatives'] += tn
        accum_info['false_positives'] += fp
        accum_info['false_negatives'] += fn

    return accum_info


def train(data, model, loss_params, optimizer, epoch, output_dir):
    train_info = do_epoch(data, model, loss_params, epoch, optimizer=optimizer)
    write_checkpoint(checkpoint_id=epoch, model=model, optimizer=optimizer, output_dir=output_dir)
    return train_info

def evaluate(data, model, loss_params, epoch):
    with torch.no_grad():
        val_info = do_epoch(data, model, loss_params, epoch, optimizer=None)
    return val_info

def do_epoch(data, model, loss_params, epoch, optimizer=None):
    if optimizer is None:
        # validation epoch
        model.eval()
    else:
        # train epoch
        model.train()

    start_time = datetime.now()

    # Iterate over batches
    accum_info = {k: 0.0 for k in (
        'ri', 
        'loss',
        'loss_ce', 
        'fscore', 
        'precision', 
        'recall', 
        'true_positives',
        'true_negatives',
        'false_positives',
        'false_negatives'
    )}

    num_insts = 0
    skipped_batches = 0
    total_size = 0
    total_selected = 0
    for batch in tqdm.tqdm(data, smoothing=0.0):
        batch = batch.to(DEVICE)
        batch_size = batch.x.shape[0]


        loss = 0
        pred = model(batch)
        ce_loss = F.binary_cross_entropy_with_logits(pred, batch.y.to(torch.float))
        loss += ce_loss
        if torch.isnan(loss):
            print(f'{loss=}')
            print(f'{torch.any(torch.isnan(pred))=}')
            import sys
            sys.exit(0)

        accum_info['loss_ce'] += ce_loss.item() * batch_size


        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accum_info = calc_metrics(batch.y, pred, accum_info)
        accum_info['loss'] += loss.item()
        num_insts += batch_size

    tp = accum_info['true_positives']
    tn = accum_info['true_negatives']
    fp = accum_info['false_positives']
    fn = accum_info['false_negatives']

    if num_insts > 0:
        accum_info['loss'] /= num_insts
        accum_info['loss_ce'] /= num_insts
        accum_info['ri'] = (tp + tn)/(tp + tn + fp + fn)
        accum_info['precision'] = tp / (tp + fp) if tp + fp != 0 else 0
        accum_info['recall'] = tp / (tp + fn) if tp + fn != 0 else 0
        accum_info['fscore'] = (2 * tp)/(2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
           
    accum_info['run_time'] = datetime.now() - start_time
    accum_info['run_time'] = str(accum_info['run_time']).split(".")[0]

    print('Skipped batches:', skipped_batches)


    return accum_info

def main():
     # Parse the command line
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    execute_training(args, config)

def execute_training(args, config):
    global DEVICE

    start_time = datetime.now()
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    config['output_dir'] = os.path.join(config['output_dir'], f'experiment_{start_time:%Y-%m-%d_%H:%M:%S}')
    os.makedirs(config['output_dir'], exist_ok=True)

    # Setup logging
    file_handler = config_logging(verbose=args.verbose, output_dir=config['output_dir'],
                   append=args.resume, rank=0)

    logging.info('Command line config: %s' % args)
    logging.info('Configuration: %s', config)
    logging.info('Saving job outputs to %s', config['output_dir'])

    # Save configuration in the outptut directory
    save_config(config)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # uncomment only for CUDA error debugging
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    torch.cuda.set_device(int(args.gpu))
    DEVICE = 'cuda:' + str(args.gpu)

    name = config['wandb']['run_name'] + f'-experiment_{start_time:%Y-%m-%d_%H:%M:%S}'
    logging.info(name)

    if args.use_wandb and not args.skip_wandb_init:
        wandb.init(
            project=config['wandb']['project_name'],
            name=name,
            tags=config['wandb']['tags'],
            config=config
        )

    # Load data
    logging.info('Loading training, validation, and test data')
    dconfig = copy.copy(config['data'])

    train_data, val_data, test_data = get_data_loaders(**dconfig)
    logging.info('Loaded %g training samples', len(train_data.dataset))
    logging.info('Loaded %g validation samples', len(val_data.dataset))
    logging.info('Loaded %g test samples', len(test_data.dataset))

    mconfig = copy.copy(config['model'])
    del mconfig['name']
    from models.gat import GAT as Model

    model = Model(
        **mconfig
    )
    model = model.to(DEVICE)

    # Optimizer
    oconfig = config['optimizer']
    params = model.parameters()
    if config['optimizer']['type'] == 'Adam':
        optimizer = torch.optim.Adam(params=params,
                lr=oconfig['learning_rate'], 
                weight_decay=oconfig['weight_decay'], 
                betas=[oconfig['beta_1'], oconfig['beta_2']],
                eps=oconfig['eps']
        )
    elif oconfig['type'] == 'SGD':
        optimizer = torch.optim.SGD(params=params, lr=oconfig['learning_rate'], momentum=oconfig['momentum'], weight_decay=oconfig['weight_decay'])
    else:
        raise NotImplementedError(f'Optimizer {config["optimizer"]["type"]} not implemented.')


    decay_rate = oconfig["learning_rate_decay_rate"]
    def lr_schedule(epoch):
        return decay_rate**epoch

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    print_model_summary(model)
    model = model.to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'The number of model parameters is {num_params}')

    # Metrics
    train_loss = np.empty(config['epochs'], float)
    train_ri = np.empty(config['epochs'], float)
    val_loss = np.empty(config['epochs'], float)
    val_ri = np.empty(config['epochs'], float)

    best_epoch = -1
    best_val_ri = 0
    best_classifier_val_ri = 0
    best_model = None
    for epoch in range(1, config['epochs'] + 1):
        train_info = train(train_data, model, {}, optimizer, epoch, config['output_dir'])
        table = make_table(
            ('Total loss', f"{train_info['loss']:.6f}"),
            ('CE loss', f"{train_info['loss_ce']:.6f}"),
            ('Rand Index', f"{train_info['ri']:.6f}"),
            ('F-score', f"{train_info['fscore']:.4f}"),
            ('Recall', f"{train_info['recall']:.4f}"),
            ('Precision', f"{train_info['precision']:.4f}"),
            ('True Positives', f"{train_info['true_positives']}"),
            ('False Positives', f"{train_info['false_positives']}"),
            ('True Negatives', f"{train_info['true_negatives']}"),
            ('False Negatives', f"{train_info['false_negatives']}"),
            ('Runtime', f"{train_info['run_time']}")
        )

        logging.info('\n'.join((
            '',
            "#" * get_terminal_columns(),
            center_text(f"Training - {epoch:4}", ' '),
            table
        )))

        train_loss[epoch-1], train_ri[epoch-1] = train_info['loss'], train_info['ri']
        if args.use_wandb:
            wandb.log({"Train Loss" : train_info['loss']}, step=epoch)
            wandb.log({"Train CE Loss" : train_info['loss_ce']}, step=epoch)
            wandb.log({"Train Accuracy" : train_info['ri']}, step=epoch)
            wandb.log({"Train Precision" : train_info['precision']}, step=epoch)
            wandb.log({"Train Recall": train_info['recall']}, step=epoch)
            wandb.log({"Train F-score": train_info['fscore']}, step=epoch)
            wandb.log({"Train Run-Time": numeric_runtime(train_info['run_time'])}, step=epoch)

        val_info = evaluate(val_data, model, {}, epoch)
        table = make_table(
            ('Total loss', f"{val_info['loss']:.6f}"),
            ('CE loss', f"{val_info['loss_ce']:.6f}"),
            ('Rand Index', f"{val_info['ri']:.6f}"),
            ('F-score', f"{val_info['fscore']:.4f}"),
            ('Recall', f"{val_info['recall']:.4f}"),
            ('Precision', f"{val_info['precision']:.4f}"),
            ('True Positives', f"{val_info['true_positives']}"),
            ('False Positives', f"{val_info['false_positives']}"),
            ('True Negatives', f"{val_info['true_negatives']}"),
            ('False Negatives', f"{val_info['false_negatives']}"),
            ('Runtime', f"{val_info['run_time']}")
        )

        logging.info('\n'.join((
            '',
            center_text(f"Validation - {epoch:4}", ' '),
            table
            )))

        if val_info['ri'] > best_classifier_val_ri:
            best_classifier_val_ri = val_info['ri']
            best_model = copy.deepcopy(model)

        if val_info['ri'] > best_val_ri:
            best_val_ri = val_info['ri']
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        val_loss[epoch-1], val_ri[epoch-1] = val_info['loss'], val_info['ri']
        if args.use_wandb:
            wandb.log({"Validation Loss" : val_info['loss']}, step=epoch)
            wandb.log({"Validation CE Loss" : val_info['loss_ce']}, step=epoch)
            wandb.log({"Validation Accuracy" : val_info['ri']}, step=epoch)
            wandb.log({"Validation Precision" : val_info['precision']}, step=epoch)
            wandb.log({"Validation Recall": val_info['recall']}, step=epoch)
            wandb.log({"Validation F-Score": val_info['fscore']}, step=epoch)
            wandb.log({"Validation Run-Time": numeric_runtime(val_info['run_time'])}, step=epoch)
            wandb.log({"Best Classifier Validation Accuracy": best_classifier_val_ri}, step=epoch)
            wandb.log({"Best Validation Accuracy": best_val_ri}, step=epoch)


        if args.early_stopping and epoch >= args.early_stopping_epoch and best_classifier_val_ri < args.early_stopping_accuracy:
            break

        lr_scheduler.step()
        
    
    del train_data, val_data


    logging.info(f'Best validation accuracy: {best_val_ri:.4f}, best epoch: {best_epoch}.')
    logging.info(f'Training runtime: {str(datetime.now() - start_time).split(".")[0]}')

    test_info = evaluate(test_data, best_model, {}, config['epochs'] + 1)
    table = make_table(
        ('Total loss', f"{test_info['loss']:.6f}"),
        ('CE loss', f"{test_info['loss_ce']:.6f}"),
        ('Rand Index', f"{test_info['ri']:.6f}"),
        ('F-score', f"{test_info['fscore']:.4f}"),
        ('Recall', f"{test_info['recall']:.4f}"),
        ('Precision', f"{test_info['precision']:.4f}"),
        ('True Positives', f"{test_info['true_positives']}"),
        ('False Positives', f"{test_info['false_positives']}"),
        ('True Negatives', f"{test_info['true_negatives']}"),
        ('False Negatives', f"{test_info['false_negatives']}"),
        ('Runtime', f"{test_info['run_time']}")
    )
    logging.info('\n'.join((
        '',
        center_text(f"Test", ' '),
        table
        )))


    if args.use_wandb:
        wandb.log({"Test Loss" : test_info['loss']}, step=config['epochs'] + 1)
        wandb.log({"Test CE Loss" : test_info['loss_ce']})
        wandb.log({"Test Accuracy" : test_info['ri']})
        wandb.log({"Test Precision" : test_info['precision']})
        wandb.log({"Test Recall": test_info['recall']})
        wandb.log({"Test F-Score": test_info['fscore']})
        wandb.log({"Test Run-Time": numeric_runtime(test_info['run_time'])})

    # Saving to disk
    if args.save:
        output_dir = os.path.join(config['output_dir'], 'summary')
        i = 0
        while True:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)  # raises error if dir already exists
                break
            i += 1
            output_dir = output_dir[:-1] + str(i)
            if i > 9:
                logging.info(f'Cannot save results on disk. (tried to save as {output_dir})')
                return

        logging.info(f'Saving all to {output_dir}')
        torch.save(best_model.state_dict(), os.path.join(output_dir, "exp_model.pt"))
        shutil.copyfile(__file__, os.path.join(output_dir, 'main.py'))
        shutil.copytree('models/', os.path.join(output_dir, 'models/'))
        results_dict = {'train_loss': train_loss,
                        'train_ri': train_ri,
                        'val_loss': val_loss,
                        'val_ri': val_ri}
        df = pd.DataFrame(results_dict)
        df.index.name = 'epochs'
        df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
        best_dict = {'best_val_ri': best_val_ri, 'best_epoch': best_epoch}
        best_df = pd.DataFrame(best_dict, index=[0])
        best_df.to_csv(os.path.join(output_dir, "best_val_results.csv"), index=False)

    logging.shutdown()



if __name__ == '__main__':
    main()
