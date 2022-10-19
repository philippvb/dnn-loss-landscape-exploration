
import os
import json
import pickle


import numpy as np
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import uuid

import pprint
import pandas as pd
import sys


import datasets
import optimizers
import models
import metrics
import model_configurations
import copy







def train(experiment_dictionary, save_directory_base, data_directory, name=None):

    gpu_exists = torch.cuda.is_available()
    name = name if name else str(uuid.uuid1())
    save_directory = os.path.join(save_directory_base, name)

    os.makedirs(save_directory, exist_ok=False)

    # init Tensorboard 
    writer = SummaryWriter(save_directory)


    # save configuration
    with open(os.path.join(save_directory, 'exp_dict.json'), 'w') as outfile:
        json.dump(experiment_dictionary, outfile)
    pprint.pprint(experiment_dictionary)
    print('Experiment saved in %s' % save_directory)


    # set seeds
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)

    # load datasets
    train_set = datasets.getDataSet(experiment_dictionary['dataset'], data_directory, train_flag=True)

    train_loader = torch.utils.data.DataLoader(train_set, drop_last=True, shuffle=True, batch_size=experiment_dictionary['batch_size'])

    val_set = datasets.getDataSet(experiment_dictionary['dataset'], data_directory, train_flag=False)


    # load model and optimizer
    model = models.get_Model(experiment_dictionary['model'], experiment_dictionary['dataset'])
    if gpu_exists:
        model = nn.DataParallel(model)
        model.cuda()

    opt = optimizers.get_optimizer(experiment_dictionary['optimizer'], model.parameters())

    scheduler = None
    scheduler_name = experiment_dictionary['optimizer'].get("scheduler", None)

    if scheduler_name is not None:
        if scheduler_name is "step":
            scheduler = torch.optim.lr_scheduler.StepLR(opt, experiment_dictionary["optimizer"].get("step"), 0.1)

        if scheduler_name is "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=experiment_dictionary["optimizer"].get("cycle"))

    loss_function = metrics.get_metric(experiment_dictionary['loss_func'])



    minimum_list = []
    model_list = []
    model_list.append(model)

    minimum_0 = create_minimum(model)


    # create checkpoint
    model_path = os.path.join(save_directory, 'model.pth')
    opt_path = os.path.join(save_directory, 'opt_state_dict.pth')
    score_list_path = os.path.join(save_directory, 'score_list.pkl')

   
    score_list = []
    initial_loss = metrics.compute_metric_on_dataset(model, train_set,
                                                metric_dict=experiment_dictionary["loss_func"], cuda=gpu_exists)

    initial_accuracy = metrics.compute_metric_on_dataset(model, val_set,
                                                        metric_dict=experiment_dictionary["acc_func"], cuda=gpu_exists)

    writer.add_scalar('Train_loss', initial_loss, 0)
    writer.add_scalar('Validation_accuracy', initial_accuracy, 0)
    writer.add_scalar('Ensemble_accuracy', initial_accuracy, 0)
    writer.flush()

    score_dict = {"epoch": 0}
    score_dict["train_loss"] = initial_loss
    score_dict["val_acc"] = initial_accuracy
    score_dict["batch_size"] =  train_loader.batch_size
    score_list += [score_dict]
    next_epoch = 1





    print('Starting experiment at epoch %d/%d' % (next_epoch, experiment_dictionary['max_epochs']))

    for epoch in range(next_epoch, experiment_dictionary['max_epochs'] + 1):

        print('\n\n Starting epoch %d' %epoch)
        gradient_size=0

        # start training
        model.train()
        print("Epoch %d - Training model with %s..." % (epoch, experiment_dictionary["loss_func"].get("name")))
        print("Current learn rate is %f" % get_lr(opt))

        start_time = time.time()
        for images,labels in train_loader:#tqdm.tqdm(train_loader):
            if gpu_exists:
                images, labels = images.cuda(), labels.cuda()
            
            opt.zero_grad()
            loss = loss_function(model, images, labels, minimum_list)
            loss.backward()
            gradient_size+=metrics.computegradientsize(model)
            opt.step()

        end_time = time.time()


        # evaluate model
        
        train_loss = metrics.compute_metric_on_dataset(model, train_set,
                                                metric_dict=experiment_dictionary["loss_func"], cuda=gpu_exists)

        val_acc = metrics.compute_metric_on_dataset(model, val_set,
                                                        metric_dict=experiment_dictionary["acc_func"], cuda=gpu_exists)

        ensemble_acc = metrics.ensemble_accuracy(model_list, val_set, cuda=gpu_exists)

        with torch.no_grad():
            distance_list = [metrics.computedistance(minimum, model).data for minimum in minimum_list]
            total_distance = np.sum(distance_list)
            print('Current distance to last minimum checkpoint: %f' % total_distance)



        # log evaluation
        
        writer.add_scalar('Train_loss', train_loss, epoch)
        writer.add_scalar('Validation_accuracy', val_acc, epoch)
        writer.add_scalar('Ensemble_accuracy', ensemble_acc, epoch)
        writer.add_scalar('train_epoch_time', end_time - start_time, epoch)
        writer.add_scalar('lr', get_lr(opt), epoch)
        writer.add_scalar('Gradient_size', gradient_size, epoch)
        writer.add_scalar('Distance_to_0', metrics.computedistance(minimum_0, model).data, epoch)
        for distance, i in zip(distance_list, range(len(distance_list))):
            writer.add_scalar('distance' + str(i), distance, epoch)
            writer.add_scalar('similarity' + str(i), torch.exp(-1* experiment_dictionary["loss_func"].get("width") * distance), epoch)

        writer.flush()


        score_dict = {"epoch": epoch}
        score_dict["train_loss"] = train_loss
        score_dict["val_acc"] = val_acc
        score_dict["batch_size"] =  train_loader.batch_size
        score_dict["train_epoch_time"] = end_time - start_time
        #score_dict["distance"] = distance
        score_list += [score_dict]

        with open(score_list_path, 'wb') as outfile:
            pickle.dump(score_list, outfile)
        torch.save(model.state_dict(), model_path)
        torch.save(opt.state_dict(), opt_path)


        # prepare for next epoch
        if (epoch-next_epoch+1) % experiment_dictionary["loss_func"].get("step") == 0 and epoch is not 1:
            
            # save state for ensemble
            model_name = 'model' + str(epoch) + '.pth'
            model_checkpoint_path = os.path.join(save_directory, model_name)
            torch.save(model.state_dict(), model_checkpoint_path)
            newmodel = copy.deepcopy(model)
            newmodel.cuda()
            model_list.append(newmodel)

            # add current value as minimum
            new_minimum = create_minimum(model)
            # minimum_list.clear()
            minimum_list.append(new_minimum)
            

        if scheduler:
            scheduler.step()
        
        if experiment_dictionary["optimizer"].get("scheduler") is "step" and (epoch-next_epoch+1) % experiment_dictionary["optimizer"].get("cycle") == 0:
                set_lr(opt, experiment_dictionary['optimizer'].get("lr"))


        # flush printer
        sys.stdout.flush()
        

    print('Experiment completed')
    print('Printing summary')
    print(pd.DataFrame(score_list))




# functions

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def create_minimum(model):
    minimum = []
    for param in model.parameters():
        newparams = param.clone().detach().to(device='cuda') #use clone to clone and detach to clear gradient
        minimum.append(newparams)

    return minimum


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
            param_group['lr'] = lr




if __name__ == '__main__':
        parser = argparse.ArgumentParser()

        parser.add_argument('-n', '--name')
        parser.add_argument('-sb', '--savedir_base', default='./results')
        parser.add_argument('-d', '--datadir', default='./data')
        parser.add_argument('-en', '--experiment_name', required=True, nargs='+')

        args = parser.parse_args()

        for experiment in args.experiment_name:
            train(model_configurations.EXP_GROUPS[experiment], 
            args.savedir_base, 
            args.datadir,
            args.name)