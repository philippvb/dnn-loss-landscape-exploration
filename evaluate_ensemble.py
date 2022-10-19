import torch 
import models
import os
import torch.nn as nn
import datasets
import numpy as np
from collections import Counter
import argparse


@torch.no_grad()
def evaluate_ensemble(model_path, dataset, ensemble):
    model_list = []

    if ensemble:
        for name_extension in range(150, 751, 150):
            model_name = os.path.join(model_path[0], 'model' + str(name_extension) + '.pth')
            model =  nn.DataParallel(models.get_Model('mobileNetV2', 'CIFAR-10'))
            model.load_state_dict(torch.load(model_name))
            model.eval()
            model_list.append(model)
        print(('Loaded {} ensemble models from ' + model_path[0]).format(len(model_list)))

    else:
        model1_name = os.path.join(model_path[0], 'model.pth')
        model1 =  nn.DataParallel(models.get_Model('mobileNetV2', 'CIFAR-10'))
        model1.load_state_dict(torch.load(model1_name))
        model_list.append(model1)
        print('Loaded first model from ' + model_path[0])

        model2_name = os.path.join(model_path[1], 'model.pth')
        model2 =  nn.DataParallel(models.get_Model('mobileNetV2', 'CIFAR-10'))
        model2.load_state_dict(torch.load(model2_name))
        model_list.append(model2)
        print('Loaded second model from ' + model_path[1])


    loader = torch.utils.data.DataLoader(dataset, drop_last=False, batch_size=1024) 
    
    same_prediction_list = np.array([])

    
    for images, labels in loader:
        prediction_list = np.zeros([0, images.shape[0]]) # np.array([])
        for model in model_list:
                prediction = model(images).argmax(dim=1).cpu().numpy()
                prediction_list = np.append(prediction_list, [prediction], axis=0)
            
        same_prediction = [Counter(prediction_list[:, i]).most_common(1)[0][1] for i in range(images.shape[0])]
        same_prediction_list = np.append(same_prediction_list, same_prediction)
            
    prediction_table = Counter(same_prediction_list)
    print(prediction_table)



if __name__ == '__main__':
        parser = argparse.ArgumentParser()

        parser.add_argument('-nd', '--networkdir', nargs='+')
        parser.add_argument('-ens', '--ensemble', default=False)
        parser.add_argument('-d', '--datadir', default='./data')

        args = parser.parse_args()

        evaluate_ensemble(args.networkdir, datasets.getDataSet('cifar10', args.datadir, train_flag=False), args.ensemble)