import torch 
from torch.utils.data import DataLoader

def get_metric(metric_dict, eval_flag=False):
    loss_function = None
    name = metric_dict.get("name")
    distance = metric_dict.get("distance", False)
    distance_factor = metric_dict.get("factor", 1)
    distance_width = metric_dict.get("width", 0.01)


    if name == "softmax_accuracy":
        loss_function =  softmax_accuracy

    elif name == "softmax_loss":
        loss_function =  softmax_loss

    if distance and not eval_flag:
        print("used distance function")
        if metric_dict.get("distance_func") is "tanh":
            return distance_wrapper(loss_function, distance_factor, distance_width, Tanh_kernel)
        else:
            return distance_wrapper(loss_function, distance_factor, distance_width, RBF_kernel)
    else:
        return remove_distance(loss_function)

    



@torch.no_grad()
def compute_metric_on_dataset(model, dataset, metric_dict, cuda):
    metric_function = get_metric(metric_dict, True)
    metric_name = metric_dict.get("name")
    
    model.eval()

    loader = DataLoader(dataset, drop_last=False, batch_size=1024)
    print("> Computing %s..." % (metric_name))

    score_sum = 0.
    for images, labels in loader: # tqdm.tqdm(loader):
        if cuda:
            images, labels = images.cuda(), labels.cuda()

        score_sum += metric_function(model, images, labels, None).item() * images.shape[0] # multiplies by how many images in batch
            
    score = float(score_sum / len(loader.dataset))

    print('The current %s value is %f' %(metric_name, score))

    return score


@torch.no_grad()
def ensemble_accuracy(model_list, dataset, cuda):
    metric_function = softmax_accuracy_ensemble
    metric_name = "ensemble softmax accuracy"
    
    for model in model_list:
        model.eval()

    loader = DataLoader(dataset, drop_last=False, batch_size=1024)
    print("> Computing %s..." % (metric_name))

    score_sum = 0.
    for images, labels in loader:
        if cuda:
            images, labels = images.cuda(), labels.cuda()

        score_sum += metric_function(model_list, images, labels).item() * images.shape[0] 
            
    score = float(score_sum / len(loader.dataset))

    print('The current %s value is %f' %(metric_name, score))
    print("The ensemble consists of %d networks" % len(model_list))

    return score




def softmax_loss(model, images, labels):
    logits = model(images)
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(logits, labels.view(-1))
    return loss



def computedistance(minimum, model):
    distance = 0
    for minparam, param in zip(minimum, model.parameters()):
        dif = torch.add(-param, minparam)
        distance += torch.sum(torch.mul(dif, dif))
    return distance


@torch.no_grad()
def computegradientsize(model):
    size=0
    for param in model.parameters():
        size += torch.sum(torch.mul(param.grad, param.grad))
    
    return size



def softmax_accuracy(model, images, labels):
    logits = model(images)
    pred_labels = logits.argmax(dim=1)
    acc = (pred_labels == labels).float().mean()

    return acc

def softmax_accuracy_ensemble(model_list, images, labels):
    logits= torch.zeros(model_list[0](images).size(), device='cuda:0')
    for model in model_list:
        logits += model(images)

    logits = logits/len(model_list) # maybe not even necessary
    pred_labels = logits.argmax(dim=1)
    acc = (pred_labels == labels).float().mean()

    return acc




def remove_distance(loss_function):

    def loss_with_distance(model, images, labels, minimum_list):
        return loss_function(model, images, labels)

    return loss_with_distance


def merge_distance(minimum_list):
    minimum = []

    for i in range(len(minimum_list[0])):
        element = [elem[i] for elem in minimum_list]
        minimum.append(sum(element)/len(minimum_list))

    return minimum


def RBF_kernel(x, width):
    return torch.exp(-1 * width * x)

def Tanh_kernel(x, width):
    return 1 - torch.tanh(width * x)


def distance_wrapper(loss_function, distance_factor, distance_width, kernel_function):

    def loss_distance(model, images, labels, minimum_list):
        loss = 0
        classify_loss = loss_function(model, images, labels)
        loss += classify_loss
        for minimum in minimum_list:
            loss += distance_factor/len(minimum_list) * kernel_function(computedistance(minimum, model), distance_width)  # divided by length to be the same as before

        return loss

    return loss_distance
