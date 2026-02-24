
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc
from captum.attr import *
import quantus
import os
import shutil
import time
import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import kornia
from skimage.metrics import structural_similarity as ssim
import pandas as pd 


def compute_accuracy_batch(model, test_loader, device):
    model.eval()
    accuracy = []
    with  torch.no_grad():
        for step, (images, labels) in enumerate(test_loader):
            images, labels= images.to(device), labels.to(device)
            prediction = torch.nn.functional.softmax(model(images), dim=1)
            output = (np.argmax(prediction.cpu().numpy(), axis=1) == labels.cpu().numpy())
            acc = np.mean(output)
            accuracy.append(acc)

    return np.mean(accuracy), np.std(accuracy)



def filter_and_compute_sparsity(model, test_loader, device):
    
    sparsity = quantus.Sparseness(disable_warnings=True, return_aggregate=True)
    score_sparsity = []

    for i, (x_batch, y_batch) in enumerate(test_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        #print(len(x_batch))
        outputs = model(x_batch)
        predictions = torch.argmax(outputs, dim=1)
        correct_mask = predictions == y_batch
        #print(correct_mask)
        x_batch = x_batch[correct_mask]
        y_batch = y_batch[correct_mask]
        #print(len(x_batch))
        x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()
        scores = sparsity(
                model= model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=None,
                s_batch=None,
                device=device,
                explain_func= quantus.explain, 
                explain_func_kwargs = {"method": "Saliency", "softmax": False})
        score_sparsity.extend(scores)
        if len(score_sparsity) > 1000:
            break 
    return np.nanmean(score_sparsity), np.nanstd(score_sparsity)


def filter_and_compute_output_stability(model, test_loader, device):
    
    metrics = quantus.RelativeOutputStability(
        nr_samples = 5,
         return_aggregate=False,
        disable_warnings=True,
    )
    score = []
    
    for i, (x_batch, y_batch) in enumerate(test_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        outputs = model(x_batch)
        predictions = torch.argmax(outputs, dim=1)
        correct_mask = predictions == y_batch
        x_batch = x_batch[correct_mask]
        y_batch = y_batch[correct_mask]
        x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()
        scores = metrics(
                model= model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=None,
                s_batch=None,
                device=device,
                explain_func= quantus.explain, 
                explain_func_kwargs = {"method": "Saliency", "softmax": False})
        scores2 = np.nanmean(scores)
        score.append(scores2)
        if len(score) > 1000:
            break 
    score2=score
    return math.log(np.nanmean(score2), 10)


def filter_and_compute_road(model, test_loader, device):

    faithfulness = quantus.ROAD(
    noise=0.01,
    perturb_func=quantus.perturb_func.noisy_linear_imputation,
    percentages=list(range(1, 100, 5)),
    display_progressbar=False,
)


    score_faithfulness = []

    for i, (x_batch, y_batch) in enumerate(test_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        #print(len(x_batch))
        outputs = model(x_batch)
        predictions = torch.argmax(outputs, dim=1)
        correct_mask = predictions == y_batch
        #print(correct_mask)
        x_batch = x_batch[correct_mask]
        y_batch = y_batch[correct_mask]
        #print(len(x_batch))
        x_batch, y_batch = x_batch.cpu().numpy(), y_batch.cpu().numpy()
        scores = faithfulness(
                model= model,
                x_batch=x_batch,
                y_batch=y_batch,
                a_batch=None,
                s_batch=None,
                device=device,
                explain_func= quantus.explain,
                explain_func_kwargs = {"method": "Saliency", "softmax": False})
        #print(scores)
        
        score_faithfulness.append(scores)
        if len(score_faithfulness) > 1000:
            break
    average_values = {}
    for d in score_faithfulness:
        for key, value in d.items():
            if key in average_values:
                average_values[key] += value
            else:
                average_values[key] = value

    # Divide the sum by the number of dictionaries to get the average
    num_dicts = len(score_faithfulness)
    for key in average_values.keys():
        average_values[key] /= num_dicts

    # Display the average values
    print("Average Values:")
    for key, value in average_values.items():
        print(f"{key}: {value}")


def compute_score(h1, h2, method, metric_name):

    if metric_name.lower() == 'ssim':
        if method.lower() == 'mp':
            data_range = 1
        else:
            data_range = 2
        out = ssim(h1, h2, data_range=data_range, win_size=5)

    return out

def make_noise(model, x_batch, y_batch, stdev):
    x_batch_np = x_batch.data.cpu().numpy()
    noise = np.random.normal(0, stdev, x_batch_np.shape).astype(np.float32)
    x_plus_noise = x_batch_np + noise
    x_plus_noise = np.clip(x_plus_noise, 0, 1)
    noisy_batch = torch.from_numpy(x_plus_noise).to(x_batch.device)

    #get model prediction on noisy images
    model.eval()
    with torch.no_grad():
      noisy_predictions = model(noisy_batch)

    # Filter images based on correct predictions
    correct_predictions = torch.argmax(noisy_predictions, dim=1) == y_batch
    filtered_noisy_batch = noisy_batch[correct_predictions]
    filtered_original_batch = x_batch[correct_predictions]
    label = y_batch[correct_predictions]


    return filtered_original_batch, filtered_noisy_batch, label

def filter_and_compute_SSIM(model, test_loader, device):
    for noise in [0.01, 0.03, 0.05, 0.07, 0.09,0.1, 0.15,0.2]:
        scores = []
        for i, (x_batch, y_batch) in enumerate(test_loader):
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                new_img, new_noisy_img, label = make_noise(model, x_batch, y_batch, noise)
                a_org = Saliency(model).attribute(inputs=new_img, target=label).sum(axis=1).cpu().numpy()
                a_noisy = Saliency(model).attribute(inputs=new_noisy_img, target=label).sum(axis=1).cpu().numpy()
                for im1, im2 in zip(a_org,a_noisy ):
                    score = compute_score(im1, im2, 'grad', 'ssim')
                    scores.append(score)
                if len(scores)>1000:
                    break
        print('SSIM for noise {} is {}.'.format(noise, sum(scores)/len(scores)))



def compute_road_aopc_morf(models):
    """
    Computes the AOPC (Area Over Perturbation Curve) for the ROAD scores. Input is a dictionary called models 
    where keys are model names and values are dictionaries with percentage of features perturbed as keys and corresponding model accuracy 
    as values. The function returns a dataframe containing sorted AOPC MoRF scores for each model.
    """
    aopc_scores = {}
    
    for model_name, model_performance in models.items():
        percent_perturbed = np.array(list(model_performance.keys()))  # Feature perturbation percentages
        accuracy = np.array(list(model_performance.values()))  # Model accuracy at each perturbation step
        
        initial_acc = accuracy[0]  # Accuracy before any perturbation
        aopc_values = initial_acc - accuracy  # Compute accuracy difference at each step
        
        # Compute AOPC using summation method
        aopc_morf = np.sum(aopc_values) / (len(percent_perturbed) + 1)
        
        aopc_scores[model_name] = aopc_morf
    
    # Convert to DataFrame and sort by descending order
    aopc_df = pd.DataFrame.from_dict(aopc_scores, orient='index', columns=['AOPC MoRF Score'])
    aopc_df = aopc_df.sort_values(by='AOPC MoRF Score', ascending=False)
    
    return aopc_df