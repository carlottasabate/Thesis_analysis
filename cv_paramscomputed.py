#!/usr/bin/env python
#$ -j Y
#$ -cwd
#$ -V
import ast
import getopt
import scipy
from linescanning import (
    prf,
    utils,
    dataset,
    plotting,
    preproc,
    optimal
)
import numpy as np
import nibabel as nb
import os
from scipy import io
import sys
import warnings
import json
import pickle
import mkl
mkl.set_num_threads=1
standard_max_threads = mkl.get_max_threads()
from datetime import datetime, timedelta
from linescanning.prf import *
import math
import matplotlib.image as mpimg
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from past.utils import old_div
from prfpy.stimulus import *
from prfpy.fit import *
from prfpy.model import *
from prfpy import timecourse
import random
from scipy.ndimage import rotate
from scipy import signal, io
import subprocess
import time
import yaml
import cortex
warnings.filterwarnings('ignore')
opj = os.path.join
opd = os.path.dirname


'''
### Function to return hrf convolution
######################################
'''

def create_hrf(prf_stim, hrf_params = [1.0, 1.0, 0.0]):
    hrf = np.array(
        [
            np.ones_like(hrf_params[1])*hrf_params[0] *
            spm_hrf(
                tr=prf_stim.TR,
                oversampling=1,
                time_length=40)[...,np.newaxis],
            hrf_params[1] *
            spm_time_derivative(
                tr=prf_stim.TR,
                oversampling=1,
                time_length=40)[...,np.newaxis],
            hrf_params[2] *
            spm_dispersion_derivative(
                tr=prf_stim.TR,
                oversampling=1,
                time_length=40)[...,np.newaxis]]).sum(axis=0)                    

    return hrf.T



'''
######################################################
### Function to return predictions based on parameters
######################################################
'''

def return_prediction(mod, prf_stim, dm, mu_x, mu_y, prf_size,  prf_amplitude, bold_baseline, srf_amplitude = None, srf_size = None):
    
    # create the rfs
    prf = np.rot90(rf.gauss2D_iso_cart(x=prf_stim.x_coordinates[..., np.newaxis],
                                    y=prf_stim.y_coordinates[..., np.newaxis],                        
                                    mu=(mu_x, mu_y),
                                    sigma=prf_size).T, axes=(1, 2))
    
    dm = prf_stim.design_matrix
    
    neural_tc = timecourse.stimulus_through_prf(prf, dm, prf_stim.dx)

    tc = Model.convolve_timecourse_hrf(self = 0, tc = neural_tc, hrf = create_hrf(prf_stim))#np.array([1,1,0]))
    
    if mod == 'dog':
        # surround receptive field
        srf = np.rot90(rf.gauss2D_iso_cart(x=prf_stim.x_coordinates[..., np.newaxis],
                                        y=prf_stim.y_coordinates[...,np.newaxis],        
                                        mu=(mu_x, mu_y),
                                        sigma=srf_size).T, axes=(1, 2))

        neural_tc = prf_amplitude[..., np.newaxis] * timecourse.stimulus_through_prf(prf, dm, prf_stim.dx) - \
            srf_amplitude[..., np.newaxis] * \
            timecourse.stimulus_through_prf(srf, dm, prf_stim.dx)

        tc = Model.convolve_timecourse_hrf(self = 0, tc = neural_tc, hrf = create_hrf(prf_stim))


    return bold_baseline[..., np.newaxis] + timecourse.filter_predictions(tc,
            filter_type = 'tc',
            filter_params = {})
    
    
'''
##################
Load Design matrix
##################
'''
def load_dm(cut_volumes = True):

    #### load design matrix ###
    screen_size_cm =39.3
    screen_distance_cm=210
    grid_nr = 20
    TR= 1.5
    
    if cut_volumes:
        n_volumes = 5
    else:
        n_volumes = 0

    design = prf.read_par_file(opj(opd(opd(prf.__file__)), '/data1/projects/Meman1/projects/pilot/code', 'design_task-2R.mat'))

    prf_stim = PRFStimulus2D(screen_size_cm = screen_size_cm,
                                screen_distance_cm = screen_distance_cm,
                                design_matrix = design[:, :, n_volumes:], # remove first 5 volumes
                                TR = TR,
                                task_names ='2R')
    
    return design, prf_stim



'''
#####################################################################################
### Function to calculate the predictions and cv rsq (used in cross_validate function)
####################################################################################
'''

def cvs(test_data, dog_g_model, design, mod):
    screen_size_cm =39.3
    screen_distance_cm=210
    grid_nr = 20
    TR= 1.5
    print(mod)

#   if mod == 'gauss':
#       rsq_mask = dog_g_model.gaussian_fitter.rsq_mask
#   else:
#       rsq_mask = dog_g_model.dog_fitter.rsq_mask

    prf_stim = PRFStimulus2D(screen_size_cm = screen_size_cm,
                              screen_distance_cm = screen_distance_cm,
                              design_matrix = design,
                              TR = TR,
                              task_names ='2R')

    test_predictions = return_prediction(mod, prf_stim, design, *list(dog_g_model[:,:-1].T))
   
    if test_predictions.shape[1] == 225:
        test_predictions = test_predictions[:,5:]
    if mod == 'gauss':
        test_predictions = test_predictions/14
    
    #calculate CV-rsq        
    #   CV_rsq = np.nan_to_num(1-np.sum((test_data[rsq_mask]-test_predictions)**2, axis=-1)/(test_data.shape[-1]*test_data[rsq_mask].var(-1)))
    CV_rsq = np.nan_to_num(1-np.sum((test_data-test_predictions)**2, axis=-1)/(test_data.shape[-1]*test_data.var(-1)))
  
  #calcualte CV-correlation
  #CV_rsq = np.zeros(self.rsq_mask.sum())
  #for i in range(len(CV_rsq)):
  #    CV_rsq[i] = np.nan_to_num(pearsonr(test_data[self.rsq_mask][i],np.nan_to_num(test_predictions[i]))[0])

  # self.iterative_search_params[self.rsq_mask,-1] = CV_rsq
    return CV_rsq, test_predictions




'''
###################################
## Fuction for the cross-validation

- **It performs the gaussian and DoG (specifying the constraint ['tc', 'bgfs']) for the training set, calculates the predictions for the test set and the rsq of it**

- **It returns a df with the rsq train and test and the predictions**
##################################
'''

def cross_validate(train, test, constraint_dog, design, sub, ses, roi_tag, njobs, cv, mod): # m_prf_tc_data.T[lbl_true[:10]]

    output_dir = f'/data1/projects/Meman1/projects/pilot/derivatives/prf/sub-{sub}/ses-{ses}_train-test'
    if mod=='gauss':
        output_base = f'sub-{sub}_ses-{ses}_task-2R_roi-{roi_tag}_model-{mod}_stage-iter_desc-prf_params.pkl'
    else:
        output_base = f'sub-{sub}_ses-{ses}_task-2R_roi-{roi_tag}_{constraint_dog}_model-{mod}_stage-iter_desc-prf_params.pkl'
    
    file = opj(output_dir,output_base)
    #utils.verbose(f"file: {file}", verbose)
    prf_results = prf.read_par_file(file)  
      
    
    rsq_test = None
    rsq_train = None
    df_rsq = None
    test_pred = None
    
    if cv: 
            rsq_test, test_pred = cvs(test, prf_results, design, mod) 
    
    return rsq_test, test_pred





####################################################################

def main(argv):
    
    #### PLACEBO / MEMANTINE
    sub         = None #'001'
    ses         = None #2
    task        = '2R'
    outputdir   = f'/data1/projects/Meman1/projects/pilot/derivatives/prf/sub-{sub}/ses-{ses}_train-test'
    inputdir    = f'/data1/projects/Meman1/projects/pilot/derivatives/pybest/sub-{sub}/ses-{ses}/unzscored'
    png_dir     = None
    space       = 'fsnative'
    giftis      = False
    fit_hrf     = False
    verbose     = True
    n_pix       = 100
    clip_dm     = [0,0,0,0]
    file_ending = None
    psc         = True
    cut_vols    = 5 #0
    roi_tag     = None
    save_grid   = True
    grid_bounds = True   
    merge_sessions = False
    cv          = False #True
    overwrite   = False
    njobs       = None
    tr          = 1.5
    mod         = 'dog'

    try:
        opts = getopt.getopt(argv,"ghs:n:t:o:i:p:m:x:u:c:v:",["help", "sub=", "model=", "ses=", "task=", "out=", "in=", "png=", "params=", "grid", "space=", "hrf", "n_pix=", "clip=", "verbose", "file_ending=", "zscore", "overwrite",  "tc", "bgfs", "no_fit", "raw", "cut_vols=", "v1", "v2", "v3", "no_grid", "merge_ses", "cv", "gauss" ,"njobs="])[0]#"constr=",
    except getopt.GetoptError:
        print("ERROR while reading arguments; did you specify an illegal argument?")
        print(main.__doc__)
        sys.exit(2)
    print('blahhhhhhhhhhhhh')
    print(mkl.get_max_threads())
    for opt, arg in opts:
        if opt in ("-s", "--sub"):
            sub = arg
        elif opt in ("-n", "--ses"):
            ses = int(arg)
        elif opt in ("-t", "--task"):
            task = arg
        elif opt in ("-o", "--out"):
            outputdir = arg
        elif opt in ("-i", "--in"):
            inputdir = arg
        elif opt in ("-p", "--png"):
            png_dir = arg
        elif opt in ("-m", "--model"):
            model = arg
        elif opt in ("-u", "--space"):
            space = arg
        elif opt in ("-g", "--grid"):
            grid_only = True
        elif opt in ("--hrf"):
            fit_hrf = True
        elif opt in ("--n_pix"):
            n_pix = int(arg)
        elif opt in ("-v", "--verbose"):
            verbose = True
        elif opt in ("--clip"):
            clip_dm = list(ast.literal_eval(arg))
        elif opt in ("--file_ending"):
            file_ending = arg
        elif opt in ("--zscore"):
            psc = False
        elif opt in ("--raw"):
            psc = False            
        elif opt in ("--overwrite"):
            overwrite = True
        elif opt in ("--gauss"):
            mod = "gauss"
        elif opt in ("--tc"):
            constraints = "tc"
        elif opt in ("--bgfs"):
            constraints = "bgfs"
        elif opt in ("--no_fit"):
            do_fit = False
        # elif opt in ("--constr"):
        #     constraints = utils.string2list(arg)
        elif opt in ("--cut_vols"):
            cut_vols = int(arg)
        elif opt in ("--v1"):
            lbl = "V1.custom"
            roi_tag = "V1"
        elif opt in ("--v2"):
            lbl = "V2.custom"
            roi_tag = "V2"
        elif opt in ("--v3"):
            lbl = "V3.custom"
            roi_tag = "V3"
        elif opt in ("--njobs"):
            njobs = int(arg)
        elif opt in ("--no_bounds"):
            grid_bounds = False  
        elif opt in ("--merge_ses"):
            merge_sessions = True     
        elif opt in ("--cv"):
            cv = True          

    

    # Procede to run 

    if sub =='001':
        lbl = f'custom.{roi_tag}' # custom.V1 // roi? - it creates a mask - taking only the instance from tc in the roi mask - vertices that belong to ROI
    #else:
    #    lbl = 'V1.custom'

    print(sub, ses)
   
    
    outputdir   = f'/data1/projects/Meman1/projects/pilot/derivatives/prf/sub-{sub}/ses-{ses}_train-test'
    inputdir    = f'/data1/projects/Meman1/projects/pilot/derivatives/pybest/sub-{sub}/ses-{ses}/unzscored'
    
    design, prf_stim = load_dm()

    # load median runs, design matrix without the first 5 seconds, and select roi
    # design, lbl_true, train_data, test_data = load_tc(sub, ses, task, outputdir, inputdir, space, verbose,
    #                                                 clip_dm, file_ending, psc, cut_vols, lbl, roi_tag, merge_sessions, cv, giftis)

    # np.save(opj(outputdir, f'{roi_tag}_data.npy'),lbl_true)

    # compute training and test doing prfModelFitting and the return_predictions using the test data, saving the parameters in prf derivatives 
    if cv:
        
        prediction = pd.DataFrame()
        cvr2 = pd.DataFrame()
        rsqs_cv_odd = pd.DataFrame()
        rsqs_cv_even = pd.DataFrame()
        models = ['gauss', 'dogTC', 'dogLBFGS'] # setting up all 3 model conditions
        
        out = f"sub-{sub}" + f"_ses-{ses}" + f"_task-{task}" + f"_roi-{roi_tag}"
        train_data = np.load(opj(outputdir, f'{out}_hemi-LR_desc-avg_bold.npy'))
        test_data = np.load(opj(outputdir, f'{out}_hemi-LR_desc-avg_bold_test.npy'))
        
        for mods in models:
            
            if mods == 'dogTC':
                mod = 'dog'
                constraints = 'tc'
            elif mods == 'dogLBFGS':
                mod = 'dog'
                constraints = 'bgfs'
            else:
                mod = 'gauss'
            
            print(mod, constraints)
            
            rsq_test_even, test_pred_even = cross_validate(train_data.T, test_data.T, constraints, design, sub, ses, roi_tag, njobs, cv, mod) # df_rsq, rsq_train, rsq_test, test_pred
            
            rsq_test_odd, test_pred_odd = cross_validate(test_data.T, train_data.T, constraints, design, sub, ses, roi_tag, njobs, cv, mod) # df_rsq, rsq_train, rsq_test, test_pred  
            
            ### get prediction to plot specific voxel
            if roi_tag == 'V1':
                voxel = 154186
            elif roi_tag == 'V2':
                voxel = 151711
            else: # V3
                voxel = 152852
            #### save prediction of specific voxel for examplification in thesis, using odd runs as training set
            prediction[f'testpred_even_{roi_tag}_{mod}_{constraints}'] = test_pred_even[voxel] 
            
            # save cvr2 for each model
            cvr2[f'cvr2_{roi_tag}_ses-{ses}_sub-{sub}_{mod}_{constraints.upper()}'] = np.mean([rsq_test_even, rsq_test_odd], axis =0) # test R2 of each run set - to calculate the average   
            
            rsqs_cv_even[f'cvr2_{mod}_{constraints}_even'] = rsq_test_even
            rsqs_cv_odd[f'cvr2_{mod}_{constraints}_odd'] = rsq_test_odd
                    
        
        cvr2.to_csv(f'/data1/projects/Meman1/projects/pilot/derivatives/prf/sub-{sub}/ses-{ses}_train-test/cvr2_{roi_tag}_ses-{ses}_sub-{sub}_allmods.csv')
        rsqs_cv_odd.to_csv(f'/data1/projects/Meman1/projects/pilot/derivatives/prf/sub-{sub}/ses-{ses}_train-test/rsqs_odd_{roi_tag}_ses-{ses}_sub-{sub}_allmods.csv')
        rsqs_cv_even.to_csv(f'/data1/projects/Meman1/projects/pilot/derivatives/prf/sub-{sub}/ses-{ses}_train-test/rsqs_even_{roi_tag}_ses-{ses}_sub-{sub}_allmods.csv')
        prediction[f'test_odd_{roi_tag}_{ses}'] = test_data.T[voxel]
        prediction.to_csv(f'/data1/projects/Meman1/projects/pilot/derivatives/prf/sub-{sub}/ses-{ses}_train-test/cvr2_prediction_{roi_tag}_ses-{ses}_sub-{sub}_allmods.csv')
        
    
    else:
        _, _, _, _ = cross_validate(train_data.T, None, constraints, design, sub, ses, roi_tag, njobs, cv, mod)  # without cv
        

if __name__ == "__main__":
    main(sys.argv[1:])





