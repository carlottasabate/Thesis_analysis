# Thesis_analysis
Exploring the effects of Memantine in hindering feedback signals using fMRI as a non-invasive biomarker

This module presents the scripts used to analyze the population receptive fields (pRFs) in the early visual cortex by fitting the DoG model, an extension of the Gaussian. Here we optimize the parameters by minimizing the residual sum of squares (RSS) using a 2-stage coarse-to-fine search. 
This process is computationally performed using the prfpy Python package \citep{aqil2021divisive} and the linescanning pipeline \citep{jheij2021}. The **spinoza_setup** file contains the necessary information to link the packages to the environment used in the Spinoza server. The parameters optimization script can be found in the **cv_paramscomputed.py** file. 

## Grid-fit, best-fitting initialization parameters analysis

The first stage, also known as grid-fit, consists of finding the best-fitting initial parameters for each vertex (i.e. population of neurons). 
The aim is to reduce the computational time during optimization by finding adequate initialization parameters. 

The grid of values and boundaries set up to obtain accurate results are found in **prf_analysis.yml**. We set up a grid of 20 values for each parameter, precisely position and size ($x_0$, $y_0$, $\sigma_1$) for the Gaussian, and include the surrounding size ($\sigma_2$) and signal amplitude of the negative Gaussian ($\beta_2$) for the DoG. Additionally, some constraints are included in order to guarantee a good fitting procedure.
The fits are restricted to positive pRFs for the Gaussian and for the positive linear Gaussian function of the DoG. Thus, boundaries for the pRF amplitude $\beta_1$ are defined from 0 to 100. 
This means that if the estimated pRF amplitude is outside the range, the fitting will be discarded, ensuring that the boundaries are followed.

All possible combinations of the free parameters are fitted to the corresponding model. The combination that best predicts the recorded BOLD signal in the vertex (minimum residuals, minimum distance between predicted and actual signal), will be selected and used for the second stage. 
The first stage also allows us to reduce the amount of vertices/pRFs that need to be measured by excluding from the second step, those whose those that explain less than 10\% of the total variance ($R^2$). Such vertices are postulated to exhibit a lack of substantial variability.

## Iterative-fit, optimal parameters analysis

Once the best fitting initialization parameters are found, these are used to start the second stage, also known as iterative fit.
During the iterative fit, an optimizer is used to iterate over the fitting procedure to find the best combination of parameters that minimize the RSS.

The two most common optimizers are the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) and the Trust-region Constrained (TC).

The L-BFGS is an algorithm used to minimize the error function without constraints. 
This technique uses an approximation of the Hessian matrix to find the direction that minimizes the objective function, and a continuous evaluation of the gradient.
It is especially suitable for high-dimensional data due to its limited memory regarding previous iterations. L-BFGS is a computationally feasible and fast method in the fitting procedure of all vertices. It has been proven that having a lower iteration cost, makes the optimization more efficient while still giving favorable convergence \citep{malouf2002comparison, liu1989limited}. The second optimizer, TC, is generally used in pRF mapping \citep{zuiderbaan2012modeling, aqil2021divisive, harvey2011relationship} due to its capability to compute more robust results, converging to a single solution regardless of the initialization parameter values \citep{rafati2018improving}.

The choice of optimizer can influence the outcomes of the optimization process, prompting the need for occasional exploration to balance the time invested and the optimization quality achieved. Irrespective of the chosen optimizer for this phase, the fitting procedure empowers us to derive the best-suited parameters for measuring the population receptive field (pRF) in every vertex. This is the reason why the optimizers were analyzed in a cvR2 scheme with the results of Subject 001. The script **cv_paramcomputed.py** contains the functions and procedure used to obtain the cvR2, while the results were inspected in the **single-subject.ipynb** file.
