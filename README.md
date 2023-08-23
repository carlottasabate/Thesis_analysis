# Thesis_analysis
Exploring the effects of Memantine in hindering feedback signals using fMRI as a non-invasive biomarker

\subsection{Parameters optimization}
 %Implementation/
\label{sec: paramoptimization}

\begin{figure*}[h]
    \centering
    \includegraphics[width = 15cm]{Figures/methods/dogauss_workflow.jpeg}
    \caption{A flow chart describing the pRF linear models estimation procedure (Gaussian and DoG), taken from Zuiderbaan et al 2012. The pRF linear model is
calculated for every voxel independently. See text for details.}
    \label{fig:doggaussworkflow}
\end{figure*}

The optimal pRF parameters of either model are found by minimizing the residual sum of squares (RSS,  \ref{eq:eqrss}, \ref{fig:doggaussworkflow}) using a 2-stage coarse-to-fine search. 
In other words, we want to minimize the difference between the recorded neural signal and the predicted response by adjusting the pRF model parameters.

\begin{equation}
    RSS = \sum_t (y(t)-\hat{y}(t))^2
    \label{eq:eqrss}
\end{equation}
\intertext{where t corresponds to each time-point in the time-series BOLD signal.}

\vspace{.5cm}

This process is computationally performed using the \textit{prfpy} Python package \citep{aqil2021divisive} and the \textit{linescanning} pipeline \citep{jheij2021}.

\subsubsection{Grid-fit, best-fitting initialization parameters analysis}

The first stage, also known as \textit{grid-fit}, consists on finding the best-fitting initial parameters for each vertex (i.e. population of neurons). 
The aim is to reduce the computational time during optimization by finding adequate initialization parameters. 

We set up a grid of 20 values for each parameter, precisely position and size ($x_0$, $y_0$, $\sigma_1$) for the Gaussian, and include the surrounding size ($\sigma_2$) and signal amplitude of the negative Gaussian ($\beta_2$) for the DoG.

Additionally, some constraints are included in order to guarantee a good fitting procedure.
The fits are restricted to positive pRFs for the Gaussian and for the positive linear Gaussian function of the DoG. Thus, boundaries for the pRF amplitude $\beta_1$ are defined from 0 to 100. 
This means that if the estimated pRF amplitude is outside the range, the fitting will be discarded, ensuring that the boundaries are followed.

The grid values are based on the polarity (i.e. polar angle with respect to the fixation point in which the stimulus is located,  \ref{eq:polar}) and eccentricity (i.e. distance of the stimulus from the fixation point,  \ref{eq:ecc}) with respect to the center position of the pRF.

A vector of 20 equally distanced polar angles ($\theta$) within the visual space (0 - $2\pi$) is created as well as for the eccentricity (0 - $5^\circ$). 

A linear transformation is applied to obtain the possible center position values of the pRF ( \ref{eq:eqx0}, \ref{eq:eqy0}). 

The size grid is built from the maximum eccentricity size ( \ref{eq:eqsize}). 

All possible combinations of the free parameters are fitted to the corresponding model. The combination that best predicts the recorded BOLD signal in the vertex (minimum residuals, minimum distance between predicted and actual signal), will be selected and used for the second stage. 
The first stage also allows us to reduce the amount of vertices/pRFs that need to be measured by excluding from the second step, those whose those that explain less than 10\% of the total variance ($R^2$). Such vertices are postulated to exhibit a lack of substantial variability.

\subsubsection{Iterative-fit, optimal parameters analysis}

Once the best fitting initialization parameters are found, these are used to start the second stage, also known as \textit{iterative fit}.

During the iterative fit, an optimizer is used to iterate over the fitting procedure to find the best combination of parameters that minimize the RSS.


The two most common optimizers are the Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) and the Trust-region Constrained (TC).

The L-BFGS is an algorithm used to minimize the error function without constraints. 
This technique uses an approximation of the Hessian matrix to find the direction that minimizes the objective function, and a continuous evaluation of the gradient.
%This technique does not require the computation of the Hessian matrix at each iteration step, in order to find the direction that minimizes the error function (i.e. take advantage of the curvature information provided by the Hessian to converge more efficiently towards the minimum.), but an approximation of it. It uses a continuous evaluation of the gradient.
Additionally, it is especially suitable for high-dimensional data due to its limited memory regarding previous iterations. %that makes it suitable for high-dimensional data. 
L-BFGS is a computationally feasible and fast method in the fitting procedure of all vertices. 
It has been proven that having a lower iteration cost, makes the optimization more efficient while still giving favorable convergence \citep{malouf2002comparison, liu1989limited}. %Nevertheless, in order to account for possible overfitting and convergence failure, the $R^2$ metric was computed to quantify the variance explained by the optimal linear Gaussian model of each voxel.  Furthermore, the second, fourth and sixth runs were averaged and used to test the optimal parameters of each voxel. R2 scores were compared with the training set (odd runs), expecting a similar result. 

The second optimizer, TC, is generally used in pRF mapping \citep{zuiderbaan2012modeling, aqil2021divisive, harvey2011relationship} due to its capability to compute more robust results, converging to a single solution regardless of the initialization parameter values \citep{rafati2018improving}.

If the actual function in the new point is similar to the quadratic model, the new point is accepted as the next iterate and the radius might be increased to explore a larger region, otherwise the step is rejected and the radius is reduced to limit the search closer to the current point. 

The choice of optimizer can influence the outcomes of the optimization process, prompting the need for occasional exploration to balance the time invested and the optimization quality achieved. Irrespective of the chosen optimizer for this phase, the fitting procedure empowers us to derive the best-suited parameters for measuring the population receptive field (pRF) in every vertex.
