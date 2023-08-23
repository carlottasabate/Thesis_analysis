# Analysis of the effects of Memantine

The optimizer with the best performance was used to estimate the refined DoG parameters for each pRF in areas V1-V3 for the placebo and Memantine conditions. 

We analyzed the effects of Memantine by comparing Memantine and placebo sessions results. Particularly, we analyzed two shape metrics, across visual areas %and eccentricity based on the profile of the pRFs (i.e. shape of the pRF) rather than the fitting parameters. Instead of relying on the fitted parameters of the DoG model, which lacks direct interpretability due to the intricate interaction between its positive and negative components, we utilized this specific feature of the pRF shape.

We used the full width at minimum (FWMIN) to measure the size of the inhibitory surround of the pRF. It is defined as the width of the visual field between the two minima of the pRF profile.
We used the full-width at half maximum (FWHM) to describe the width (size) of the positive part of the receptive field. It specifically measures the width of a peak at half of its maximum value. 
We statistically analyzed the differences between drugs for both metrics.

The file **single-subject_analysis.ipynb** also contains the analysis of these metrics for Subject 001 results.

## Group analysis

The differences between drug conditions were statistically analyzed across subjects (i.e.  Subjects 001, 005, 007, 008, 010 and 012) performing a group analysis. The script can be found **group_analysis_dog.ipynb**. Additionally the Gaussian model results were analyzed in **group_analysis_gauss**.

Each fitted vertex that was able to explain more than 10\% of the pRF response variance and had eccentricity values between 0.5 and $4.5^\circ$, was taken into account for the analysis. 
To investigate the potential presence of inhibitory surround on the DoG model, we calculated the profile of each pRF per participant and session. Additionally, we assessed whether the amplitude of the suppressive surround is influenced by the drug and visual area.
Both metrics, namely FWMIN and FWHM were statistically investigated across participants to compare potential differences in the size of the inhibitory surround and the positive part of the pRF, between drug conditions in the early visual cortex. 
