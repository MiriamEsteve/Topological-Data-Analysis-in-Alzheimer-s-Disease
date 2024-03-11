import pandas as pd

# Read dataset

path = "/home/miriam/alzheimer/OpenNeuro_Alzheimer/CSV/"
control_tag = ["AD", "CN", "FTD"]
i = 1
dict_AD = []
dict_CN = []
dict_FTD = []

j = 0
control_tag_j = control_tag[j]
dict_AD.append(pd.read_csv(path + control_tag_j + "/sub-00" + str(i) + "_CN.csv", index_col=0))

########################################################################################################################

#########  Install datalab in Linux
# @sudo apt-get update
# sudo apt-get install datalad

######### Use data
# datalad clone https://github.com/OpenNeuroDatasets/ds004504.git
# cd ds004504
# datalad get .
# datalad install -r ///openneuro/ds004504


########################################################################################################################

import mne

# Replace 'path_to_your_eeg_data.edf' with the path to your EEG data file
file_path = './data.edf'

# Load the EEG data
raw = mne.io.read_raw_edf(file_path, preload=True)

# Plot the first few seconds of data
raw.plot(duration=5, n_channels=raw.info['nchan'], scalings='auto')


########################################################################################################################
##### https://mne.tools/stable/index.html #########
# Load or create the EEG sensor locations
# For this example, we'll assume the sensor locations are correctly set in the data
# If not, you would need to set them using `raw.set_montage()` with a Montage object

# Load the standard MRI brain (FreeSurfer subject 'fsaverage')
subjects_dir = mne.datasets.sample.data_path() + '/subjects'
brain_model = 'fsaverage'

# Load or create the EEG sensor locations
# For this example, we'll assume the sensor locations are correctly set in the data
# If not, you would need to set them using `raw.set_montage()` with a Montage object
# Use the fsaverage brain
subjects_dir = mne.datasets.sample.data_path() + '/subjects'
subject = 'fsaverage'

# Create or load source space
src = mne.setup_source_space(subject, spacing='oct6', subjects_dir=subjects_dir, add_dist=False)

# Create or load BEM model
model = mne.make_bem_model(subject=subject, subjects_dir=subjects_dir)
bem = mne.make_bem_solution(model)

## Step 4: Compute Forward Model
# Load the transformation file for EEG sensor coregistration
trans = 'path_to-your_coreg-trans.fif'  # This is typically obtained through the coregistration GUI in MNE

# The forward model uses the head model and sensor locations to predict how brain currents are projected onto the EEG sensors:
fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=1)

## Step 5: Compute Inverse Operator and Estimate Sources
# The inverse operator is used to estimate the sources from the EEG data:
fwd = mne.make_forward_solution(raw.info, trans=trans, src=src, bem=bem, eeg=True, mindist=5.0, n_jobs=1)

## Step 5: Compute Inverse Operator and Estimate Sources
# The inverse operator is used to estimate the sources from the EEG data:
# Compute the noise covariance matrix from the pre-stimulus period
from mne import compute_covariance
cov = compute_covariance(raw, tmin=None, tmax=0.)

# Compute the inverse operator
from mne.minimum_norm import make_inverse_operator, apply_inverse
inverse_operator = make_inverse_operator(raw.info, fwd, cov, loose=0.2, depth=0.8)

# Estimate sources
stc = apply_inverse(raw, inverse_operator, lambda2=1.0/9.0, method='dSPM')

## Step 6: Visualize the Brain Activity
# Now, you can visualize the estimated brain activity:
# Visualize the brain activity
brain = stc.plot(subjects_dir=subjects_dir, subject=subject, hemi='both', time_viewer=True)
