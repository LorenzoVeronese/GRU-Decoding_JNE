import os
import subprocess


##### Download Betas
for sub in [1, 2, 5, 7]:
    directory = 'data/01_nsd_betas/ppdata/subj_{}/func1pt8mm/betas_fithrf_GLMdenoise_RR'.format(sub)
    if not os.path.exists(directory):
        os.makedirs(directory)
    for sess in range(1,41):
        os.system('wget https://natural-scenes-dataset.s3.amazonaws.com/nsddata_betas/ppdata/subj{:02}/func1pt8mm/betas_fithrf_GLMdenoise_RR/betas_session{:02}.nii.gz -P data/01_nsd_betas/ppdata/subj_{}/func1pt8mm/betas_fithrf_GLMdenoise_RR/'.format(sub,sess,sub))

##### Download ROIs
roi_files = ['thalamus.nii.gz', 'streams.nii.gz', 'rh.thalamus.nii.gz', 'rh.streams.nii.gz', 'rh.prf-visualrois.nii.gz', 
             'rh.prf-eccrois.nii.gz', 'rh.nsdgeneral.nii.gz', 'rh.MTL.nii.gz', 'rh.Kastner2015.nii.gz', 'rh.HCP_MMP1.nii.gz', 
             'rh.floc-words.nii.gz', 'rh.floc-places.nii.gz', 'rh.floc-faces.nii.gz', 'rh.floc-bodies.nii.gz', 'rh.corticalsulc.nii.gz', 
             'prf-visualrois.nii.gz', 'prf-eccrois.nii.gz', 'nsdgeneral.nii.gz', 'MTL.nii.gz', 'lh.thalamus.nii.gz', 'lh.streams.nii.gz', 
             'lh.prf-visualrois.nii.gz', 'lh.prf-eccrois.nii.gz', 'lh.nsdgeneral.nii.gz', 'lh.MTL.nii.gz', 'lh.Kastner2015.nii.gz', 'lh.HCP_MMP1.nii.gz', 
             'lh.floc-words.nii.gz', 'lh.floc-places.nii.gz', 'lh.floc-faces.nii.gz', 'lh.floc-bodies.nii.gz', 'lh.corticalsulc.nii.gz', 
             'Kastner2015.nii.gz', 'HCP_MMP1.nii.gz', 'floc-words.nii.gz', 'floc-places.nii.gz', 'floc-faces.nii.gz', 'floc-bodies.nii.gz', 'corticalsulc.nii.gz']
for sub in [1,2,5,7]:
    directory = 'data/00_nsd_misc/ppdata/subj_{}/func1pt8mm/roi/'.format(sub)
    if not os.path.exists(directory):
        os.makedirs(directory)

    # URL of the folder to download
    for file in roi_files:
        os.system('wget https://natural-scenes-dataset.s3.amazonaws.com/nsddata/ppdata/subj{:02}/func1pt8mm/roi/{} -P data/00_nsd_misc/ppdata/subj_{}/func1pt8mm/roi/'.format(sub, file, sub))

##### Download Stimuli
#os.system('wget https://natural-scenes-dataset.s3.amazonaws.com/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5 -P data/02_nsd_stimuli/stimuli/nsd/')
