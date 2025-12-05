# NB: TO BE RUN WITH 'data' AS WORKING DIRECTORY

import os
import sys
import numpy as np
import h5py
import scipy.io as spio
import nibabel as nib

import argparse
parser = argparse.ArgumentParser(description='Argument Parser')
parser.add_argument("-sub", "--sub", help="Subject Number", default=1)
parser.add_argument("-betas_type", "--betas_type", help="Chosen Betas type (it is a prefix, i.e. subj space '', MNI space)'MNI'", default='MNI')
parser.add_argument("-chosen_roi", "--chosen_roi", help="Chosen ROI", default='nsdgeneral')
#parser.add_argument("-free_disk", "--free_disk", help="Remove redundant files (1) or not (0)", default=0)
args = parser.parse_args()
sub=int(args.sub)
betas_type=str(args.betas_type)
if betas_type != '':
    betas_type = betas_type + '_'
chosen_roi=str(args.chosen_roi)
print(chosen_roi)
print(betas_type)
#free_disk = int(args.free_disk)
#assert sub in [1,2,5,7]



##### FUNCTION FOR MATRIX BETA-STIMULUS
def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    def _check_keys(d):
        '''
        checks if entries in dictionary are mat-objects. If yes
        todict is called to change them to nested dictionaries
        '''
        for key in d:
            if isinstance(d[key], spio.matlab.mio5_params.mat_struct):
                d[key] = _todict(d[key])
        return d

    def _todict(matobj):
        '''
        A recursive function which constructs from matobjects nested dictionaries
        '''
        d = {}
        for strg in matobj._fieldnames:
            elem = matobj.__dict__[strg]
            if isinstance(elem, spio.matlab.mio5_params.mat_struct):
                d[strg] = _todict(elem)
            elif isinstance(elem, np.ndarray):
                d[strg] = _tolist(elem)
            else:
                d[strg] = elem
        return d

    def _tolist(ndarray):
        '''
        A recursive function which constructs lists from cellarrays
        (which are loaded as numpy ndarrays), recursing into the elements
        if they contain matobjects.
        '''
        elem_list = []
        for sub_elem in ndarray:
            if isinstance(sub_elem, spio.matlab.mio5_params.mat_struct):
                elem_list.append(_todict(sub_elem))
            elif isinstance(sub_elem, np.ndarray):
                elem_list.append(_tolist(sub_elem))
            else:
                elem_list.append(sub_elem)
        return elem_list
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)



##### UTILITY FUNCTION FOR RAM MEMORY MONITORING
def print_mem_usage():
    import psutil

    process = psutil.Process()
    memory_bytes = process.memory_info().rss
    memory_megabytes = memory_bytes / (1024 * 1024)
    print("RAM memory used:", memory_megabytes, "MB")

stim_order_f = '00_nsd_misc/experiments/nsd/nsd_expdesign.mat'
stim_order = loadmat(stim_order_f)



##### SELECT IDS FOR TRAIN AND TEST DATA
print("Selecting ids for training and test data")
sig_train = {}
sig_test = {}
num_sessions = 40 #37
num_trials = num_sessions*750
for idx in range(num_trials):
    ''' nsdId as in design csv files'''
    nsdId = stim_order['subjectim'][sub-1, stim_order['masterordering'][idx] - 1] - 1
    if stim_order['masterordering'][idx]>1000: #first 1000 are images that are repeated among subjects
        if nsdId not in sig_train:
            sig_train[nsdId] = []
        sig_train[nsdId].append(idx)
    else: # the frist 1000 images, the ones repates, are used as test set
        if nsdId not in sig_test:
            sig_test[nsdId] = []
        sig_test[nsdId].append(idx)

train_im_idx = list(sig_train.keys())
test_im_idx = list(sig_test.keys())



##### LOAD BETAS
print("Loading fMRI data (masking the ROIs)")
if betas_type == 'MNI_': #MNI space
    print('    Choosing ROI in MNI space:')
    roi_dir = '00_nsd_misc/rois/MNI/'
else: #subj space
    print('    Choosing ROI in subject space:')
    roi_dir = '00_nsd_misc/rois/subj_{}/'.format(sub)
betas_dir = '01_nsd_betas/subj_{}/'.format(sub)

print('        '+betas_type + f'{chosen_roi}.nii')
mask_filename = betas_type + f'{chosen_roi}.nii'
mask = nib.load(roi_dir+mask_filename).get_fdata()
num_voxel = mask[mask>0].shape[0]

fmri = np.zeros((num_trials, num_voxel)).astype(np.float32)
for i in range(num_sessions): #37
    beta_filename = betas_type+"betas_session{:02d}.nii.gz".format(i+1) #betas_type is the prefix (e.g. MNI if I want betas in MNi space)
    beta_f = nib.load(betas_dir+beta_filename).get_fdata().astype(np.float32)
    fmri[i*750:(i+1)*750] = beta_f[mask>0].transpose() #from 81x104x83x750 to 750x83x104x81
    del beta_f
    print(i+1)

#if(free_disk): 
#    os.system('rm ' + betas_dir + '*')
#    print("Betas removed from the disk.")
print("Shape of the betas:", len(fmri), len(fmri[0]))


##### LOAD STIMULI
print("Loading stimuli")
f_stim = h5py.File('02_nsd_stimuli/nsd/nsd_stimuli.hdf5', 'r')
stim = f_stim['imgBrick'][:]

print_mem_usage()

print("Stimuli are loaded.")

#if(free_disk): 
#    os.system('rm 03_nsd_stimuli/stimuli/nsd/nsd_stimuli.hdf5')
#    print("Stimuli removed from the disk")



##### SPLIT TRAIN-TEST AND SAVE BETAS AND STIMULI
### Train
num_train, num_test = len(train_im_idx), len(test_im_idx)
vox_dim, im_dim, im_c = num_voxel, 425, 3
fmri_array = np.zeros((num_train,vox_dim))
stim_array = np.zeros((num_train,im_dim,im_dim,im_c))
for i, idx in enumerate(train_im_idx):
    # Load stimulus image
    stim_array[i] = f_stim['imgBrick'][idx]
    
    # Load and process fMRI data
    fmri_data = []
    for sig_idx in sorted(sig_train[idx]):
        fmri_data.append(fmri[sig_idx])
    fmri_array[i] = np.mean(fmri_data, axis=0) #average over repeated images (the time is already averaged)
    #print(i)

os.makedirs('04_processed_data/subj_{}/'.format(sub), exist_ok=True)

np.save('04_processed_data/subj_{}/betas_train.npy'.format(sub),fmri_array ) # 10 sessions => 0.512350944 GB - 37 sessions => 1.114391456 GB
np.save('04_processed_data/subj_{}/stimuli_train.npy'.format(sub),stim_array ) # 10 sessions => 17.656455128 GB - 37 sessions => 38.403765128 GB

print("Train betas and stimuli saved.")

### Test
fmri_array = np.zeros((num_test,vox_dim))
stim_array = np.zeros((num_test,im_dim,im_dim,im_c))
for i,idx in enumerate(test_im_idx):
    stim_array[i] = stim[idx]
    fmri_array[i] = fmri[sorted(sig_test[idx])].mean(0)
    #print(i)

f_stim.close()

np.save('04_processed_data/subj_{}/betas_test.npy'.format(sub),fmri_array ) # 10 sessions => 0.052329600 GB - 37 sessions => 0.123527872 GB
np.save('04_processed_data/subj_{}/stimuli_test.npy'.format(sub),stim_array ) # 10 sessions => 1.803360128 GB - 37 sessions => 4.256970128 GB

print("Test betas and stimuli saved.")

num_train, num_test = len(train_im_idx), len(test_im_idx)



##### LOAD ANNOTATIONS
print("Loading caption data")
annots_cur = np.load('03_nsd_annots/COCO_73k_annots_curated.npy')

#print(annots_cur.shape)
#print(type(annots_cur), type(annots_cur[0]), type(annots_cur[0][0]))
#print(annots_cur.dtype)

### Train
captions_array = np.empty((num_train, 5),dtype=annots_cur.dtype)
for i,idx in enumerate(train_im_idx):
    captions_array[i,:] = annots_cur[idx,:]
    #print(i)
np.save('04_processed_data/subj_{}/annotations_train.npy'.format(sub),captions_array ) # 10 sessions => 0.020365128 GB - 37 sessions => 0.44295128 GB

### Test
captions_array = np.empty((num_test, 5),dtype=annots_cur.dtype)
for i,idx in enumerate(test_im_idx):
    captions_array[i,:] = annots_cur[idx,:]
    #print(i)
np.save('04_processed_data/subj_{}/annotations_test.npy'.format(sub),captions_array ) # 10 sessions => 0.002080128 GB - 37 sessions - 0.004910128 GB

print("Caption data are saved.")