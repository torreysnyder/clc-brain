import os
import numpy as np
import pandas as pd
import random
import nibabel as nib
from sklearn.linear_model import LinearRegression
from fracridge import FracRidgeRegressorCV
from scipy.stats import pearsonr as corr
from tqdm import tqdm
import matplotlib
from matplotlib import pyplot as plt
from nilearn import datasets
from nilearn import plotting

directory = 'C:/Users/torre/OneDrive/Desktop/RP2/sub3_nsd_images/training_images'
train_img_list = []
nsd_ids = []
for filename in os.listdir(directory):
    if filename.startswith('train-'):
        train_img_list.append(filename)
        nsd_id_str = filename[-9:-4]
        nsd_id = int(nsd_id_str)
        nsd_ids.append(nsd_id)
lh_fmri = np.load('lh_training_fmri (3).npy')
rh_fmri = np.load('rh_training_fmri (3).npy')
print(nsd_ids)
print(len(nsd_ids))

nsd_embeddings = pd.DataFrame()
embeddings = np.load('layer8_all_nouns_embeddings_data.npy', allow_pickle=True)
sorted_embeddings = pd.DataFrame(embeddings)
print(sorted_embeddings.shape)
#word_count = np.load('subject_wordcount_data.npy', allow_pickle=True)
#wordcount_df = pd.DataFrame(word_count)
#sorted_embeddings.insert(1, 'words_removed', wordcount_df.iloc[:, 4])
sorted_embeddings[sorted_embeddings.columns[0]] = sorted_embeddings[sorted_embeddings.columns[0]].astype(int)
#sorted_embeddings = sorted_embeddings.sort_values(by=sorted_embeddings.columns[0])
fmri_index = []
print((sorted_embeddings.iloc[:, 0]))
for value in sorted_embeddings.iloc[:, 0]:
    if value in nsd_ids:
        nsd_embeddings = nsd_embeddings.append(sorted_embeddings[sorted_embeddings.iloc[:, 0] == value])
        fmri_index.append(nsd_ids.index(value))
np.save('nsd_embeddings.npy', nsd_embeddings)
print(len(fmri_index))
print(fmri_index)

lh_fmri = lh_fmri[fmri_index]
rh_fmri = rh_fmri[fmri_index]
num_train = int(np.round(len(fmri_index)/100*90))
idxs = np.arange(len(fmri_index))
idxs_train, idxs_val = idxs[:num_train], idxs[num_train:]

lh_fmri_train = lh_fmri[idxs_train]
lh_fmri_val = lh_fmri[idxs_val]
rh_fmri_train = rh_fmri[idxs_train]
rh_fmri_val = rh_fmri[idxs_val]

sbert_embeddings = np.load('nsd_embeddings.npy', allow_pickle=True)
sbert_embeddings = pd.DataFrame(sbert_embeddings)
print(sbert_embeddings.shape)
sbert_embeddings1 = sbert_embeddings.iloc[:, 3:]
sbert_embeddings1.to_csv('sbert_embeddings.csv')
sbert_embeddings_train = np.array(sbert_embeddings1.head(len(idxs_train)))
sbert_embeddings_val = np.array(sbert_embeddings1.tail(len(idxs_val)))
#wordcount_val = sbert_embeddings.iloc[:, 1].tail(len(idxs_val))
#wordcount_df = pd.DataFrame(wordcount_val, columns=["nsdid", "words_removed"])
#print(wordcount_val)
#print(wordcount_val.mean())
#wordcount_val.to_json("wordcount_val.json")
#filtered_wordcount = wordcount_df[wordcount_df.iloc[:, 1] < 1]
#print(filtered_wordcount)
#print(sbert_embeddings_train.shape)
#print(lh_fmri_train.shape)
fmri_data = pd.DataFrame(lh_fmri_train)
fmri_data.to_csv('fmri_data.csv')
reg_lh = FracRidgeRegressorCV().fit(sbert_embeddings_train, lh_fmri_train)
reg_rh = FracRidgeRegressorCV().fit(sbert_embeddings_train, rh_fmri_train)
lh_fmri_val_pred = reg_lh.predict(sbert_embeddings_val)
rh_fmri_val_pred = reg_rh.predict(sbert_embeddings_val)
#print(lh_fmri_val_pred.shape)
lh_correlation = np.zeros(lh_fmri_val_pred.shape[1])
for v in tqdm(range(lh_fmri_val_pred.shape[1])):
    lh_correlation[v] = corr(lh_fmri_val_pred[:, v], lh_fmri_val[:, v])[0]
print(lh_correlation.shape)
rh_correlation = np.zeros(rh_fmri_val_pred.shape[1])
for v in tqdm(range(rh_fmri_val_pred.shape[1])):
    rh_correlation[v] = corr(rh_fmri_val_pred[:,v], rh_fmri_val[:,v])[0]
#roi_mapping_file = nib.load('Kastner2015 (1).nii')
#roi_map = roi_mapping_file.get_fdata()
##df = pd.DataFrame(roi_map)
#print(df)
#kastner_header = roi_mapping_file.header
#print(kastner_header)
#roi_name_map = roi_mapping_file.get_fdata()
#roi_names = []
#lh_challenge_roi_file = nib.load('lh.Kastner2015 (1).nii')
#lh_challenge_roi = lh_challenge_roi_file.get_fdata()
#lh_roi_correlation = []
#for r1 in range(len(lh_challenge_roi)):
#    for r2 in roi_name_map[r1]:
#        if r2[0] != 0:
#            roi_names.append(r2[1])
#            lh_roi_idx = np.where(lh_challenge_roi[r1] == r2[0])[0]
#print(count)
#roi_names.append('All vertices')
#lh_roi_correlation.append(lh_correlation)
#lh_mean_roi_correlation = [np.mean(lh_roi_correlation[r]) for r in range(len(lh_roi_correlation))]
#plt.figure(figsize=(18,6))
#x = np.arange(len(roi_names))
#width = 0.30
#plt.bar(x, lh_mean_roi_correlation, width)
#plt.ylim(bottom=0, top=1)
#plt.xlabel('ROIs')
#plt.xticks(ticks=x, labels=roi_names)
#plt.ylabel('Mean Pearson\'s $r$')
#plt.show()
#voxel = np.argmax(lh_correlation)
#plt.scatter(lh_fmri_val_pred[:, voxel], lh_fmri_val[:, voxel])
#z = np.polyfit(lh_fmri_val_pred[:, voxel], lh_fmri_val[:, voxel], 1)
#p = np.poly1d(z)
#plt.plot(lh_fmri_val_pred[:, voxel], p(lh_fmri_val_pred[:, voxel]), "r--")
#text = f"y={z[0]:0.3f}x{z[1]:+0.3f}"
#plt.gca().text(0.05, 0.95, text)
#plt.ylim(-2, 2.5)
#plt.show()
#hemisphere = 'left'
#roi_dir = 'lh.all-vertices_fsaverage_space (9).npy'
#fsaverage_all_vertices = np.load(roi_dir)
#fsaverage_correlation = np.zeros(len(fsaverage_all_vertices))
#if hemisphere == 'left':
#    fsaverage_correlation[np.where(fsaverage_all_vertices)[0]] = lh_correlation
#fsaverage = datasets.fetch_surf_fsaverage('fsaverage')
#view = plotting.view_surf(
#    surf_mesh=fsaverage['infl_'+hemisphere],
#    surf_map=fsaverage_correlation,
#    bg_map=fsaverage['sulc_'+hemisphere],
#    threshold=1e-14,
#    cmap='cold_hot',
#    colorbar=True,
#    title='Encoding accuracy'
#)
#view.save_as_html('lh_sub_surface_plot.html')
roi_mapping_file = ['mapping_prf-visualrois (3).npy', 'mapping_streams (3).npy', 'mapping_floc-bodies (3).npy', 'mapping_floc-places (3).npy']
roi_name_map = []
for r in roi_mapping_file:
    roi_name_map.append(np.load(r, allow_pickle=True).item())
lh_challenge_roi_file = ['lh.prf-visualrois_challenge_space (3).npy', 'lh.streams_challenge_space (3).npy', 'lh.floc-bodies_challenge_space (3).npy', 'lh.floc-places_challenge_space (3).npy']
rh_challenge_roi_file = ['rh.prf-visualrois_challenge_space (3).npy', 'rh.streams_challenge_space (3).npy', 'rh.floc-bodies_challenge_space (3).npy', 'rh.floc-places_challenge_space (3).npy']
lh_challenge_roi = []
rh_challenge_roi = []
for r in range(len(lh_challenge_roi_file)):
    lh_challenge_roi.append(np.load(lh_challenge_roi_file[r]))
    rh_challenge_roi.append(np.load(rh_challenge_roi_file[r]))
roi_names = []
lh_roi_correlation = []
rh_roi_correlation = []
for r1 in range(len(lh_challenge_roi)):
    for r2 in roi_name_map[r1].items():
        if r2[0] != 0:
            roi_names.append(r2[1])
            lh_roi_idx = np.where(lh_challenge_roi[r1] == r2[0])[0]
            rh_roi_idx = np.where(rh_challenge_roi[r1] == r2[0])[0]
            lh_roi_correlation.append(lh_correlation[lh_roi_idx])
            rh_roi_correlation.append(rh_correlation[rh_roi_idx])
roi_names.append('All vertices')
lh_roi_correlation.append(lh_correlation)
rh_roi_correlation.append(rh_correlation)
lh_mean_roi_correlation = [np.mean(lh_roi_correlation[r]) for r in range(len(lh_roi_correlation))]
rh_mean_roi_correlation = [np.mean(rh_roi_correlation[r]) for r in range(len(rh_roi_correlation))]
print(lh_mean_roi_correlation)
print(rh_mean_roi_correlation)
plt.figure(figsize=(18,6))
x = np.arange(len(roi_names))
width = 0.30
plt.bar(x - width/2, lh_mean_roi_correlation, width, label='Left Hemisphere')
plt.bar(x + width/2, rh_mean_roi_correlation, width, label='Right Hemisphere')
plt.xlabel('ROIs')
plt.xticks(ticks=x, labels=roi_names)
plt.ylabel('Mean Pearson\'s $r$')
plt.show()


