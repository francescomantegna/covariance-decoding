
import numpy as np
from scipy.stats import iqr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from nilearn.plotting import plot_connectome
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.inspection import permutation_importance
plt.ion()

n_sources = 98 
n_permutations = 1000

subject_number = 1
inputdir = 'datarep/'
filename_template = 'subject_%s_covariance_source.npz'

data = np.load(inputdir + filename_template % (subject_number), allow_pickle=True)
Xs = data['x']
ys = data['y']

covidx = np.tril_indices(n_sources,k=-1)
Ds = np.array([Xs[ii][covidx] for ii in range(len(Xs))])

clf = LogisticRegression(solver='lbfgs', C=0.01, max_iter=10000, tol=1.0e-3)
model = clf.fit(Ds, ys)
model.score(Ds, ys)
coefficients = model.coef_
face_features = np.clip(coefficients,a_min=0,a_max=None)
place_features = np.clip(coefficients,a_min=None,a_max=0)

print(' ')
print('permutation feature importance')
imp = permutation_importance(model, Ds, ys, n_repeats=n_permutations, random_state=0, scoring='roc_auc', n_jobs=10)
threshold = iqr(np.abs(imp['importances_mean'][np.nonzero(imp['importances_mean'])[0]]),rng=(5,95))
mask = np.abs(imp['importances_mean']) > threshold

face_feature_matrix = np.zeros((n_sources,n_sources))
face_feature_matrix[covidx] = face_features * mask
face_feature_matrix.T[covidx] = face_features * mask
place_feature_matrix = np.zeros((n_sources,n_sources))
place_feature_matrix[covidx] = place_features * mask
place_feature_matrix.T[covidx] = place_features * mask

beta_coefs = face_feature_matrix - np.abs(place_feature_matrix)

atlas = np.load(inputdir + 'atlas_labels.npz', allow_pickle=True)
labelnames = atlas['names']
labelpos = atlas['pos']
labelcolors = atlas['colors']

fig = plt.figure(figsize=(9,8), constrained_layout=True)
gs = GridSpec(2, 2, height_ratios=[1.0, 1.0], hspace=-0.1, wspace=-0.1, figure=fig)

ax_top = fig.add_subplot(gs[0, :])
plot_connectome(beta_coefs, np.array(labelpos), edge_threshold='95%', edge_vmin=-0.03, edge_vmax=0.03, node_color=labelcolors,
				title=None, edge_cmap='bwr', figure=fig, axes=ax_top, display_mode='lr', node_size=15, alpha=1.0, colorbar=False)
face_line = Line2D([0], [0], color='red', linestyle='-', label='face')
place_line = Line2D([0], [0], color='blue', linestyle='-', label='place')
ax_top.legend(handles=[face_line, place_line], loc=9, fontsize=16)

left_sub = np.tril(beta_coefs[:49,:49])

ax_bottom_left = fig.add_subplot(gs[1, 0])
ax_bottom_left.imshow(left_sub, aspect='auto',vmin=-0.03, vmax=0.03, cmap='bwr')
ax_bottom_left.set_xticks(np.arange(49),labelnames[:49],rotation=270)
for tick_label, color in zip(ax_bottom_left.get_xticklabels(), labelcolors[:49]):
    tick_label.set_color(color)
ax_bottom_left.set_yticks(np.arange(49),labelnames[:49])
for tick_label, color in zip(ax_bottom_left.get_yticklabels(), labelcolors[:49]):
    tick_label.set_color(color)
ax_bottom_left.tick_params(labelsize=6)
ax_bottom_left.spines['right'].set_visible(False)
ax_bottom_left.spines['top'].set_visible(False)
ax_bottom_left.yaxis.set_ticks_position('left')
ax_bottom_left.xaxis.set_ticks_position('bottom')
ax_bottom_left.set_title('left hemisphere')

right_sub = np.tril(beta_coefs[49:,49:])

ax_bottom_right = fig.add_subplot(gs[1, 1])
ax_bottom_right.imshow(right_sub, aspect='auto', vmin=-0.03, vmax=0.03, cmap='bwr')
ax_bottom_right.set_xticks(np.arange(49),labelnames[49:],rotation=270)
for tick_label, color in zip(ax_bottom_right.get_xticklabels(), labelcolors[49:]):
    tick_label.set_color(color)
ax_bottom_right.set_yticks(np.arange(49),labelnames[49:])
for tick_label, color in zip(ax_bottom_right.get_yticklabels(), labelcolors[:49]):
    tick_label.set_color(color)
ax_bottom_right.tick_params(labelsize=6)
ax_bottom_right.spines['right'].set_visible(False)
ax_bottom_right.spines['top'].set_visible(False)
ax_bottom_right.yaxis.set_ticks_position('left')
ax_bottom_right.xaxis.set_ticks_position('bottom')
ax_bottom_right.set_title('right hemisphere')
