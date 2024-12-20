
import os
import pickle
import numpy as np
from time import time
from common import *
import matplotlib.pyplot as plt
from load_fif2 import load_dataset2
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, permutation_test_score

if __name__ == '__main__':

	np.random.seed(0)

	class_labels = ['Face', 'Place']
	scorings = 'roc_auc' 
	n_folds = 5
	shuffle = False
	remove_outliers = True

	t_min = -0.25  
	t_max = 6.25  
	time_window_size = 0.5 
	time_step = 0.5  
	epsilon = 0.01 
	t_mins = np.arange(t_min, t_max - time_window_size + epsilon, time_step)
	t_maxs = np.arange(t_min + time_window_size, t_max + epsilon, time_step)
	time_windows = list(zip(t_mins, t_maxs))

	runs_id = [1, 2, 3, 4]
	eye_tracking = True
	eye_speed_max = 300.0  # degrees / s
	eye_pupil_max = 0.5
	return_epochs = False
	mag_grad = True
	tangent_space = False
	psd_decoding = False
	rating_min = 3
	p_value_threshold = 0.05
	n_permutations = 1000
	outdir = 'decoding_output/timeresolved/'

	solver = 'lbfgs'
	tol = 1.0e-3
	n_jobs = 10

	for ss in range(3,11):

		filename_template = filename_templates[ss]
		inputdir = 'preprocessing/' + filename_template[34:38] + '/'

		print(' ')
		print('subject %s' % filename_template[34:38])

		results_time = []
		time_significant_idx = []
		for t_start, t_stop in time_windows:

			print(' ')
			print('time window %.2f - %.2f' % (t_start, t_stop))

			X, y_original, visual_cues, ratings, self_reportings, runs, timesteps, picks, trl_id, meg_info = load_dataset2(filename_template,
																															directory=inputdir,
																															runs_id=runs_id,
																															t_start=t_start,
																															t_stop=t_stop,
																															shuffle=shuffle,
																															rating_min=rating_min,
																															eye_tracking=eye_tracking,
																															eye_speed_max=eye_speed_max,
																															eye_pupil_max=eye_pupil_max,
																															mag_grad=mag_grad,
																															return_epochs=return_epochs)

			le = LabelEncoder()
			le.fit(y_original)
			y = le.transform(y_original)
			X = np.nan_to_num((X - X.mean(2)[:, :, None]) / X.std(2)[:, :, None])
			n_trials = X.shape[0]

			print(" ")
			print("Cross-validated %s using signal in time" % scorings)
			clf = LogisticRegression(solver=solver, tol=tol)
			cv = StratifiedKFold(n_splits=n_folds, random_state=0, shuffle=True)
			print(cv)  
			X_time = np.hstack([X[:, ch, :] for ch in range(X.shape[1])])

			print("Computing %s permutations for %s time score" % (n_permutations, scorings))
			t0 = time()
			score, permutation_scores, p_value = permutation_test_score(clf, X_time, y, scoring=scorings, cv=cv, n_permutations=n_permutations, n_jobs=n_jobs)
			print("%s sec" % (time() - t0))
			print("p_value = %s" % p_value)
			results = score
			idx_time = p_value <= p_value_threshold
			print("time %s : %s" % (scorings, results))
			results_time.append(results)
			time_significant_idx.append(idx_time)

			print(" ")
			output_file = '%s_decoding_output_%.2f_%.2f.pickle' % (filename_template[34:38], t_start, t_stop)
			dirname = outdir + filename_template[34:38] + '/'
			if not os.path.exists(dirname):
				os.makedirs(dirname)
			mdict = {}
			mdict['time_win'] = [t_start, t_stop]
			mdict['score_time'] = score
			mdict['p_value_time'] = p_value
			mdict['permutation_time'] = permutation_scores
			mdict['mask_time'] = idx_time
			mdict['trial_number'] = n_trials
			pickle.dump(mdict, open(dirname + output_file, 'wb'), protocol=2)

