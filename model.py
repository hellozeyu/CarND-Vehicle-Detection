from sklearn.cross_validation import cross_val_score
from sklearn.svm import LinearSVC
from bayes_opt import BayesianOptimization
from vehicle_detect import extract_features
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import numpy as np
import glob
import pickle
import time

cars = glob.glob('./data/vehicles/**/*.png', recursive=True)
notcars = glob.glob('./data/non-vehicles/**/*.png', recursive=True)

color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (32, 32) # Spatial binning dimensions
hist_bins = 32    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off

car_features = extract_features(cars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)
notcar_features = extract_features(notcars, color_space=color_space,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=spatial_feat,
                        hist_feat=hist_feat, hog_feat=hog_feat)

X = np.vstack((car_features, notcar_features)).astype(np.float64)
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))


# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
    scaled_X, y, test_size=0.2, random_state=rand_state)

def svccv(C):
    val = cross_val_score(
        LinearSVC(C=C, random_state=2),
        X_train, y_train, 'f1', cv=2
    ).mean()

    return val

gp_params = {"alpha": 1e-5}
svcBO = BayesianOptimization(svccv,
        {'C': (0.001, 100)})
svcBO.maximize(n_iter=10, **gp_params)
print('-' * 53)
print('Final Results')
print(svcBO.res['max']['max_params'])
print('SVC: %f' % svcBO.res['max']['max_val'])

best_para = svcBO.res['max']['max_params']

print('Using:',orient,'orientations',pix_per_cell,
    'pixels per cell and', cell_per_block,'cells per block')
print('Feature vector length:', len(X_train[0]))
# Use a linear SVC
# svc = LinearSVC(C=best_para['C'])
svc = LinearSVC()
# Check the training time for the SVC
t=time.time()
svc.fit(X_train, y_train)
t2 = time.time()
print(round(t2-t, 2), 'Seconds to train SVC...')
# Check the score of the SVC
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
accuracy_score = svc.score(X_test, y_test)
# Check the prediction time for a single sample
t=time.time()

dist_pickle = {}
dist_pickle["svc"] = svc
dist_pickle["scaler"] = X_scaler
dist_pickle["orient"] = orient
dist_pickle["pix_per_cell"] = pix_per_cell
dist_pickle["cell_per_block"] = cell_per_block
dist_pickle["spatial_size"] = spatial_size
dist_pickle["hist_bins"] = hist_bins
dist_pickle["color_space"] = color_space
dist_pickle["accuracy_score"] = accuracy_score
dist_pickle['hog_channel'] = hog_channel

pickle.dump( dist_pickle, open( "svc_pickle.p", "wb" ) )
