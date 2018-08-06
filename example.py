from ImageFeatures import image_features
import pandas as pd
import time
import glob
t0 = time.time()
image_features.extract_image_feats('results.csv',glob.glob('test_data/*.jpg'),n_jobs=2)
total = time.time()-t0
print("took {} seconds".format(total))