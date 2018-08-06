# -*- coding: utf-8 -*-
"""Image Features module.

This module is used to extract image features that can be later used for ad clicking prediction
  as in https://maths-people.anu.edu.au/~johnm/courses/mathdm/talks/dimitri-clickadvert.pdf .

Example:
        $ python example.py
"""
import os
import math
import numpy as np
import cv2
import imutils
import math
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed,ThreadPoolExecutor
import pandas as pd

def segment_image_watershed(image):
	"""
	Args:
		image (numpy array): input colored image

	Returns:
		list: watershed output labels.
	"""
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
	# compute the exact Euclidean distance from every binary
	# pixel to the nearest zero pixel, then find peaks in this
	# distance map
	D = ndimage.distance_transform_edt(thresh)
	localMax = peak_local_max(D, indices=False, min_distance=20,labels=thresh)
	markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
	labels = watershed(-D, markers, mask=thresh)
	return labels


def calculate_image_simplicity(image,c_threshold = 0.01,nchannels=3,nbins =8):
	"""
	Args:
		image (numpy array): input colored image
		c_threshold (float 0-1): threshold on the maximum of the histogram value to be used in the output simplicity feature
		nchannel(int): 3 for colored images and 1 for grayscale
		nbins(int): number of bins used to calculate histogram

	Returns:
		tuple: returns 2 features representing image simplicity .
	"""
	feature_1 =0
	max_bin = -1
	max_channel = -1
	bin_index = -1
	for channel in  range(nchannels):
		hist = cv2.calcHist(image, [channel], None,[nbins],[0,256])
		maximum = hist.max()
		feature_1 += np.sum([1 if hist[i]>=(c_threshold*maximum) else 0 for i in range(8)])

		if max_bin<maximum:
			max_bin = maximum
			max_channel = channel
			bin_index = np.where(hist == max_bin)[0]

	feature_2 = max_bin *100.0 /  image.flatten().shape[0]
	return feature_1,feature_2	

def get_segmented_image(image,labels,segment_id):
	"""
	Args:
		image (numpy array): input colored image
		labels : output labels from watershed calling from segment_image_watershed

	Returns:
		tuple: returns 2 features representing image simplicity .
	"""
	mask = np.zeros(image.shape, dtype="uint8")
	mask[labels == segment_id] = 1
	return image *mask

def image_basic_segment_stats(image):
	"""
	Args:
		image (numpy array): input colored image

	Returns:
		tuple: returns segmentation statistics (10 features as in the paper of ad clicking).
	"""
	labels = segment_image_watershed(image)
	n_segments =  len(np.unique(labels)) - 1
	regions_size = []
	max_region = -1
	max_region_index = -1
	for segment_label in range(1,n_segments+1):
		n_pixels = np.count_nonzero(labels == segment_label)
		regions_size.append(n_pixels)
		if n_pixels>max_region:
			max_region = n_pixels
			max_region_index = segment_label

	regions_size.sort()
	if len(regions_size)>=2:
		contrast_segments_size = regions_size[-1]-regions_size[0]
		ratio_largest_component = regions_size[-1]*100.0 / image.flatten().shape[0]
		ratio_second_largest_component = regions_size[-2]*100 / image.flatten().shape[0]
	else:
		contrast_segments_size = -1
		ratio_largest_component = -1
		ratio_second_largest_component = -1
	image_segmented = get_segmented_image(image,labels,max_region_index)
	hue_1,hue_2,hue_3 = image_hue_histogram(image_segmented)
	bright_1,bright_2,bright_3,_ = image_brightness(image_segmented)
	return n_segments,contrast_segments_size,ratio_largest_component\
	,ratio_second_largest_component,hue_1,hue_2,hue_3\
	,bright_1,bright_2,bright_3

def image_face_feats(image):
	"""
	Args:
		image (numpy array): input colored image

	Returns:
		int: number of faces in the input image based on pretrained opencv haarcascade for face and eyes .
	"""
	current_dir = os.path.dirname(os.path.realpath(__file__))
	face_cascade = cv2.CascadeClassifier(os.path.join(current_dir,'models','haarcascade_frontalface_default.xml'))
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	eye_cascade = cv2.CascadeClassifier(os.path.join(current_dir,'models','haarcascade_eye.xml'))
	eyes = eye_cascade.detectMultiScale(gray)
	nfaces = 0
	if (len(eyes)/2)>len(faces):
		nfaces = len(eyes)/2
	else:
		nfaces = len(faces)
	return nfaces

def image_sift_feats(image):
	"""
	Args:
		image (numpy array): input colored image

	Returns:
		int: number of keypoints from sift
	"""
	gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	kp = sift.detect(gray,None)
	return len(kp)

def image_rgb_simplicity(image):
	"""
	Args:
		image (numpy array): input colored rgb image

	Returns:
		image simplicity features 
	"""
	return calculate_image_simplicity(image)

def image_hsv_simplicity(image):
	"""
	Args:
		image (numpy array): input colored RGB image

	Returns:
		image simplicity features for HSV images
	"""
	image =  cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	return calculate_image_simplicity(image,0.05,1,20)

def image_hue_histogram(image):
	"""
	Args:
		image (numpy array): input colored  image

	Returns:
		image  features from hue histogram
	"""
	image =  cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	(H,S,V) =  cv2.split(image.astype("float"))
	hist = cv2.calcHist(image, [0], None,[20],[0,256])
	c_threshold = 0.01
	maximum = hist.max()
	feature_1 = np.sum([1 if hist[i]>=(c_threshold*maximum) else 0 for i in range(20)])
	max_2 = -1

	for i in range(20):
		if hist[i]==maximum:
			continue
		if hist[i]>max_2:
			max_2 = hist[i]
	feature_2 = maximum-max_2

	return feature_1,feature_2[0],np.std(H)


def image_grayscale_simplicity(image):
	"""
	Args:
		image (numpy array): input colored  image

	Returns:
		image simplicity features based on grayscale image
	"""
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	std = np.std(image)
	hist = cv2.calcHist(image, [0], None,[256],[0,256])
	maximum = hist.max()
	c_threshold = 0.01
	feature_2 = np.sum([1 if hist[i]>=(c_threshold*maximum) else 0 for i in range(256)])
	prune = int((2.5*1.0*255)/100)
	hist = hist[prune:255-prune]
	features_1 = 0
	for itm in hist:
		if itm>0:
			features_1+=1
	

	return features_1,feature_2,std

def image_sharpness(image):
	"""
	Args:
		image (numpy array): input colored  image

	Returns:
		image sharpness features on grayscale image
	"""
	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	return cv2.Laplacian(image, cv2.CV_64F).var()


def image_contrast(image):
	"""
	Args:
		image (numpy array): input colored  image

	Returns:
		image contrast features on HSV image
	"""
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	(Y,U,V) =  cv2.split(image.astype("float"))
	std = np.std(Y)
	maximum = Y.max()
	minimum = Y.min()
	if (maximum-minimum)<=0:
		return 0
	return std*1.0/(maximum-minimum)

def image_saturation(image):
	"""
	Args:
		image (numpy array): input colored  image

	Returns:
		image saturation features on HSV image
	"""
	image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
	(H,S,V) =  cv2.split(image.astype("float"))
	mean = np.mean(S)
	std = np.std(S)
	max_saturation = S.max()
	min_saturation = S.min()
	return mean,std,max_saturation,min_saturation

def image_brightness(image):
	"""
	Args:
		image (numpy array): input colored  image

	Returns:
		image brightness features on YUV image
	"""
	image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
	(Y,U,V) =  cv2.split(image.astype("float"))
	mean = np.mean(Y)
	std = np.std(Y)
	max_brightness = Y.max()
	min_brightness = Y.min()
	return mean,std,max_brightness,min_brightness


def image_colorfulness(image):
	"""
	Args:
		image (numpy array): input colored  image

	Returns:
		image colorfullness features as discussed in the paper
	"""
	# split the image into its respective RGB components
	(B, G, R) = cv2.split(image.astype("float"))
 
	# compute rg = R - G
	rg = np.absolute(R - G)
 
	# compute yb = 0.5 * (R + G) - B
	yb = np.absolute(0.5 * (R + G) - B)
 
	# compute the mean and standard deviation of both `rg` and `yb`
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))
 
	# combine the mean and standard deviations
	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
 
	# derive the "colorfulness" metric and return it
	return stdRoot + (0.3 * meanRoot)

def parallel_process(array, function, n_jobs=3, use_kwargs=False, front_num=1):
    """
        A parallel version of the map function with a progress bar. 

        Args:
            array (array-like): An array to iterate over.
            function (function): A python function to apply to the elements of array
            n_jobs (int, default=3): The number of cores to use
            use_kwargs (boolean, default=False): Whether to consider the elements of array as dictionaries of 
                keyword arguments to function 
            front_num (int, default=3): The number of iterations to run serially before kicking off the parallel job. 
                Useful for catching bugs
        Returns:
            [function(array[0]), function(array[1]), ...]
    """
    #We run the first few iterations serially to catch bugs
    if front_num > 0:
        front = [function(**a) if use_kwargs else function(a) for a in array[:front_num]]
    #If we set n_jobs to 1, just run a list comprehension. This is useful for benchmarking and debugging.
    if n_jobs==1:
        return front + [function(**a) if use_kwargs else function(a) for a in tqdm(array[front_num:])]
    #Assemble the workers
    with ThreadPoolExecutor(max_workers=n_jobs) as pool:
        #Pass the elements of array into function
        if use_kwargs:
            futures = [pool.submit(function, **a) for a in array[front_num:]]
        else:
            futures = [pool.submit(function, a) for a in array[front_num:]]
        kwargs = {
            'total': len(futures),
            'unit': 'it',
            'unit_scale': True,
            'leave': True
        }
        #Print out the progress as tasks complete
        for f in tqdm(as_completed(futures), **kwargs):
            pass
    out = []
    #Get the results from the futures. 
    for i, future in tqdm(enumerate(futures)):
        try:
            out.append(future.result())
        except Exception as e:
        	print(e)
    return front + out

def get_image_all_feats(image_path,img_width = 256):
	"""
	Args:
		img_path (str): takes image path as input
		img_width (int): with to which the image is resized

	Returns:
		a tuple of all features from the previous functions including file name at the start
	"""
	image = cv2.imread(image_path)
	image = imutils.resize(image,width=img_width)
	n_segments,contrast_segments_size,ratio_largest_component,ratio_second_largest_component,segment_hue_1,segment_hue_2,segment_hue_3,segment_bright_1,segment_bright_2,segment_bright_3 = image_basic_segment_stats(image)
	n_faces = image_face_feats(image)
	n_sift = image_sift_feats(image)
	rgb_simple_1,rgb_simple_2 = image_rgb_simplicity(image)
	hsv_simple_1,hsv_simple_2 = image_hsv_simplicity(image)
	gray_simple_1,gray_simple_2,gray_simple_3 = image_grayscale_simplicity(image)
	hue_hist_1,hue_hist_2,hue_hist_3 = image_hue_histogram(image)
	sharpness = image_sharpness(image)
	contrast = image_contrast(image)
	colorful = image_colorfulness(image)
	sat_1,sat_2,sat_3,sat_4 = image_saturation(image)
	bright_1,bright_2,bright_3,bright_4 = image_brightness(image) 
	return os.path.basename(image_path),\
	n_segments,contrast_segments_size,ratio_largest_component,ratio_second_largest_component,\
	segment_hue_1,segment_hue_2,segment_hue_3,\
	segment_bright_1,segment_bright_2,segment_bright_3,n_faces,n_sift,rgb_simple_1,rgb_simple_2,hsv_simple_1,hsv_simple_2,\
	hue_hist_1,hue_hist_2,hue_hist_3,gray_simple_1,gray_simple_2,gray_simple_3 ,sharpness,contrast,colorful,\
	sat_1,sat_2,sat_3,sat_4,bright_1,bright_2,bright_3,bright_4

def extract_image_feats(out_name,file_list,n_jobs=3):
	"""
	Args:
		out_name (str): name of the output file results
		file_list (list): list of input files

	Returns:
		write to out_name a dataframe including image name and all extracted features
	"""
	scores = parallel_process(file_list,get_image_all_feats,n_jobs=n_jobs)
	image_data = pd.DataFrame( scores,columns=['image','n_segments','contrast_segments_size','ratio_largest_component'\
			,'ratio_second_largest_component','segment_hue_1','segment_hue_2','segment_hue_3','segment_bright_1',\
			'segment_bright_2','segment_bright_3','n_faces','n_sift','rgb_simple_1','rgb_simple_2','hsv_simple_1','hsv_simple_2',\
			'hue_hist_1','hue_hist_2','hue_hist_3','gray_simple_1','gray_simple_2','gray_simple_3',\
			'sharpness', 'contrast','colorful','sat_1','sat_2','sat_3','sat_4','bright_1','bright_2','bright_3','bright_4'])
	image_data.to_csv(out_name,index=False)