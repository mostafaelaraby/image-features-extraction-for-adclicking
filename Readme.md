# Image Features Extractor
This project used to extract features from input folder images , these features are mainly used for ad clicking predictions .
These features were used in https://www.kaggle.com/c/avito-demand-prediction contest.



### Prerequisites
Install all packages in the requirements.txt


> pip install -r requirements.txt


## Getting Started
Run example.py to get a results.csv file including all features extracted from input test_data Folder.

## Available functions    
> image = cv2.imread(image_path)
### calculate image simplicity
Used to calculate simplicity of input image.

    calculate_image_simplicity(image,c_threshold = 0.01,nchannels=3,nbins =8)

### image basic segment stats
Used to extract basic image segmentation statistics (tuple of 10 features).

    image_basic_segment_stats(image)

### image face feats
Used to extract number of faces from input image using pretrained HaarCascade from opencv.

    image_face_feats(image)

### image sift feats
number of sift keypoints extracted from input image

    image_sift_feats(image)

### image rgb simplicity
get image simplicity feature from RGB image

    image_rgb_simplicity(image)

### image hsv simplicity
get image simplicity features from hsv image

    image_hsv_simplicity(image)

### image hue histogram
image features from histogram of HSV images

    image_hue_histogram(image)

### image grayscale simplicity
used for simplicity features on grayscale images
    
    image_grayscale_simplicity(image)

### image sharpness
used to calculate image sharpness score

    image_sharpness(image)

## image contrast 
used to calculate image contrast score

    image_contrast(image)

## image saturation
used to calculate image saturation
> image_saturation(image)

## image brightness
used to calculate image brightness score
> image_brightness(image)

## image colorfulness
used to calculate colorfulness score based on the paper

    image_colorfulness(image)

## Extract image feats
used to calculate all previous features and put it in a dataframe saved to csv


    extract_image_feats(out file name , input file list of images, number of parallel jobs)

## Results 
Results scored on https://www.kaggle.com/c/avito-demand-prediction contest
These features with simple LightGBM model it got me (Root Mean Squared Error (RMSE):<br>
- 0.2207 on public leaderboard
- 0.2246 on private leaderboard



## References
- Dimitri Ad Clicking prediction paper [https://maths-people.anu.edu.au/~johnm/courses/mathdm/talks/dimitri-clickadvert.pdf](https://maths-people.anu.edu.au/~johnm/courses/mathdm/talks/dimitri-clickadvert.pdf)
- Opencv Haar Cascade  model
 



