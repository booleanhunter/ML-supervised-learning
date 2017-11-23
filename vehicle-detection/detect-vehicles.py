import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
import pickle
import copy
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from skimage.feature import hog
from sklearn.externals import joblib
from scipy import ndimage as ndi
from moviepy.editor import VideoFileClip
from collections import deque
from sklearn.cross_validation import train_test_split

import numpy as np
import cv2
from skimage.feature import hog

def convert_color(img, conv='YCrCb'):
    if conv == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'HLS':
        return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

"""
Template matching is not a particularly robust method for finding vehicles unless you know exactly what your target object looks like.
However, raw pixel values are still quite useful to include in your feature vector in searching for cars.
While it could be cumbersome to include three color channels of a full resolution image, you can perform 
spatial binning on an image and still retain enough information to help in finding vehicles.
"""
def bin_spatial(img, size=(32, 32)):
    color1 = cv2.resize(img[:,:,0], size).ravel()
    color2 = cv2.resize(img[:,:,1], size).ravel()
    color3 = cv2.resize(img[:,:,2], size).ravel()
    return np.hstack((color1, color2, color3))

"""
An image template is useful for detecting things that do not vary much in their appearance - for example, icons of emojis. 
But for most real world objects that do appear in different forms, orientation, and sizes, this technique does not work quite well. 
In template matching, you depend on raw color values laid out in a specific order, and that can vary a lot. So you need to find some transformations that are robust to changes in appearance. One such transform is to compute a histogram of color values for an image.
When you compare the histogram of a known object with the regions of a test image, locations with a similar color distribution 
will reveal a close match. So we are no longer sensitive to a perfect arrangement of pixels. So objects that appear in slightly 
different orientations and sizes will still be a match.
"""
# Define a function to compute color histogram features
# NEED TO CHANGE bins_range if reading .png files with mpimg!
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


# Define a function to return HOG features and visualization
"""
The scikit-image hog() function takes in a single color channel or grayscaled image as input, as well as various parameters. 
These parameters include orientations, pixels_per_cell and cells_per_block.

The number of orientations is specified as an integer, and represents the number of orientation bins that the gradient 
information will be split up into in the histogram. Typical values are between 6 and 12 bins.

The pixels_per_cell parameter specifies the cell size over which each gradient histogram is computed. This paramater is 
passed as a 2-tuple so you could have different cell sizes in x and y, but cells are commonly chosen to be square.

The cells_per_block parameter is also passed as a 2-tuple, and specifies the local area over which the histogram counts in a given 
cell will be normalized. Block normalization is not necessarily required, but generally leads to a more robust feature set.

There is another optional power law or "gamma" normalization scheme set by the flag transform_sqrt. 
This type of normalization may help reduce the effects of shadows or other illumination variation, but will cause an error 
if your image contains negative values (because it's taking the square root of image values).
"""
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(imgs, cspace='RGB', orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # normalize the pixels.
        #image = image.astype(np.float32)/255
        # apply color conversion.
        feature_image = convert_color(image, cspace)

        spatial_features = bin_spatial(feature_image, size=spatial_size)
        file_features.append(spatial_features)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
        file_features.append(hist_features)
        # Append the new feature vector to the features list

        # Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        file_features.append(hog_features)
        # Append the new feature vector to the features list.
        features.append(np.concatenate(file_features))
    # Return list of feature vectors
    return features

# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                    xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]
    # Compute the span of the region to be searched
    xspan = x_start_stop[1] - x_start_stop[0]
    yspan = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per step in x/y
    nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
    ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
    # Compute the number of windows in x/y
    nx_windows = np.int(xspan/nx_pix_per_step) - 1
    ny_windows = np.int(yspan/ny_pix_per_step) - 1
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    # Note: you could vectorize this step, but in practice
    # you'll be considering windows one by one with your
    # classifier, so looping makes sense
    for ys in range(ny_windows):
        for xs in range(nx_windows):
            # Calculate window position
            startx = xs*nx_pix_per_step + x_start_stop[0]
            endx = startx + xy_window[0]
            starty = ys*ny_pix_per_step + y_start_stop[0]
            endy = starty + xy_window[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Convert the coordinates into integer values first.
        x1, y1 = bbox[0]
        x2, y2 = bbox[1]

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        bbox_int = []
        bbox_int.append((x1, y1))
        bbox_int.append((x2, y2))
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox_int[0], bbox_int[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

"""
Now, we define two new functions: single_img_features() and search_windows(). 
We can use these to search over all the windows defined by your slide_windows(), extract features at each window position, 
and predict with our classifier on each set of features.
"""
# Define a function to extract features from a single image window
# This function is very similar to extract_features()
# just for a single image rather than list of images
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9,
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):

    img = img.astype(np.float32)/255
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# Define a function you will pass an image
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB',
                    spatial_size=(32, 32), hist_bins=32,
                    hist_range=(0, 256), orient=9,
                    pix_per_cell=8, cell_per_block=2,
                    hog_channel=0, spatial_feat=True,
                    hist_feat=True, hog_feat=True):

    #1) Create an empty list to receive positive detection windows
    on_windows = []
    #2) Iterate over all windows in the list
    for window in windows:
        #3) Extract the test window from original image
        test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        #4) Extract features for that window using single_img_features()
        features = single_img_features(test_img, color_space=colorspace,
                            spatial_size=spatial_size, hist_bins=hist_bins,
                            orient=orient, pix_per_cell=pix_per_cell,
                            cell_per_block=cell_per_block,
                            hog_channel=hog_channel, spatial_feat=True,
                            hist_feat=True, hog_feat=True)
        #5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))

        #6) Predict using your classifier
        prediction = clf.predict(test_features)
        #7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    #8) Return windows for positive detections
    return on_windows

# Convert windows to heatmap numpy array.
def create_heatmap(windows, image_shape):
    background = np.zeros(image_shape[:2])
    for window in windows:
        background[window[0][1]:window[1][1], window[0][0]:window[1][0]] += 1
    return background

# find the nonzero areas from a heatmap and
# turn them to windows
def find_windows_from_heatmap(image):
    hot_windows = []
    # Threshold the heatmap
    thres = 0
    image[image <= thres] = 0
    # Set labels
    labels = ndi.label(image)
    # iterate through labels and find windows
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        hot_windows.append(bbox)
    return hot_windows

"""
Now that we've got several feature extraction methods in your toolkit, we're almost ready to train a classifier, 
but first, as in any machine learning application, we need to normalize your data. Python's sklearn package provides 
you with the StandardScaler() method to accomplish this task. To read more about how you can choose different normalizations 
with the StandardScaler() method, check out the documentation
"""
def combine_boxes(windows, image_shape):
    hot_windows = []
    image = None
    if len(windows)>0:
        # Create heatmap with windows
        image = create_heatmap(windows, image_shape)
        # find boxes from heatmap
        hot_windows = find_windows_from_heatmap(image)
    # return new windows
    return hot_windows


# Divide up into cars and notcars
car_images = glob.glob('./images/vehicles/vehicles/*/*png')
non_car_images = glob.glob('./images/non-vehicles/non-vehicles/*/*png')
cars = []
notcars = []
for image in car_images:
    cars.append(image)

for image in non_car_images:
    notcars.append(image)


colorspace = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 8
pix_per_cell = 8
cell_per_block = 2
hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16)
hist_bins = 32
hist_range=(0, 256)
train_model = True
#HLS, 4, 8, 95+
#YCrCb, 4, 8, 95+
filename_train = './classifier.joblib.pkl'
filename_scaler = './scaler.joblib.pkl'

# parameters for GridSearchCV
#grid_search_parameters = {'kernel':('linear', 'rbf', 'poly'), 'C':[0.001, 0.01, 0.1, 1, 10], 'gamma':  [0.001, 0.01, 0.1, 1]}
if train_model:
    t=time.time()
    car_features = extract_features(cars, cspace=colorspace, orient=orient,
                        pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                        hog_channel=hog_channel)
    notcar_features = extract_features(notcars, cspace=colorspace, orient=orient,
                            pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                            hog_channel=hog_channel)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract HOG features...')

    print("car feature shape: ", len(car_features))
    print("non-car feature shape: ", len(notcar_features))
        # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
    scaled_X = X_scaler.transform(X)

# Define the labels vector



# Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    # Use a linear SVC
    clf = svm.SVC(kernel='linear', C=0.001, gamma=0.001)
    # Check the training time for the SVC
    t=time.time()
    #clf = GridSearchCV(svc, grid_search_parameters)
    clf.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(clf.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    n_predict = 10
    print('My SVC predicts: ', clf.predict(X_test[0:n_predict]))
    print('For these',n_predict, 'labels: ', y_test[0:n_predict])
    t2 = time.time()
    print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')

        # save the trained model
    _ = joblib.dump(clf, filename_train, compress=9)
    _ = joblib.dump(X_scaler, filename_scaler, compress=9)
else:
    # load the trained model
    clf = joblib.load(filename_train)
    X_scaler = joblib.load(filename_scaler)


def process_image(image):
    """
    Pipeline to detect and track vehicles across images of video frames
    """
    draw_image = np.copy(image)

    windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[400, 640],
                    xy_window=(96, 96), xy_overlap=(0.75, 0.75))

    windows += slide_window(image, x_start_stop=[32, None], y_start_stop=[400, 610],
                    xy_window=(144, 144), xy_overlap=(0.75, 0.75))
    windows += slide_window(image, x_start_stop=[410, 1280], y_start_stop=[390, 540],
                    xy_window=(192, 192), xy_overlap=(0.75, 0.75))

    hot_windows = search_windows(image, windows, clf, X_scaler, color_space=colorspace,
                        spatial_size=spatial_size, hist_bins=hist_bins,
                        orient=orient, pix_per_cell=pix_per_cell,
                        cell_per_block=cell_per_block,
                        hog_channel=hog_channel, spatial_feat=True,
                        hist_feat=True, hog_feat=True)


    #draw_image = draw_boxes(draw_image, hot_windows, color=(255, 0, 0), thick=6)
    combined_windows = combine_boxes(hot_windows, image.shape)
    filtered_windows = []
    # no car detection yet, create new detections and add them to the list.
    if len(detections) == 0:
        for window in combined_windows:
            box_points = get_box_points(window)
            new_car = Detection()
            new_car.add(box_points)
            detections.append(new_car)
            window_img = draw_boxes(draw_image, filtered_windows, color=(0, 0, 255), thick=6)
            return window_img
    else:
        boxes_copy = copy.copy(combined_windows)
        # Run thorugh all the existing detections and see if any new detections
        # matche with them.
        # if match is found add to the detection.
        # If not found decrease the confidence of the previous detection.
        non_detected_cars_idxs = []
        for car_idx, car in enumerate(detections):
            match_found = False
            box_detection_idx = 0
            for idx, box in enumerate(boxes_copy):
                box_points = get_box_points(box)
                if car.match_detection(box_points):
                    match_found = True
                    if car.consecutive_detection >= min_consecutive_detection:
                        average_box = car.average_detections()
                        filtered_windows.append(((average_box[0],average_box[1]),(average_box[2], average_box[3])))

		    # remove after the match.
                    box_detection_idx = idx
                    # Match for the car is found, break the inner loop
                    break

            # Match not found for the previous detection, decrease its confidence.
            # The delete detections is true, remove the detection from the list of previous detections.
            if not match_found:
                delete_Detection = car.failed_detect()
                if delete_Detection:
                    non_detected_cars_idxs.append(car_idx)
                else:
                    average_box = car.average_detections()
                    filtered_windows.append(((average_box[0],average_box[1]),(average_box[2], average_box[3])))
            else:
                # Delete the detected box from the list of boxes to be mathched.
                del boxes_copy[box_detection_idx]

        # Remove all the undetected cars from the list of detections using thier saved index.
        if len(non_detected_cars_idxs) > 0:
             non_detected_cars_idxs =  non_detected_cars_idxs[::-1]
             for i in non_detected_cars_idxs:
                del detections[i]

        # Add the unmatched boxes to the detections array.
        for box in boxes_copy:
            box_points = get_box_points(box)
            new_car = Detection()
            new_car.add(box_points)
            detections.append(new_car)

            # If the match is not found decrease the confidence of the detection.



    window_img = draw_boxes(draw_image, filtered_windows, color=(0, 0, 255), thick=6)

    return window_img

def get_box_points(box):
    """
    Takes in box points of form ((x1,y1), (x2, y2)) and converts it to form
    [x1, y1, x2, y2].
    """
    box_points = []
    x1, y1 = box[0]
    x2, y2 = box[1]

    box_points.append(x1)
    box_points.append(y1)
    box_points.append(x2)
    box_points.append(y2)
    return box_points


margin = 100
min_consecutive_detection = 8
max_allowed_miss = 4
confidence_thresh = 10

def is_within_margin(a, b):
    if abs(a-b) > margin:
        return False
    return True

class Detection():
    def __init__(self):
        # the box coordinates in the form [x1,y1,x2,y2]
        self.last_box = []
        # number of consecutive frames in which the car has been detected.
        self.consecutive_detection = 0
        # number of consecutive frames in which the car has not been found.
        self.consecutive_miss = 0
        # the box coordinates of last n detections in the form deque([[x1, y1, x2, y2], [x1, y1, x2, y2], [x1, y1, x2, y2]...], maxlen=5)
        self.last_n_detections = deque(maxlen=10)
        # [avg x1 , avg y1, avg x2, avgy2] of last n detections.
        self.average_box = []

    def add(self, box):
        """
        box argument should be of format [x1, y1, x2, y2]
        """
        self.last_box = box
        self.consecutive_detection =  self.consecutive_detection + 1
        self.last_n_detections.append(box)
        self.average_detections()
        # set the previous count of consecutive misses to 0.
        self.consecutive_miss = 0

    def average_detections(self):
        """
        Find the mean of detections in the deque.
        """

        self.average_box = np.mean(self.last_n_detections, axis=0)
        return self.average_box

    def match_detection(self, box):
        """
        Checks whether the box is very close/similar to the [x1, y1, x2, y2]
        box argument should be of format [x1, y1, x2, y2]
        """
        i = 0
        for point in box:
            # see if all the points in the box lies within the margin of the last detection.

            if not is_within_margin(point, self.last_box[i]):
                return False
            i = i + 1
        # If the match found then add it to the detection.
        self.add(box)
        return True

    def failed_detect(self):
         delete_detection = True
         self.consecutive_miss = self.consecutive_miss + 1
         # In case the car doesn't get for more than 3 frames consecutively we discard the
         # object.
         if self.consecutive_miss  > max_allowed_miss:
             return delete_detection
        # This helps remove the stray false positives which doesn't get detected in
        # consecutive frames.
         if self.consecutive_detection < min_consecutive_detection:
             return delete_detection


         # Wait till you the miss becomes greater than max_allowed_miss.
         return False


# array of Detection class.
detections = []
# output video directory
video_output = './project_video_class_full.mp4'
# input video directory
clip1 = VideoFileClip("./video-tracking-output.mp4")
# video process pipline
video_clip = clip1.fl_image(process_image)
# write processed files
video_clip.write_videofile(video_output, audio=False)