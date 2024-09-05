import numpy as np
import cv2
from sklearn.linear_model import LinearRegression
from skimage.feature import hog
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from keras.applications import VGG16


data = np.load('training_images.npz', allow_pickle=True)
images = data['images']
pts = data['points']
print(images.shape, pts.shape)
# images.shape = (2811, 252, 252, 3) meaning 2811 images 252 x 252 pixels using RGB (3 channels)
# pts.shape = (2811, 50, 2) meaning 2811 points (one set of points for each image). Each image made of 50 points. Each point made of (x, y) coordinate (2)

test_data = np.load('test_images.npz', allow_pickle=True)
test_images = test_data['images']
print(test_images.shape)

example_data = np.load('examples.npz', allow_pickle=True)
example_images = example_data['images']
print(example_images.shape)


def visualise_pts(img, pts):
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.plot(pts[:, 0], pts[:, 1], '+r')
    plt.show()


def euclid_dist(pred_pts, gt_pts):
    """
      Calculate the euclidean distance between pairs of points
      :param pred_pts: The predicted points
      :param gt_pts: The ground truth points
      :return: An array of shape (no_points,) containing the distance of each predicted point from the ground truth
    """
    import numpy as np
    pred_pts = np.reshape(pred_pts, (-1, 2))
    gt_pts = np.reshape(gt_pts, (-1, 2))
    return np.sqrt(np.sum(np.square(pred_pts - gt_pts), axis=-1))


def save_as_csv(points, location='.'):
    """
    Save the points out as a .csv file
    :param points: numpy array of shape (no_test_images, no_points, 2) to be saved
    :param location: Directory to save results.csv in. Default to current working directory
    """
    assert points.shape[0] == 554, 'wrong number of image points, should be 554 test images'
    assert np.prod(points.shape[
                   1:]) == 50 * 2, 'wrong number of points provided. There should be 50 points with 2 values (x,y) per point'
    np.savetxt(location + '/results.csv', np.reshape(points, (points.shape[0], -1)), delimiter=',')


# -------------------------------- MY CODE ----------------------------------------------- #


def normalize_images(images):
    """
    Normalize images using OpenCV and NumPy.
    Each image is first converted to a floating point type, then normalized to have a mean of 0 and standard deviation of 1.
    :param images: numpy array of images
    :return: numpy array of normalized images
    """
    normalized_images = []
    for img in images:
        # Convert to floating point
        img_float = img.astype(np.float32)
        # Normalize the image - mean 0 and standard deviation 1
        norm_img = cv2.normalize(img_float, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        normalized_images.append(norm_img)
    return np.array(normalized_images)


def convert_to_grayscale(images):
    """
    Convert images to grayscale using OpenCV.
    :param images: numpy array of images
    :return: numpy array of grayscale images
    """
    grayscale_images = []
    for img in images:
        # Ensure the image is in the correct format (BGR)
        if img.ndim == 3 and img.shape[2] == 3:
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_img = img  # if the image is already in one channel (grayscale)
        grayscale_images.append(gray_img)

    return np.array(grayscale_images)


def extract_features(image, points, window_size=5):
    """
    Extracts pixel intensity features around each landmark point.
    :param image: The image (assumed to be in grayscale).
    :param points: Landmark points for the image.
    :param window_size: The size of the window around each point to consider.
    :return: A flattened array of pixel intensities around each point.
    """
    features = []
    for (x, y) in points:
        x, y = int(x), int(y)
        patch = image[max(0, y-window_size):y+window_size+1, max(0, x-window_size):x+window_size+1]
        features.extend(patch.flatten())
    return np.array(features)


def extract_hog_features(image, points, window_size=5):
    """
    Extracts HOG features around each landmark point.
    :param image: The image (assumed to be in grayscale).
    :param points: Landmark points for the image.
    :param window_size: The size of the window around each point to consider.
    :return: A flattened array of HOG features around each point.
    """
    features = []
    for (x, y) in points:
        x, y = int(x), int(y)
        patch = image[max(0, y-window_size):y+window_size+1, max(0, x-window_size):x+window_size+1]
        # Extract HOG features from the patch
        hog_features = hog(patch, pixels_per_cell=(2, 2), cells_per_block=(1, 1), feature_vector=True)
        features.extend(hog_features)
    return np.array(features)


def train_regressor(images, points, mean_shape):
    X = []
    Y = []
    for img, pts in zip(images, points):
        features = extract_hog_features(img, mean_shape)
        if len(features) != expected_feature_length:
            print(f"Unexpected feature length: {len(features)}")
            continue
        X.append(features)
        Y.append((pts - mean_shape).flatten())
    regressor = LinearRegression()
    regressor.fit(X, Y)
    return regressor


def apply_cascade(image, regressor, initial_shape):
    features = extract_hog_features(image, initial_shape)
    if len(features) != expected_feature_length:
        print(f"Unexpected feature length for prediction: {len(features)}")
        return initial_shape
    shape_update = regressor.predict([features])
    new_shape = initial_shape + np.reshape(shape_update, (-1, 2))
    return new_shape


def create_cnn_model(input_shape):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(100)  # 50 landmark points
    ])
    return model


def create_vgg_model(input_shape):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    base_model.trainable = False
    model = Sequential([
        base_model,
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(100)  # 50 landmark points
    ])
    return model


def preprocess_images(images):
    # Preprocess images for the CNN model: normalize and grayscale
    normalized_images = normalize_images(images)
    grayscale_images = convert_to_grayscale(normalized_images)
    return grayscale_images


def predict_landmarks_cnn(model, images):
    # Predict landmarks using the CNN model
    predictions = model.predict(images)
    return predictions


def visualize_predictions(images, predictions):
    count = 0
    for img, pts in zip(images, predictions):
        # Reshape the predictions to a 2D array (50 points, 2 coordinates each)
        pts_reshaped = pts.reshape(-1, 2)
        visualise_pts(img, pts_reshaped)
        count += 1
        if count == 3:
            break


def create_lip_mask(image, lip_points):
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, [lip_points], (255, 255, 255))
    return mask


def color_lips(image, mask, color=(0, 255, 0), intensity=0.5):
    colored_image = np.copy(image)
    colored_image[np.where((mask == [255, 255, 255]).all(axis=2))] = [0, 255, 0]
    return colored_image


expected_feature_length = 11250
# Preprocess training images for CNN
preprocessed_training_images = preprocess_images(images)
# Flatten the points for training
flattened_points = pts.reshape(pts.shape[0], -1)
print(flattened_points.shape)

# Preprocess test images
preprocessed_test_images = preprocess_images(test_images)

# Load or create your CNN model
cnn_model = create_cnn_model(input_shape=(252, 252, 1))  # Grayscale images, hence 1 channel
cnn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])
cnn_model.fit(preprocessed_training_images, flattened_points, batch_size=32, epochs=10, validation_split=0.1)
cnn_model.save_weights('cnn_model_weights.h5')
#cnn_model.load_weights('cnn_model_weights.h5')

# Train the cascaded regression model (if not already trained)
mean_shape = np.mean(pts, axis=0)  # Calculate the mean shape from the training data
regressor = train_regressor(preprocessed_training_images, pts, mean_shape)
refined_prediction = []
# Test the combined model
for test_img in preprocessed_test_images:
    # Initial prediction with CNN
    initial_prediction = cnn_model.predict(np.expand_dims(test_img, axis=0))
    initial_prediction_reshaped = initial_prediction.reshape(-1, 2)

    # Refinement with cascaded regression
    refined_prediction = apply_cascade(test_img, regressor, initial_prediction_reshaped)

    # Visualization
    visualise_pts(test_img, refined_prediction)
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_GRAY2RGB) if len(test_img.shape) == 2 else test_img
    lip_indices = list(range(30, 50))  # Lip landmarks from 30 to 49
    lip_points = refined_prediction[lip_indices].reshape((-1, 1, 2)).astype(np.int32)
    lip_mask = create_lip_mask(test_img_rgb, lip_points)
    green_lips_image = color_lips(test_img_rgb, lip_mask)

    # Display or save the result
    cv2.imshow('Green Lips', green_lips_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


expected_feature_length = 22689
# Similarly for VGG model, ensure the images are preprocessed correctly for the VGG model
# VGG model requires 3-channel input
vgg_model = create_vgg_model(input_shape=(252, 252, 3))  # VGG expects 3 channels
vgg_model.save_weights('vgg_model_weights.h5')
#vgg_model.load_weights('vgg_model_weights.h5')  # Load weights if you have a trained model

mean_shape = np.mean(pts, axis=0)  # Calculate the mean shape from the training data
regressor = train_regressor(preprocessed_training_images, pts, mean_shape)

vgg_refined_prediction = []
# Test the combined model
for test_img in preprocessed_test_images:
    # Initial prediction with CNN
    vgg_initial_prediction = predict_landmarks_cnn(vgg_model, normalize_images(test_images))  # Normalize but do not grayscale for VGG
    vgg_initial_prediction_reshaped = vgg_initial_prediction.reshape(-1, 2)

    # Refinement with cascaded regression
    vgg_refined_prediction = apply_cascade(test_img, regressor, vgg_initial_prediction_reshaped)

    # Visualization
    visualise_pts(test_img, vgg_refined_prediction)
    test_img_rgb = cv2.cvtColor(test_img, cv2.COLOR_GRAY2RGB) if len(test_img.shape) == 2 else test_img
    lip_indices = list(range(30, 50))  # Lip landmarks from 30 to 49
    lip_points = vgg_refined_prediction[lip_indices].reshape((-1, 1, 2)).astype(np.int32)
    lip_mask = create_lip_mask(test_img_rgb, lip_points)
    green_lips_image = color_lips(test_img_rgb, lip_mask)

    # Display or save the result
    cv2.imshow('Green Lips', green_lips_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


save_as_csv(refined_prediction, "test_points.csv")
save_as_csv(vgg_refined_prediction, "test_points.csv")
