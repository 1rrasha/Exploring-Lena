# Rasha Mansour-1210773
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
lena_img = cv2.imread('lena.jpg')

# Convert to grayscale
gray_lena = cv2.cvtColor(lena_img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_lena.jpg', gray_lena)

# Convert to binary
def custom_thresholding(image, threshold_value, max_value):
    binary_image = np.zeros_like(image)
    rows, cols = image.shape
    for i in range(rows):
        for j in range(cols):
            if image[i, j] > threshold_value:
                binary_image[i, j] = max_value
            else:
                binary_image[i, j] = 0
    return binary_image

# Thresholding
binary_lena = custom_thresholding(gray_lena, 127, 255)
cv2.imwrite('binary_lena.jpg', binary_lena)

# Downscale the image
small_gray_lena = cv2.resize(gray_lena, (256, 256), interpolation=cv2.INTER_AREA)
cv2.imwrite('small_gray_lena.jpg', small_gray_lena)


# Analyze the grayscale image
mean = np.mean(gray_lena)
std_dev = np.std(gray_lena)
entropy = -np.sum((gray_lena / 255.0) * np.log2(gray_lena / 255.0 + 1e-6))
histogram = cv2.calcHist([gray_lena], [0], None, [256], [0, 256])

# Normalize histogram
hist_norm = histogram.ravel() / histogram.max()
Q = hist_norm.cumsum()

# Compute normalized histogram
hist_norm = histogram.ravel() / histogram.sum()

# Compute cumulative histogram
cumulative_hist = np.cumsum(hist_norm)

# Enhance the image contrast by 50%
alpha = 1.5 # Contrast control (1.0-3.0)
beta = 0    # Brightness control (0-100)
contrast_enhanced_lena = cv2.convertScaleAbs(gray_lena, alpha=alpha, beta=beta)
cv2.imwrite('contrast_enhanced_lena.jpg', contrast_enhanced_lena)

# Create horizontally flipped version
# Get the dimensions of the image
height, width = gray_lena.shape

# Create an empty image of the same size
flipped_lena = np.zeros_like(gray_lena)

# Iterate through each row
for i in range(height):
    # Reverse the order of the columns
    flipped_lena[i, :] = gray_lena[i, ::-1]

# Save the flipped image
cv2.imwrite('flipped_lena.jpg', flipped_lena)

# Apply a Gaussian blur to the flipped image
# Define the Gaussian kernel size and standard deviation
kernel_size = (5, 5)
sigma = 0

# Create a 2D Gaussian kernel
kernel = cv2.getGaussianKernel(kernel_size[0], sigma)
kernel_2d = kernel * kernel.T

# Pad the image to handle borders
padding_size = kernel_size[0] // 2
flipped_lena_padded = cv2.copyMakeBorder(flipped_lena, padding_size, padding_size, padding_size, padding_size, cv2.BORDER_CONSTANT)

# Convolve the image with the Gaussian kernel
blurred_flipped_lena = cv2.filter2D(flipped_lena_padded, -1, kernel_2d)

# Save the blurred flipped lena image
cv2.imwrite('blurred_flipped_lena.jpg', blurred_flipped_lena)

# Create negative version of the grayscale image
negative_lena = 255 - gray_lena
cv2.imwrite('negative_lena.jpg', negative_lena)

# Custom "Crop" function
def crop_image(image, start_row, start_col, end_row, end_col):
    return image[start_row:end_row, start_col:end_col]

# coordinates of the bounding box around the eyes
eye_start_row = 230
eye_start_col = 240
eye_end_row = 285  
eye_end_col = 360

# Crop function to extract the region around the eyes
cropped_lena = crop_image(gray_lena, eye_start_row, eye_start_col, eye_end_row, eye_end_col)
cv2.imwrite('cropped_lena.jpg', cropped_lena)

# Calculate histogram of the entire grayscale image
hist_full = cv2.calcHist([gray_lena], [0], None, [256], [0, 256])

# Define the histogram of the specific strip you're looking for (you can use the histogram of the cropped image)
hist_strip = cv2.calcHist([cropped_lena], [0], None, [256], [0, 256])

# Define a function for histogram matching
def histogram_matching(hist_full, hist_strip):
    min_diff = float('inf')
    best_match_index = 0
    
    # Iterate through possible starting positions for the strip
    for i in range(len(hist_full) - len(hist_strip)):
        # Calculate sum of squared differences (SSD) between the histograms
        diff = np.sum((hist_full[i:i+len(hist_strip)] - hist_strip)**2)
        
        # Update if current difference is smaller
        if diff < min_diff:
            min_diff = diff
            best_match_index = i
    
    return best_match_index

# Find the best match
best_match_index = histogram_matching(hist_full, hist_strip)

# Calculate corresponding coordinates for the best match
best_match_start_row = eye_start_row
best_match_start_col = eye_start_col
best_match_end_row = eye_end_row
best_match_end_col = eye_end_col

# Extract the best match region from the original grayscale image
best_match_region = gray_lena[best_match_start_row:best_match_end_row, best_match_start_col:best_match_end_col]

# Display the best match region
plt.imshow(best_match_region, cmap='gray')
cv2.imwrite('best_match_region.jpg', best_match_region)
plt.title('Best Match Region')
plt.axis('off')
plt.show()

# Display results
plt.figure(figsize=(12, 12))

# Original Image
plt.subplot(4, 4, 1)
plt.imshow(cv2.cvtColor(lena_img, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Grayscale Image
plt.subplot(4, 4, 2)
plt.imshow(gray_lena, cmap='gray')
plt.title('Grayscale Image')
plt.axis('off')

# Binary Image
plt.subplot(4, 4, 3)
plt.imshow(binary_lena, cmap='gray')
plt.title('Binary Image')
plt.axis('off')

# Small Gray Image
plt.subplot(4, 4, 4)
plt.imshow(small_gray_lena, cmap='gray')
plt.title('Downscaling Grayscale Image')
plt.axis('off')

# Enhanced Contrast Image
plt.subplot(4, 4, 5)
plt.imshow(contrast_enhanced_lena, cmap='gray')
plt.title('Contrast Enhanced Image')
plt.axis('off')

# Flipped Image
plt.subplot(4, 4, 6)
plt.imshow(flipped_lena, cmap='gray')
plt.title('Flipped Image')
plt.axis('off')

# Blurred Flipped Image
plt.subplot(4, 4, 7)
plt.imshow(blurred_flipped_lena, cmap='gray')
plt.title('Blurred Flipped Image')
plt.axis('off')

# Negative Image
plt.subplot(4, 4, 8)
plt.imshow(negative_lena, cmap='gray')
plt.title('Negative Image')
plt.axis('off')

# Cropped Image
plt.subplot(4, 4, 9)
plt.imshow(cropped_lena, cmap='gray')
plt.title('Cropped Image')
plt.axis('off')

# Histogram
plt.subplot(4, 4, 10)
plt.hist(gray_lena.ravel(), 256, [0, 256])
plt.title('Grayscale Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Frequency')

# Normalized Histogram
plt.subplot(4, 4, 11)
plt.plot(hist_norm, color='black')
plt.title('Normalized Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Probability')

# Cumulative Histogram
plt.subplot(4, 4, 12)
plt.plot(cumulative_hist, color='black')
plt.title('Cumulative Histogram')
plt.xlabel('Pixel Intensity')
plt.ylabel('Cumulative Probability')

plt.tight_layout()
plt.show()

# Output the mean, std_dev, and entropy for verification
print("Mean:", mean)
print("Standard Deviation:", std_dev)
print("Entropy:", entropy)