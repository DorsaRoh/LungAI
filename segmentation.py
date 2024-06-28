import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "000000 (6).png"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Apply thresholding to segment the lungs (assuming lungs are the darkest part)
_, thresholded = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY_INV)

# Perform morphological operations to remove small noise and fill holes
kernel = np.ones((5,5), np.uint8)
morphed = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)
morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)

# Find contours of the lungs
contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask for the lungs
lung_mask = np.zeros_like(image)
cv2.drawContours(lung_mask, contours, -1, 255, thickness=cv2.FILLED)

# Apply the mask to the original image
isolated_lungs = cv2.bitwise_and(image, image, mask=lung_mask)

# Find bounding box of the largest contour (assumed to be the lungs)
largest_contour = max(contours, key=cv2.contourArea)
x, y, w, h = cv2.boundingRect(largest_contour)

# Increase the bounding box size slightly for better cropping
padding = 20
x = max(0, x - padding)
y = max(0, y - padding)
w = min(image.shape[1] - x, w + 2 * padding)
h = min(image.shape[0] - y, h + 2 * padding)

# Crop the region containing the lungs
cropped_lungs = isolated_lungs[y:y+h, x:x+w]

# Resize the cropped image to the original image size
resized_lungs = cv2.resize(cropped_lungs, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

# Save the result
output_path = "/mnt/data/final_resized_lungs_v2.png"
cv2.imwrite(output_path, resized_lungs)

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('Original Image')
plt.imshow(image, cmap='gray')
plt.subplot(1, 3, 2)
plt.title('Isolated Lungs')
plt.imshow(isolated_lungs, cmap='gray')
plt.subplot(1, 3, 3)
plt.title('Resized Lungs')
plt.imshow(resized_lungs, cmap='gray')
plt.show()
