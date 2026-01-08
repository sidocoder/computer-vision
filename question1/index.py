import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_canny_assignment_full(image_path):
    # --- STEP 1 & 2: INPUT ---
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        print("Error: Image not found. Check the file path.")
        return

    # --- STEP 3: GRAYSCALE ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # --- STEP 4: GAUSSIAN BLUR (Noise Reduction) ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # --- STEP 5: GRADIENT MAGNITUDE (Sobel) ---
    # Manually calculating gradients to demonstrate the intensity changes
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    # Calculate Magnitude: sqrt(dx^2 + dy^2)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    magnitude = np.uint8(np.absolute(magnitude))

    # --- STEP 6: HYSTERESIS THRESHOLDING (Canny Low vs High) ---
    # Low Threshold (Wide): Captures more faint boundaries
    canny_low = cv2.Canny(blurred, 50, 150)
    # High Threshold (Tight): Captures only strong structural boundaries
    canny_high = cv2.Canny(blurred, 200, 250)

    # --- FINAL OUTPUT DISPLAY ---
    plt.figure(figsize=(18, 10))

    images = [cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), gray, blurred, magnitude, canny_low, canny_high]
    titles = ["1. Original Image", "2. Gray Scale", "3. Gaussian Blurred", 
              "4. Gradient Magnitude", "5. Canny (Low/Wide)", "6. Canny (High/Tight)"]

    for i in range(6):
        plt.subplot(2, 3, i+1)
        plt.imshow(images[i], cmap='gray' if i > 0 else None)
        plt.title(titles[i])
        plt.axis('off')

    plt.tight_layout()
    plt.show()
    
run_canny_assignment_full('image.png')