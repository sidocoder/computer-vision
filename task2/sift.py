import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_sift(img1_path, img2_path):
    # Step 1: Load and Preprocess
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None: return print("Error loading images.")

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Step 2: SIFT Detection & Description
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    # Step 3: Brute-Force Matching (L2 Norm)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    
    # Step 4: Draw Red line Matches
    res = cv2.drawMatches(gray1, kp1, gray2, kp2, matches[:100], None, 
                           matchColor=(255, 0, 0), 
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    #visualization
    plt.figure(figsize=(15, 7))
    plt.title("Algorithm A: SIFT Feature Matching")
    plt.imshow(res)
    plt.axis('off')
    plt.show()

run_sift('photo2.jpg', 'photo1.jpg')