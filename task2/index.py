import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_feature_matching_red_lines(img1_path, img2_path):
    # Step 1: Load the Input Images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    
    if img1 is None or img2 is None:
        print("Error: Could not load images.")
        return

    # Step 2: Convert to Grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # --- SIFT Detection ---
    sift = cv2.SIFT_create()
    kp1_sift, des1_sift = sift.detectAndCompute(gray1, None)
    kp2_sift, des2_sift = sift.detectAndCompute(gray2, None)
    bf_sift = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches_sift = sorted(bf_sift.match(des1_sift, des2_sift), key=lambda x: x.distance)
    
    # Draw Matches with RED lines (matchColor=(255, 0, 0) because plt uses RGB)
    sift_res = cv2.drawMatches(gray1, kp1_sift, gray2, kp2_sift, matches_sift[:100], None, 
                                matchColor=(255, 0, 0), # Red color in RGB
                                singlePointColor=None,
                                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # --- ORB Detection ---
    orb = cv2.ORB_create(nfeatures=1000)
    kp1_orb, des1_orb = orb.detectAndCompute(gray1, None)
    kp2_orb, des2_orb = orb.detectAndCompute(gray2, None)
    bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches_orb = sorted(bf_orb.match(des1_orb, des2_orb), key=lambda x: x.distance)
    
    # Draw Matches with RED lines
    orb_res = cv2.drawMatches(gray1, kp1_orb, gray2, kp2_orb, matches_orb[:100], None, 
                               matchColor=(255, 0, 0), # Red color in RGB
                               singlePointColor=None,
                               flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Visualization
    plt.figure(figsize=(15, 12))
    plt.subplot(2, 1, 1)
    plt.title("SIFT Matching", color='black', fontsize=14)
    plt.imshow(sift_res) 
    plt.axis('off')

    plt.subplot(2, 1, 2)
    plt.title("ORB Matching", color='black', fontsize=14)
    plt.imshow(orb_res)
    plt.axis('off')

    plt.subplots_adjust(hspace=0.3)
    plt.show()

run_feature_matching_red_lines('photo2.jpg', 'photo1.jpg')