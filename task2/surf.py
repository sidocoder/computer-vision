import cv2
import numpy as np
import matplotlib.pyplot as plt

def run_orb(img1_path, img2_path):
    # Step 1: Load and Preprocess
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    if img1 is None or img2 is None: return print("Error loading images.")

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Step 2: ORB Detection & Description
    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Step 3: Brute-Force Matching (Hamming Distance)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(bf.match(des1, des2), key=lambda x: x.distance)
    
    # Step 4: Draw Red Matches
    res = cv2.drawMatches(gray1, kp1, gray2, kp2, matches[:100], None, 
                           matchColor=(255, 0, 0), 
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    #visualization
    plt.figure(figsize=(15, 7))
    plt.title("Algorithm B: ORB Feature Matching (SURF Alternative)")
    plt.imshow(res)
    plt.axis('off')
    plt.show()

run_orb('photo2.jpg', 'photo1.jpg')