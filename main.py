import cv2
import glob
import numpy as np
from affine_ransac import Ransac
from align_transform import Align
from affine_transform import Affine

print("Hay nhap duong dan: ")
path1 = input("Your image path: ")
original = cv2.imread(path1)

# Sift
sift = cv2.xfeatures2d.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Load images
all_images_to_compare = []
titles = []
rate_max=0
bb=[]
name =""
print("Hay nhap duong dan folder chua anh: ")
path = input("Your image folder path: ")

def is_not_blank(s):
    return bool(s and not s.isspace())

for f in glob.iglob(path+"\*"):
    image = cv2.imread(f)
    titles.append(f)
    all_images_to_compare.append(image)
for image_to_compare, title in zip(all_images_to_compare, titles):
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)
    matches = flann.knnMatch(desc_1, desc_2, k=2)
    percentage_similarity = []
    good_points = []

    for m, n in matches:
        if m.distance <0.6*n.distance:
            good_points.append(m)

    number_keypoints = 0
    if len(kp_1) >= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)

    print("Title: " + title)
    rate= len((good_points)) / number_keypoints * 100
    percentage_similarity.append(rate)
    print("Similarity: " + str(int(rate)) + "\n")
    if rate >rate_max:
        rate_max=rate
        # name =image_to_compare.title()
        bb=image_to_compare
        name =title

print ("Max value perentage : ", rate_max)

if (is_not_blank(name)):
    print ("ton tai anh reference trong anh scene")
    print(name)
    kp_2, desc_2 = sift.detectAndCompute(bb, None)
    index_params = dict(algorithm=0, trees=5)
    search_params = dict()
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(desc_1, desc_2, k=2)
    good_points = []

    for m, n in matches:
        if m.distance < 0.6*n.distance:
            good_points.append(m)
    result = cv2.drawMatches(original, kp_1, bb, kp_2, good_points, None)

    af = Affine()

    outlier_rate = 0.9
    _, _, pts_s, pts_t = af.create_test_case(outlier_rate)
    K = 3

    idx = np.random.randint(0, pts_s.shape[1], (K, 1))
    A_test, t_test = af.estimate_affine(pts_s[:, idx], pts_t[:, idx])

    rs = Ransac(K=3, threshold=1)
    residual = rs.residual_lengths(A_test, t_test, pts_s, pts_t)
    A_rsc, t_rsc, inliers = rs.ransac_fit(pts_s, pts_t)
    print(A_rsc, '\n', t_rsc)

    source_path = path1
    target_path = name

    al = Align(source_path, target_path, threshold=1)

    print ("Tim dc ket qua")

    cv2.imshow("referenced image", original)
    cv2.imshow("scene Image", bb)
    cv2.imshow("matched points of reference & scene image", result)
    cv2.imshow("image after affine transform", al.align_image())

    cv2.waitKey(0)
    cv2.destroyAllWindows()



