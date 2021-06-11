import cv2
import numpy as np
from affine_ransac import Ransac
from affine_transform import Affine

RATIO = 0.8

class Align():

    def __init__(self, source_path, target_path,
                 K=3, threshold=1):
        self.source_path = source_path
        self.target_path = target_path
        self.K = K
        self.threshold = threshold

    def read_image(self, path, mode=1):

        return cv2.imread(path, mode)

    def extract_SIFT(self, img):

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Extract key points & descriptors
        sift = cv2.xfeatures2d.SIFT_create()
        kp, desc = sift.detectAndCompute(img_gray, None)

        # Extract positions of key points
        kp = np.array([p.pt for p in kp]).T

        return kp, desc

    def match_SIFT(self, desc_s, desc_t):
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(desc_s, desc_t, k=2)
        fit_pos = np.array([], dtype=np.int32).reshape((0, 2))

        matches_num = len(matches)
        for i in range(matches_num):
            if matches[i][0].distance <= RATIO * matches[i][1].distance:
                temp = np.array([matches[i][0].queryIdx,
                                 matches[i][0].trainIdx])
                fit_pos = np.vstack((fit_pos, temp))
        return fit_pos

    def affine_matrix(self, kp_s, kp_t, fit_pos):
        kp_s = kp_s[:, fit_pos[:, 0]]
        kp_t = kp_t[:, fit_pos[:, 1]]
        _, _, inliers = Ransac(self.K, self.threshold).ransac_fit(kp_s, kp_t)
        kp_s = kp_s[:, inliers[0]]
        kp_t = kp_t[:, inliers[0]]
        A, t = Affine().estimate_affine(kp_s, kp_t)
        M = np.hstack((A, t))

        return M

    def warp_image(self, source, target, M):
        rows, cols, _ = target.shape
        warp = cv2.warpAffine(source, M, (cols, rows))
        merge = np.uint8(target * 0.5 + warp * 0.5)

        return merge

    def align_image(self):
        img_source = self.read_image(self.source_path)
        img_target = self.read_image(self.target_path)
        kp_s, desc_s = self.extract_SIFT(img_source)
        kp_t, desc_t = self.extract_SIFT(img_target)
        fit_pos = self.match_SIFT(desc_s, desc_t)
        M = self.affine_matrix(kp_s, kp_t, fit_pos)
        return self.warp_image(img_source, img_target, M)
