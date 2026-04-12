from skimage.feature import graycomatrix, graycoprops

patch_size = 32

features = []

for i in range(0, img.shape[0], patch_size):
    for j in range(0, img.shape[1], patch_size):
        
        patch = img[i:i+patch_size, j:j+patch_size]

        if patch.shape != (patch_size, patch_size):
            continue

        glcm = graycomatrix(patch,
                            distances=[1],
                            angles=[0],
                            levels=256,
                            symmetric=True,
                            normed=True)

        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]

        # entropy
        p = glcm[:, :, 0, 0]
        p = p[p > 0]
        entropy = -np.sum(p * np.log(p))

        features.append([contrast, homogeneity, correlation, entropy])
