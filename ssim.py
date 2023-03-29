import numpy as np
from skimage.metrics import structural_similarity
def SSIM(x_image, y_image, max_value=255.0, win_size=3, use_sample_covariance=True):
    x = np.asarray(x_image, np.float32)
    y = np.asarray(y_image, np.float32)
    return structural_similarity(x,y, win_size=win_size, data_range=max_value, multichannel=(x.ndim>2))