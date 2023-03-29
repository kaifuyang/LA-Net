from skimage.metrics import peak_signal_noise_ratio
def PSNR(x_image, y_image):
    return peak_signal_noise_ratio(x_image, y_image,data_range=255.0)