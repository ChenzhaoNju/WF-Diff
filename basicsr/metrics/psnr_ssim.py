import cv2
import numpy as np
from skimage import transform
from scipy import ndimage
import lpips
from basicsr.metrics.metric_util import reorder_image, to_y_channel
from basicsr.utils.registry import METRIC_REGISTRY
# import kornia.color as color
# import torch
# import math


@METRIC_REGISTRY.register()
def calculate_psnr(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the PSNR calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    mse = np.mean((img - img2)**2)
    if mse == 0:
        return float('inf')
    return 20. * np.log10(255. / np.sqrt(mse))


def _ssim(img, img2):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    Args:
        img (ndarray): Images with range [0, 255] with order 'HWC'.
        img2 (ndarray): Images with range [0, 255] with order 'HWC'.

    Returns:
        float: ssim result.
    """

    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img, -1, window)[5:-5, 5:-5]
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean()


@METRIC_REGISTRY.register()
def calculate_ssim(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate SSIM (structural similarity).

    Ref:
    Image quality assessment: From error visibility to structural similarity

    The results are the same as that of the official released MATLAB code in
    https://ece.uwaterloo.ca/~z70wang/research/ssim/.

    For three-channel images, SSIM is calculated for each channel and then
    averaged.

    Args:
        img (ndarray): Images with range [0, 255].
        img2 (ndarray): Images with range [0, 255].
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the SSIM calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: ssim result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f'Wrong input_order {input_order}. Supported input_orders are "HWC" and "CHW"')
    img = reorder_image(img, input_order=input_order)
    img2 = reorder_image(img2, input_order=input_order)
    img = img.astype(np.float64)
    img2 = img2.astype(np.float64)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, ...]

    if test_y_channel:
        img = to_y_channel(img)
        img2 = to_y_channel(img2)

    ssims = []
    for i in range(img.shape[2]):
        ssims.append(_ssim(img[..., i], img2[..., i]))
    return np.array(ssims).mean()

@METRIC_REGISTRY.register()
def calculate_uciqe(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    img_bgr =img

    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)  # Transform to Lab color space

    # if nargin == 1:                                 # According to training result mentioned in the paper:
    coe_metric = [0.4680, 0.2745, 0.2576]      # Obtained coefficients are: c1=0.4680, c2=0.2745, c3=0.2576.
    img_lum = img_lab[..., 0]/255
    img_a = img_lab[..., 1]/255
    img_b = img_lab[..., 2]/255

    img_chr = np.sqrt(np.square(img_a)+np.square(img_b))              # Chroma

    img_sat = img_chr/np.sqrt(np.square(img_chr)+np.square(img_lum))  # Saturation
    aver_sat = np.mean(img_sat)                                       # Average of saturation

    aver_chr = np.mean(img_chr)                                       # Average of Chroma

    var_chr = np.sqrt(np.mean(abs(1-np.square(aver_chr/img_chr))))    # Variance of Chroma

    dtype = img_lum.dtype                                             # Determine the type of img_lum
    if dtype == 'uint8':
        nbins = 256
    else:
        nbins = 65536

    hist, bins = np.histogram(img_lum, nbins)                        # Contrast of luminance
    cdf = np.cumsum(hist)/np.sum(hist)

    ilow = np.where(cdf > 0.0100)
    ihigh = np.where(cdf >= 0.9900)
    tol = [(ilow[0][0]-1)/(nbins-1), (ihigh[0][0]-1)/(nbins-1)]
    con_lum = tol[1]-tol[0]

    quality_val = coe_metric[0]*var_chr+coe_metric[1]*con_lum + coe_metric[2]*aver_sat         # get final quality value
    # print("quality_val is", quality_val)
    return quality_val
    # image = img
    # # image = cv2.imread(image)
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # RGB转为HSV
    # H, S, V = cv2.split(hsv)
    # delta = np.std(H) / 180
    # # 色度的标准差
    # mu = np.mean(S) / 255  # 饱和度的平均值
    # # 求亮度对比值
    # n, m = np.shape(V)
    # number = math.floor(n * m / 100)
    # v = V.flatten() / 255
    # v.sort()
    # bottom = np.sum(v[:number]) / number
    # v = -v
    # v.sort()
    # v = -v
    # top = np.sum(v[:number]) / number
    # conl = top - bottom
    # uciqe = 0.4680 * delta + 0.2745 * conl + 0.2576 * mu
    # return uciqe


def _uicm(img):
    img = np.array(img, dtype=np.float64)
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    RG = R - G
    YB = (R+G)/2 -B
    K = R.shape[0]*R.shape[1]
    RG1 = RG.reshape(1,K)
    RG1 = np.sort(RG1)
    alphaL = 0.1
    alphaR = 0.1
    RG1 = RG1[0,int(alphaL*K+1):int(K*(1-alphaR))]
    N = K* (1-alphaR-alphaL)
    meanRG = np.sum(RG1)/N
    deltaRG = np.sqrt(np.sum((RG1-meanRG)**2)/N)

    YB1 = YB.reshape(1,K)
    YB1 = np.sort(YB1)
    alphaL = 0.1
    alphaR = 0.1
    YB1 = YB1[0,int(alphaL*K+1):int(K*(1-alphaR))]
    N = K* (1-alphaR-alphaL)
    meanYB = np.sum(YB1) / N
    deltaYB = np.sqrt(np.sum((YB1 - meanYB)**2)/N)
    uicm = -0.0268*np.sqrt(meanRG**2+meanYB**2)+ 0.1586*np.sqrt(deltaYB**2+deltaRG**2)
    return uicm

def _uiconm(img):
    img = np.array(img, dtype=np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    patchez = 5
    m = R.shape[0]
    n = R.shape[1]
    if m%patchez != 0 or n%patchez != 0:
        x = int(m-m%patchez+patchez)
        y = int(n-n%patchez+patchez)
        R = transform.resize(R,(x,y))
        G = transform.resize(G, (x, y))
        B = transform.resize(B, (x, y))
    m = R.shape[0]
    n = R.shape[1]
    k1 = m /patchez
    k2 = n /patchez
    AMEER = 0
    for i in range(0,m,patchez):
        for j in range(0,n,patchez):
            sz = patchez
            im = R[i:i+sz,j:j+sz]
            Max = np.max(im)
            Min = np.min(im)
            if (Max != 0 or Min != 0) and Max != Min:
                AMEER = AMEER + np.log((Max-Min)/(Max+Min))*((Max-Min)/(Max+Min))
    AMEER = 1/(k1*k2) *np.abs(AMEER)
    AMEEG = 0
    for i in range(0,m,patchez):
        for j in range(0,n,patchez):
            sz = patchez
            im = G[i:i+sz,j:j+sz]
            Max = np.max(im)
            Min = np.min(im)
            if (Max != 0 or Min != 0) and Max != Min:
                AMEEG = AMEEG + np.log((Max-Min)/(Max+Min))*((Max-Min)/(Max+Min))
    AMEEG = 1/(k1*k2) *np.abs(AMEEG)
    AMEEB = 0
    for i in range(0,m,patchez):
        for j in range(0,n,patchez):
            sz = patchez
            im = B[i:i+sz,j:j+sz]
            Max = np.max(im)
            Min = np.min(im)
            if (Max != 0 or Min != 0) and Max != Min:
                AMEEB = AMEEB + np.log((Max-Min)/(Max+Min))*((Max-Min)/(Max+Min))
    AMEEB = 1/(k1*k2) *np.abs(AMEEB)
    uiconm = AMEER +AMEEG +AMEEB
    return uiconm

def _uism(img):
    img = np.array(img, dtype=np.float64)
    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    hx = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    hy = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])

    SobelR = np.abs(ndimage.convolve(R, hx, mode='nearest')+ndimage.convolve(R, hy, mode='nearest'))
    SobelG = np.abs(ndimage.convolve(G, hx, mode='nearest')+ndimage.convolve(G, hy, mode='nearest'))
    SobelB = np.abs(ndimage.convolve(B, hx, mode='nearest')+ndimage.convolve(B, hy, mode='nearest'))
    patchez = 5
    m = R.shape[0]
    n = R.shape[1]
    if m%patchez != 0 or n%patchez != 0:
        x = int(m - m % patchez + patchez)
        y = int(n - n % patchez + patchez)
        SobelR = transform.resize(SobelR, (x, y))
        SobelG = transform.resize(SobelG, (x, y))
        SobelB = transform.resize(SobelB, (x, y))
    m = SobelR.shape[0]
    n = SobelR.shape[1]
    k1 = m /patchez
    k2 = n /patchez
    EMER = 0
    for i in range(0,m,patchez):
        for j in range(0,n,patchez):
            sz = patchez
            im = SobelR[i:i+sz,j:j+sz]
            Max = np.max(im)
            Min = np.min(im)
            if Max != 0 and Min != 0:
                EMER = EMER + np.log(Max/Min)
    EMER = 2/(k1*k2)*np.abs(EMER)

    EMEG = 0
    for i in range(0,m,patchez):
        for j in range(0,n,patchez):
            sz = patchez
            im = SobelG[i:i+sz,j:j+sz]
            Max = np.max(im)
            Min = np.min(im)
            if Max != 0 and Min != 0:
                EMEG = EMEG + np.log(Max/Min)
    EMEG = 2/(k1*k2)*np.abs(EMEG)
    EMEB = 0
    for i in range(0,m,patchez):
        for j in range(0,n,patchez):
            sz = patchez
            im = SobelB[i:i+sz,j:j+sz]
            Max = np.max(im)
            Min = np.min(im)
            if Max != 0 and Min != 0:
                EMEB = EMEB + np.log(Max/Min)
    EMEB = 2/(k1*k2)*np.abs(EMEB)
    lambdaR = 0.299
    lambdaG = 0.587
    lambdaB = 0.114
    uism = lambdaR * EMER + lambdaG * EMEG + lambdaB * EMEB
    return uism

@METRIC_REGISTRY.register()
def calculate_uiqm(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    x = img
    x = x.astype(np.float32)
    c1 = 0.0282; c2 = 0.2953; c3 = 3.5753
    uicm   = _uicm(x)
    uism   = _uism(x)
    uiconm = _uiconm(x)
    uiqm = (c1*uicm) + (c2*uism) + (c3*uiconm)
    return uiqm


lpips_model = lpips.LPIPS(net='alex') 
@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    img_tensor = lpips.im2tensor(img)
    img2_tensor = lpips.im2tensor(img2)
    return lpips_model(img_tensor, img2_tensor).item()

