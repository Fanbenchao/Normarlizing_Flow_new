import numpy as np
import scipy.io as scio
import glob
import os
from os import path
import h5py
import gc

def pack_raw(raw_im):
    """Packs Bayer image to 4 channels (h, w) --> (h/2, w/2, 4)."""
    # pack Bayer image to 4 channels
    im = np.expand_dims(raw_im, axis=2)
    img_shape = im.shape
    # print('img_shape: ' + str(img_shape))
    h = img_shape[0]
    w = img_shape[1]
    out = np.concatenate((im[0:h:2, 0:w:2, :],
                          im[0:h:2, 1:w:2, :],
                          im[1:h:2, 1:w:2, :],
                          im[1:h:2, 0:w:2, :]), axis=2)

    del raw_im
    gc.collect()

    return out

def unpack_raw(raw4ch):
    """Unpacks 4 channels to Bayer image (h/2, w/2, 4) --> (h, w)."""
    img_shape = raw4ch.shape
    h = img_shape[0]
    w = img_shape[1]
    # d = img_shape[2]
    bayer = np.zeros([h * 2, w * 2], dtype=np.float32)
    # bayer = raw4ch
    # bayer.reshape((h * 2, w * 2))
    bayer[0::2, 0::2] = raw4ch[:, :, 0]
    bayer[0::2, 1::2] = raw4ch[:, :, 1]
    bayer[1::2, 1::2] = raw4ch[:, :, 2]
    bayer[1::2, 0::2] = raw4ch[:, :, 3]
    return bayer

def load_metadata(meta_path):
    """Loads metadata from file."""
    meta = scio.loadmat(meta_path)
    # meta = meta[list(meta.keys())[3]]  # 3rd key: 'metadata'
    meta = meta['metadata']  # key: 'metadata'
    return meta[0, 0]

def get_nlf(metadata):
    nlf = metadata['UnknownTags'][7, 0][2][0][0:2]
    # print('nlf shape = %s' % str(nlf.shape))
    return nlf

def load_one_tuple_images(filepath_tuple):
    in_path = filepath_tuple[0]  # index 0: input noisy image path
    gt_path = filepath_tuple[1]  # index 1: ground truth image path
    meta_path = filepath_tuple[2]  # index 3: metadata path

    # raw = loadmat(in_path)  # (use this for .mat files without -v7.3 format)
    with h5py.File(in_path, 'r') as f:  # (use this for .mat files with -v7.3 format)
        raw = f[list(f.keys())[0]]  # use the first and only key
        # input_image = np.transpose(raw)  # TODO: transpose?
        input_image = np.expand_dims(pack_raw(raw), axis=0)
        input_image = np.nan_to_num(input_image)
        input_image = np.clip(input_image, 0.0, 1.0)

    with h5py.File(gt_path, 'r') as f:
        gt_raw = f[list(f.keys())[0]]  # use the first and only key
        # gt_image = np.transpose(gt_raw)  # TODO: transpose?
        gt_image = np.expand_dims(pack_raw(gt_raw), axis=0)
        gt_image = np.nan_to_num(gt_image)
        gt_image = np.clip(gt_image, 0.0, 1.0)

    # with h5py.File(var_path, 'r') as f:
    #     var_raw = f[list(f.keys())[0]]  # use the first and only key
    #     var_image = np.expand_dims(pack_raw(var_raw), axis=0)
    #     np.nan_to_num(var_image)
    metadata = load_metadata(meta_path)

    nlf0, nlf1 = get_nlf(metadata)

    fparts = in_path.split('/')
    sdir = fparts[-3]
    if len(sdir) != 30:
        sdir = fparts[-2]  # if subdirectory does not exist
    iso = np.array(float(sdir[12:17]))[np.newaxis]
    # max_iso = 3200.0
    # iso = iso / max_iso  # - 0.5  # TODO: is this okay?
    cam = np.array(float(['IP', 'GP', 'S6', 'N6', 'G4'].index(sdir[9:11])))[np.newaxis]
#     shutter = np.array(float(['00050','00060','00100','00160','00200','00400','00800','01000','01600','02000'].index(sdir[18:23])))[np.newaxis]
#     if int(sdir[18:23])<=60:
#         shutter = np.array([0.0])
#     elif int(sdir[18:23])>60 and int(sdir[18:23])<=320:
#         shutter = np.array([1.0])
#     elif int(sdir[18:23])>320 and int(sdir[18:23])<=750:
#         shutter = np.array([2.0])
#     elif int(sdir[18:23])>750 and int(sdir[18:23])<=2500:
#         shutter = np.array([3.0])
#     else:
#         shutter = np.array([4.0])
#     shutter = np.array(float(['0','1','2','3','4'].index(sdir[18:23])))[np.newaxis]
    # use noise layer instead of noise image TODO: just to be aware of this crucial step
    input_image = input_image - gt_image

    # fix NLF(noise level )
    # nlf0 = sys.float_info.epsilon if nlf0 <= 0 else nlf0
    nlf0 = 1e-6 if nlf0 <= 0 else nlf0
    # nlf1 = sys.float_info.epsilon if nlf1 <= 0 else nlf1
    nlf1 = 1e-6 if nlf1 <= 0 else nlf1
    nlf0 = np.array(nlf0)[np.newaxis]
    nlf1 = np.array(nlf1)[np.newaxis]
    return input_image, gt_image, nlf0, nlf1, iso, cam, sdir

def sample_indices_uniform(h, w, ph, pw, shuf=False, n_pat_per_im=None):
    """Uniformly sample patch indices from (0, 0) up to (h, w) with patch height and width (ph, pw) """
    ii = []
    jj = []
    n_p = 0
    for i in np.arange(0, h - ph + 1, ph):
        for j in np.arange(0, w - pw + 1, pw):
            ii.append(i)
            jj.append(j)
            n_p += 1
            if (n_pat_per_im is not None) and (n_p == n_pat_per_im):
                break
        if (n_pat_per_im is not None) and (n_p == n_pat_per_im):
            break
    if shuf:
        ii, jj = shuffle(ii, jj)
    return ii, jj, n_p

def process(data_path,index,patch_size):
    data_list = []
#     flag = 0
    for i in index:
#         flag +=1
        idx = '%04d'%i
        input_path = glob.glob(path.join(data_path,idx+'*'))
        clean_path,meta_path,noise_path = glob.glob(path.join(input_path[0],'*'))
        noise, gt, nlf0,nlf1, iso, shutter , sdir = load_one_tuple_images([clean_path,noise_path,meta_path])
        _,H,W,_ = noise.shape
        ii, jj, n_p = sample_indices_uniform(H, W, patch_size, patch_size)
        pid = 0
        for (i, j) in zip(ii, jj):
            in_patch = noise[:,i:i + patch_size, j:j + patch_size, :]
            gt_patch = gt[:,i:i + patch_size, j:j + patch_size, :]
            pat_dict = {'in': in_patch, 'gt': gt_patch, 'nlf0': nlf0,
                        'nlf1': nlf1, 'iso': iso, 'shutter': shutter,
                        'filename': sdir, 'pid': pid}
            pid += 1
            data_list.append(pat_dict)
#         if flag ==3:
#             break
        print(sdir)
#         break
    return data_list