# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
import openslide
import h5py
import random
import pandas as pd
import cv2

try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None
        random.seed(random_seed)
        
        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])
            print("Using max images:", len(self._raw_idx))
        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        resolution      = None, # Ensure specific resolution, None = highest available.
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
                image = pyspng.load(f.read())
            else:
                image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        fname = 'dataset.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        return labels

#----------------------------------------------------------------------------
    
class WSICoordDataset(Dataset):  
    def __init__(self,
        wsi_dir,                   # Path to WSI directory.
        coord_dir,             # Path to h5 coord database.
        process_list = None,  #Dataframe path of WSIs to process and their seg_levels/downsample levels that correspond to the coords
        wsi_exten = '.svs',
        max_coord_per_wsi = 'inf',
        resolution      = 256, # Ensure specific resolution.
        desc = None,
        rescale_mpp = True,
        desired_mpp = 0.25,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self.wsi_dir = wsi_dir
        self.wsi_exten = wsi_exten
        self.coord_dir = coord_dir
        self.max_coord_per_wsi = max_coord_per_wsi
        if process_list is None:
            self.process_list = None
        else:
            self.process_list = pd.read_csv(process_list)
        self.patch_size = resolution
        self.rescale_mpp = rescale_mpp
        self.desired_mpp = desired_mpp
        #Implement labels here..
        self.coord_dict, self.wsi_names = self.createCoordDict(self.wsi_dir, self.wsi_exten, self.coord_dir, self.max_coord_per_wsi, self.process_list, self.patch_size)
        
        if desc is None:
            name = str(self.coord_dir)
        else:
            name = desc
        self.coord_size = len(self.coord_dict)  # get the size of coord dataset
        print('Number of WSIs:', len(self.wsi_names))
        print('Number of patches:', self.coord_size)
        # self.wsi = None
        # self.wsi_open = None
        
        self._all_fnames = os.listdir(self.wsi_dir)
        
        raw_shape = [self.coord_size] + list(self._load_raw_image(0).shape)
        print('Raw shape of dataset:', raw_shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)
    
    @staticmethod
    def createCoordDict(wsi_dir, wsi_exten, coord_dir, max_coord_per_wsi, process_list, patch_size):
        if process_list is None:
            #Only use WSI that have coord files....
            all_coord_files = sorted([x for x in os.listdir(coord_dir) if x.endswith('.h5')])
        else:
            #Only use WSI that coord files aren't excluded and are in coord_dir
            wsi_plist = list(process_list.loc[~process_list['exclude_ids'].isin(['y','yes','Y']),'slide_id'])
            coord_plist = sorted([x.split(wsi_exten)[0]+'.h5' for x in wsi_plist])
            all_coord_files = sorted([x for x in os.listdir(coord_dir) if x.endswith('.h5') and x in coord_plist])
        #Get WSI filenames from path that have coord files/in process list
        wsi_names = sorted([w for w in os.listdir(wsi_dir) if w.endswith(wsi_exten) and w.split(wsi_exten)[0]+'.h5' in all_coord_files])
                
        #Get corresponding coord h5 files using WSI paths
        h5_names = [wsi_name.split(wsi_exten)[0]+'.h5' for wsi_name in wsi_names]
        #Loop through coord files, get coord length, randomly choose X coords for each wsi (max_coord_per_wsi)
        coord_dict = {}
        wsi_number = 0
        for h5, wsi_name in zip(h5_names, wsi_names):
            #All h5 paths must exist....
            h5_path = os.path.join(coord_dir, h5)
            with h5py.File(h5_path, "r") as f:
                dset = f['coords']
                max_len = len(dset)
                if max_len < float(max_coord_per_wsi):
                    #Return all coords
                    coords = dset[:]
                else:
                    #Randomly select X coords
                    rand_ind = np.sort(random.sample(range(max_len), int(max_coord_per_wsi)))
                    coords = dset[rand_ind]
            #Check that coordinates and patch resolution is within the dimensions of the WSI... slow but only done once at beginning
            wsi = openslide.OpenSlide(os.path.join(wsi_dir, wsi_name))
            #Get the desired seg level for the patching based on process list
            if process_list is not None:
                seg_level = process_list.loc[process_list['slide_id']==wsi_name,'seg_level'].iloc[0]
                #if seg_level != 0:
                #    print('{} for {}'.format(seg_level, wsi_name))
            else:
                seg_level = 0
                
            dims = wsi.level_dimensions[seg_level]
            # print(wsi_name)
            for i,coord in enumerate(coords):
                #Check that coordinates are inside dims
                changed = False
            #   old_coord = coord.copy()
                if coord[0]+patch_size > dims[0]:
                    coord[0] = dims[0]-patch_size
                #   print('X not in bounds, adjusting')
                    changed = True
                if coord[1]+patch_size > dims[1]:
                    coord[1] = dims[1]-patch_size
                #   print('Y not in bounds, adjusting')
                    changed = True
                if changed:
                #   print("Changing coord {} to {}".format(old_coord, coord))
                    coords[i] = coord
            
            #Store as dictionary with tuples {0: (coord, wsi_number), 1: (coord, wsi_number), etc.}
            dict_len = len(coord_dict)
            for i in range(coords.shape[0]):
                coord_dict[i+dict_len] = (coords[i], wsi_number)
            
            #Storing number/index because smaller size than string
            wsi_number += 1
            
        return coord_dict, wsi_names 
    
    @staticmethod    
    def adjPatchOOB(wsi_dim, coord, patch_size):
        #wsi_dim = (wsi_width, wsi_height)
        #coord = (x, y) with y axis inverted or point (0,0) starting in top left of image
        #patchsize = integer for square patch only
        #assume coord starts at (0,0) in line with original WSI,
        #therefore the patch is only out-of-bounds if the coord+patchsize exceeds the WSI dimensions
        #check dimensions, adjust coordinate if out of bounds
        coord = (int(coord[0]), int(coord[1])) 
        if coord[0]+patch_size > wsi_dim[0]:
            coord[0] = int(wsi_dim[0] - patch_size)
        
        if coord[1]+patch_size > wsi_dim[1]:
            coord[1] = int(wsi_dim[1] - patch_size) 
        
        return coord

    def scalePatch(self, wsi, coord, input_mpp=0.5, desired_mpp=0.25, patch_size=512, eps=0.05, level=0):
        desired_mpp = float(desired_mpp)
        input_mpp = float(input_mpp)
        factor = desired_mpp/input_mpp
        #Openslide get dimensions of full WSI
        dims = wsi.level_dimensions[0]
        if input_mpp > desired_mpp + eps or input_mpp < desired_mpp - eps:
            #print('scale by {:.2f} factor'.format(factor))
            # if factor > 1
            #input mpp must be smaller and therefore at higher magnification (e.g. desired 40x vs input 60x) and vice versa
            #approach: shrink a larger patch by factor to the desired patch size or enlarge a smaller patch to desired patch size
            scaled_psize = int(patch_size*factor)
            #check and adjust dimensions of coord based on scaled patchsize
            coord = self.adjPatchOOB(dims, coord, scaled_psize)
            adj_patch = np.array(wsi.read_region(coord, level, (scaled_psize, scaled_psize)).convert('RGB'))
            #shrink patch down to desired mpp if factor > 1
            #enlarge if factor < 1
            patch = cv2.resize(adj_patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            return patch
        else: 
            #print('skip scaling factor {:.2f}. input um per pixel ({}) within +/- {} of desired MPP ({}).'.format(factor, input_mpp, eps, desired_mpp))
            coord = self.adjPatchOOB(dims, coord, patch_size)
            patch = np.array(wsi.read_region(coord, level, (patch_size, patch_size)).convert('RGB'))
            return patch
            
    def _load_raw_image(self, raw_idx):
        coord, wsi_num = self.coord_dict[raw_idx % self.coord_size]
        wsi_name = self.wsi_names[wsi_num]
        #print('opening {}'.format(wsi_name))
        img_path = os.path.join(self.wsi_dir, wsi_name)
        wsi = openslide.OpenSlide(img_path)
        #Check if WSI already open... does this really help performance?
        #Can't be pickled.... bad for multiprocessing in this case
        # if self.wsi_open is None or self.wsi_open != self.wsi_names[wsi_num]:
            # self.wsi = openslide.OpenSlide(img_path)
            # self.wsi_open = self.wsi_names[wsi_num]
        mpp = None
        if self.process_list is not None:
            seg_level = self.process_list.loc[self.process_list['slide_id']==wsi_name,'seg_level'].iloc[0]
            try:
                mpp = self.process_list.loc[self.process_list['slide_id']==wsi_name,'MPP'].iloc[0]
            except Exception as e:
                pass
            #if seg_level != 0:
            #    print('{} for {}'.format(seg_level, wsi_name))
        else:
            seg_level = 0
        
        if self.rescale_mpp:
            if mpp is None:
                try:
                    mpp = wsi.properties['openslide.mpp-x']
                except Exception as e:
                    print(e)
                    print(wsi_name)
                    raise ValueError('Cannot find slide MPP from process list or Openslide properties. Set rescale_mpp to False to avoid this error or add slide MPPs to process list.')
            img = self.scalePatch(wsi=wsi, coord=coord, input_mpp=mpp, desired_mpp=self.desired_mpp, patch_size=self.patch_size, level=seg_level) 
        else:
            img = np.array(wsi.read_region(coord, seg_level, (self.patch_size, self.patch_size)).convert('RGB'))
        # img = img.transpose(2, 0, 1) # HWC => CHW
        img = np.moveaxis(img, 2, 0) # HWC => CHW
        return img
    
    def __getstate__(self):
        return dict(super().__getstate__())
    
    # def _open_file(self, fname):
    #     return open(os.path.join(self.wsi_dir, fname), 'rb')

    # def close(self):
    #     try:
    #         if self._zipfile is not None:
    #             self._zipfile.close()
    #     finally:
    #         self._zipfile = None
    
    #Not implemented
    def _load_raw_labels(self):
        return None
        # fname = 'dataset.json'
        # if fname not in self._all_fnames:
        #     return None
        # with self._open_file(fname) as f:
        #     labels = json.load(f)['labels']
        # if labels is None:
        #     return None
        # labels = dict(labels)
        # labels = [labels[fname.replace('\\', '/')] for fname in self.wsi_names]
        # labels = np.array(labels)
        # labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])
        # return labels

#----------------------------------------------------------------------------
