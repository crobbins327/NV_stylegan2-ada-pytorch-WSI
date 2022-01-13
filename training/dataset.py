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
from training.wsi_utils import isWhitePatch_S, isBlackPatch_S
import cv2
import pyvips
import time

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
        rescale_mpp = False,
        desired_mpp = 0.25,
        check_white_black = False,
        random_seed = 0,
        load_mode = 'openslide',
        make_all_pipelines = False,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        try:
            random.seed(random_seed)
        except Exception as e:
            print(e)
            random.seed(0)
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
        self.check_white_black = check_white_black
        self.load_mode = load_mode
        self.make_all_pipelines = make_all_pipelines
        #Implement labels here..
        #Need to load the wsi_pipelines after init for multiprocessing?
        self.wsi_pipelines = None
        self.coord_dict, self.wsi_names, self.wsi_props = self.createWSIData()
        
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
        
        raw_shape = [self.coord_size] + list(self._load_raw_image(0, load_one=True).shape)
        print('Raw shape of dataset:', raw_shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, random_seed=random_seed, **super_kwargs)
        #Trying to resolve picking of this dictionary for multiprocessing.....
        #Maybe there's a better way... maybe just load one image or add a 'test' parameter?
        del self.wsi_pipelines
        self.wsi_pipelines = None
        
    def createWSIData(self):
        if self.process_list is None:
            #Only use WSI that have coord files....
            all_coord_files = sorted([x for x in os.listdir(self.coord_dir) if x.endswith('.h5')])
        else:
            #Only use WSI that coord files aren't excluded and are in coord_dir
            wsi_plist = list(self.process_list.loc[~self.process_list['exclude_ids'].isin(['y','yes','Y']),'slide_id'])
            coord_plist = sorted([os.path.splitext(x)[0]+'.h5' for x in wsi_plist])
            all_coord_files = sorted([x for x in os.listdir(self.coord_dir) if x.endswith('.h5') and x in coord_plist])
        #Get WSI filenames from path that have coord files/in process list
        wsi_names = sorted([w for w in os.listdir(self.wsi_dir) if w.endswith(tuple(self.wsi_exten)) and os.path.splitext(w)[0]+'.h5' in all_coord_files])
            
        #Get corresponding coord h5 files using WSI paths
        h5_names = [os.path.splitext(wsi_name)[0]+'.h5' for wsi_name in wsi_names]
        #Loop through coord files, get coord length, randomly choose X coords for each wsi (max_coord_per_wsi)
        coord_dict = {}
        wsi_props = {}
        # wsi_number = 0
        for h5, wsi_name in zip(h5_names, wsi_names):
            #All h5 paths must exist....
            h5_path = os.path.join(self.coord_dir, h5)
            with h5py.File(h5_path, "r") as f:
                attrs = dict(f['coords'].attrs)
                seg_level = int(attrs['patch_level'])
                dims = attrs['level_dim']
                #patch_size = attrs['patch_size']
                dset = f['coords']
                max_len = len(dset)
                if max_len < float(self.max_coord_per_wsi):
                    #Return all coords
                    coords = dset[:]
                else:
                    #Randomly select X coords
                    rand_ind = np.sort(random.sample(range(max_len), int(self.max_coord_per_wsi)))
                    coords = dset[rand_ind]
            #Check that coordinates and patch resolution is within the dimensions of the WSI... slow but only done once at beginning
            if self.check_white_black:
                wsi = openslide.OpenSlide(os.path.join(self.wsi_dir, wsi_name))

            #Get the desired seg level for the patching based on process list
            mpp = None
            seg_level = 0
            if self.process_list is not None:
                seg_level = int(self.process_list.loc[self.process_list['slide_id']==wsi_name,'seg_level'].iloc[0])
                if self.rescale_mpp and 'MPP' in self.process_list.columns:
                    mpp = float(self.process_list.loc[self.process_list['slide_id']==wsi_name,'MPP'].iloc[0])
                    seg_level = 0
                #if seg_level != 0:
                #    print('{} for {}'.format(seg_level, wsi_name))
            if self.rescale_mpp and mpp is None:
                try:
                    wsi = openslide.OpenSlide(os.path.join(self.wsi_dir, wsi_name))
                    mpp = float(wsi.properties['openslide.mpp-x'])
                    seg_level = 0
                except Exception as e:
                    print(e)
                    print(wsi_name)
                    raise ValueError('Cannot find slide MPP from process list ["MPP"] or Openslide properties. Set rescale_mpp to False to avoid this error or add slide MPPs to process list')
            
            del_index = []
            # print(wsi_name)
            for i,coord in enumerate(coords):
                #Check that coordinates are inside dims
                changed = False
            #   old_coord = coord.copy()
                if coord[0]+self.patch_size > dims[0]:
                    coord[0] = dims[0]-self.patch_size
                #   print('X not in bounds, adjusting')
                    changed = True
                if coord[1]+self.patch_size > dims[1]:
                    coord[1] = dims[1]-self.patch_size
                #   print('Y not in bounds, adjusting')
                    changed = True
                if changed:
                #   print("Changing coord {} to {}".format(old_coord, coord))
                    coords[i] = coord
                if self.check_white_black:
                    patch = np.array(wsi.read_region(coord, seg_level, (self.patch_size, self.patch_size)).convert('RGB'))
                    #print('Checking if patch is white or black...')
                    if isBlackPatch_S(patch, rgbThresh=20, percentage=0.05) or isWhitePatch_S(patch, rgbThresh=220, percentage=0.5):
                        #print('Removing coord because patch is black or white...')
                        #print(i)
                        del_index.append(i)
            
            if len(del_index) > 0:
                print('Removing {} coords that have black or white patches....'.format(len(del_index)))
                coords = np.delete(coords, del_index, axis=0)    
            
            #Store as dictionary with tuples {0: (coord, wsi_number), 1: (coord, wsi_number), etc.}
            dict_len = len(coord_dict)
            for i in range(coords.shape[0]):
                #Make key a string so that it is less likely to have hash collisions...
                coord_dict[str(i+dict_len)] = (coords[i], wsi_name)
            wsi_props[wsi_name] = (seg_level, mpp)
            #Storing number/index because smaller size than string??
            # wsi_number += 1
            
        return coord_dict, wsi_names, wsi_props
    
    @staticmethod    
    def adjPatchOOB(wsi_dim, coord, patch_size):
        #wsi_dim = (wsi_width, wsi_height)
        #coord = (x, y) with y axis inverted or point (0,0) starting in top left of image
        #patchsize = integer for square patch only
        #assume coord starts at (0,0) in line with original WSI,
        #therefore the patch is only out-of-bounds if the coord+patchsize exceeds the WSI dimensions
        #check dimensions, adjust coordinate if out of bounds
        coord = [int(coord[0]), int(coord[1])] 
        if coord[0]+patch_size > wsi_dim[0]:
            coord[0] = int(wsi_dim[0] - patch_size)
        
        if coord[1]+patch_size > wsi_dim[1]:
            coord[1] = int(wsi_dim[1] - patch_size) 
        
        return tuple(coord)

    def scalePatch(self, wsi, dims, coord, input_mpp=0.5, desired_mpp=0.25, patch_size=512, eps=0.05, level=0):
        desired_mpp = float(desired_mpp)
        input_mpp = float(input_mpp)
        factor = desired_mpp/input_mpp
        #Openslide get dimensions of full WSI
        # dims = wsi.level_dimensions[0]
        if input_mpp > desired_mpp + eps or input_mpp < desired_mpp - eps:
            #print('scale by {:.2f} factor'.format(factor))
            # if factor > 1
            #input mpp must be smaller and therefore at higher magnification (e.g. desired 40x vs input 60x) and vice versa
            #approach: shrink a larger patch by factor to the desired patch size or enlarge a smaller patch to desired patch size
            scaled_psize = int(patch_size*factor)
            #check and adjust dimensions of coord based on scaled patchsize
            coord = self.adjPatchOOB(dims, coord, scaled_psize)
            adj_patch = self._load_patch(wsi, level, coord, scaled_psize, dims=dims)
            #shrink patch down to desired mpp if factor > 1
            #enlarge if factor < 1
            #Could implement fully in vips...
            patch = cv2.resize(adj_patch, (patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
            return patch
        else: 
            #print('skip scaling factor {:.2f}. input um per pixel ({}) within +/- {} of desired MPP ({}).'.format(factor, input_mpp, eps, desired_mpp))
            coord = self.adjPatchOOB(dims, coord, patch_size)
            patch = self._load_patch(wsi, level, coord, patch_size, dims=dims)
            return patch
    
    @staticmethod
    def fetch(region, patch_size, x, y):
        return region.fetch(x, y, patch_size, patch_size)
    
    def vips_readRegion(self, region, level, patch_size, x, y, mode="RGBA", ref_level=0):
        #Assumes that region is at desired level, therefore coordinates x,y need to be downsampled
        #because x,y coords are referenced at level=0
        assert isinstance(level, int)
        assert isinstance(ref_level, int)
        if level < 0:
            level = 0
        if ref_level < 0:
            ref_level = 0
        downsample = 2**(level-ref_level)
        x,y = x//downsample, y//downsample
        patch = self.fetch(region, patch_size, int(x), int(y))
        return PIL.Image.frombuffer(mode, (patch_size, patch_size), patch, 'raw', mode, 0, 1)
    
    def vips_crop(self, wsi_vips, level, patch_size, x, y, mode="RGBA", ref_level=0):
        #Assumes that region is at desired level, therefore coordinates x,y need to be downsampled
        #because x,y coords are referenced at level=0
        assert isinstance(level, int)
        assert isinstance(ref_level, int)
        if level < 0:
            level = 0
        if ref_level < 0:
            ref_level = 0
        downsample = 2**(level-ref_level)
        x,y = x//downsample, y//downsample
        patch = wsi_vips.crop(int(x), int(y), patch_size, patch_size)
        return PIL.Image.frombuffer(mode, (patch_size, patch_size), patch.write_to_memory(), 'raw', mode, 0, 1)
    
    def _load_wsi_pipelines(self, load_one_wsi_name = None):
        #Create all the image pipelines in a dictionary
        wsi_pipelines = {}
        
        if load_one_wsi_name is not None:
            load_WSIs = [load_one_wsi_name]
        else:
            load_WSIs = self.wsi_names
        
        #vips method
        if self.load_mode == 'vips':
            for wsi_name in load_WSIs:
                seg_level, mpp = self.wsi_props[wsi_name]
                if os.path.splitext(wsi_name)[1]=='.tiff' or os.path.splitext(wsi_name)[1]=='.tif':
                    wsi_vips = pyvips.Image.new_from_file(os.path.join(self.wsi_dir, wsi_name))
                else:                
                    wsi_vips = pyvips.Image.openslideload(os.path.join(self.wsi_dir, wsi_name), level=seg_level)
                reg = pyvips.Region.new(wsi_vips)
                if seg_level==0:
                    dims = [wsi_vips.width, wsi_vips.height, wsi_vips.bands]
                else:
                    assert isinstance(seg_level, int)
                    if seg_level < 0:
                        seg_level = 0
                    downsample = 2**seg_level
                    dims = [int(wsi_vips.width*downsample), int(wsi_vips.height*downsample), wsi_vips.bands]
                wsi_pipelines[wsi_name] = (reg, dims)
            return wsi_pipelines
        elif self.load_mode == 'vips-crop':
            for wsi_name in load_WSIs:
                seg_level, mpp = self.wsi_props[wsi_name]
                if os.path.splitext(wsi_name)[1]=='.tiff' or os.path.splitext(wsi_name)[1]=='.tif':
                    wsi_vips = pyvips.Image.new_from_file(os.path.join(self.wsi_dir, wsi_name))
                else:                
                    wsi_vips = pyvips.Image.openslideload(os.path.join(self.wsi_dir, wsi_name), level=seg_level)
                # reg = pyvips.Region.new(wsi_vips)
                if seg_level==0:
                    dims = [wsi_vips.width, wsi_vips.height, wsi_vips.bands]
                else:
                    assert isinstance(seg_level, int)
                    if seg_level < 0:
                        seg_level = 0
                    downsample = 2**seg_level
                    dims = [int(wsi_vips.width*downsample), int(wsi_vips.height*downsample), wsi_vips.bands]
                wsi_pipelines[wsi_name] = (wsi_vips, dims)
            return wsi_pipelines
        #openslide method (slower than vips!)
        elif self.load_mode == 'openslide':
            for wsi_name in load_WSIs:
                seg_level, mpp = self.wsi_props[wsi_name]
                wsi = openslide.OpenSlide(os.path.join(self.wsi_dir, wsi_name))
                dims = wsi.level_dimensions[seg_level]
                wsi_pipelines[wsi_name] = wsi, dims
            return wsi_pipelines
        else:
            raise ValueError('Load mode {} not supported! Modes {} are supported.'.format(self.load_mode, ['vips', 'openslide']))

    def _load_one_wsi(self, wsi_name):        
        #vips method
        if self.load_mode == 'vips':
            seg_level, mpp = self.wsi_props[wsi_name]
            wsi_vips = pyvips.Image.openslideload(os.path.join(self.wsi_dir, wsi_name), level=seg_level, access='sequential')
            reg = pyvips.Region.new(wsi_vips)
            if seg_level==0:
                dims = [wsi_vips.width, wsi_vips.height, wsi_vips.bands]
            else:
                assert isinstance(seg_level, int)
                if seg_level < 0:
                    seg_level = 0
                downsample = 2**seg_level
                dims = [int(wsi_vips.width*downsample), int(wsi_vips.height*downsample), wsi_vips.bands]
            return reg, dims
        elif self.load_mode == 'vips-crop':
            seg_level, mpp = self.wsi_props[wsi_name]
            wsi_vips = pyvips.Image.openslideload(os.path.join(self.wsi_dir, wsi_name), level=seg_level, access='sequential')
            # reg = pyvips.Region.new(wsi_vips)
            if seg_level==0:
                dims = [wsi_vips.width, wsi_vips.height, wsi_vips.bands]
            else:
                assert isinstance(seg_level, int)
                if seg_level < 0:
                    seg_level = 0
                downsample = 2**seg_level
                dims = [int(wsi_vips.width*downsample), int(wsi_vips.height*downsample), wsi_vips.bands]
            return wsi_vips, dims
        #openslide method (slower than vips!)
        elif self.load_mode == 'openslide':
            seg_level, mpp = self.wsi_props[wsi_name]
            wsi = openslide.OpenSlide(os.path.join(self.wsi_dir, wsi_name))
            dims = wsi.level_dimensions[seg_level]
            return wsi, dims
        else:
            raise ValueError('Load mode {} not supported! Modes {} are supported.'.format(self.load_mode, ['vips', 'vips-crop', 'openslide']))
    
    def _load_patch(self, wsi, level, coord, patch_size, dims=None):
        if self.load_mode == 'openslide':
            patch = np.array(wsi.read_region(coord, level, (patch_size, patch_size)).convert('RGB'))
        elif self.load_mode == 'vips':
            if dims[2] == 4:
                patch = np.array(self.vips_readRegion(wsi, level, patch_size, coord[0], coord[1], mode='RGBA').convert('RGB'))
            elif dims[2] == 3:
                #print('pyvips opened image as RGB')
                patch = np.array(self.vips_readRegion(wsi, level, patch_size, coord[0], coord[1], mode='RGB'))
            else:
                raise ValueError('Mode for image (RGB or RGBA) not specified/supported. Cannot use vips to open and scale patches..')
        elif self.load_mode == 'vips-crop':
            if dims[2] == 4:
                patch = np.array(self.vips_crop(wsi, level, patch_size, coord[0], coord[1], mode='RGBA').convert('RGB'))
            elif dims[2] == 3:
                #print('pyvips opened image as RGB')
                patch = np.array(self.vips_crop(wsi, level, patch_size, coord[0], coord[1], mode='RGB'))
            else:
                raise ValueError('Mode for image (RGB or RGBA) not specified/supported. Cannot use vips to open and scale patches..')
        else:
            raise ValueError('Load mode {} not supported! Modes {} are supported.'.format(self.load_mode, ['vips', 'openslide']))
        return patch
    
    #Is this part of the code slowing the iteration time down??      
    def _load_raw_image(self, raw_idx, load_one=False):
        #lookup on a hashtable/dict should not matter if there are 5 entries or 5 million
        #coord_start = time.time()
        coord, wsi_name = self.coord_dict[str(raw_idx % self.coord_size)]
        #coord_end = time.time()
        #print('Coord lookup time:', coord_end-coord_start)
        # wsi_name = self.wsi_names[wsi_num]
        #prop_start = time.time()
        seg_level, mpp = self.wsi_props[wsi_name]
        #prop_end = time.time()
        #print('prop lookup time:', prop_end-prop_start)
        #Had to do it this way because you can't pickle openslide/pyvips WSIs for multiprocessing...
        #Why does using the pipeline consume RAM, but when you initially create the pipeline(s) it does not consume as much RAM?
        if self.make_all_pipelines:
            #pipelines_start = time.time()
            if self.wsi_pipelines is None:
                #load pipelines first
                if load_one:
                    #For the test image for init
                    self.wsi_pipelines = self._load_wsi_pipelines(load_one_wsi_name=wsi_name)
                else:
                    self.wsi_pipelines = self._load_wsi_pipelines()

            if self.load_mode=='openslide':
                #Cannot copy or deepcopy openslide objects, and the objects accumulate RAM in the wsi_pipelines dict
                wsi, dims = self.wsi_pipelines[wsi_name]
            else:
                #Perhaps copying the object from the dict uses less RAM? The object should be deleted at the end of this function's scope, and it won't accumulate RAM being stored in dictionary...?
                wsi, dims = self.wsi_pipelines[wsi_name][0].copy(), self.wsi_pipelines[wsi_name][1] 
           # pipelines_end = time.time()
            #print('Time to load WSI after making all pipelines:', pipelines_end-pipelines_start)
        else:
         #   pipelines_start = time.time()
            wsi, dims = self._load_one_wsi(wsi_name)
          #  pipelines_end = time.time()
            #print('Time to remake pipeline and load 1 WSI image:', pipelines_end-pipelines_start)
        #patch_start = time.time()
        if self.rescale_mpp:
            if mpp is None and self.load_mode == 'openslide':
                try:
                    mpp = wsi.properties['openslide.mpp-x']
                except Exception as e:
                    print(e)
                    print(wsi_name)
                    raise ValueError('Cannot find slide MPP from process list or Openslide properties. Set rescale_mpp to False to avoid this error or add slide MPPs to process list.')
            elif mpp is None and self.load_mode != 'openslide':
                raise ValueError('Cannot find slide MPP from process list. Cannot use mode load_mode {} if MPP not in process list. Change load_mode to "openslide", set rescale_mpp to False, or slide MPPs to process list to avoid error.'.format(self.load_mode))
            #print(wsi, coord, seg_level, mpp)
            img = self.scalePatch(wsi=wsi, dims=dims, coord=coord, input_mpp=mpp, desired_mpp=self.desired_mpp, patch_size=self.patch_size, level=seg_level) 
        else:
            #Loading time may depend on disk and file location. Scratch60 may be an HDD whereas project is a SSD....
            img = self._load_patch(wsi, seg_level, coord, self.patch_size, dims=dims)
        # img = img.transpose(2, 0, 1) # HWC => CHW
        img = np.moveaxis(img, 2, 0) # HWC => CHW
        #patch_end = time.time()
        #print('Time to load patch:', patch_end-patch_start)
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
