import os
import numpy as np
import tensorflow.keras as keras
import skimage.io
import albumentations as A
import cv2

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, img_path, mask_path, batch_size=32, dim=(800,800), 
                 n_channels=1, shuffle=True, augmentation=True, **kwargs):
        # [FIX] Call super init for Keras 3 compatibility
        super().__init__(**kwargs)
        
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.img_path = img_path
        self.mask_path = mask_path
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.augmentation = augmentation
        
        # Albumentations pipeline
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.75),
            A.VerticalFlip(p=0.75),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=0, p=0.75, border_mode=cv2.BORDER_CONSTANT),
            A.Rotate(limit=270, p=0.75, border_mode=cv2.BORDER_CONSTANT),
            A.RandomRotate90(p=0.75),
            A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.15, 0.15), p=0.75),
            A.GaussNoise(var_limit=(0.0, 200.0), p=0.75),
            A.Defocus(radius=(1, 3), p=0.75)
        ])
        
        # Initialize indexes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        # [NOTE] If you have fewer images than batch_size, this returns 0 and crashes.
        # Ensure len(self.list_IDs) >= self.batch_size
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        X, y = self.__data_generation(list_IDs_temp)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def read_image(self, path):
        img = skimage.io.imread(path, as_gray=True)
        return img

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim), dtype=np.uint8)
        y = np.empty((self.batch_size, *self.dim), dtype=np.uint8)

        for i, ID in enumerate(list_IDs_temp):
            # Load Image
            X[i,] = self.read_image(os.path.join(self.img_path, ID))
            
            # Load Mask (Assumes same filename but .tif extension)
            # Adjust this replacement logic if your filenames differ
            # Example: "image_01.jpg" -> "image_01.tif"
            if ID.lower().endswith('.jpg'):
                mask_name = ID.replace('.jpg', '.tif').replace('.JPG', '.tif')
            elif ID.lower().endswith('.png'):
                mask_name = ID.replace('.png', '.tif')
            else:
                mask_name = ID + '.tif' # Fallback
                
            y[i] = self.read_image(os.path.join(self.mask_path, mask_name))

            if self.augmentation:
                augmented = self.transform(image=X[i,], mask=y[i,])
                X[i,] = augmented['image']
                y[i,] = augmented['mask']

        X = np.expand_dims(X, axis=3)
        y = np.expand_dims(y, axis=3).astype(bool)

        return X, y
