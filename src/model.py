from tensorflow.keras.layers import (Conv2D, MaxPooling2D, concatenate, Input, 
                                     Lambda, Add, Activation, UpSampling2D, 
                                     BatchNormalization)
from tensorflow.keras import Model

def build_rd_unet(input_shape=(800, 800, 1), normalization='min_max', print_summary=True):
    """
    Builds a Residual Dilated U-Net model.
    """
    inputs = Input(input_shape)

    # Normalize input data
    if normalization == 'min_max':
        # Assumes input is 0-255, scales to 0-1
        normalized_inputs = Lambda(lambda x: x / 255)(inputs)
    elif normalization == 'batchnorm':
        normalized_inputs = BatchNormalization()(inputs)
    else:
        normalized_inputs = inputs

    # --- Helper for Residual Block ---
    def res_block(x, filters):
        # Path A
        conv = Conv2D(filters, (3, 3), kernel_initializer='he_normal', padding='same', use_bias=False)(x)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)
        conv = Conv2D(filters, (3, 3), activation=None, kernel_initializer='he_normal', padding='same', use_bias=False)(conv)
        conv = BatchNormalization()(conv)
        
        # Path B (Skip connection)
        skip = Conv2D(filters, (1, 1), activation=None, kernel_initializer='he_normal', padding='same', use_bias=False)(x)
        skip = BatchNormalization()(skip)
        
        # Add & Activate
        out = Add()([conv, skip])
        out = Activation('relu')(out)
        return out

    # --- Downsampling ---
    conv1 = res_block(normalized_inputs, 8)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = res_block(pool1, 16)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = res_block(pool2, 32)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = res_block(pool3, 48)
    pool4 = MaxPooling2D((2, 2))(conv4)

    # --- Dilated Bridge ---
    bridge = pool4
    # Dilations: 1, 2, 4, 8, 16
    for rate in [1, 2, 4, 8, 16]:
        bridge = Conv2D(64, (3, 3), dilation_rate=rate, kernel_initializer='he_normal', padding='same', use_bias=False)(bridge)
        bridge = BatchNormalization()(bridge)
        bridge = Activation('relu')(bridge)
    
    # Final bridge conv
    bridge = Conv2D(64, (3, 3), kernel_initializer='he_normal', padding='same')(bridge)
    bridge = BatchNormalization()(bridge)
    bridge = Activation('relu')(bridge)

    # --- Upsampling ---
    up8 = UpSampling2D(size=(2, 2))(bridge)
    up8 = concatenate([up8, conv4])
    conv8 = res_block(up8, 48)

    up9 = UpSampling2D(size=(2, 2))(conv8)
    up9 = concatenate([up9, conv3])
    conv9 = res_block(up9, 32)

    up10 = UpSampling2D(size=(2, 2))(conv9)
    up10 = concatenate([up10, conv2])
    conv10 = res_block(up10, 16)

    up11 = UpSampling2D(size=(2, 2))(conv10)
    up11 = concatenate([up11, conv1])
    conv11 = res_block(up11, 8)

    # --- Output ---
    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv11)

    model = Model(inputs=[inputs], outputs=[outputs], name="RD_UNet")

    if print_summary:
        model.summary()
    
    return model
