"""
U-Net Model for Road Segmentation
Deep learning architecture for binary segmentation of roads from satellite imagery.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import cv2


class UNetModel:
    """
    U-Net architecture for road segmentation.
    
    Architecture:
    - Encoder: 4 downsampling blocks
    - Bottleneck: 2 conv layers
    - Decoder: 4 upsampling blocks with skip connections
    - Output: Binary mask (128x128x1)
    """
    
    def __init__(self, input_shape=(128, 128, 3)):
        self.input_shape = input_shape
        self.model = self._build_unet()
        
    def _conv_block(self, inputs, filters, kernel_size=3):
        """Convolutional block: Conv -> BatchNorm -> ReLU -> Conv -> BatchNorm -> ReLU"""
        x = layers.Conv2D(filters, kernel_size, padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        return x
    
    def _encoder_block(self, inputs, filters):
        """Encoder block: Conv block -> MaxPooling"""
        x = self._conv_block(inputs, filters)
        p = layers.MaxPooling2D(pool_size=(2, 2))(x)
        return x, p
    
    def _decoder_block(self, inputs, skip_features, filters):
        """Decoder block: UpConv -> Concatenate -> Conv block"""
        x = layers.Conv2DTranspose(filters, (2, 2), strides=2, padding='same')(inputs)
        x = layers.Concatenate()([x, skip_features])
        x = self._conv_block(x, filters)
        return x
    
    def _build_unet(self):
        """Build lightweight U-Net architecture for faster training"""
        inputs = layers.Input(shape=self.input_shape)
        
        # Encoder (reduced filters: 32->64->128->256)
        s1, p1 = self._encoder_block(inputs, 32)      # 128x128 -> 64x64
        s2, p2 = self._encoder_block(p1, 64)          # 64x64 -> 32x32
        s3, p3 = self._encoder_block(p2, 128)         # 32x32 -> 16x16
        s4, p4 = self._encoder_block(p3, 256)         # 16x16 -> 8x8
        
        # Bottleneck (reduced from 1024 to 512)
        b = self._conv_block(p4, 512)                 # 8x8
        
        # Decoder
        d1 = self._decoder_block(b, s4, 256)          # 8x8 -> 16x16
        d2 = self._decoder_block(d1, s3, 128)         # 16x16 -> 32x32
        d3 = self._decoder_block(d2, s2, 64)          # 32x32 -> 64x64
        d4 = self._decoder_block(d3, s1, 32)          # 64x64 -> 128x128
        
        # Output
        outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(d4)
        
        model = models.Model(inputs=inputs, outputs=outputs, name='U-Net-Lite')
        return model
    
    def compile_model(self, learning_rate=1e-3):
        """Compile model with optimizer and loss (faster learning rate)"""
        # Enable mixed precision for faster training on compatible hardware
        try:
            from tensorflow.keras import mixed_precision
            policy = mixed_precision.Policy('mixed_float16')
            mixed_precision.set_global_policy(policy)
        except:
            pass  # Fall back to float32 if mixed precision not available
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', self._dice_coefficient, self._iou_metric]
        )
    
    @staticmethod
    def _dice_coefficient(y_true, y_pred, smooth=1e-6):
        """Dice coefficient metric"""
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        return (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    
    @staticmethod
    def _iou_metric(y_true, y_pred, smooth=1e-6):
        """Intersection over Union metric"""
        y_true_f = tf.reshape(y_true, [-1])
        y_pred_f = tf.reshape(y_pred, [-1])
        intersection = tf.reduce_sum(y_true_f * y_pred_f)
        union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - intersection
        return (intersection + smooth) / (union + smooth)
    
    def train(self, train_images, train_masks, epochs=50, batch_size=8, validation_split=0.2):
        """
        Train the U-Net model
        
        Args:
            train_images: numpy array (N, 128, 128, 3), normalized to [0, 1]
            train_masks: numpy array (N, 128, 128), binary {0, 1}
            epochs: number of training epochs
            batch_size: batch size for training
            validation_split: fraction of data for validation
            
        Returns:
            history: training history
        """
        # Normalize images to [0, 1]
        train_images = train_images.astype(np.float32) / 255.0
        
        # Expand mask dimensions to (N, 128, 128, 1)
        train_masks = np.expand_dims(train_masks, axis=-1).astype(np.float32)
        
        # Data augmentation
        data_gen = keras.preprocessing.image.ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='reflect'
        )
        
        # Callbacks (reduced patience for faster training)
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,  # Reduced from 10
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,  # Reduced from 5
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train
        history = self.model.fit(
            train_images, train_masks,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        return history
    
    def predict(self, image):
        """
        Predict road mask for a single image
        
        Args:
            image: numpy array (128, 128, 3), RGB image [0, 255]
            
        Returns:
            mask: numpy array (128, 128), probability mask [0.0, 1.0]
        """
        # Normalize
        img_normalized = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        # Predict
        pred = self.model.predict(img_batch, verbose=0)
        
        # Remove batch and channel dimensions
        # Return PROBABILITIES (not binary), let the app handle thresholding
        mask_prob = pred[0, :, :, 0]
        
        return mask_prob
    
    def save_model(self, filepath):
        """Save model weights"""
        self.model.save_weights(filepath)
    
    def load_model(self, filepath):
        """Load model weights"""
        self.model.load_weights(filepath)
    
    def summary(self):
        """Print model summary"""
        return self.model.summary()


class UNetSegmenter:
    """
    Wrapper class for U-Net to match the interface of RoadSegmenter
    """
    
    def __init__(self):
        self.model_type = 'unet'
        self.unet = UNetModel()
        self.unet.compile_model()
        self.history = None
        
    def train(self, images, masks, epochs=50, batch_size=8):
        """Train U-Net model"""
        self.history = self.unet.train(images, masks, epochs=epochs, batch_size=batch_size)
        return self.history
    
    def predict(self, image):
        """Predict road mask"""
        return self.unet.predict(image)
    
    def get_history(self):
        """Get training history"""
        return self.history
