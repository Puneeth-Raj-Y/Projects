import numpy as np
import cv2
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

class RoadSegmenter:
    def __init__(self, model_type='svm'):
        self.model_type = model_type
        if model_type == 'svm':
            # Using a smaller max_iter to ensure quick training for the demo
            self.model = make_pipeline(StandardScaler(), SVC(probability=True, kernel='rbf', cache_size=1000, max_iter=2000))
        elif model_type == 'xgboost':
            self.model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
        else:
            raise ValueError("Model type must be 'svm' or 'xgboost'")

    def _extract_features(self, images):
        """
        Flattens images to pixels with RGB + HSV features.
        """
        # images shape: (N, H, W, 3). Assumes RGB.
        features_list = []
        for img in images:
            # Add HSV features (Hue, Saturation, Value)
            # Roads are often distinct in Saturation (low) and Value (varies)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            
            # Combine RGB + HSV -> 6 channels
            combined = np.concatenate([img, hsv], axis=2) # (H, W, 6)
            features_list.append(combined.reshape(-1, 6))
            
        return np.vstack(features_list)

    def _prepare_labels(self, masks):
        """
        Flattens masks to (N_pixels,)
        """
        # masks shape: (N, H, W)
        # Flatten: (N*H*W,)
        return masks.flatten()

    def train(self, images, masks):
        """
        Trains the model on the provided images and masks.
        """
        X = self._extract_features(images)
        y = self._prepare_labels(masks)
        
        
        # Class balancing and Subsampling
        # Roads are rare (~5%), so we must balance the dataset or the model will just predict "Background" for everything.
        
        # Find indices of road and background
        road_indices = np.where(y > 0)[0]
        bg_indices = np.where(y == 0)[0]
        
        n_road = len(road_indices)
        n_bg = len(bg_indices)
        
        # Limit total samples to 50,000 for speed, but keep balanced
        max_samples = 50000
        
        if n_road == 0:
            print("Warning: No road pixels found in training data!")
            # Fallback to random sampling if no roads (shouldn't happen with valid data)
            if len(y) > max_samples:
                idx = np.random.choice(len(y), max_samples, replace=False)
                X = X[idx]
                y = y[idx]
        else:
            # We want roughly 50/50 split if possible, or use all roads if small
            target_per_class = max_samples // 2
            
            # Sample roads
            if n_road > target_per_class:
                road_sample = np.random.choice(road_indices, target_per_class, replace=False)
            else:
                road_sample = road_indices # Take all roads
            
            # Sample background (match road count or hit limit)
            n_bg_target = min(n_bg, max_samples - len(road_sample))
            if n_bg > n_bg_target:
                bg_sample = np.random.choice(bg_indices, n_bg_target, replace=False)
            else:
                bg_sample = bg_indices
                
            # Combine
            idx = np.concatenate([road_sample, bg_sample])
            np.random.shuffle(idx)
            
            X = X[idx]
            y = y[idx]
            
        self.model.fit(X, y)

    def predict(self, image):
        """
        Predicts road mask for a single image.
        Returns:
            mask: (H, W) binary mask
        """
        # Image shape: (H, W, 3)
        H, W, C = image.shape
        
        # Features must match training features (6 channels now)
        X = self._extract_features(np.array([image]))
        
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(X)
            # Return probability of class 1 (Road)
            # Shape (H*W, ) -> (H, W)
            return probs[:, 1].reshape(H, W)
        else:
            # Fallback for models without probability (e.g. standard SVM without probability=True)
            y_pred = self.model.predict(X)
            return y_pred.reshape(H, W).astype(float)
