import streamlit as st
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from data_generator import generate_synthetic_data
from model import RoadSegmenter
from unet_model import UNetSegmenter

st.set_page_config(page_title="Road Extraction AI", layout="wide")

st.title("🛰️ Automated Road Extraction from Satellite Images")
st.markdown("Use Machine Learning (SVM / XGBoost / U-Net) to identify roads in satellite imagery.")

# Sidebar for controls
st.sidebar.header("Configuration")
model_choice = st.sidebar.selectbox("Choose Model", ["SVM", "xgboost", "U-Net"])
train_size = st.sidebar.slider("Training Samples", 10, 5000, 50)

# U-Net specific parameters
if model_choice == "U-Net":
    st.sidebar.subheader("U-Net Parameters")
    epochs = st.sidebar.slider("Training Epochs", 5, 100, 10)  # Reduced default from 20 to 10
    batch_size = st.sidebar.slider("Batch Size", 4, 32, 16)    # Increased from 8 to 16 for faster training
else:
    epochs = 10
    batch_size = 16

# State management
if 'model' not in st.session_state:
    st.session_state['model'] = None
if 'model_type' not in st.session_state:
    st.session_state['model_type'] = None

# 1. Data Generation Section
st.header("1. Data Preparation")
data_src = st.radio("Select Data Source", ["Synthetic (Generated)", "Real Data (From Folder)"])

if data_src == "Synthetic (Generated)":
    if st.button("Generate Synthetic Training Data"):
        with st.spinner("Generating data..."):
            images, masks = generate_synthetic_data(num_samples=train_size)
            st.session_state['train_images'] = images
            st.session_state['train_masks'] = masks
            st.success(f"Generated {train_size} images!")

else:
    st.markdown("Provide absolute paths to your data folders. Images must be JPG/PNG.")
    c1, c2 = st.columns(2)
    img_path = c1.text_input("Images Folder Path", value=os.path.abspath(os.path.join("data", "train", "images")))
    mask_path = c2.text_input("Masks Folder Path", value=os.path.abspath(os.path.join("data", "train", "masks")))
    
    if st.button("Load Real Data"):
        if not os.path.exists(img_path) or not os.path.exists(mask_path):
            st.error("One or both paths do not exist!")
        else:
            with st.spinner("Loading and resizing images..."):
                # Simple loader
                import glob
                image_files = sorted(glob.glob(os.path.join(img_path, "*.*")))
                mask_files = sorted(glob.glob(os.path.join(mask_path, "*.*")))
                
                # Filter for images
                image_files = [f for f in image_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                mask_files = [f for f in mask_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                # Limit to selected size
                image_files = image_files[:train_size]
                mask_files = mask_files[:train_size]
                
                if len(image_files) != len(mask_files) or len(image_files) == 0:
                    st.error(f"Found {len(image_files)} images and {len(mask_files)} masks. Counts must match and be > 0.")
                else:
                    loaded_imgs = []
                    loaded_masks = []
                    for img_f, mask_f in zip(image_files, mask_files):
                        # Load Image
                        img = cv2.imread(img_f)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        img = cv2.resize(img, (128, 128))
                        loaded_imgs.append(img)
                        
                        # Load Mask
                        mask = cv2.imread(mask_f, cv2.IMREAD_GRAYSCALE)
                        mask = cv2.resize(mask, (128, 128))
                        _, mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)
                        loaded_masks.append(mask)
                    
                    st.session_state['train_images'] = np.array(loaded_imgs)
                    st.session_state['train_masks'] = np.array(loaded_masks)
                    st.success(f"Loaded {len(loaded_imgs)} real images successfully!")

if 'train_images' in st.session_state:
    st.image(st.session_state['train_images'][0], caption="Sample Training Image", width=300)
    st.image(st.session_state['train_masks'][0] * 255, caption="Sample Ground Truth Mask", width=300)

# 2. Training Section
st.header("2. Model Training")
if st.button("Train Model"):
    if 'train_images' not in st.session_state:
        st.error("Please generate data first!")
    else:
        if model_choice == "U-Net":
            with st.spinner(f"Training U-Net (this may take a few minutes)..."):
                # Create progress containers
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Clear Keras session to prevent graph conflicts
                import tensorflow as tf
                tf.keras.backend.clear_session()
                
                # Train U-Net
                segmenter = UNetSegmenter()
                
                # Custom callback for progress
                class StreamlitCallback(tf.keras.callbacks.Callback):
                    def on_epoch_end(self, epoch, logs=None):
                        progress = (epoch + 1) / epochs
                        progress_bar.progress(progress)
                        status_text.text(f"Epoch {epoch+1}/{epochs} - Loss: {logs['loss']:.4f} - Val Loss: {logs.get('val_loss', 0):.4f}")
                
                # Add custom callback
                history = segmenter.train(
                    st.session_state['train_images'], 
                    st.session_state['train_masks'],
                    epochs=epochs,
                    batch_size=batch_size
                )
                
                st.session_state['model'] = segmenter
                st.session_state['model_type'] = model_choice
                st.session_state['training_history'] = history
                
                progress_bar.progress(1.0)
                status_text.text("Training completed!")
                st.success(f"U-Net Model Trained Successfully!")
                
                # Plot training history
                if history is not None:
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    # Loss plot
                    ax1.plot(history.history['loss'], label='Training Loss')
                    ax1.plot(history.history['val_loss'], label='Validation Loss')
                    ax1.set_xlabel('Epoch')
                    ax1.set_ylabel('Loss')
                    ax1.set_title('Training and Validation Loss')
                    ax1.legend()
                    ax1.grid(True)
                    
                    # Metrics plot
                    ax2.plot(history.history['accuracy'], label='Accuracy')
                    if 'dice_coefficient' in history.history:
                        ax2.plot(history.history['dice_coefficient'], label='Dice Coefficient')
                    ax2.set_xlabel('Epoch')
                    ax2.set_ylabel('Metric')
                    ax2.set_title('Training Metrics')
                    ax2.legend()
                    ax2.grid(True)
                    
                    st.pyplot(fig)
                    plt.close()
        else:
            with st.spinner(f"Training {model_choice}..."):
                segmenter = RoadSegmenter(model_type=model_choice.lower())
                segmenter.train(st.session_state['train_images'], st.session_state['train_masks'])
                st.session_state['model'] = segmenter
                st.session_state['model_type'] = model_choice
                st.success(f"{model_choice} Model Trained Successfully!")

# 3. Inference Section
st.header("3. Test the Model")

test_mode = st.radio("Select Test Image Source:", ["Upload Image", "Random Sample from Test Set"])
test_img = None

if test_mode == "Upload Image":
    test_file = st.file_uploader("Upload a satellite image", type=['jpg', 'png', 'jpeg'])
    if test_file is not None:
        file_bytes = np.asarray(bytearray(test_file.read()), dtype=np.uint8)
        test_img = cv2.imdecode(file_bytes, 1)
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
        test_img = cv2.resize(test_img, (128, 128))

elif test_mode == "Random Sample from Test Set":
    test_dir = os.path.abspath(os.path.join("data", "test", "images"))
    if st.button("Load Random Test Image"):
        if os.path.exists(test_dir):
            files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.png'))]
            if files:
                import random
                chosen = random.choice(files)
                path = os.path.join(test_dir, chosen)
                # st.info(f"Loaded: {chosen}")
                
                # Load and process
                loaded = cv2.imread(path)
                loaded = cv2.cvtColor(loaded, cv2.COLOR_BGR2RGB)
                loaded = cv2.resize(loaded, (128, 128))
                st.session_state['test_image_cache'] = loaded
            else:
                st.error("No images found in test folder!")
        else:
             st.error(f"Test Data folder not found at: {test_dir}")
    
    # Use cached image
    if 'test_image_cache' in st.session_state:
        test_img = st.session_state['test_image_cache']

# Manage prediction state
if 'prediction_done' not in st.session_state:
    st.session_state['prediction_done'] = False
if 'final_mask' not in st.session_state:
    st.session_state['final_mask'] = None

if st.button("Run Prediction"):
    if st.session_state['model'] is None:
        st.error("Train a model first!")
    elif test_img is None:
        st.error("Load or upload an image first!")
    else:
        # Predict
        pred_mask = st.session_state['model'].predict(test_img)
        st.session_state['final_mask'] = pred_mask
        st.session_state['prediction_img'] = test_img
        st.session_state['prediction_done'] = True

if st.session_state['prediction_done'] and st.session_state['final_mask'] is not None and 'prediction_img' in st.session_state:
    raw_pred = st.session_state['final_mask']
    test_img = st.session_state['prediction_img']
    
    # --- Post-Processing ---
    st.subheader("Post-Processing & Sensitivity")
    col1, col2 = st.columns(2)
    
    # Add Sensitivity Slider
    threshold = col1.slider("Sensitivity Threshold", 0.0, 1.0, 0.4, 0.05)
    
    # --- Robust Normalization ---
    # Normalize probabilities to 0-1 range to ensure we can always find *something*
    # even if the model is very unsure (low absolute probability).
    p_min, p_max = raw_pred.min(), raw_pred.max()
    if p_max > p_min:
        normalized_pred = (raw_pred - p_min) / (p_max - p_min)
    else:
        normalized_pred = raw_pred
        
    st.caption(f"Debug: Model Confidence Range: [{p_min:.3f}, {p_max:.3f}] -> Normalized to [0.0, 1.0]")
    
    use_cleanup = col2.checkbox("Apply Noise Reduction", value=True)
    kernel_size = col2.slider("Noise Filter Strength", 1, 10, 3)
    
    # Apply Threshold on NORMALIZED data
    pred_mask = (normalized_pred > threshold).astype(np.uint8)
    
    final_mask = pred_mask
    
    # Auto-disable cleanup if very few pixels found
    if np.sum(final_mask > 0) < 50:
        use_cleanup = False
        st.caption("ℹ️ Cleanup auto-disabled due to low signal.")
        
    if use_cleanup:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        # Opening: Removes small white noise
        final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
        # Closing: Fills small black holes
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

    # Display results
    c1, c2 = st.columns(2)
    with c1:
        st.image(test_img, caption="Input Image", use_container_width=True)
    with c2:
        st.image(final_mask * 255, caption="Predicted Road Mask", use_container_width=True)
        
    if use_cleanup:
        st.caption("ℹ️ 'Noise Reduction' removes scattered pixels. Adjust 'Strength' to smooth more aggressively.")

    # --- 4. Flood Simulation Section ---
    st.header("4. Flood Simulation & Routing")
    st.markdown("Simulate a flood event and find the shortest safe path.")
    
    # Store final_mask in session state for flood simulation
    st.session_state['flood_mask'] = final_mask
    
    # We need the predicted mask to build the network
    if final_mask is not None and np.sum(final_mask > 0) > 10:
        # Initialize Router
        from router import FloodRouter
        
        # Build router with the mask
        try:
            router = FloodRouter(final_mask)
            
            if len(router.graph.nodes) < 2:
                st.warning("⚠️ No road segments detected (Black Mask). The flood simulation requires a predicted road network.")
                st.info("Try: \\n 1. **Lower the Sensitivity Threshold** (try 0.3-0.4) \\n 2. **Disable Noise Reduction** temporarily \\n 3. Training with more samples")
            else:
                col_sim1, col_sim2 = st.columns(2)
                
                with col_sim1:
                    st.subheader("Route Configuration")
                    # Start/End Points (normalized 0-127)
                    start_y = st.slider("Start Y", 0, 127, 10, key="sy")
                    start_x = st.slider("Start X", 0, 127, 10, key="sx")
                    end_y = st.slider("End Y", 0, 127, 110, key="ey")
                    end_x = st.slider("End X", 0, 127, 110, key="ex")
                    
                    st.markdown("---")
                    st.subheader("Flood Zone")
                    flood_y = st.slider("Flood Center Y", 0, 127, 64, key="fy")
                    flood_x = st.slider("Flood Center X", 0, 127, 64, key="fx")
                    flood_radius = st.slider("Flood Radius", 5, 50, 15, key="fr")
                    
                    st.markdown("---")
                    # Dedicated button for finding routes
                    find_route = st.button("🔍 Find Alternative Route", type="primary", use_container_width=True)
                
                start_pt = (start_y, start_x)
                end_pt = (end_y, end_x)
                flood_center = (flood_y, flood_x)
                
                # Initialize route state
                if 'route_result' not in st.session_state:
                    st.session_state['route_result'] = None
                
                # Calculate routes when button is clicked
                if find_route:
                    with st.spinner("Finding routes..."):
                        # 1. Calculate Standard Path (without flood)
                        path_normal, msg_normal = router.find_path(start_pt, end_pt)
                        
                        # 2. Get Flooded Nodes
                        flooded_nodes = router.get_flooded_nodes(flood_center, flood_radius)
                        
                        # 3. Calculate Alternative Path (avoiding flood)
                        path_alt, msg_alt = router.find_path(start_pt, end_pt, flooded_nodes=flooded_nodes)
                        
                        # Store results
                        st.session_state['route_result'] = {
                            'path_normal': path_normal,
                            'path_alt': path_alt,
                            'msg_normal': msg_normal,
                            'msg_alt': msg_alt,
                            'flood_center': flood_center,
                            'flood_radius': flood_radius,
                            'start_pt': start_pt,
                            'end_pt': end_pt,
                            'test_img': test_img.copy()
                        }
                
                # Display results if available
                if st.session_state['route_result'] is not None:
                    result = st.session_state['route_result']
                    path_normal = result['path_normal']
                    path_alt = result['path_alt']
                    
                    # --- Visualization ---
                    # Create detailed visualization
                    res_img = result['test_img'].copy()
                    
                    # Draw Flood Zone (Semi-transparent Red Circle)
                    overlay = res_img.copy()
                    cv2.circle(overlay, (result['flood_center'][1], result['flood_center'][0]), 
                              result['flood_radius'], (255, 0, 0), -1)
                    alpha = 0.3
                    res_img = cv2.addWeighted(overlay, alpha, res_img, 1 - alpha, 0)
                    
                    # Draw Original Path (if exists and different from alternative)
                    if path_normal and path_normal != path_alt:
                        pts_normal = np.array([(p[1], p[0]) for p in path_normal], np.int32)
                        pts_normal = pts_normal.reshape((-1, 1, 2))
                        # Yellow path with thicker line
                        cv2.polylines(res_img, [pts_normal], False, (255, 255, 0), 3)
                    
                    # Draw Alternative Path (HIGHLIGHTED)
                    if path_alt:
                        pts_alt = np.array([(p[1], p[0]) for p in path_alt], np.int32)
                        pts_alt = pts_alt.reshape((-1, 1, 2))
                        # Bright Cyan/Aqua path with thick line for visibility
                        cv2.polylines(res_img, [pts_alt], False, (0, 255, 255), 3)
                    
                    # Draw Points (smaller markers)
                    # Start = Green, End = Red
                    cv2.circle(res_img, (result['start_pt'][1], result['start_pt'][0]), 3, (0, 255, 0), -1)
                    cv2.circle(res_img, (result['start_pt'][1], result['start_pt'][0]), 4, (255, 255, 255), 1)
                    cv2.circle(res_img, (result['end_pt'][1], result['end_pt'][0]), 3, (255, 0, 0), -1)
                    cv2.circle(res_img, (result['end_pt'][1], result['end_pt'][0]), 4, (255, 255, 255), 1)
                    
                    with col_sim2:
                        st.image(res_img, caption="Flood Simulation Result", use_container_width=True)
                        
                        # Legend
                        st.markdown("""
                        **Legend:**
                        - 🟢 **Green**: Start Point
                        - 🔴 **Red**: End Point  
                        - 🟡 **Yellow**: Original Path (blocked by flood)
                        - 🔵 **Cyan (Highlighted)**: Alternative Safe Route
                        - 🔴 **Red Zone**: Flooded Area
                        """)
                        
                        # Status messages
                        if path_alt:
                            st.success(f"✅ Alternative Route Found! Length: {len(path_alt)} pixels")
                            if path_normal and len(path_normal) != len(path_alt):
                                detour = len(path_alt) - len(path_normal)
                                st.info(f"📏 Detour: +{detour} pixels ({detour/len(path_normal)*100:.1f}% longer)")
                        elif path_normal:
                            st.warning("⚠️ Original path blocked by flood, and no alternative route exists!")
                            st.error("The road network is disconnected. Try adjusting flood parameters.")
                        else:
                            st.error("❌ No path found at all (disconnected road network).")
                            st.info("Try lowering the sensitivity threshold or disabling noise reduction.")
                else:
                    with col_sim2:
                        st.info("👆 Click 'Find Alternative Route' to start simulation")
                        
        except Exception as e:
            st.error(f"Error building road network: {str(e)}")
            st.info("Try adjusting the sensitivity threshold or noise reduction settings.")
    else:
        st.warning("⚠️ No road segments detected. The flood simulation requires a predicted road network.")
        st.info("**Try:** \\n 1. **Lower the Sensitivity Threshold** (try 0.3-0.4) \\n 2. **Disable Noise Reduction** temporarily \\n 3. Train the model with more data")

