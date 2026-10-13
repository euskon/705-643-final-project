# Climbing Route Detection: 2D vs. 3D LiDAR Model Evaluation

This document outlines a 2D Convolutional Neural Network (CNN) approach for detecting climbing routes from LiDAR data, adapted from the methodology of Rafique et al. (2022). It also provides an experimental framework for using this 2D model as a baseline to evaluate a heavier, more complex 3D point cloud model.

## Part 1: The 2D CNN Baseline Approach

Instead of processing raw 3D point clouds, this approach converts the 3D data into 2D images using a "virtual camera" oriented toward the crag based on rough GPS coordinates. 

### Data Preparation and Encoding
1.  **Horizontal Projection:** Project the 3D point cloud into a 2D depth map from a horizontal or profile perspective (facing the cliff), rather than a top-down view. This prevents vertical cliffs and overhangs from collapsing into a single pixel.
2.  **2-Channel Gradient Calculation:** Instead of standard RGB images or single-value slope calculations, compute the elevation/depth gradient using a central difference method. Create a 2-channel image representing the **X-direction gradient** and the **Y-direction gradient**. This preserves the strict directional steepness required to identify climbable faces.
3.  **Normalization:** Apply **Gaussian normalization** to the gradient images. Do not use Instance Normalization for gradients, as it requires varying scaling factors that disrupt consistency across patches.

### Model Architecture & Training
*   **Architecture:** Use a modified U-Net architecture (commonly implemented in PyTorch) configured for binary image segmentation. 
*   **Loss Function Tuning:** Address the severe class imbalance (climbable rock is a tiny percentage of the total area) by applying heavily weighted cross-entropy loss. For example, assign a weight of $1.0$ to the target (route) pixels and $0.05$ to the background pixels. This forces the model to actually learn the route features rather than lazily predicting "background" for a low loss score.

---

## Part 2: Experimental Framework (2D vs. 3D Evaluation)

Training this 2D CNN alongside your heavier 3D point cloud model (e.g., PointNet++, VoxelNet, or MinkowskiEngine) provides a perfect baseline. It allows you to scientifically justify whether the massive increase in compute power required for 3D processing is actually yielding better results.

Here are the specific experiments you can run to compare them:

### Experiment 1: The Occlusion & Overhang Test (Recall Comparison)
*   **Hypothesis:** The 3D model will have significantly higher recall on overhanging routes and complex cave systems, which might still suffer from occlusion in a 2D projection.
*   **Method:** Isolate a validation set of crags known for heavy overhangs, caves, or arêtes. Compare the intersection-over-union (IoU) and recall of the 2D model vs. the 3D model. If the 2D model performs equally well, your virtual camera projection is sufficient, and the 3D model's complexity is unnecessary.

### Experiment 2: GPS Noise Robustness
*   **Hypothesis:** The 3D model will be more robust to inaccurate GPS coordinates.
*   **Method:** Intentionally inject spatial noise (e.g., shift the pin by 10m, 50m, 100m) into the crag GPS coordinates during inference. 
*   **Why it matters:** The 2D model relies on the rough GPS pin to orient the "virtual camera". If the pin is wrong, the camera angle is wrong, and the 2D projection might miss the wall entirely. A 3D model processing a larger voxel grid or point cloud radius might find the route regardless of minor GPS drift.

### Experiment 3: Cost-Benefit Analysis (Compute vs. Accuracy)
*   **Hypothesis:** The 3D model will achieve slightly higher accuracy, but at an exponentially higher computational cost.
*   **Method:** Track the training time, memory consumption (VRAM), and inference time per crag for both models. Plot this against the accuracy/IoU metrics. 
*   **Decision Metric:** If the 3D model achieves a 3% bump in accuracy but takes 15x longer to process a point cloud, you might decide the 2D U-Net is the more viable solution for scaling your application to process mountain ranges at scale.

### Experiment 4: Crag-Level Aggregate Distributions
*   **Method:** Using the macro-level experiments mentioned previously (Route Count Regression and Earth Mover's Distance on Heights), compare the aggregate statistics predicted by the 2D model versus the 3D model against the Mountain Project ground truth. 
*   **Analysis:** Does processing the raw 3D geometry allow the model to better estimate the vertical span (height) of a route compared to the 2D gradient map?