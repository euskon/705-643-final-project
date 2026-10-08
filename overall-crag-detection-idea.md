Using Entwine Point Tile (EPT) data to hunt for climbing crags is a perfect machine learning use case. The data gives you raw 3D spatial geometries, and you can build a pipeline that specifically searches for the type of climbing you want—like filtering out low-angle slabs to specifically target vertical or slightly overhanging cliff faces.

Here is how you can build the pipeline from raw EPT ingestion to 3D segmentation using Python and PyTorch.

---

## 1. Data Ingestion (EPT to Tensors)

EPT is highly optimized for streaming, but PyTorch needs tensors. You can use **PDAL (Point Data Abstraction Library)** in Python to read the EPT data over the network and convert it into NumPy arrays, which are then easily cast to PyTorch tensors.

You don't need the entire landscape; you can query PDAL using a bounding box for a specific canyon or ridge, downloading only the X, Y, and Z coordinates (and sometimes return intensity) for that area.

## 2. Geometric Pre-Processing & Filtering

Before feeding data to a neural network, you want to use the raw geometry to filter out the noise. Not every rock is climbable.

* **Ground/Vegetation Removal:** Use PDAL's built-in filtering algorithms (like SMRF) to classify and strip away the ground layer, scree fields, and trees, isolating just the bare rock structures.
* **Surface Normals & Slope Angles:** This is where you target specific climbing styles. By calculating the surface normals of the point cloud, you can find the slope angle of the rock face. To find prime sport climbing walls and eliminate low-angle slabs or ledges, you can filter the dataset to only retain points with a slope gradient between roughly **85 degrees (vertical) and 110 degrees (overhanging)**.

## 3. 3D Segmentation Model (PyTorch)

Once you have the filtered point clouds of steep rock faces, you need a neural network to segment out the *clean, climbable* sections from the loose, chossy, or fractured sections.

Standard 2D CNNs don't work on unordered 3D point clouds. Instead, you'll want to use an architecture designed for this:

* **PointNet++:** This network learns hierarchical features directly from the raw point cloud. It groups local points, extracts features, and can output a per-point classification score (e.g., 1 for "solid rock face", 0 for "fractured rock").
* **Sparse Convolutions (Minkowski Engine):** For massive geographic areas, converting the point cloud into a sparse voxel grid and running 3D sparse convolutions is often much faster and highly memory-efficient in PyTorch.

## 4. Clustering into Route-Scale Crags

The neural network will output millions of individual points labeled as "good climbing." However, a single 1-meter square of good rock isn't a route.

You can run an algorithm like **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** on the positively labeled points. By setting the minimum cluster size to represent roughly 15 to 30 meters of vertical height, DBSCAN will group the points into continuous, contiguous walls—giving you the exact GPS coordinates and dimensions of a candidate crag.