<!-- Page limit: under 2 pages when pasted into the MVA2027 template (two-column), references included. -->

# Do Raw 3D Points Help? Finding Climbable Rock in Airborne Laser Scans

**Yixian Li, Zach Freund** — Johns Hopkins University, EN.705.643

## Abstract

Rock climbers still find new cliffs by word of mouth and by walking, even though most of the United States has been laser-scanned from aircraft and the scans are free. We ask whether a neural network can pick out climbable cliffs in those scans, and in particular whether it needs the raw 3D points or whether a simple top-down height map is enough. Our labels come from a public website where climbers record existing climbs, which gives rough locations only. Using several climbing areas in Arkansas, we will compare a 3D point network against a 2D image network on the same locations and test both on an area neither has seen; a proof of concept on one area (18.7 million scanned points, 530 recorded climbs) confirmed that the data is accessible and the pipeline runs. We expect the two to perform similarly overall, with the 3D model ahead only on the steepest cliffs that the scanner captured well.

## 1. Introduction and motivation

A climbing area starts when someone notices a cliff, walks to it and decides the rock is good. Websites list the climbs that came out of that process, but nothing lists the cliffs nobody has looked at. Land managers have the opposite need: cliffs are nesting sites for birds of prey and home to rare plants, and it is hard to protect cliffs that were never mapped.

The data to do better exists. LiDAR (light detection and ranging) measures distance with a laser: a scanner fires a pulse, times the reflection, and works out the 3D position where the pulse landed. Flown on an aircraft and aimed at the ground, it produces a *point cloud*: millions of 3D points tracing the terrain. A US government programme has scanned most of the country this way. The usual way to use it is to flatten the points into a *digital elevation model* (DEM), a top-down image whose pixel values are ground height. A DEM is small and easy to feed to an image network, but it stores one height per pixel, so a vertical cliff becomes a single sharp edge and an overhang cannot be represented at all. Raw points keep that shape, at the cost of far more data and a less familiar model.

Whether that cost is worth paying is unclear, because a plane looks straight down and sees little of a vertical face. In our proof-of-concept data the scanner typically recorded rock at fewer than half of the heights along a climb, and for only 8.5% of climbs did it record rock along at least 80% of the height.

We also have to make our own labels. No dataset marks climbable rock in a scan, and labelling one by hand would mean sending a climber to every cliff. The closest substitute is Mountain Project, a website where climbers record existing climbs, and it is a rough guide at best. It gives one coordinate for a whole section of cliff, often 10–30 m from the rock, so that 142 climbs can share a single point; we have to find the cliff in the scan ourselves and place examples along it. It also records only where people do climb. A cliff with no entry may be poor rock or simply unvisited, so we cannot call it a negative. The negatives we can trust, such as the flat ground above a cliff and the rubble below it, look nothing like a cliff, which makes the task easier than it should be.

Our contributions are:

- A direct comparison of a 3D point network and a 2D height-map network for finding climbable rock, on identical locations and splits.
- Two tests that are hard to pass by accident: a test set of cliffs only, so a model cannot score well just by telling cliffs from flat ground, and a whole climbing area held out from training.
- A labelling pipeline that turns rough, per-cliff locations from a climbing website into training examples.

MVA suits this work because it is an application study: the question is which representation is worth deploying, under the labels and sensor that are really available.

## 2. Related work

**Networks for point clouds.** PointNet++ [1] is the deep neural network we use for 3D input. It takes an unordered set of points, groups nearby points into small neighbourhoods, summarises each one, and repeats this at larger scales, much as an image network builds up from small patches. It is usually pretrained on ModelNet40 [2], a collection of 3D models of everyday objects such as chairs and lamps. Li et al. [3] ran PointNet++ and two other 3D networks on airborne scans to label each point as building, vegetation and so on. All three beat a classical random-forest classifier, and PointNet++ was not the strongest. Varney et al. [4] released DALES, forty airborne scenes with half a billion points labelled by hand into eight such classes. Both rely on dense manual labels for common object classes. We have only rough locations, and a class no benchmark contains.

**Rock and cliffs.** Azhari et al. [5] detect cracks in a cliff with a point network, and Zoumpekas et al. [6] detect rockfall. Both use scanners on the ground facing the cliff, which see the rock directly. From the air, Jones [7] mapped 428 km of cliff in Tennessee using fixed slope thresholds on a DEM. We ask whether a learned model can go beyond a slope threshold using airborne data.

**Finding rare terrain features.** Albrecht et al. [8] find ancient ruins in airborne scans from few, noisy labels, and Rafique et al. [9] find sinkholes, reporting that the gradient of the DEM works far better than aerial photos. Both flatten the scan to 2D first. That is reasonable for ruins and sinkholes, which are gentle shapes; cliffs are the case where flattening should hurt most, which is why we test it. Stoker and Miller [10] document the accuracy of the scans we use.

## 3. Data and models

We will gather data for several climbing areas in Arkansas. So far we have run a proof of concept on one of them, Horseshoe Canyon Ranch near Jasper; the other areas have not been chosen or downloaded yet.

- **USGS 3DEP scans.** Free airborne LiDAR scans from the US Geological Survey's national programme. In the proof of concept: the 2017 survey of the area, 18.66 M points over 1.5 × 1.7 km (7.3 per m²).
- **Mountain Project.** A website where climbers record climbs, giving one coordinate per cliff section. In the proof of concept: 530 climbs at 33 coordinates.
- **Training examples.** Locations we label climbable or not, each with the scan around it. In the proof of concept: 354 climbable and 1,093 not, from 30 cliff sections.

Examples are split by cliff section, never at random, because neighbouring examples overlap. In the proof of concept that gave 18 sections for training (845 examples), 6 for validation (291) and 6 for testing (311). One whole area will be kept out of training and used only for the final test. The height maps for the 2D model are computed from the same scans, not downloaded separately.

**Pretrained models.** For 3D, a PointNet++ checkpoint trained on ModelNet40 (from the open-source yanx27 implementation, MIT licence). For 2D, a ResNet-18 trained on ImageNet.

**Proof of concept.** We ran the proof of concept to assess feasibility. It confirmed that the data is accessible: the scan, the climb list and the PointNet++ checkpoint were all downloaded and loaded in our notebook. The model also appeared to run as expected. The checkpoint loads cleanly (77 weight tensors; only the new two-class output layer starts from scratch), and a two-epoch trial trains and evaluates end to end, reaching 0.67 balanced accuracy on test cliffs. That number only shows the code runs.

## 4. Method and training plan

**Task.** Given a map location, predict whether the rock there is climbable. Every model sees the same 10 m-wide area around the same locations.

**Labels.** For each Mountain Project coordinate we search the scan within 30 m for steep rock and place one positive example per recorded climb along it. Negatives are taken 12 m and 20 m in front of and behind the cliff, and at random spots at least 40 m from any climb. Steep rock with no recorded climb is left out of training. In the proof of concept we could place 429 of the 530 climbs this way, and dropped 75 of those because the scanner saw too little rock there. Three cliff sections were excluded because the website's coordinate was a placeholder.

**3D model.** PointNet++ with its output layer replaced by a two-class one. The input is every scanned point within 5 m of the location, sampled to 1,024 points.

**2D model.** We build a DEM of the same area at 0.5 m per pixel and compute its gradient in x and y, following [9]. A ResNet-18 takes the two gradient images as input.

**In-between model.** The same ResNet-18 on a three-channel image holding the lowest height, highest height and point count per pixel. This keeps a little vertical information at 2D cost and shows how much of any gap it closes.

**Evaluation.** We report balanced accuracy and AUROC on the test cliffs. Because our "not climbable" examples are mostly flat ground or rubble, a model could do well by detecting cliffs alone. Our main result therefore uses a second test set made only of cliffs: those near recorded climbs against steep rock far from any. Slope cannot separate these, so a difference there reflects something more. We also break results down by how much of each cliff the scanner saw, to test our expectation that 3D helps only where the face is visible.

Test cliffs from an area the model trained on share its rock type, vegetation and scan settings. The final test is therefore the held-out area. We fix every model, setting and decision threshold using the other areas, then run the models once on the held-out area and report the same metrics there.

**Further experiments.** For PointNet++: random initialisation, pretrained with only the final layers trained, and fully fine-tuned. For both models: a larger input area (10 m against 30 m), and weighting each cliff section equally in the loss so the largest one does not dominate.

**Training.** Class-weighted cross-entropy, AdamW optimiser (learning rate 1e-3, weight decay 1e-4), cosine schedule over 60 epochs with early stopping, batch size 32. Inputs are randomly rotated about the vertical axis, rescaled by up to 10% and jittered. Training is cheap: about 28 s per epoch on a laptop in our trial run, so under half an hour for a full run. Every result is repeated over three random splits, since the number of test cliff sections is small.

| Week | Milestone |
|---|---|
| 1 | Extend the pipeline to several areas; download and label the Arkansas areas |
| 2 | Full PointNet++ runs, three initialisations, three splits |
| 3 | DEM pipeline; 2D and in-between models |
| 4 | Cliff-only test set; input-area and loss-weighting experiments; freeze all models |
| 5 | Single final run on the held-out area; breakdown by scanner coverage; error analysis |
| 6 | Paper, code and model release |

## 5. Collaboration and resources

Code: https://github.com/euskon/705-643-final-project. Models and training examples: Hugging Face (link to be added). Zach Freund owns the scan and labelling pipeline; Yixian Li owns the 2D models and evaluation. Each of us works on a personal branch and merges by pull request, and every run saves its metrics to a file that we compare in a shared notebook.

## References

[1] C. R. Qi, L. Yi, H. Su, L. J. Guibas. PointNet++: Deep hierarchical feature learning on point sets in a metric space. NeurIPS, 2017.
[2] Z. Wu et al. 3D ShapeNets: A deep representation for volumetric shapes. CVPR, 2015.
[3] N. Li, O. Kähler, N. Pfeifer. A comparison of deep learning methods for airborne lidar point clouds classification. IEEE JSTARS, 14:6467–6486, 2021.
[4] N. Varney, V. K. Asari, Q. Graehling. DALES: A large-scale aerial LiDAR data set for semantic segmentation. CVPR Workshops, 2020.
[5] F. Azhari, C. Sennersten, M. Milford, T. Peynot. PointCrack3D: Crack detection in unstructured environments using a 3D-point-cloud-based deep neural network. arXiv:2111.11615, 2021.
[6] T. Zoumpekas et al. End-to-end intelligent framework for rockfall detection. arXiv:2102.06491, 2021.
[7] K. Jones. An inventory and mapping of cliffs within the South Cumberland Plateau region of Tennessee. M.S. thesis, Univ. of Tennessee at Chattanooga, 2018.
[8] C. M. Albrecht et al. Learning and recognizing archeological features from LiDAR data. IEEE Big Data, 2019.
[9] M. U. Rafique, J. Zhu, N. Jacobs. Automatic segmentation of sinkholes using a convolutional neural network. Earth and Space Science, 2022.
[10] J. M. Stoker, B. Miller. The accuracy and consistency of 3D Elevation Program data: A systematic analysis. Remote Sensing, 14(4):940, 2022.
