You are spot on. Your intuition exactly mirrors a core machine learning concept called **Hard Negative Mining**. If you only show the model Kansas, it just learns that "flat equals bad" and might classify a steep dirt hill or a city building as a 5.11b sport route.

To build a robust decision boundary, your combination approach is exactly how you should structure the dataset:

**1. The "Easy" Macro Negatives (The Baseline)**
You absolutely need terrain that looks fundamentally different, but you can skip the perfectly flat plains. Instead, sample steep, heavily forested hillsides or rolling dunes. This teaches the network to differentiate between a 60-degree dirt slope with a tree canopy and a true geological formation.

**2. The Local Perpendicular Negatives (Contextual Awareness)**
Your tangent idea is brilliant. Think about the topology of classic limestone bands like Cactus Cliff—you have a sheer vertical face, a sudden flat plateau on top, and a rocky scree field directly behind where the belayer stands. By calculating the surface normal of the climbing face and stepping back 10 to 20 meters perpendicularly, you capture these adjacent non-climbable zones. This teaches the model the spatial *context* of a crag, not just the wall itself.

**3. The "Hard" Negatives (The Borderline Choss)**
This is where the model earns its keep. You need geological features that are mathematically vertical but practically unclimbable—like crumbling shale cliffs, loose dirt embankments, or highway road cuts. You can source these by finding steep terrain intersecting with specific geological survey maps known for poor rock quality, or by sampling sheer cliffs alongside rivers that are known to be unstable dirt rather than solid rock.

**4. Avoiding the "False Negative" Trap**
You correctly identified the danger of assuming anything just outside a known route cluster is bad rock. If you do that, the model learns to penalize unclimbed, high-quality rock—defeating the entire purpose of the tool.

To solve this, you use an **ignore label** during training. If you find a steep, solid-looking rock face near a known crag that simply lacks GPS route data, you don't label it 0 (unclimbable) and you don't label it 1 (climbable). You assign it a special value (like -1) that tells PyTorch to completely ignore those specific points when calculating the loss function. By masking out the unknowns, the model is only penalized when it misidentifies your explicit, ground-truth positive and negative examples.