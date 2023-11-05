# 🟢 gaussian_splatting_notes
The text version of my explanatory stream (Chinese with English CC) on gaussian splatting https://youtube.com/live/1buFrKUaqwM

# 📑 Introduction
This guide aims at deciphering the formulae in the rasterization process (*forward* and *backward*). **It is only focused on these two parts**, and I want to provide as many details as possible since here lies the core of the algorithm. I will paste relative code from the [original repo](https://github.com/graphdeco-inria/gaussian-splatting) to help you identify where to look at.

Before continuing, please read the [original paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf) of how the gaussian splatting algorithm works in a big picture. Also note that the full algorithm has other important parts such as point densification and pruning which *won't* be covered in this article since I think those parts are relatively easier to understand.

## Forward pass
The forward pass consists of two parts:
1.  Compute the attributes of each gaussian
2.  Compute the color of each pixel

### 1. Compute the attributes of each gaussian

Each gaussian holds the following *raw* attributes:

```python3
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py#L47-L52
self._xyz = torch.empty(0)            # world coordinate
self._features_dc = torch.empty(0)    # diffuse color
self._features_rest = torch.empty(0)  # spherical harmonic coefficients
self._scaling = torch.empty(0)        # 3d scale
self._rotation = torch.empty(0)       # rotation expressed in quaternions
self._opacity = torch.empty(0)        # opacity

# they are initialized as empty tensors then assigned with values on
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/scene/gaussian_model.py#L215
```

To project the gaussian onto a 2D image, we must go through some more computations to transform the attribues to 2D:

### 2. Compute the color of each pixel
