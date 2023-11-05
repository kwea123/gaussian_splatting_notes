# 🟢 Gaussian Splatting Notes (WIP)
The text version of my explanatory stream (Chinese with English CC) on gaussian splatting https://youtube.com/live/1buFrKUaqwM

# 📖 Table of contents

- [Introduction](#-introduction)
- [Foward pass](#%EF%B8%8F-forward-pass)
  - placeholder
- Backward pass
  - placeholder

# 📑 Introduction
This guide aims at deciphering the formulae in the rasterization process (*forward* and *backward*). **It is only focused on these two parts**, and I want to provide as many details as possible since here lies the core of the algorithm. I will paste related code from the [original repo](https://github.com/graphdeco-inria/gaussian-splatting) to help you identify where to look at.

Before continuing, please read the [original paper](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/3d_gaussian_splatting_high.pdf) of how the gaussian splatting algorithm works in a big picture. Also note that the full algorithm has other important parts such as point densification and pruning which *won't* be covered in this article since I think those parts are relatively easier to understand.

# ➡️ Forward pass
The forward pass consists of two parts:
1.  Compute the attributes of each gaussian
2.  Compute the color of each pixel

## 1. Compute the attributes of each gaussian

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

### 1-1. Compute derived attributes (radius, uv, cov2D)

First, from `scaling` and `rotation`, we can compute *3D covariance* from the formula

$\Sigma = RSS^TR^T \quad \text{Eq. 6}$ where
```cuda
// https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L134-L138
glm::mat3 R = glm::mat3(
  1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
  2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
  2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
);
```
and
```cuda
// https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L121-L124
glm::mat3 S = glm::mat3(1.0f); // S is a diagonal matrix
S[0][0] = mod * scale.x;
S[1][1] = mod * scale.y;
S[2][2] = mod * scale.z;
```
Note that `S` is multiplied with a scale factor `mod` that is kept as `1.0` during training.

In inference, this value (`scaling_modifier`) and be modified on
```python3
# https://github.com/graphdeco-inria/gaussian-splatting/blob/main/gaussian_renderer/__init__.py#L18
def render(..., scaling_modifier = 1.0, ...):
```
to control the scale of the gaussians. In their demo they showed how it looks by setting this number to something <1 (shrinking the size). Theoretically this value can also be set >1 to increase the size.

------------------------
⚠️ quote from the paper ⚠️
> An obvious approach would be to directly optimize the covariance matrix Σ to obtain 3D Gaussians that represent the radiance field. However, covariance matrices have physical meaning only when they are positive semi-definite. For our optimization of all our pa- rameters, we use gradient descent that cannot be easily constrained to produce such valid matrices, and update steps and gradients can very easily create invalid covariance matrices.

The design of optimizing the 3D covariance by decomposing it to `R` and `S` separately is not a random choice. It is a trick we call "reparametrization". By making it expressed as $RSS^TR^T$, it is guaranteed to be **always** positive semi-definite (matrix of the form $A^TA$ is always positive semi-definite).

------------------------

Next, we need to get 3 things: `radius`, `uv` and `cov2D` (equivalently its inverse `conic`) which are the 2D attributes of a gaussian projected on an image.

We can get the `cov2D` by $\Sigma' = JW\Sigma W^TJ^T \quad \text{Eq. 5}$
```cuda
// https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L99-L106
glm::mat3 T = W * J;
glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);
glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;
```


### 1-2. Compute which tile each gaussian covers

## 2. Compute the color of each pixel
