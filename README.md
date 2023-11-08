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

If you see sections starting with 💡, it's something I think important to understand.

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

To project the gaussian onto a 2D image, we must go through some more computations to transform the attributes to 2D:

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
💡 quote from the paper 💡
> An obvious approach would be to directly optimize the covariance matrix Σ to obtain 3D Gaussians that represent the radiance field. However, covariance matrices have physical meaning only when they are positive semi-definite. For our optimization of all our pa- rameters, we use gradient descent that cannot be easily constrained to produce such valid matrices, and update steps and gradients can very easily create invalid covariance matrices.

The design of optimizing the 3D covariance by decomposing it to `R` and `S` separately is not a random choice. It is a trick we call "reparametrization". By making it expressed as $RSS^TR^T$, it is guaranteed to be **always** positive semi-definite (matrix of the form $A^TA$ is always positive semi-definite).

------------------------

Next, we need to get 3 things: `radius`, `uv` and `cov` (2D covariance, or equivalently its inverse `conic`) which are the 2D attributes of a gaussian projected on an image.

We can get `cov` by $\Sigma' = JW\Sigma W^TJ^T \quad \text{Eq. 5}$
```cuda
// https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L99-L106
glm::mat3 T = W * J;
glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);
glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;
```

Let's put ![1](https://github.com/graphdeco-inria/gaussian-splatting/assets/11364490/2819c95a-e216-4352-8739-90c692b13c91) (remember the 2D and 3D covariance matrices are symmetric) for the calculation that we're going to do in the following.

Its inverse `conic` (honestly I don't know why they've chosen such a bad variable name, calling it `cov_inv` would've been 100x better) can be expressed as ![1](https://github.com/graphdeco-inria/gaussian-splatting/assets/11364490/6cefc42e-273b-4b30-8eab-1db944670f3e) (actually it's a very useful thing to remember: to invert a 2D matrix, you invert the diagonal, put negative signs on the off-diagonal entries and finally put a `1/det` in front of everything).
```cuda
// https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L219
float det = (cov.x * cov.z - cov.y * cov.y);
// https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L222-L223
float det_inv = 1.f / det;
float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };  // since the covariance matrix is symmetric, we only need to save the upper triangle
```

--------------------------------
💡 A small trick to ensure the numerical stability of the inverse of `cov` 💡
```cuda
// https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L110-L111
cov[0][0] += 0.3f;
cov[1][1] += 0.3f;
```
By construction, `cov` is only positive *semi-* definite (recall that it's in the form $A^TA$) which is not sufficient for this matrix to be *invertible* (which we need it to be because we need to calculate Eq. 4).

Here we add `0.3` to the diagonal to make it invertible. Why is this true? Let's put $cov = A^TA$; adding some positive value to the diagonal means adding $\lambda I$ to the matrix ($\lambda$ is the value we add, and $I$ is the identity matrix), so $cov = A^TA + \lambda I$. Now for any vector $x$, if we compute $x^T \cdot cov \cdot x$, it is equal to $x^TA^TAx + \lambda x^Tx = ||Ax||^2 + \lambda ||x||^2$ which is **strictly positive**. Why are we computing this quantity? This is actually the definition of a matrix being **positive definite** (note that we have gotten rid of the *semi-*) which means not only it's invertible, but also all of its eigenvalues are strictly positive.

--------------------------------

Having `cov` in hand, we can now proceed to compute the `radius` of a gaussian.

Theoretically, when projecting an ellipsoid onto an image, you get an *ellipse*, not a circle. However, storing the attributes of an ellipse is much more complicated: you need to store the center, the long and short axis lengths and the orientation; whereas for a circle, you only need its center and the radius. Therefore, the authors choose to approximate the projection with a circle circumscribing the ellipse (see the following figure). This is what the `radius` attribute represents.

<img width="277" alt="" src="https://github.com/lumalabs/luma-pynerf/assets/11364490/63f25c15-18cd-4be9-8e61-cc5db715c308">

How to get the `radius` from `cov`? Let's make analogy from the 1-dimensional case.

Imagine we have a 1D gaussian like the following:

![image](https://github.com/lumalabs/luma-pynerf/assets/11364490/b50d4359-dc23-4ded-8107-4c2165e55e50)

How can we define the "radius" of such a gaussian? Intuitively, it is some value $r$ that we expect that if we crop the graph from $-r$ to $r$, it still covers most of the graph. Following this intuition and our high-school math knowledge, it is not difficult to come up with the value $r = 3 \cdot \sqrt{var}$ where $var$ is the variation of this gaussian (btw, this covers 99.73% of the gaussian).

Fortunately, the analogy applies to *any* dimension, just be aware that the "radius" is different along each axis (remember there are two axes in an ellipse).

We said $r = 3 \cdot \sqrt{var}$. How to, then, get the $var$ of a 2D gaussian given its covariance matrix? It is the **two eigenvalues** of the covariance matrix. Therefore, the problem now comes down to the calculation of the two eigenvalues.

I could've given you the answer directly, but out of personal preference (I ❤️ linear-algebra), I want to detail it more. First of all, for a square matrix $A$ we say it has eigenvalue $\lambda$ with the associated eigenvector $x$ if $\lambda$ and $x$ satisfy $Ax = \lambda x, x \neq 0$. There are as many eigenvalues (and associated eigenvectors) as the dimension of $A$ if we operate in the domain of complex numbers.

In general, to calculate *all* eigenvalues of $A$, we solve the equation $det(A-λ\cdot I) = 0$ (the variable being $λ$). If we
replace with the `cov` matrix we have above, this equation can be expressed as $(a-λ)(c-λ)-b^2 = 0$ which is a quadratic equation that all of us are familiar with.

The solutions (eigenvalues) are `lambda1` and `lambda2` in the following code
```cuda
// https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L219
float det = (cov.x * cov.z - cov.y * cov.y);  // this is a*c - b*b in our expression
...
// https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L229-L231
float mid = 0.5f * (cov.x + cov.z);
float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));  // I'm not too sure what 0.1 serves here
float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
```
Then we finally get `radius` as 3 times the square root of the bigger eigenvalue:
```cuda
https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L232
float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));  // ceil() to make it at least 1 because we operate in pixel space
```

Last thing, which is probably the most obvious, is the `uv` (image coordinates) of the gaussian. It is done via a simple projection from the 3D center:
```cuda
// https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L197-L200
float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
float4 p_hom = transformPoint4x4(p_orig, projmatrix);
float p_w = 1.0f / (p_hom.w + 0.0000001f);
float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };
...
// https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu#L233
float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };  // I like to call it uv
```

Phew, we finally got the three quantities we need to know: **radius, uv and conic**. Let's move on to the next part.

### 1-2. Compute which tile each gaussian covers

## 2. Compute the color of each pixel
