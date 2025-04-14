import numpy as np
import array_api_strict as xp
import parallelproj_pybind as pp

img_dim = (2, 3, 4)

# img_dim = xp.asarray([2, 3, 4], dtype=xp.int32)
voxsize = xp.asarray([4.0, 3.0, 2.0], dtype=xp.float32)

img_origin = (-0.5 * xp.asarray(img_dim, dtype=xp.float32) + 0.5) * voxsize

# Read the image from file
img = xp.reshape(xp.asarray(np.loadtxt("img.txt", dtype=np.float32)), img_dim)

# Read the ray start and coordinates from file
vstart = xp.reshape(xp.asarray(np.loadtxt("vstart.txt", dtype=np.float32)), (2, 5, 3))
vend = xp.reshape(xp.asarray(np.loadtxt("vend.txt", dtype=np.float32)), (2, 5, 3))

# Calculate the start and end coordinates in world coordinates
xstart = vstart * voxsize + img_origin
xend = vend * voxsize + img_origin

# Allocate memory for forward projection results
img_fwd = xp.zeros(xstart.shape[0], dtype=xp.float32)

# Perform forward projection
pp.joseph3d_fwd(xstart, xend, img, img_origin, voxsize, img_fwd)

# Read the expected forward values from file
expected_fwd_vals = xp.reshape(
    xp.asarray(np.loadtxt("expected_fwd_vals.txt", dtype=np.float32)), img_fwd.shape
)

# Check if we got the expected results
eps = 1e-7

assert (
    xp.max(xp.abs(img_fwd - expected_fwd_vals)) < eps
), "Forward projection test failed."

# Test the back projection
bimg = xp.zeros(img_dim, dtype=xp.float32)
ones = xp.ones(img_fwd.shape, dtype=xp.float32)
pp.joseph3d_back(xstart, xend, bimg, img_origin, voxsize, ones)

print(bimg)

ip1 = float(xp.sum(img * bimg))
ip2 = float(xp.sum(img_fwd * ones))

print(ip1, ip2)

assert abs(ip1 - ip2) / abs(ip1) < eps, "Back projection test failed."
