import cv2 as cv
import imageio
import numpy as np
import os

def create_silhouettes():

    silhouettes = np.empty((37, 576, 720), dtype=np.uint8)
    for i in range(37):
        k = i
        if k < 10:
            k = f"0{k}"
        image = cv.imread(f'data/images/viff.0{k}.ppm')
        silhouette = get_dino_silhouettes(image)
        silhouettes[i,...] = silhouette

    with open("silhouettes.npy", "wb") as f:
        np.save(f, silhouettes)

def get_dino_silhouettes(image):

    b,g,r = cv.split(image)
    
    # apply binary thresholding
    _, thresh1 = cv.threshold(b, 112, 1, cv.THRESH_BINARY_INV)
    _, thresh2 = cv.threshold(g, 200, 1, cv.THRESH_BINARY)
    _, thresh3 = cv.threshold(r, 215, 1, cv.THRESH_BINARY)

    # Combine channels
    silhouette = thresh1 + thresh2 + thresh3
    mask = silhouette > 0
    silhouette[mask] = 1
    # Remove some garbage
    silhouette[:, 680:] = 0
    silhouette[:3, :] = 0

    # Close and floodfill
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4,4))
    silhouette = cv.morphologyEx(silhouette, cv.MORPH_CLOSE, kernel) 
    silhouette = cv.morphologyEx(silhouette, cv.MORPH_CLOSE, kernel) 

    mask = np.zeros((576+2, 720+2), np.uint8)
    cv.floodFill(silhouette, mask, seedPoint=(0,0), newVal=1)
    silhouette = np.bitwise_not(mask[1:-1,1:-1])

    return silhouette

def create_gif(dirname: str, outname, fps):
    images = []
    for file_name in sorted(os.listdir(dirname)):
        if file_name.endswith('.tif'):
            file_path = os.path.join(dirname, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave(f'./animation/{outname}.gif', images, fps=fps)


def load_point_cloud(fname: str):
    pc = np.empty((0))
    with open(fname, "rb") as f:
        pc = np.load(f)

    return pc

def load_cameras(fname: str):

    _N_IMAGES = 36
    GT_C = []

    with open(fname) as f:
        f.readline()
        for _ in range(_N_IMAGES):
            vals = list(map(float, f.readline().split(',')[1:]))
            C = np.array(vals).reshape((3, 4))
            GT_C.append(C)

    return GT_C


"""
======================================
"""


"""
The following utility functions are from for Computer Exercise 3: Optimization in TSBB15 Computer Vision

The functions in this module mostly map directly to their
equivalent in the MATLAB toolbox for CE3.


Written by Hannes Ovrén, 2016
Minor updates by Johan Edstedt, 2020
"""

def homog(x):
    """Homogenous representation of a N-D point
    
    Parameters
    ----------------
    x : (N, 1) or (N, ) array
        The N-dimensional point
    
    Returns
    ----------------
    xh : (N+1, 1) or (N+1, ) array
        The point x with an extra row with a '1' added
    """
    is2d = (x.ndim == 2)
    if not is2d:
        x = x.reshape(-1,1)
    d, n = x.shape
    X = np.empty((d+1, n))
    X[:-1, :] = x
    X[-1, :] = 1
    return X if is2d else X.ravel()


def project(x, C):
    """Project 3D point
    
    Parameters
    --------------
    x : (3,1) or (3,) array
        A 3D point in world coordinates
    C : (3, 4) matrix
        Camera projection matrix
    
    Returns
    -------------------
    y : (2, 1) array
        Projected image point
    """
    if not C.shape == (3,4):
        raise ValueError('C is not a valid camera matrix')
    X = homog(x)
    y = np.dot(C, X)
    y /= y[2]
    return y[:2]


def cross_matrix(v):
    """Compute cross product matrix for a 3D vector
    
    Parameters
    --------------
    v : (3,) array
        The input vector
        
    returns
    --------------
    V_x : (3,3) array
        The cross product matrix of v such that V_x b == v x b
    """
    v = v.ravel()
    if not v.size == 3:
        raise ValueError('Can only handle 3D vectors')
    
    return np.array([[0, -v[2], v[1]],
                     [v[2],  0, -v[0]],
                     [-v[1], v[0], 0]])
    

def fmatrix_from_cameras(C1, C2):
    """Fundamental matrix from camera pair
    
    Parameters
    ------------------
    C1 : (3, 4) array
        Camera 1
    C2 : (3, 4) array
        Camera 2
        
    Returns
    ---------------------
    F : (3,3) array
        Fundamental matrix corresponding to C1 and C2
    """
    U, s, V = np.linalg.svd(C2) # Note: C2 = U S V  (V already transposed)
    n = V[3, :]
    e = np.dot(C1, n)
    C2pinv = np.linalg.pinv(C2)
    F = np.dot(cross_matrix(e), np.dot(C1, C2pinv))
    return F


def triangulate_optimal(C1, C2, x1, x2):
    """Optimal trinagulation of 3D point
    
    Parameters
    ------------------
    C1 : (3, 4) array
        First camera
    C2 : (3, 4) array
        Second camera
    x1 : (2,) array
        Image coordinates in first camera
    x2 : (2,) array
        Image coordinates in second camera
    
    Returns
    ------------------
    X : (3, 1) array
        The triangulated 3D point
    """
    move_orig = lambda x: np.array([[1., 0., x[0]],[0., 1., x[1]], [0., 0., 1.]])
    T1 = move_orig(x1)
    T2 = move_orig(x2)
    
    # Find and transform F
    F = fmatrix_from_cameras(C1, C2)
    F = np.dot(T1.T, np.dot(F, T2))
    
    # Extract epipoles
    # Normalize to construct rotation matrix
    e1, e2 = fmatrix_epipoles(F)
    e1 /= np.linalg.norm(e1)
    e2 /= np.linalg.norm(e2)
    
    R_from_epipole = lambda e: np.array([[e[0], e[1], 0],[-e[1], e[0], 0],[0,0,1]])
    R1 = R_from_epipole(e1)
    R2 = R_from_epipole(e2)
    
    F = np.dot(R1, np.dot(F, R2.T))
    
    # Build polynomial, with code from Klas Nordberg
    f1 = f2 = 1 # Note: Matlab implementation assumed e1, e2 homogeneous, and fk=e[-1]
    a = F[1,1]
    b = F[1,2]
    c = F[2,1]
    d = F[2,2]
    k1 = b * c - a * d
    g = [a * c * k1 * f2**4, #coefficient for t^6
         (a**2 + c**2 * f1**2)**2 + k1 * (b * c + a * d) * f2**4, #coefficient for t^5
         4 * (a**2 + c**2 * f1**2) * (a * b + c * d * f1**2) + \
         2 * a * c * k1 * f2**2 + b * d * k1 * f2**4,         
         2 * (4 * a * b * c * d * f1**2 + a**2 * (3 * b**2 + d**2 * (f1-f2) * (f1 + f2)) + \
         c**2 * (3 * d**2 * f1**4 + b**2 * (f1**2 + f2**2))),
         -a**2 * c * d + a * b *(4 * b**2 + c**2 + 4*d**2 * f1**2 - 2 * d**2 * f2**2) + \
         2 * c * d * (2 * d**2 * f1**4 + b**2 * (2 * f1**2 + f2**2)),
         b**4 - a**2 * d**2 + d**4 * f1**4 + b**2 * (c**2 + 2 * d**2 * f1**2),
         b * d * k1]
    
    # Find roots of the polynomial
    r = np.real(np.roots(g))
    
    # Check each point
    s = [t**2 / (1 + f2**2 * t**2) + (c * t + d)**2 / ((a * t + b)**2 + \
         f1**2 * (c * t + d)**2) for t in r]
         
    # Add value at asymptotic point
    s.append(1. / f2**2 + c**2 / (a**2 + f1**2 * c**2))
    
    # Check two possible cases
    i_min = np.argmin(s)
    if i_min < r.size:
        # Not point at infinity
        tmin = r[i_min]
        l1 = np.array([-f1 * (c * tmin + d), a * tmin + b, c * tmin + d])
        l2 = np.array([tmin * f2, 1, -tmin])
    else:
        # Special case: tmin = tinf
        l1 = np.array([-f1 * c, a, c])
        l2 = np.array([f2, 0., -1.])
    
    # Find closest points to origin
    find_closest = lambda l: np.array([-l[0] * l[2], 
                                       -l[1] * l[2], 
                                       l[0]**2 + l[1]**2]).reshape(-1,1)
    x1new = find_closest(l1)
    x2new = find_closest(l2)
    
    # Transfer back to original coordinate system
    x1new = np.dot(T1, np.dot(R1.T, x1new))
    x2new = np.dot(T2, np.dot(R2.T, x2new))

    # Find 3D point with linear method on new coordinates
    X = triangulate_linear(C1, C2, x1new, x2new)
    
    return X


def triangulate_linear(C1, C2, x1, x2):
    """Linear trinagulation of 3D point
    
    Parameters
    ------------------
    C1 : (3, 4) array
        First camera
    C2 : (3, 4) array
        Second camera
    x1 : (2,) array
        Image coordinates in first camera
    x2 : (2,) array
        Image coordinates in second camera
    
    Returns
    ------------------
    X : (3, 1) array
        The triangulated 3D point
    """
    if x1.shape[0] == 2:
        x1 = homog(x1)
        x2 = homog(x2)
    M = np.vstack([np.dot(cross_matrix(x1), C1),
                  np.dot(cross_matrix(x2), C2)])
    U, s, V = np.linalg.svd(M)
    X = V[-1,:]
    return X[:3] / X[-1]                 
    
    
def fmatrix_epipoles(F):
    """Epipoles of a fundamental matrix
    
    Parameters
    -------------------
    F : (3,3) array
        Fundamental matrix
    
    Returns
    -------------------
    e1 : (2,1) array
        Epipole 1
    e2 : (2,1) array
        Epipole 2
    """
    U, s, V = np.linalg.svd(F)
    e1 = U[:,-1]
    e2 = V[-1,:]
    
    e1 /= e1[-1]
    e2 /= e2[-1]
    
    return e1[:2], e2[:2]