import numpy as np
from matplotlib import pyplot as plt
from utils import load_cameras, load_point_cloud, project, create_silhouettes, create_gif

INSIDE, OUTSIDE, OVERLAP, NOT_SET = 0, 1, 2, 3


class Projection_Labler:
    """
    Class to label a cube as inside, outside or overlapping silhuettes
    of an object.
    """

    def __init__(self, cameras):
        self.N_CAMS, self.RMAX, self.CMAX = (37, 576, 720)
        
        self.cameras = cameras
        self.silhouettes = np.empty((self.N_CAMS, self.RMAX, self.CMAX), dtype=np.uint8)

        with open("data/silhouettes.npy", "rb") as f:
            self.silhouettes = np.load(f)
            print(f"Silhouettes loaded")

        self.silhouettes[self.silhouettes == 254] = 0
        self.silhouettes[self.silhouettes == 255] = 1


    def label(self, points):

        assert(points.shape[0] == (3))

        label = INSIDE

        for j in range(36):
            # Project corners onto the image plane of the silhuette
            projected = np.round(project(points, self.cameras[j])).astype(int)
            # Create a filled area from the resulting square
            area = self.fill_area(projected)
            # Count the overlap of the area with the silhuette
            inside = np.count_nonzero(self.silhouettes[j][area] > 0)

            count_area = np.count_nonzero(area)

            # Adjust label accordingly
            if inside == 0:
                label = OUTSIDE
                break
            elif inside != count_area:
                label = OVERLAP

        return label

    def fill_area(self, projected_points):
        x_min = np.maximum(np.min(projected_points[0,:]), 0)
        y_min = np.maximum(np.min(projected_points[1,:]), 0)
        x_max = np.minimum(np.max(projected_points[0,:]), self.CMAX)
        y_max = np.minimum(np.max(projected_points[1,:]), self.RMAX)

        mask = np.zeros((self.RMAX, self.CMAX)).astype(int)
        mask[y_min:y_max,x_min:x_max] = 1

        return mask > 0


class Octree:
    def __init__(self, point, size, labler, depth = 0, label=NOT_SET):
        self.children = []
        self.depth = depth
        self.label = label
        self.labler = labler
        self.point = point
        self.size = size

    def subdivide(self):

        if self.label == INSIDE:
            return

        x, y, z = self.point

        # Split space into 8 octants
        new_size = self.size / 2

        p_new = np.empty((8,3))

        p_new[0,:] = np.array([x,y,z])
        p_new[1,:] = np.array([x+new_size,y,z])
        p_new[2,:] = np.array([x,y+new_size,z])
        p_new[3,:] = np.array([x,y,z+new_size])
        p_new[4,:] = np.array([x+new_size,y+new_size,z])
        p_new[5,:] = np.array([x+new_size,y,z+new_size])
        p_new[6,:] = np.array([x,y+new_size,z+new_size])
        p_new[7,:] = np.array([x+new_size,y+new_size,z+new_size])


        for i in range(8):
            p = p_new[i]
            label = self.label

            xi, yi, zi = p
            # Create corners for the current point
            corners = np.empty((8,3))
            corners[0,:] = np.array([xi,yi,zi])
            corners[1,:] = np.array([xi+new_size,yi,zi])
            corners[2,:] = np.array([xi,yi+new_size,zi])
            corners[3,:] = np.array([xi,yi,zi+new_size])
            corners[4,:] = np.array([xi+new_size,yi+new_size,zi])
            corners[5,:] = np.array([xi+new_size,yi,zi+new_size])
            corners[6,:] = np.array([xi,yi+new_size,zi+new_size])
            corners[7,:] = np.array([xi+new_size,yi+new_size,zi+new_size])
            label = self.labler.label(corners.T)

            # Dont create voxels outside the silhouettes
            if label == OUTSIDE:
                continue

            vox = Octree(p, new_size, self.labler, depth=self.depth + 1, label=label)
            self.children.append(vox)


def subdivide_voxel(root, depth):

    if root.depth == depth:
        return
    else:
        root.subdivide()
        for vox in root.children:
            subdivide_voxel(vox, depth)

def get_leaf_points(root, depth):
    """
    Find "leaf" points of an octree at a given depth
    """

    if depth != root.depth and len(root.children) != 0:
        points = np.empty(3)
        for child in root.children:
            points = np.vstack((points, get_leaf_points(child, depth)))

        points = points[1:, :]
        return points
    else:
        return root.point

def create_octree(point_cloud, cameras) -> Octree:
    
    x_min = np.min(point_cloud[:,0])
    y_min = np.min(point_cloud[:,1])
    z_min = np.min(point_cloud[:,2])
    x_max = np.max(point_cloud[:,0])
    y_max = np.max(point_cloud[:,1])
    z_max = np.max(point_cloud[:,2])

    start_point = np.floor(np.array([x_min, y_min, z_min])*100)/100
    size = np.ceil(np.max([x_max-x_min, y_max-y_min, z_max-z_min])*100)/100

    labler = Projection_Labler(cameras)

    # Initialize Octree
    root = Octree(start_point, size, labler)

    return root


def visualize_octree(root, depth):
    """
    Visualize 3D points with space carving. Create Octree representation subdivided to a choosen depth

    ### Parameters:
    root: Octree root
    depth: subdivide depth, level of detail

    """
    
    points = get_leaf_points(root, depth).reshape((-1,3))

    # Normalize cloud points between 0 - 1
    points += -np.min(points, axis=0)
    points /= np.max(points) if not np.max(points) == 0 else 1.0

    # Numb voxels in each dimensio
    n_voxels = np.power(2,depth)

    # Scale to match number of voxels in each dim
    points *= n_voxels - 1
    points = np.round(points).astype(int)

    # Prepare ind
    vox = np.zeros((n_voxels, n_voxels, n_voxels))
    
    # Fill voxels from the base points
    vox[points[:,0], points[:,1], points[:,2]] = 1
    
    _, (ax) = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'auto'})
    ax.set_title(f"Octree generated voxels, depth: {depth}")
    ax.voxels(vox)
    ax.view_init(elev=10, azim=-45)
    plt.savefig(fname=f"1_{depth}", dpi=200)
    ax.view_init(elev=10, azim=128)
    plt.savefig(f"2_{depth}", dpi=200)
    ax.view_init(elev=10, azim=150)
    plt.savefig(f"3_{depth}", dpi=200)



def visualize_pointcloud(point_cloud):
    """
    Visualize 3D points.

    ### Parameters:
    point_cloud: 3D points cloud shape (numb_points, 3)

    """

    point_cloud += -np.min(point_cloud, axis=0)
    point_cloud /= np.max(point_cloud) 
    
    fig, (ax) = plt.subplots(1, 1, subplot_kw={'projection':'3d', 'aspect':'auto'})
    ax.set_title("Point cloud")
    ax.scatter(point_cloud[:, 0], point_cloud[:, 1], point_cloud[:, 2], s=1, c='r', zorder=10)
    ax.view_init(elev=10, azim=-45)
    plt.savefig(fname=f"PC_1", dpi=200)
    ax.view_init(elev=10, azim=128)
    plt.savefig(fname=f"PC_2", dpi=200)
    ax.view_init(elev=10, azim=150)
    plt.savefig(fname=f"PC_3", dpi=200)
    #plt.show()



if __name__ == "__main__":

    # create_silhouettes()

    # # Load point cloud
    point_cloud = load_point_cloud("data/point_cloud.npy")

    # visualize_pointcloud(point_cloud)

    # # Load camera matrices
    cameras = load_cameras("data/cameras.csv")

    octree = create_octree(point_cloud, cameras)

    subdivide_voxel(octree, 1)
    visualize_octree(octree, depth=0)
    visualize_octree(octree, depth=2)
    # visualize_octree(octree, depth=3)
    # visualize_octree(octree, depth=4)
    # visualize_octree(octree, depth=5)
    # visualize_octree(octree, depth=8)
    # visualize_octree(octree, depth=7)
    
    # visualize_octree(octree, depth=8)
