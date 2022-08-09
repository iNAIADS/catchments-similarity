import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.colors as mcolors

COLORS = list(mcolors.TABLEAU_COLORS.keys())

def add_dummy_points(points):
    print(points.shape)
    x_min_point = np.min(points[:,0])
    y_min_point = np.min(points[:,1])
    x_max_point = np.max(points[:,0])
    y_max_point = np.max(points[:,1])

    # add 4 distant dummy points
    points_with_dummies = np.append(points,
        [[x_min_point-1,y_min_point-1],
        [x_max_point+1,y_min_point-1],
        [x_max_point+1,y_max_point+1],
        [x_min_point-1,y_max_point+1]],
        axis = 0)
    return points_with_dummies

def compute_voronoi_tesselation(points):

    vor = Voronoi(points)
    vor_regions = vor.regions
    vor_vertices = vor.vertices
    complete_regions = [vor_regions[i] for i in vor.point_region[:-4]]
    complete_polygons = [[vor.vertices[j] for j in complete_regions[i]] for i in range(len(complete_regions))]
    
    return complete_polygons

def make_plot(points, polygons, bbox):

    fig, ax = plt.subplots()
    # colorize
    for i in range(len(points)):
        point = points[i,:]
        ax.fill(*zip(*polygons[i]), color=COLORS[i%(len(COLORS))])
        ax.scatter(point[0], point[1], color=COLORS[i%(len(COLORS))], edgecolors='k', zorder=7)

    # fix the range of axes
    plt.xlim([bbox[0][0],bbox[1][0]]), plt.ylim([bbox[0][1],bbox[2][1]])

    plt.savefig("voronoi")
    plt.close()

def wrapper(points, do_plot=False):
    points_with_dummies = add_dummy_points(points)   
    bbox = points_with_dummies[-4:,:] 
    complete_polygons = compute_voronoi_tesselation(points_with_dummies)
    if do_plot:
        make_plot(points, complete_polygons, bbox)

    return complete_polygons, bbox

if __name__ == '__main__':

    points = np.random.rand(15,2)

    wrapper(points, do_plot=True)