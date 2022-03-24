import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, PathPatch
from scipy.sparse import csr_matrix, coo_matrix
from scipy.sparse.csgraph import dijkstra
from timeit import default_timer as timer
from mpl_toolkits.mplot3d import axes3d
import mpl_toolkits.mplot3d.art3d as art3d

matplotlib.use("TkAgg")  # for plotting, had to sudo apt-get install python3-tk


# TODO: edit section

class DistanceGraph:
    cs_graph = None
    dist_matrix = None
    predecessors = None

    def __init__(self, args, field, num_vertices, obstacles):
        # field is of form [m_x, m_y, m_z, l, w, h], same format as in mujoco
        # obstacles is list with entries of form: [m_x, m_y, m_z, l, w, h], same format as in mujoco
        # Up to 10000 nodes can be handled memorywise (computation time non-problematic in this case)
        # x_spaces * y_spaces * z_spaces should not increase beyond 8000

        [m_x, m_y, m_z, l, w, h] = field
        self.args = args

        # Get min and max coordinate values, number of spaces in each direction, obstacles and z_penalty

        self.x_min = m_x - l
        self.y_min = m_y - w
        self.z_min = m_z - h

        self.x_max = m_x + l
        self.y_max = m_y + w
        self.z_max = m_z + h

        assert len(num_vertices) == 3

        self.n_x = num_vertices[0]
        self.n_y = num_vertices[1]
        self.n_z = num_vertices[2]

        self.obstacles = obstacles

        # Compute space between vertices in each direction

        self.dx = (self.x_max - self.x_min) / (self.n_x - 1)
        self.dy = (self.y_max - self.y_min) / (self.n_y - 1)
        self.dz = (self.z_max - self.z_min) / (self.n_z - 1)

        # lower_band and diff

        self.lower_band = np.array([self.x_min, self.y_min, self.z_min])
        self.diff = np.array([self.dx, self.dy, self.dz])
        self.threshold = 1e-4
        self.hash = [1, self.n_x, self.n_x * self.n_y]

        # total number of vertices

        self.n = self.n_x * self.n_y * self.n_z

        # initialize obstacle_vertices matrix
        # has size of the n_x x n_y x n_z, can be indexed by i,j,k
        # 0 entry means "no obstacle", 1 entry means "obstacle", at the corresponding vertex

        self.obstacle_vertices = np.zeros((self.n_x + 1, self.n_y + 1, self.n_z + 1))

        # boundaries initialized as obstacle
        self.obstacle_vertices[-1, :, :] = 1
        self.obstacle_vertices[:, -1, :] = 1
        self.obstacle_vertices[:, :, -1] = 1

        # previous: array of indices used to iterate through the vertices
        self.previous = list()
        for a in [-1, 0, 1]:
            for b in [-1, 0, 1]:
                self.previous.append([a, b, -1])
        for a in [-1, 0, 1]:
            self.previous.append([a, -1, 0])
        self.previous.append([-1, 0, 0])

        # check vertex density criterion (obstacles only detectable if e.g. l>dx/2)
        graph_okay = True
        n_x_min = 0
        n_y_min = 0
        n_z_min = 0
        for [m_x, m_y, m_z, l, w, h] in self.obstacles:
            n_x_min = max(n_x_min, (self.x_max - self.x_min) / (2 * l) + 1)
            n_y_min = max(n_y_min, (self.y_max - self.y_min) / (2 * w) + 1)
            n_z_min = max(n_z_min, (self.z_max - self.z_min) / (2 * h) + 1)
            if l <= self.dx / 2 or w <= self.dy / 2 or h <= self.dz / 2:
                graph_okay = False

        # print section
        if self.args:
            self.args.logger.info("Created DistanceGraph with: ")
            self.args.logger.info(
                "\tField: x: [{}, {}], y: [{}, {}], z: [{}, {}]".format(self.x_min, self.x_max, self.y_min, self.y_max,
                                                                        self.z_min, self.z_max))
            self.args.logger.info("\tObstacles:")
            for obstacle in obstacles:
                self.args.logger.info("\t\t{}".format(obstacle))
            self.args.logger.info(
                "\tNumber of vertices: n_x: {}, n_y: {}, n_z: {}".format(self.n_x, self.n_y, self.n_z))
            self.args.logger.info("\tTotal number of vertices: {}".format(self.n))
            self.args.logger.info("\tDx: {}, Dy: {}, Dz: {}".format(self.dx, self.dy, self.dz))
            self.args.logger.info(
                "\tRequired number of vertices: n_x > {}, n_y > {}, n_z > {}".format(n_x_min, n_y_min, n_z_min))

        if not graph_okay:
            raise Exception("Vertex density is not high enough, requirements see above")

    def is_obstacle(self, x, y, z):
        # checks whether a point (x, y, z) lies inside an obstacle
        for [m_x, m_y, m_z, l, w, h] in self.obstacles:
            # l, w, h are increased to make sure that the edges of the obstacles are considered as well
            l += 0.02
            w += 0.02
            h += 0.02
            if m_x - l <= x <= m_x + l and m_y - w <= y <= m_y + w and m_z - h <= z <= m_z + h:
                return True
        return False

    def gridpoint2vertex(self, gridpoint) -> int:
        # converts gridpoint representation [i, j, k] to vertex ID
        node = np.dot(gridpoint, self.hash)
        return int(node)

    def vertex2gridpoint(self, vertex) -> (int, int, int):
        # converts vertex ID to gridpoint representation [i, j ,k]
        k = np.floor(vertex / (self.n_x * self.n_y))
        new = vertex % (self.n_x * self.n_y)
        j = np.floor(new / self.n_x)
        i = vertex % self.n_x
        return i, j, k

    def gridpoint2coords(self, gridpoint) -> (int, int, int):
        # converts gridpoint representation [i, j, k] to coords representation [x, y, z]
        [i, j, k] = gridpoint
        x = self.x_min + i * self.dx
        y = self.y_min + j * self.dy
        z = self.z_min + k * self.dz
        return x, y, z

    def coords2gridpoint(self, coords) -> (int, int, int):
        # converts coords representation [x, y, z] to gridpoint representation [i, j, k]
        [x, y, z] = coords
        if not (
                self.x_min - self.threshold <= x < self.x_max
                and self.y_min - self.threshold <= y < self.y_max
                and self.z_min - self.threshold <= z < self.z_max):
            return None
        return np.rint((np.array(coords) - self.lower_band) / self.diff)

    def compute_cs_graph(self):
        # create cs_graph as a sparse matrix of size [num_nodes, num_nodes],
        # where the entry cs_graph[node_a, node_b] == True only if there is a connection between node_a and node_b
        # only connecting nodes that do not lie within an obstacle
        if self.args:
            self.args.logger.info("Computing {}x{} cs_graph ...".format(self.n, self.n))
        start = timer()
        row = list()
        col = list()
        data = list()

        # mark nodes lying inside an obstacle as 1 in self.obstacle_nodes
        for i in range(self.n_x):
            for j in range(self.n_y):
                for k in range(self.n_z):
                    x, y, z = self.gridpoint2coords([i, j, k])
                    if self.is_obstacle(x, y, z):  # if point is black
                        self.obstacle_vertices[i, j, k] = 1

        # connect only non-obstacle vertices with edges
        # the edge's weight corresponds to the distance between the vertices
        # edges are stored in lists row and col
        for i in range(self.n_x):
            for j in range(self.n_y):
                for k in range(self.n_z):
                    for a, b, c in self.previous:
                        if self.obstacle_vertices[i, j, k] == 0:
                            if self.obstacle_vertices[i + a, j + b, k + c] == 0:  # i.e. is white
                                basevertex = self.gridpoint2vertex([i, j, k])
                                connectvertex = self.gridpoint2vertex([i + a, j + b, k + c])
                                # distance d between two vertices
                                d = np.sqrt(
                                    a * a * self.dx * self.dx + b * b * self.dy * self.dy + c * c * self.dz * self.dz)
                                row.append(basevertex)
                                col.append(connectvertex)
                                data.append(d)
        # create cs_graph as a sparse matrix,
        # where the entry cs_graph[node_a, node_b] == True only if there is a connection between vertex_a and vertex_b
        self.cs_graph = csr_matrix((data, (row, col)), shape=(self.n, self.n))
        end = timer()
        if self.args:
            self.args.logger.info("\tdone after {} secs".format(end - start))

    def compute_dist_matrix(self, compute_predecessors=False):
        # create a distance_matrix dist_matrix from self.cs_graph by using dijkstra shortest path algorithm
        # dist_matrix is fully populated of size n x n
        # the entry dist_matrix[vertex_a, vertex_b] contains the shortest distance on cs_graph between vertex_a and vertex_b
        if self.args:
            self.args.logger.info("Computing {}x{} dist_matrix ...".format(self.n, self.n))
        start = timer()
        if self.cs_graph is None:
            raise Exception("No CS_Graph available!")
        if compute_predecessors:
            self.dist_matrix, self.predecessors = dijkstra(self.cs_graph, directed=False, return_predecessors=True)
        else:
            self.dist_matrix = dijkstra(self.cs_graph, directed=False, return_predecessors=False)
        end = timer()
        if self.args:
            self.args.logger.info("\t done after {} secs".format(end - start))

    def get_dist_grid(self, coords1, coords2, return_path=False):
        gridpoint1 = self.coords2gridpoint(coords1)
        gridpoint2 = self.coords2gridpoint(coords2)
        if gridpoint2 is None or gridpoint1 is None:
            return np.inf
        else:
            return abs(gridpoint1[0] - gridpoint2[0]) + abs(gridpoint1[1] - gridpoint2[1]) + abs(
                gridpoint1[2] - gridpoint2[2])

    def get_dist(self, coords1, coords2, return_path=False):
        # get the shortest distance between coord1 and coords2 (each of form [x, y, z]) on cs_graph
        # in case a predecessor matrix has been calculated, one can also return the shortest path

        if self.dist_matrix is None:
            raise Exception("No dist_matrix available!")

        # transfer coords [x, y, z] to the closest node in gridpoint representation [i, j, k]
        gridpoint1 = self.coords2gridpoint(coords1)
        gridpoint2 = self.coords2gridpoint(coords2)

        # if gridpoint is not in grid, assume there is no connection (shortest distance = inf)
        if gridpoint1 is None or gridpoint2 is None:
            return np.inf, None

        # transfer gridpoint representation to vertex ID
        vertex_a = self.gridpoint2vertex(gridpoint1)
        vertex_b = self.gridpoint2vertex(gridpoint2)

        if not return_path:
            return self.dist_matrix[vertex_a][vertex_b], None
        else:
            if self.predecessors is None:
                raise Exception("No predecessors available!")
            path = []
            current_node = vertex_b
            path.append(self.gridpoint2coords(self.vertex2gridpoint(current_node)))
            while current_node != vertex_a:
                current_node = self.predecessors[vertex_a, current_node]
                # if there is no path, dijkstra writes -9999 to predecessor matrix
                if current_node == -9999:
                    if self.args:
                        self.args.logger.info("No path!")
                    return self.dist_matrix[vertex_a][vertex_b], None
                path.append(self.gridpoint2coords(self.vertex2gridpoint(current_node)))
            return self.dist_matrix[vertex_a][vertex_b], path

    def plot_goals(self, goals=None, colors=None, azim=-12, elev=15, show=False, save_path='test', extra=None):
        # Plot goals with different options
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        co_graph = coo_matrix(self.cs_graph)
        # scatter plot boundaries of field
        x_array = [self.x_min, self.x_min, self.x_min, self.x_min, self.x_max, self.x_max, self.x_max, self.x_max]
        y_array = [self.y_min, self.y_min, self.y_max, self.y_max, self.y_min, self.y_min, self.y_max, self.y_max]
        z_array = [self.z_min, self.z_max, self.z_min, self.z_max, self.z_min, self.z_max, self.z_min, self.z_max]
        ax.scatter(x_array, y_array, z_array, c='b')
        # plots obstacle
        for [m_x, m_y, m_z, l, w, h] in self.obstacles:
            # top
            side1 = Rectangle((m_x - l, m_y - w), 2 * l, 2 * w, color=[0, 0, 1, 0.1])
            ax.add_patch(side1)
            art3d.pathpatch_2d_to_3d(side1, z=m_z + h, zdir="z", )
            # bottom
            side1 = Rectangle((m_x - l, m_y - w), 2 * l, 2 * w, color=[0, 0, 1, 0.1])
            ax.add_patch(side1)
            art3d.pathpatch_2d_to_3d(side1, z=m_z - h, zdir="z")
            # back
            side1 = Rectangle((m_y - w, m_z - h), 2 * w, 2 * h, color=[0, 0, 1, 0.1])
            ax.add_patch(side1)
            art3d.pathpatch_2d_to_3d(side1, z=m_x + l, zdir="x")
            # front
            side1 = Rectangle((m_y - w, m_z - h), 2 * w, 2 * h, color=[0, 0, 1, 0.1])
            ax.add_patch(side1)
            art3d.pathpatch_2d_to_3d(side1, z=m_x - l, zdir="x")
            # right
            side1 = Rectangle((m_x - l, m_z - h), 2 * l, 2 * h, color=[0, 0, 1, 0.1])
            ax.add_patch(side1)
            art3d.pathpatch_2d_to_3d(side1, z=m_y + w, zdir="y")
            # left
            side1 = Rectangle((m_x - l, m_z - h), 2 * l, 2 * h, color=[0, 0, 1, 0.1])
            ax.add_patch(side1)
            art3d.pathpatch_2d_to_3d(side1, z=m_y - w, zdir="y")

        # plot goals:
        for i in range(len(goals)):
            current_goals = goals[i]
            current_color = colors[i]
            for goal in current_goals:
                x = goal[0]
                y = goal[1]
                z = goal[2]
                ax.scatter([x], [y], [z], c=current_color)

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_zlim(0.4, 0.8)
        if extra == 1:
            ax.set_yticks([0.5, 0.7, 0.9, 1.1])
            ax.set_zticks([0.4, 0.5, 0.6, 0.7, 0.8])

        ax.view_init(elev=elev, azim=azim)
        if show:
            plt.show()
        plt.savefig(save_path + ".pdf")

    def plot_graph(self, path=None, graph=False, obstacle_vertices=False, goals=None, save_path='test', show=False,
                   azim=-12, elev=15, extra=None):
        # Plot graph with different options
        if self.args:
            self.args.logger.info("Plotting ...")
        if self.cs_graph is None:
            raise Exception("No cs_graph available")
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        co_graph = coo_matrix(self.cs_graph)
        # scatter plot boundaries of field
        x_array = [self.x_min, self.x_min, self.x_min, self.x_min, self.x_max, self.x_max, self.x_max, self.x_max]
        y_array = [self.y_min, self.y_min, self.y_max, self.y_max, self.y_min, self.y_min, self.y_max, self.y_max]
        z_array = [self.z_min, self.z_max, self.z_min, self.z_max, self.z_min, self.z_max, self.z_min, self.z_max]
        ax.scatter(x_array, y_array, z_array, c='b')
        # plots obstacle
        for [m_x, m_y, m_z, l, w, h] in self.obstacles:
            # top
            side1 = Rectangle((m_x - l, m_y - w), 2 * l, 2 * w, color=[0, 0, 1, 0.1])
            ax.add_patch(side1)
            art3d.pathpatch_2d_to_3d(side1, z=m_z + h, zdir="z", )
            # bottom
            side1 = Rectangle((m_x - l, m_y - w), 2 * l, 2 * w, color=[0, 0, 1, 0.1])
            ax.add_patch(side1)
            art3d.pathpatch_2d_to_3d(side1, z=m_z - h, zdir="z")
            # back
            side1 = Rectangle((m_y - w, m_z - h), 2 * w, 2 * h, color=[0, 0, 1, 0.1])
            ax.add_patch(side1)
            art3d.pathpatch_2d_to_3d(side1, z=m_x + l, zdir="x")
            # front
            side1 = Rectangle((m_y - w, m_z - h), 2 * w, 2 * h, color=[0, 0, 1, 0.1])
            ax.add_patch(side1)
            art3d.pathpatch_2d_to_3d(side1, z=m_x - l, zdir="x")
            # right
            side1 = Rectangle((m_x - l, m_z - h), 2 * l, 2 * h, color=[0, 0, 1, 0.1])
            ax.add_patch(side1)
            art3d.pathpatch_2d_to_3d(side1, z=m_y + w, zdir="y")
            # left
            side1 = Rectangle((m_x - l, m_z - h), 2 * l, 2 * h, color=[0, 0, 1, 0.1])
            ax.add_patch(side1)
            art3d.pathpatch_2d_to_3d(side1, z=m_y - w, zdir="y")

        if path:
            for i in range(len(path) - 1):
                a = path[i]
                b = path[i + 1]
                X, Y, Z = [a[0], b[0]], [a[1], b[1]], [a[2], b[2]]
                ax.plot(X, Y, Z, c=[1, 0, 0, 1])

        # plot graph edges
        if graph:
            for i, j, v in zip(co_graph.row, co_graph.col, co_graph.data):
                a = self.gridpoint2coords(self.vertex2gridpoint(i))
                b = self.gridpoint2coords(self.vertex2gridpoint(j))
                X, Y, Z = [a[0], b[0]], [a[1], b[1]], [a[2], b[2]]
                ax.plot(X, Y, Z, c=[0, 0, 0, 0.2])
        # scatter plot vertices that are marked as black (with obstacle)
        if obstacle_vertices:
            for i in range(self.n_x):
                for j in range(self.n_y):
                    for k in range(self.n_z):
                        x, y, z = self.gridpoint2coords([i, j, k])
                        if self.obstacle_vertices[i, j, k] == 0:
                            ax.scatter([x], [y], [z], c=[0, 0, 0, 0.3])
        # plot goals:
        if goals:
            for goal in goals:
                x = goal[0]
                y = goal[1]
                z = goal[2]
                ax.scatter([x], [y], [z], c='green')

        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        if extra == 1:
            ax.set_xticks([1.1, 1.3, 1.5])
            ax.set_zticks([0.4, 0.5, 0.6, 0.7, 0.8])

        ax.view_init(elev=elev, azim=azim)
        if show:
            plt.show()
        plt.savefig(save_path + ".pdf")
        if self.args:
            self.args.logger.info("\tdone")
