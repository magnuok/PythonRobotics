"""

Path planning Sample Code with RRT*

author: Atsushi Sakai(@Atsushi_twi)

"""

import math
import os
import sys
import numpy as np

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../RRT/")

try:
    from rrt import RRT
except ImportError:
    raise

def static_var(**kwargs):
    """
    This function creates decorator. Used for counting recursive calls.
    """
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

class prettyfloat(float):
    """
    Class for printing float with given precision
    """
    def __repr__(self):
        return "%0.2f" % self

class RRTStar(RRT):
    """
    Class for RRT Star planning
    """

    class Node(RRT.Node):
        def __init__(self, x, y, alpha):
            super().__init__(x, y, alpha)
            self.cost = 0.0
            self.kappa = 0.0
            self.d = float("Inf")

    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis,
                 path_resolution,
                 goal_sample_rate,
                 max_iter,
                 connect_circle_dist,
                 max_alpha,
                 max_kappa
                 ):
        super().__init__(start, goal, obstacle_list,
                         rand_area, expand_dis, path_resolution, goal_sample_rate, max_iter)
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Random Sampling Area [min,max]

        """
        self.connect_circle_dist = connect_circle_dist
        self.max_alpha = max_alpha
        self.max_kappa = max_kappa
        self.goal_node = self.Node(goal[0], goal[1], goal[2])

    def planning(self, animation=True, search_until_max_iter=True):
        """
        rrt star path planning

        animation: flag for animation on or off
        search_until_max_iter: search until max iteration for path improving or not
        """
        
        self.node_list = [self.start]
                
        # TODO: Informed RRT*
        # max length we expect to find in our 'informed' sample space, starts as infinite
        cBest = float('inf')
        solutionSet = set()
        path = None
         # Computing the sampling space
        cMin = math.sqrt(pow(self.start.x - self.goal.x, 2)
                         + pow(self.start.y - self.goal.y, 2))
        xCenter = np.array([[(self.start.x + self.goal.x) / 2.0],
                            [(self.start.y + self.goal.y) / 2.0], [0]])
        a1 = np.array([[(self.goal.x - self.start.x) / cMin],
                       [(self.goal.y - self.start.y) / cMin], [0]])

        etheta = math.atan2(a1[1], a1[0])
        # first column of idenity matrix transposed
        id1_t = np.array([1.0, 0.0, 0.0]).reshape(1, 3)
        M = a1 @ id1_t
        U, S, Vh = np.linalg.svd(M, True, True)
        C = np.dot(np.dot(U, np.diag(
            [1.0, 1.0, np.linalg.det(U) * np.linalg.det(np.transpose(Vh))])), Vh)

        # Sample space is defined by cBest
        # cMin is the minimum distance between the start point and the goal
        # xCenter is the midpoint between the start and the goal
        # cBest changes when a new path is found
        
        for i in range(self.max_iter):

            rnd_node = self.get_random_node()
            # Get nearest index of rnd node.
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)

            new_node = self.steer(self.node_list[nearest_ind], rnd_node, self.expand_dis)

            if self.check_collision(new_node, self.obstacle_list):
                # Find nodes nearby new_node
                near_inds = self.find_near_nodes(new_node)                
                new_node = self.choose_parent(new_node, near_inds)
                if new_node:
                    self.node_list.append(new_node)
                    self.rewire(new_node, near_inds)

            if animation and i % 1 == 0:
                self.draw_graph(rnd_node)

            if (not search_until_max_iter) and new_node:  # check reaching the goal
                last_index = self.search_best_goal_node()
                if last_index:
                    return self.generate_final_cost(last_index)

        print("Reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_path_values(last_index)

        return None

    def choose_parent(self, new_node, filtered_inds):
        # If no one close, return None and throw away the new_node
        if not filtered_inds:
            return None

        # Search through near_inds and find minimum cost
        costs = []
        for i in filtered_inds:
            near_node = self.node_list[i]
            t_node = self.steer(near_node, new_node)
            if t_node and self.check_collision(t_node, self.obstacle_list):
                costs.append(self.calc_new_cost(near_node, new_node))
            else:
                costs.append(float("inf"))  # the cost of collision node
        
        min_cost = min(costs)

        if min_cost == float("inf"):
            print("There is no good path.(min_cost is inf)")
            return None

        # Set parent to the one found with lowest cost
        min_ind = filtered_inds[costs.index(min_cost)]
        new_node = self.steer(self.node_list[min_ind], new_node)
        new_node.parent = self.node_list[min_ind]
        new_node.cost = min_cost

        new_node = self.update_node_values(new_node)

        return new_node

    def update_node_values(self, node):
        """
        Updates values for new or rewired node.
        """
        node.alpha = math.atan2(node.y - node.parent.y, node.x - node.parent.x)

        d, _ = self.calc_distance_and_angle(node.parent, node)
        node.d = d

        if d == 0 or node.parent.d == 0 or abs(self.ssa(node.parent.alpha - node.alpha)) > self.max_alpha:
            node.kappa = float("Inf")
        else:
            node.kappa = (2*math.tan(abs(self.ssa(node.parent.alpha - node.alpha)))) / min(node.parent.d, node.d)
        
        return node

    def search_best_goal_node(self):
        dist_to_goal_list = [self.calc_dist_to_goal(n.x, n.y) for n in self.node_list]
        goal_inds = [dist_to_goal_list.index(i) for i in dist_to_goal_list if i <= self.expand_dis]

        safe_goal_inds = []
        for goal_ind in goal_inds:
            t_node = self.steer(self.node_list[goal_ind], self.goal_node)
            if self.check_collision(t_node, self.obstacle_list):
                safe_goal_inds.append(goal_ind)

        if not safe_goal_inds:
            return None

        min_cost = min([self.node_list[i].cost for i in safe_goal_inds])
        for i in safe_goal_inds:
            if self.node_list[i].cost == min_cost:
                return i

        return None

    def find_near_nodes(self, new_node):

        nnode = len(self.node_list) + 1

        # TODO: Find source for calculation of the radius
        r = self.connect_circle_dist * math.sqrt((math.log(nnode) / nnode))

        # First find nodes nearby
        dist_list = [(node.x - new_node.x) ** 2 +
                     (node.y - new_node.y) ** 2 for node in self.node_list]
        near_inds = [dist_list.index(i) for i in dist_list if i <= r ** 2]

        # For nearby nodes, filter nodes by angle constraint
        near_nodes = [self.node_list[i] for i in near_inds]
        angle_list = [abs( self.ssa(node.alpha - math.atan2(new_node.y - node.y, new_node.x - node.x))) for node in near_nodes]

        curvature_list = []
        for node in near_nodes:
            d, _ = self.calc_distance_and_angle(node, new_node)
            alpha_next = math.atan2(new_node.y - node.y, new_node.x - node.x)
            if d == 0 or node.d == 0 or abs(self.ssa(node.alpha - alpha_next)) > self.max_alpha:
                kappa_next = float('Inf')
            else:
                kappa_next = (2*math.tan(abs(self.ssa(node.alpha - alpha_next)))) / min(node.d, d)

            curvature_list.append(kappa_next)

        filtered_inds = [near_inds[angle_list.index(alpha)] for alpha, kappa in zip(angle_list, curvature_list) if alpha <= self.max_alpha and kappa <= self.max_kappa]

        # For nearby nodes, filter nodes by curvature constraint


        return filtered_inds

    def rewire(self, new_node, near_inds):
        """
        This function checks if the cost to the nodes in near_inds is less through new_node as compared to their older costs, 
        then its parent is changed to new_node.
        """
        for i in near_inds:
            near_node = self.node_list[i]
            edge_node = self.steer(new_node, near_node)
            if not edge_node:
                continue
            edge_node.cost = self.calc_new_cost(new_node, near_node)

            no_collision = self.check_collision(edge_node, self.obstacle_list)
            improved_cost = near_node.cost > edge_node.cost

            if no_collision and improved_cost:
                near_node = edge_node
                near_node.parent = new_node
                # Update values
                near_node = self.update_node_values(near_node)

                self.propagate_cost_to_leaves(new_node)

    def propagate_cost_to_leaves(self, parent_node):
        """
        When rewired, this function updates the costs of parents.
        """
        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

    def calc_new_cost(self, from_node, to_node):
        """
        Calculate cost functions. TODO: Now only one used at a time. Tuning and superpos. must be done.
        c_d - distance cost
        c_c - curvature cost
        c_o - obstacle cost
        """
        # Distance cost
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        c_d = from_node.cost + d

        # Obstacle cost
        c_o = from_node.cost + 1/self.get_min_obstacle_distance(to_node, self.obstacle_list)

        # Curvature cost
        alpha_next = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        if d == 0 or from_node.d == 0 or abs(self.ssa(from_node.alpha - alpha_next)) > self.max_alpha:
            c_c = float('Inf')
        else:
            RRTStar.get_sum_c_c.counter = 0
            kappa_next = (2*math.tan(abs(self.ssa(from_node.alpha - alpha_next)))) / min(from_node.d, d)

            c_c =( max(self.get_max_kappa(from_node), kappa_next) 
                + (self.get_sum_c_c(from_node) + kappa_next) / (RRTStar.get_sum_c_c.counter) )
        
        return c_d # c_d, c_o or c_c


    """ --- Utils --- """

    @staticmethod
    def ssa(angle):
        """
        Smallest signed angle. Maps angle into interval [-pi pi]
        """
        wrpd_angle = (angle + math.pi) % (2*math.pi) - math.pi
        return wrpd_angle

    @static_var(counter=0)
    def get_sum_c_c(self, from_node):
        """
        Finds sum of curvature cost, recursively. The static variable keeps track of depth/#parents
        """
        # stash counter in the function itself
        RRTStar.get_sum_c_c.counter += 1
        if from_node.parent == None:
            return 0
        return from_node.cost + self.get_sum_c_c(from_node.parent)

    @staticmethod
    def get_max_kappa(node):
        """
        Finds maximum curvature from node to root, recursively.
        """
        if node.parent == None:
            return 0
        return max(node.cost, RRTStar.get_max_kappa(node.parent))

    @staticmethod
    def get_min_obstacle_distance(node, obstacleList):
        """
        Finds minimum distance to obstacle from node.
        """
        dx_list = [ ox - node.x for (ox, oy, size) in obstacleList]
        dy_list = [ oy - node.y for (ox, oy, size) in obstacleList]
        d_list = [math.sqrt(dx * dx + dy * dy) for (dx, dy) in zip(dx_list, dy_list)]

        return min(d_list)

    def generate_final_path_values(self, goal_ind):
        path = [[self.end.x, self.end.y, math.degrees(self.end.alpha), self.end.cost, 0, 0]]
        node = self.node_list[goal_ind]
        while node.parent is not None:
            path.append([node.x, node.y, math.degrees(node.alpha), node.cost, node.d, node.kappa])
            node = node.parent
        path.append([node.x, node.y, math.degrees(node.alpha), node.cost, node.d, node.kappa])

        return path

def main():
    #print("Start " + __file__)

    show_live_animation = False
    show_final_animation = True

    # [x, y, radius]
    obstacleList = [
        (2, 10, 2),
        (8, 10, 2),
        (17, 3, 3)
        ]

    # Set Initial parameters
    rrt_star = RRTStar(start = [0, 0, 0], # [x, y, theta]
                       goal = [8, 20, math.pi/2], # [x, y, theta]
                       obstacle_list = obstacleList,
                       rand_area = [0, 20],
                       expand_dis = 1,
                       path_resolution = 0.1,
                       goal_sample_rate = 5,
                       max_iter = 1000,
                       connect_circle_dist = 20,
                       max_alpha = math.pi/2,
                       max_kappa = 2)

    path = rrt_star.planning(animation=show_live_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("Found path")
        print("\n  x,   y,  alpha,  cost,   d,  kappa")

        for i in reversed(path):
            i = map(prettyfloat, i)
            print(list(i)) 

        # Draw final path
        if show_final_animation:
            rrt_star.draw_graph()
            plt.plot([x for (x, y, alpha, cost, d, kappa) in path], [y for (x, y, alpha, cost, d, kappa) in path], '-r')
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.show()

if __name__ == '__main__':
    main()
