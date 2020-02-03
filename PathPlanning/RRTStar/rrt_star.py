"""

Path planning Sample Code with RRT*

author: Atsushi Sakai(@Atsushi_twi)

"""

import math
import os
import sys

import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.abspath(__file__)) +
                "/../RRT/")

try:
    from rrt import RRT
except ImportError:
    raise


show_live_animation = False
show_final_animation = True


def static_var(varName, value):
    def decorate(function):
        setattr(function,varName,value)
        return function
    return decorate


class RRTStar(RRT):
    """
    Class for RRT Star planning
    """

    class Node(RRT.Node):
        def __init__(self, x, y, alpha):
            super().__init__(x, y, alpha)
            self.cost = 0.0
            self.rho = 0.0
            self.d = 1.0

    def __init__(self, start, goal, obstacle_list, rand_area,
                 expand_dis=3.0,
                 path_resolution=1.0,
                 goal_sample_rate=20,
                 max_iter=300,
                 connect_circle_dist=50.0
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
        self.goal_node = self.Node(goal[0], goal[1], goal[2])

    def planning(self, animation=True, search_until_max_iter=True):
        """
        rrt star path planning

        animation: flag for animation on or off
        search_until_max_iter: search until max iteration for path improving or not
        """
        # Init node_list and include start position
        self.node_list = [self.start]

        for i in range(self.max_iter):
            #print("Iter:", i, ", number of nodes:", len(self.node_list))

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

            if animation and i % 30 == 0:
                self.draw_graph(rnd_node)

            if (not search_until_max_iter) and new_node:  # check reaching the goal
                last_index = self.search_best_goal_node()
                if last_index:
                    return self.generate_final_course(last_index)

        print("reached max iteration")

        last_index = self.search_best_goal_node()
        if last_index:
            return self.generate_final_course(last_index)

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
        new_node.alpha = math.atan2(new_node.y - new_node.parent.y, new_node.x - new_node.parent.x)
        # distance between new node and its parent
        d, _ = self.calc_distance_and_angle(new_node.parent, new_node)
        new_node.d = d

        return new_node

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
        filtered_inds = [near_inds[angle_list.index(i)] for i in angle_list if i <= math.pi/2]

        return filtered_inds

    def rewire(self, new_node, near_inds):
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
                self.propagate_cost_to_leaves(new_node)

    def calc_new_cost(self, from_node, to_node):
        # Distance cost
        d, _ = self.calc_distance_and_angle(from_node, to_node)
        distance_cost = from_node.cost + d

        # Curvature cost
        if d == 0 or from_node.d == 0:
            curvature_cost = float('Inf')
        else:
            to_node.rho = (2*math.tan(abs( from_node.alpha - to_node.alpha ))) / min(from_node.d, d)
            # initialize counter
            curvature_cost = from_node.rho + to_node.rho

            RRTStar.curvature_cost.counter = 0
            rho_sum = self.curvature_cost(from_node)
            #curvature_cost = rho_sum/RRTStar.curvature_cost.counter + to_node.rho

        return curvature_cost

    def propagate_cost_to_leaves(self, parent_node):

        for node in self.node_list:
            if node.parent == parent_node:
                node.cost = self.calc_new_cost(parent_node, node)
                self.propagate_cost_to_leaves(node)

    def ssa(self, angle):
        # Smallest signed angle. Maps angle into interval [-pi pi]
        wrpd_angle = (angle + math.pi) % (2*math.pi) - math.pi
        return wrpd_angle

    @static_var('counter', 0)
    def curvature_cost(self, from_node):
        # stash counter in the function itself
        RRTStar.curvature_cost.counter += 1
        if not from_node.parent:
            return from_node.rho
        return from_node.rho + self.curvature_cost(from_node.parent)


def main():
    print("Start " + __file__)

    # ====Search Path with RRT====

    obstacleList = [ # [x, y, radius]
        (50, 50, 10),
        (50, 70, 10),
        (70, 50, 10),
        (90, 50, 10),
        (10, 70, 10)]

    # Set Initial parameters
    rrt_star = RRTStar(start = [60, 20, 0], # [x, y, theta]
                       goal = [90, 90, 0], # [x, y, theta]
                       obstacle_list = obstacleList,
                       rand_area = [0, 100],
                       expand_dis = 10,
                       path_resolution = 1,
                       goal_sample_rate = 10,
                       max_iter = 1500,
                       connect_circle_dist = 50)
    path = rrt_star.planning(animation=show_live_animation)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")
        # Print path
        for i in reversed(path):
            print(i)

        # Draw final path
        if show_final_animation:
            rrt_star.draw_graph()
            plt.plot([x for (x, y, alpha) in path], [y for (x, y, alpha) in path], '-r')
            plt.grid(True)
            plt.pause(0.01)  # Need for Mac
            plt.show()
        
        


if __name__ == '__main__':
    main()
