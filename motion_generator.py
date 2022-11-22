import numpy as np

from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path


class MotionGenerator:

    def __init__(self, environment_state=None, agent_position=None, ignore_agents_as_obstacles=False):
        # declare attributes but set them in set_environment_state()-method
        self.env_state_raw = None
        self.agent_position = None
        self.height = None
        self.width = None
        self.env_state_graph = None
        self.cs_graph = None
        self.env_state_free_fields = None
        self.ignore_agents_as_obstacles = None
        if agent_position is not None:
            self.set_agent_position(agent_position)
        if environment_state is not None:
            self.set_environment_state(environment_state, ignore_agents_as_obstacles)
        elif environment_state:
            raise ValueError("Motion Generator also needs the agent_position for which the route should be found.")

    def reset(self, environment_state, agent_position, ignore_agents_as_obstacles=False):
        self.env_state_raw = None
        self.agent_position = None
        self.height = None
        self.width = None
        self.env_state_graph = None
        self.cs_graph = None
        self.env_state_free_fields = None
        self.ignore_agents_as_obstacles = None
        if agent_position is not None:
            self.set_agent_position(agent_position)
        if environment_state is not None:
            self.set_environment_state(environment_state, ignore_agents_as_obstacles)
        elif environment_state:
            raise ValueError("Motion Generator also needs the agent_position for which the route should be found.")

    def set_environment_state(self, environment_state, ignore_agents_as_obstacles=False):
        self.env_state_raw = environment_state

        self.height = self.env_state_raw.shape[0]
        self.width = self.env_state_raw.shape[1]
        self.env_state_graph = None
        self.cs_graph = None
        self.env_state_free_fields = None
        self.ignore_agents_as_obstacles = ignore_agents_as_obstacles
        self.create_graph()

    def set_agent_position(self, agent_position):
        self.agent_position = agent_position

    def _get_graph_index(self, position):
        """Computes the graph index for a given position."""
        index = position[0] * self.width + position[1]
        return index

    def _get_coords_from_index(self, index):
        """Computes the grid position for a given index."""
        return index // self.width, index % self.width

    def create_graph(self):
        """Creates a graph from the given environment state to fit necessary form for shortest path algorithms of scipy."""

        if self.env_state_raw is None:
            raise RuntimeError("Environment in motion generator is None.")

        # init graph
        number_of_cells = self.width * self.height
        graph_shape = (number_of_cells, number_of_cells)
        graph = np.zeros(graph_shape)

        # create feature map with 'free' fields. Each free field is represented by a 1, occupied fields represented by a 0.
        env_state_free_fields = np.copy(self.env_state_raw[:, :, 0])
        for fm in range(1, self.env_state_raw.shape[2]):
            env_state_free_fields -= self.env_state_raw[:, :, fm]

        # ensure that agent is not seen as obstacle
        env_state_free_fields[tuple(self.agent_position)] = 1
        self.env_state_free_fields = np.clip(env_state_free_fields, 0, 1)

        # ensure that NO agent is seen as obstacle
        agent_fm = self.env_state_raw[:, :, 15]
        agents_where = np.where(agent_fm == 1)
        if self.ignore_agents_as_obstacles:
            for i, _ in enumerate(agents_where):
                a_position = (agents_where[0][i], agents_where[1][i])
                self.env_state_free_fields[a_position] = 1
        '''
        else:
            # collision handling = block fields around other agents
            for i, where in enumerate(agents_where):
                if agents_where[0].shape[0] != 0:
                    a_position = (agents_where[0][i], agents_where[1][i])
                    if a_position != tuple(self.agent_position):
                        a_position_x = a_position[0]
                        a_position_y = a_position[1]
                        self.env_state_free_fields[a_position] = 0
                        if 0 <= a_position_x - 1 < self.width:
                            self.env_state_free_fields[a_position_x - 1, a_position_y] = 0
                        if 0 <= a_position_x + 1 < self.width:
                            self.env_state_free_fields[a_position_x + 1, a_position_y] = 0
                        if 0 <= a_position_y - 1 < self.height:
                            self.env_state_free_fields[a_position_x, a_position_y - 1] = 0
                        if 0 <= a_position_y + 1 < self.height:
                            self.env_state_free_fields[a_position_x - 1, a_position_y + 1] = 0
        '''

        # print(f"Free fields matrix:\n {self.env_state_free_fields}")

        # create graph as matrix representation
        for i in range(self.height):
            for j in range(self.width):
                current_field = self.env_state_free_fields[i, j]
                current_field_graph_index = self._get_graph_index((i, j))

                # if current field is free (==1), connect to free neighbors:
                if current_field:
                    # look at neighbors
                    if 0 <= i - 1 < self.width:
                        if self.env_state_free_fields[i - 1, j]:
                            neighbor_field_graph_index = self._get_graph_index((i - 1, j))
                            graph[current_field_graph_index, neighbor_field_graph_index] = 1

                    if 0 <= i + 1 < self.width:
                        if self.env_state_free_fields[i + 1, j]:
                            neighbor_field_graph_index = self._get_graph_index((i + 1, j))
                            graph[current_field_graph_index, neighbor_field_graph_index] = 1

                    if 0 <= j - 1 < self.height:
                        if self.env_state_free_fields[i, j - 1]:
                            neighbor_field_graph_index = self._get_graph_index((i, j - 1))
                            graph[current_field_graph_index, neighbor_field_graph_index] = 1

                    if 0 <= j + 1 < self.height:
                        if self.env_state_free_fields[i, j + 1]:
                            neighbor_field_graph_index = self._get_graph_index((i, j + 1))
                            graph[current_field_graph_index, neighbor_field_graph_index] = 1

        # write graph as object attribute
        self.env_state_graph = graph

        # create csr_graph
        self.cs_graph = csr_matrix(self.env_state_graph)

        # reset ignore-mode
        if self.ignore_agents_as_obstacles:
            self.ignore_agents_as_obstacles = False

    def shortest_path(self, start_position, end_position, check_for_free_neighbors=True):
        """Computes the shortest path for the given start_position (x, y) and end_position (x, y) as tuple (path, actions)."""

        # if end_position is blocked, return None
        if not self.env_state_free_fields[end_position]:
            if not check_for_free_neighbors:
                return None, ['nop']
            else:
                # return path to a free neighbor
                paths_to_neighbors = self.shortest_path_to_free_neighbors(tuple(start_position), end_position)
                # currently always returns the path to the first found neighbor. Can be extended (e.g. random choice) if needed.
                return paths_to_neighbors[0] if paths_to_neighbors else (None, ['nop'])

        start_index = self._get_graph_index(start_position)
        dist_matrix, predecessors = shortest_path(csgraph=self.cs_graph, directed=False, indices=start_index, return_predecessors=True)

        path = self._reconstruct_path_from_predecessors(predecessors, start_position, end_position)

        return path

    def shortest_path_to_free_neighbors(self, start_position, end_position):
        """Creates paths to the neighbors of the given end_position as list of tuples (path, actions)."""

        # calculate neighbors to end_position:
        free_neighbors = self.get_free_neighbors(end_position)
        paths_to_neighbors = []

        if start_position in free_neighbors:
            DIFFERENCE_TO_ORIENTATION = {
                (-1, 0): 2,
                (1, 0): 1,
                (0, -1): 3,
                (0, 1): 4
            }
            diff = np.array(start_position) - np.array(end_position)
            needed_orientation = DIFFERENCE_TO_ORIENTATION[(diff[0], diff[1])]
            if self.env_state_raw[start_position[0], start_position[1], -5 + needed_orientation] == 1:
                return [(None, ['nop'])]
            else:
                orientation_action = _get_action_from_to(start_position, end_position)
                return [([end_position], [orientation_action])]
        else:
            for neighbor in free_neighbors:
                path = self.shortest_path(start_position, neighbor, check_for_free_neighbors=False)

                if path != (None, ['nop']):
                    orientation_action = _get_action_from_to(neighbor, end_position)
                    if path[1][-1] != orientation_action:
                        path[1].append(orientation_action)
                    paths_to_neighbors.append(path)

        return paths_to_neighbors

    def _reconstruct_path_from_predecessors(self, predecessors, start_position, end_position):
        """Reconstructs the path from start_position to end_position as a list as specified
        by the predecessors-output from the shortest path algorithm."""

        path = [end_position]
        end_index = self._get_graph_index(end_position)
        start_index = self._get_graph_index(start_position)

        # if no path to the target exists
        if predecessors[end_index] < 0:
            return None, ['nop']

        current_index = end_index
        while current_index != start_index:
            path.append(self._get_coords_from_index(predecessors[current_index]))
            current_index = predecessors[current_index]

        path.reverse()
        actions = _reconstruct_actions_from_path(path)
        return path, actions

    def get_free_neighbors(self, position):
        """Returns all free neighbors to a position (x, y) in the given grid."""
        possible_neighbors = get_neighbors(position)

        # check if possible neighbors are free / part of the grid
        free_neighbors = []
        for neighbor in possible_neighbors:
            if all([0 <= n < g for n, g in zip(neighbor, self.env_state_free_fields.shape)]) and self.env_state_free_fields[neighbor]:
                free_neighbors.append(neighbor)

        return free_neighbors


def get_neighbors(position):
    """Returns all neighboring fields of the given position (ignoring grid bounds and diagonal neighbors)."""
    possible_neighbors = []

    # delta x
    for delta in [-1, 1]:
        new_pos = (position[0] + delta, position[1])
        possible_neighbors.append(new_pos)

    # delta y
    for delta in [-1, 1]:
        new_pos = (position[0], position[1] + delta)
        possible_neighbors.append(new_pos)
    return possible_neighbors


def _reconstruct_actions_from_path(path):
    """Takes the path as list of coordinates and recreates the necessary actions from that."""

    actions = []
    for i in range(len(path) - 1):
        actions.append(_get_action_from_to(path[i], path[i + 1]))
    return actions


def _get_action_from_to(pos1, pos2):
    """Returns the action needed for getting from pos1 to pos2 or orienting towards pos2 if pos2 is blocked."""
    DIFFERENCE_TO_ACTION = {
        (-1, 0): 'move_right',
        (1, 0): 'move_left',
        (0, -1): 'move_down',
        (0, 1): 'move_up'
    }

    diff = np.array(pos1) - np.array(pos2)
    diff = (diff[0], diff[1])
    return DIFFERENCE_TO_ACTION[diff]
