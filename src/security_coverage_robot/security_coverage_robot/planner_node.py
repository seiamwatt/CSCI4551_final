import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import MarkerArray, Marker
import math
import numpy as np
from collections import deque


class CellState:
    UNKNOWN = 0
    FREE = 1
    OBSTACLE = 2
    RED = 3


class PlannerNode(Node):
    def __init__(self):
        super().__init__('planner_node')

        # Parameters
        self.declare_parameter('grid_resolution', 0.5)
        self.declare_parameter('grid_width', 40)
        self.declare_parameter('grid_height', 40)
        self.declare_parameter('grid_origin_x', -10.0)
        self.declare_parameter('grid_origin_y', -10.0)
        self.declare_parameter('lidar_mark_range', 3.0)

        self.resolution = self.get_parameter('grid_resolution').value
        self.grid_w = self.get_parameter('grid_width').value
        self.grid_h = self.get_parameter('grid_height').value
        self.origin_x = self.get_parameter('grid_origin_x').value
        self.origin_y = self.get_parameter('grid_origin_y').value

        # Grid tracking
        self.cell_state = np.full(
            (self.grid_h, self.grid_w), CellState.UNKNOWN, dtype=np.int8
        )
        self.zone_color = [
            ['unknown' for _ in range(self.grid_w)] for _ in range(self.grid_h)
        ]
        self.visit_count = np.zeros((self.grid_h, self.grid_w), dtype=np.int32)
        self.required_visits = np.ones((self.grid_h, self.grid_w), dtype=np.int32)

        # Robot state
        self.robot_row = None
        self.robot_col = None
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0

        # Spanning tree
        self.parent = {}
        self.tree_built = False
        self.coverage_path = []
        self.current_path_index = 0
        self.last_tree_size = 0

        # Waypoint state — only send a new waypoint after the previous one is
        # reached (or the robot reports stuck).
        self.waiting_for_reach = False
        self.sent_waypoint_row = None
        self.sent_waypoint_col = None

        # Lidar data
        self.latest_scan = None

        # --- Subscribers ---
        self.zone_sub = self.create_subscription(
            String, '/zone_color', self.zone_callback, 10
        )
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10
        )
        self.reached_sub = self.create_subscription(
            String, '/waypoint_reached', self.reached_callback, 10
        )
        self.status_sub = self.create_subscription(
            String, '/control_status', self.status_callback, 10
        )

        # --- Publishers ---
        self.waypoint_pub = self.create_publisher(
            PoseStamped, '/coverage_waypoint', 10
        )
        self.grid_viz_pub = self.create_publisher(
            MarkerArray, '/coverage_grid', 10
        )
        self.metrics_pub = self.create_publisher(String, '/coverage_metrics', 10)
        self.waypoint_zone_pub = self.create_publisher(
            String, '/waypoint_zone', 10
        )

        # Timers
        self.plan_timer = self.create_timer(0.5, self.planning_loop)
        self.viz_timer = self.create_timer(2.0, self.publish_grid_viz)

        self.get_logger().info('Planner node started (lidar-based exploration)')

    # ------------------------------------------------------------------ #
    #  Coordinate helpers
    # ------------------------------------------------------------------ #

    def world_to_grid(self, x, y):
        col = int((x - self.origin_x) / self.resolution)
        row = int((y - self.origin_y) / self.resolution)
        return row, col

    def grid_to_world(self, row, col):
        x = self.origin_x + (col + 0.5) * self.resolution
        y = self.origin_y + (row + 0.5) * self.resolution
        return x, y

    def in_bounds(self, row, col):
        return 0 <= row < self.grid_h and 0 <= col < self.grid_w

    # ------------------------------------------------------------------ #
    #  Callbacks
    # ------------------------------------------------------------------ #

    def odom_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny, cosy)
        self.robot_row, self.robot_col = self.world_to_grid(
            self.robot_x, self.robot_y
        )

    def scan_callback(self, msg):
        """Use lidar to mark free and obstacle cells."""
        self.latest_scan = msg

        if self.robot_row is None:
            return

        max_range = self.get_parameter('lidar_mark_range').value

        for i in range(len(msg.ranges)):
            angle = msg.angle_min + i * msg.angle_increment
            r = msg.ranges[i]

            if not math.isfinite(r) or r < msg.range_min or r > msg.range_max:
                continue

            world_angle = self.robot_yaw + angle
            effective_range = min(r, max_range)

            # Mark cells along the ray as free
            step = self.resolution * 0.5
            d = 0.0
            cos_a = math.cos(world_angle)
            sin_a = math.sin(world_angle)
            while d < effective_range:
                wx = self.robot_x + d * cos_a
                wy = self.robot_y + d * sin_a
                gr, gc = self.world_to_grid(wx, wy)
                if self.in_bounds(gr, gc):
                    if self.cell_state[gr][gc] == CellState.UNKNOWN:
                        self.cell_state[gr][gc] = CellState.FREE
                d += step

            # Mark the endpoint as obstacle if the ray hit something
            if r < max_range:
                wx = self.robot_x + r * cos_a
                wy = self.robot_y + r * sin_a
                gr, gc = self.world_to_grid(wx, wy)
                if self.in_bounds(gr, gc):
                    self.cell_state[gr][gc] = CellState.OBSTACLE

    def zone_callback(self, msg):
        """Update grid with zone color from perception node.

        NOTE: Visit counting is NOT done here — it happens when a waypoint
        is reached (reached_callback), so that we only count actual coverage
        passes, not repeated color detections.
        """
        if self.robot_row is None or self.robot_col is None:
            return

        row, col = self.robot_row, self.robot_col
        if not self.in_bounds(row, col):
            return

        color = msg.data.strip().lower()
        if color == 'unknown':
            return

        old_color = self.zone_color[row][col]
        self.zone_color[row][col] = color

        if color == 'red':
            self.cell_state[row][col] = CellState.RED
            self.required_visits[row][col] = 0
            if old_color != 'red':
                self.get_logger().info(f'Cell ({row},{col}) RED — CCTV, skipping')
        elif color == 'yellow':
            if self.cell_state[row][col] != CellState.OBSTACLE:
                self.cell_state[row][col] = CellState.FREE
            self.required_visits[row][col] = 2
            if old_color != 'yellow':
                self.get_logger().info(f'Cell ({row},{col}) YELLOW — needs 2 passes')
        elif color == 'green':
            if self.cell_state[row][col] != CellState.OBSTACLE:
                self.cell_state[row][col] = CellState.FREE
            self.required_visits[row][col] = 1
        elif color == 'blue':
            if self.cell_state[row][col] != CellState.OBSTACLE:
                self.cell_state[row][col] = CellState.FREE
            # Mark as already fully explored — don't increment visit_count
            self.visit_count[row][col] = max(
                self.visit_count[row][col],
                self.required_visits[row][col]
            )
            if old_color != 'blue':
                self.get_logger().info(f'Cell ({row},{col}) BLUE — already explored')

    def reached_callback(self, msg):
        """Control node reports that a waypoint was reached.

        This is where we count an actual visit to the cell. We validate that
        the reported position matches the dispatched cell — otherwise a stale
        message could credit a visit to the wrong cell.
        """
        self.waiting_for_reach = False

        if self.sent_waypoint_row is None or self.sent_waypoint_col is None:
            return

        r, c = self.sent_waypoint_row, self.sent_waypoint_col
        self.sent_waypoint_row = None
        self.sent_waypoint_col = None

        if not self.in_bounds(r, c):
            return

        # Sanity-check: does the reported (x,y) match the cell we sent?
        try:
            rx_str, ry_str = msg.data.split(',')
            rx, ry = float(rx_str), float(ry_str)
            wx, wy = self.grid_to_world(r, c)
            if math.hypot(rx - wx, ry - wy) > self.resolution:
                self.get_logger().warn(
                    f'Stale /waypoint_reached ({rx:.2f},{ry:.2f}) '
                    f'!= cell ({r},{c}) world=({wx:.2f},{wy:.2f}) — ignoring'
                )
                return
        except (ValueError, AttributeError):
            # Older messages without coords — fall through and credit the visit
            pass

        self.visit_count[r][c] += 1
        self.get_logger().info(
            f'Cell ({r},{c}) visited — '
            f'count={self.visit_count[r][c]}/'
            f'{self.required_visits[r][c]}'
        )

    def status_callback(self, msg):
        """Control node reports stuck or skipped — drop the wait state."""
        data = msg.data
        if data.startswith('stuck'):
            self.waiting_for_reach = False
            self.get_logger().warn('Robot stuck — skipping waypoint and replanning')
            self.sent_waypoint_row = None
            self.sent_waypoint_col = None
        elif data.startswith('skipped'):
            self.waiting_for_reach = False
            self.get_logger().info('Waypoint skipped by control node')
            self.sent_waypoint_row = None
            self.sent_waypoint_col = None

    # ------------------------------------------------------------------ #
    #  Spanning tree
    # ------------------------------------------------------------------ #

    def get_neighbors(self, row, col):
        """Return traversable (FREE) 4-connected neighbors."""
        neighbors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = row + dr, col + dc
            if self.in_bounds(nr, nc) and self.cell_state[nr][nc] == CellState.FREE:
                neighbors.append((nr, nc))
        return neighbors

    def build_spanning_tree(self, start_row, start_col):
        """BFS spanning tree from the robot's current position."""
        self.parent = {}
        visited = set()
        queue = deque()
        queue.append((start_row, start_col))
        visited.add((start_row, start_col))
        self.parent[(start_row, start_col)] = None

        while queue:
            row, col = queue.popleft()
            for nr, nc in self.get_neighbors(row, col):
                if (nr, nc) not in visited:
                    visited.add((nr, nc))
                    self.parent[(nr, nc)] = (row, col)
                    queue.append((nr, nc))

        self.last_tree_size = len(self.parent)
        self.get_logger().info(f'Spanning tree: {len(self.parent)} cells')

    def generate_coverage_path(self, start_row, start_col):
        """Iterative DFS traversal of the spanning tree.

        Yellow cells (required_visits == 2) are inserted into the path a
        second time so the robot physically revisits them.
        """
        if not self.parent:
            return []

        # Build children map
        children = {}
        for node, par in self.parent.items():
            if par is not None:
                children.setdefault(par, []).append(node)

        path = []
        visited = set()

        # Iterative DFS using an explicit stack.
        # Stack items: (node, child_index)
        stack = [((start_row, start_col), 0)]
        visited.add((start_row, start_col))
        path.append((start_row, start_col))

        while stack:
            node, ci = stack[-1]
            kids = children.get(node, [])

            if ci < len(kids):
                # Move to the next unvisited child
                stack[-1] = (node, ci + 1)
                child = kids[ci]
                if child not in visited:
                    visited.add(child)
                    path.append(child)
                    stack.append((child, 0))
            else:
                # All children processed — backtrack
                stack.pop()
                if stack:
                    path.append(stack[-1][0])  # return to parent

        # Add extra visits for cells that still need passes after this path.
        # Compare planned occurrences against required_visits and any visits
        # that have already happened — only top up the difference, so we
        # never over-visit yellow cells.
        path_counts = {}
        for cell in path:
            path_counts[cell] = path_counts.get(cell, 0) + 1

        yellow_extra = []
        for cell, planned in path_counts.items():
            r, c = cell
            already = int(self.visit_count[r][c])
            required = int(self.required_visits[r][c])
            shortfall = required - (already + planned)
            if shortfall > 0:
                yellow_extra.extend([cell] * shortfall)

        if yellow_extra:
            path.extend(yellow_extra)

        return path

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #

    def needs_more_visits(self, row, col):
        if self.cell_state[row][col] in (
            CellState.OBSTACLE, CellState.RED, CellState.UNKNOWN
        ):
            return False
        return self.visit_count[row][col] < self.required_visits[row][col]

    # ------------------------------------------------------------------ #
    #  Main planning loop
    # ------------------------------------------------------------------ #

    def planning_loop(self):
        if self.robot_row is None:
            return

        # Don't send a new waypoint while waiting for the last one
        if self.waiting_for_reach:
            return

        row, col = self.robot_row, self.robot_col
        if not self.in_bounds(row, col):
            return

        # Mark current cell free if unknown
        if self.cell_state[row][col] == CellState.UNKNOWN:
            self.cell_state[row][col] = CellState.FREE

        # Count free cells to decide if tree needs rebuilding
        free_count = int(np.sum(self.cell_state == CellState.FREE))

        if free_count == 0:
            self.publish_metrics()
            return

        # Rebuild only on first run, when the path is exhausted, or when the
        # frontier has actually grown a lot — avoids thrashing on every lidar
        # scan.
        need_rebuild = (
            not self.tree_built
            or self.current_path_index >= len(self.coverage_path)
            or free_count > max(self.last_tree_size + 10,
                                int(self.last_tree_size * 1.5))
        )

        if need_rebuild:
            self.build_spanning_tree(row, col)
            self.coverage_path = self.generate_coverage_path(row, col)
            self.current_path_index = 0
            self.tree_built = True

        # Follow coverage path
        if self.current_path_index < len(self.coverage_path):
            # Skip red / obstacle / already-complete cells
            while self.current_path_index < len(self.coverage_path):
                tr, tc = self.coverage_path[self.current_path_index]
                if self.cell_state[tr][tc] in (CellState.RED, CellState.OBSTACLE):
                    self.current_path_index += 1
                    continue
                if not self.needs_more_visits(tr, tc):
                    self.current_path_index += 1
                    continue
                break
            else:
                # Exhausted the path — check for remaining uncovered cells below
                self.current_path_index = len(self.coverage_path)

            if self.current_path_index < len(self.coverage_path):
                target_row, target_col = self.coverage_path[self.current_path_index]
                self.publish_waypoint(target_row, target_col)
                self.current_path_index += 1

        # If path exhausted, check for remaining uncovered cells
        if self.current_path_index >= len(self.coverage_path):
            uncovered = []
            for r in range(self.grid_h):
                for c in range(self.grid_w):
                    if self.needs_more_visits(r, c):
                        uncovered.append((r, c))

            if uncovered:
                nearest = min(
                    uncovered,
                    key=lambda cell: abs(cell[0] - row) + abs(cell[1] - col)
                )
                self.build_spanning_tree(row, col)
                self.coverage_path = self.generate_coverage_path(row, col)
                self.current_path_index = 0
                self.get_logger().info(
                    f'Replanning: {len(uncovered)} cells still need coverage'
                )

        self.publish_metrics()

    # ------------------------------------------------------------------ #
    #  Publishers
    # ------------------------------------------------------------------ #

    def publish_waypoint(self, target_row, target_col):
        """Send a waypoint to the control node and notify the metrics node."""
        wx, wy = self.grid_to_world(target_row, target_col)

        # Waypoint for control node
        waypoint = PoseStamped()
        waypoint.header.frame_id = 'odom'
        waypoint.header.stamp = self.get_clock().now().to_msg()
        waypoint.pose.position.x = wx
        waypoint.pose.position.y = wy
        waypoint.pose.position.z = 0.0
        waypoint.pose.orientation.w = 1.0
        self.waypoint_pub.publish(waypoint)

        # Zone info for metrics node
        color = self.zone_color[target_row][target_col]
        zone_msg = String()
        zone_msg.data = f'waypoint_zone:{wx:.3f},{wy:.3f},{color}'
        self.waypoint_zone_pub.publish(zone_msg)

        # Track what we sent
        self.waiting_for_reach = True
        self.sent_waypoint_row = target_row
        self.sent_waypoint_col = target_col

        self.get_logger().info(
            f'Waypoint -> ({target_row},{target_col}) '
            f'[{color}] world=({wx:.2f},{wy:.2f})'
        )

    def publish_metrics(self):
        total_required = 0
        total_completed = 0

        for r in range(self.grid_h):
            for c in range(self.grid_w):
                if self.cell_state[r][c] in (
                    CellState.OBSTACLE, CellState.RED, CellState.UNKNOWN
                ):
                    continue
                required = self.required_visits[r][c]
                completed = min(self.visit_count[r][c], required)
                total_required += required
                total_completed += completed

        if total_required > 0:
            coverage_pct = (total_completed / total_required) * 100.0
        else:
            coverage_pct = 0.0

        msg = String()
        msg.data = (
            f'coverage:{coverage_pct:.1f}'
            f'|required:{total_required}'
            f'|completed:{total_completed}'
        )
        self.metrics_pub.publish(msg)

    def publish_grid_viz(self):
        markers = MarkerArray()

        for r in range(self.grid_h):
            for c in range(self.grid_w):
                if self.cell_state[r][c] == CellState.UNKNOWN:
                    continue

                marker = Marker()
                marker.header.frame_id = 'odom'
                marker.header.stamp = self.get_clock().now().to_msg()
                marker.ns = 'coverage_grid'
                marker.id = r * self.grid_w + c
                marker.type = Marker.CUBE
                marker.action = Marker.ADD

                wx, wy = self.grid_to_world(r, c)
                marker.pose.position.x = wx
                marker.pose.position.y = wy
                marker.pose.position.z = 0.01
                marker.scale.x = self.resolution * 0.9
                marker.scale.y = self.resolution * 0.9
                marker.scale.z = 0.02
                marker.color.a = 0.6

                color = self.zone_color[r][c]
                if self.cell_state[r][c] == CellState.OBSTACLE:
                    marker.color.r = 0.3
                    marker.color.g = 0.3
                    marker.color.b = 0.3
                elif color == 'red':
                    marker.color.r = 1.0
                    marker.color.g = 0.0
                    marker.color.b = 0.0
                elif color == 'yellow':
                    marker.color.r = 1.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                elif color == 'green':
                    marker.color.r = 0.0
                    marker.color.g = 1.0
                    marker.color.b = 0.0
                elif color == 'blue':
                    marker.color.r = 0.0
                    marker.color.g = 0.0
                    marker.color.b = 1.0
                else:
                    marker.color.r = 0.8
                    marker.color.g = 0.8
                    marker.color.b = 0.8

                # Dim cells that are fully covered
                if (not self.needs_more_visits(r, c)
                        and self.cell_state[r][c] == CellState.FREE):
                    marker.color.a = 0.3

                markers.markers.append(marker)

        self.grid_viz_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = PlannerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()