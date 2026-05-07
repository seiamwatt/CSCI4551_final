import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from nav_msgs.msg import Odometry
import math


class MetricsNode(Node):
    def __init__(self):
        super().__init__('metrics_node')

        # Use ROS clock for simulation compatibility
        self.start_time = self.get_clock().now()

        # Distance tracking
        self.total_distance = 0.0
        self.prev_x = None
        self.prev_y = None

        # Coverage data from planner
        self.coverage_pct = 0.0
        self.total_required = 0
        self.total_completed = 0

        # Zone behavior tracking
        self.correct_behaviors = 0
        self.total_behaviors = 0

        # Current zone
        self.current_zone = 'unknown'

        # Visit counts per zone type
        self.zone_stats = {
            'green': {'entered': 0, 'correct': 0},
            'yellow': {'entered': 0, 'correct': 0},
            'red': {'entered': 0, 'violations': 0},
            'blue': {'entered': 0, 'correct': 0},
        }

        # Track how many times each yellow cell has been visited
        # Key: "x,y" string from waypoint, Value: visit count
        self.yellow_cell_visits = {}

        # Waypoint tracking
        self.waypoints_reached = 0
        self.waypoints_stuck = 0
        self.waypoints_skipped = 0

        # Most recent dispatched waypoint waiting on a reach/stuck/skip event.
        # Stored as (x_str, y_str, color) so we credit correctness based on
        # what *actually* happened, not just what the planner asked for.
        self.pending_waypoint = None

        # Red zone coverage attempts (violations)
        self.red_coverage_attempts = 0

        # --- Subscribers ---
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10
        )
        self.metrics_sub = self.create_subscription(
            String, '/coverage_metrics', self.coverage_callback, 10
        )
        self.zone_sub = self.create_subscription(
            String, '/zone_color', self.zone_callback, 10
        )
        self.reached_sub = self.create_subscription(
            String, '/waypoint_reached', self.reached_callback, 10
        )
        self.status_sub = self.create_subscription(
            String, '/control_status', self.status_callback, 10
        )
        # Planner publishes zone info per waypoint:
        #   "waypoint_zone:<x>,<y>,<color>"
        self.waypoint_zone_sub = self.create_subscription(
            String, '/waypoint_zone', self.waypoint_zone_callback, 10
        )

        # Publisher for full metrics report
        self.report_pub = self.create_publisher(String, '/metrics_report', 10)

        # Timer to print metrics every 5 seconds
        self.report_timer = self.create_timer(5.0, self.publish_report)

        self.get_logger().info('Metrics node started')

    # ------------------------------------------------------------------ #
    #  Callbacks
    # ------------------------------------------------------------------ #

    def odom_callback(self, msg):
        """Track total distance traveled."""
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y

        if self.prev_x is not None and self.prev_y is not None:
            dist = math.hypot(x - self.prev_x, y - self.prev_y)
            # Only count meaningful movement to filter odometry noise
            if dist > 0.01:
                self.total_distance += dist

        self.prev_x = x
        self.prev_y = y

    def coverage_callback(self, msg):
        """Parse coverage metrics from planner node.

        Expected format: "coverage:XX.X|required:N|completed:N"
        """
        try:
            parts = msg.data.split('|')
            for part in parts:
                key, value = part.split(':')
                if key == 'coverage':
                    self.coverage_pct = float(value)
                elif key == 'required':
                    self.total_required = int(value)
                elif key == 'completed':
                    self.total_completed = int(value)
        except (ValueError, IndexError) as e:
            self.get_logger().warn(f'Failed to parse coverage metrics: {e}')

    def zone_callback(self, msg):
        """Track zone transitions detected by perception node."""
        new_zone = msg.data.strip().lower()
        if new_zone not in ('green', 'yellow', 'red', 'blue'):
            return

        # Only act on transitions
        if new_zone == self.current_zone:
            return

        self.current_zone = new_zone
        self.zone_stats[new_zone]['entered'] += 1

    def waypoint_zone_callback(self, msg):
        """Record zone behavior when the robot reaches a cell.

        Expected format: "waypoint_zone:<x>,<y>,<color>"

        Since the planner now publishes this AFTER the robot arrives,
        we credit correctness immediately here.
        """
        try:
            _, payload = msg.data.split(':', 1)
            parts = payload.split(',')
            x_str = parts[0].strip()
            y_str = parts[1].strip()
            color = parts[2].strip().lower()
        except (ValueError, IndexError):
            return

        cell_key = f'{x_str},{y_str}'
        self.total_behaviors += 1

        if color == 'red':
            self.red_coverage_attempts += 1
            self.zone_stats['red']['violations'] += 1
            self.get_logger().warn(
                f'VIOLATION: Robot visited RED zone at ({x_str}, {y_str})'
            )
        elif color == 'green':
            self.zone_stats['green']['correct'] += 1
            self.correct_behaviors += 1
        elif color == 'yellow':
            count = self.yellow_cell_visits.get(cell_key, 0) + 1
            self.yellow_cell_visits[cell_key] = count
            if count <= 2:
                self.zone_stats['yellow']['correct'] += 1
                self.correct_behaviors += 1
            else:
                self.get_logger().warn(
                    f'Yellow cell ({x_str}, {y_str}) visited {count} times (expected 2)'
                )
        elif color == 'blue':
            self.zone_stats['blue']['correct'] += 1
            self.correct_behaviors += 1
        else:
            # Unknown color — count the decision but not as correct.
            pass

    def reached_callback(self, msg):
        """Track waypoint reach count."""
        self.waypoints_reached += 1

    def status_callback(self, msg):
        """Track stuck or skipped waypoints from control node."""
        if msg.data.startswith('stuck'):
            self.waypoints_stuck += 1
            self.pending_waypoint = None
            self.get_logger().warn(f'Robot got stuck: {msg.data}')
        elif msg.data.startswith('skipped'):
            self.waypoints_skipped += 1
            self.pending_waypoint = None

    # ------------------------------------------------------------------ #
    #  Report
    # ------------------------------------------------------------------ #

    def publish_report(self):
        """Publish a full metrics report."""
        now = self.get_clock().now()
        elapsed_ns = (now - self.start_time).nanoseconds
        elapsed_s = elapsed_ns / 1e9
        minutes = int(elapsed_s // 60)
        seconds = int(elapsed_s % 60)

        # Behavior correctness percentage
        if self.total_behaviors > 0:
            behavior_pct = (self.correct_behaviors / self.total_behaviors) * 100.0
        else:
            behavior_pct = 100.0

        # Efficiency: distance per completed cell
        if self.total_completed > 0:
            dist_per_cell = self.total_distance / self.total_completed
        else:
            dist_per_cell = 0.0

        # Yellow zone completion check
        yellow_complete = sum(
            1 for v in self.yellow_cell_visits.values() if v >= 2
        )
        yellow_total = len(self.yellow_cell_visits)

        # Build report
        report_lines = [
            '========== COVERAGE METRICS ==========',
            f'Time elapsed:       {minutes}m {seconds}s',
            f'Distance traveled:  {self.total_distance:.2f} m',
            '',
            f'Coverage:           {self.coverage_pct:.1f}%',
            f'  Required passes:  {self.total_required}',
            f'  Completed:        {self.total_completed}',
            '',
            f'Waypoint stats:',
            f'  Reached:          {self.waypoints_reached}',
            f'  Stuck/failed:     {self.waypoints_stuck}',
            f'  Skipped (red):    {self.waypoints_skipped}',
            '',
            f'Behavior correctness: {behavior_pct:.1f}%',
            f'  Total decisions:    {self.total_behaviors}',
            f'  Correct:            {self.correct_behaviors}',
            f'  Red violations:     {self.red_coverage_attempts}',
            '',
            f'Zone breakdown:',
            f'  Green  - entered: {self.zone_stats["green"]["entered"]}, '
            f'correct: {self.zone_stats["green"]["correct"]}',
            f'  Yellow - entered: {self.zone_stats["yellow"]["entered"]}, '
            f'correct: {self.zone_stats["yellow"]["correct"]}, '
            f'cells 2x covered: {yellow_complete}/{yellow_total}',
            f'  Red    - entered: {self.zone_stats["red"]["entered"]}, '
            f'violations: {self.zone_stats["red"]["violations"]}',
            f'  Blue   - entered: {self.zone_stats["blue"]["entered"]}, '
            f'correct: {self.zone_stats["blue"]["correct"]}',
            '',
            f'Efficiency:',
            f'  Distance per cell: {dist_per_cell:.2f} m/cell',
            '======================================='
        ]

        report = '\n'.join(report_lines)

        # Publish
        msg = String()
        msg.data = report
        self.report_pub.publish(msg)

        # Also log to terminal
        self.get_logger().info('\n' + report)


def main(args=None):
    rclpy.init(args=args)
    node = MetricsNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()