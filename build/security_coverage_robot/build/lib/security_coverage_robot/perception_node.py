import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge
import cv2
import numpy as np
from collections import deque


class PerceptionNode(Node):
    def __init__(self):
        super().__init__('perception_node')
        self.bridge = CvBridge()

        # Subscribe to the robot's camera
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publish detected zone color
        self.zone_pub = self.create_publisher(String, '/zone_color', 10)

        # HSV thresholds — declare as parameters so they're tunable
        # NOTE: green lower = 36 to avoid overlap with yellow upper = 35
        self.declare_parameter('green_lower', [36, 80, 80])
        self.declare_parameter('green_upper', [85, 255, 255])
        self.declare_parameter('yellow_lower', [20, 80, 80])
        self.declare_parameter('yellow_upper', [35, 255, 255])
        self.declare_parameter('red_lower_1', [0, 80, 80])
        self.declare_parameter('red_upper_1', [10, 255, 255])
        self.declare_parameter('red_lower_2', [170, 80, 80])
        self.declare_parameter('red_upper_2', [180, 255, 255])
        self.declare_parameter('blue_lower', [100, 80, 80])
        self.declare_parameter('blue_upper', [130, 255, 255])

        # Minimum percentage of pixels to classify a zone
        self.declare_parameter('min_detection_ratio', 0.15)

        # Number of frames for majority-vote debouncing
        self.declare_parameter('debounce_frames', 5)

        # Floor region: fraction of image height to crop from top
        self.declare_parameter('floor_crop_ratio', 0.6)

        # Cache parameters (avoids fetching every frame)
        self._cache_parameters()

        # Register callback for runtime parameter changes
        self.add_on_set_parameters_callback(self._on_param_change)

        # Debounce buffer
        buffer_size = self.get_parameter('debounce_frames').value
        self.detection_buffer = deque(maxlen=buffer_size)

        # Track last published zone to only publish on change
        self.last_published_zone = 'unknown'

        self.get_logger().info('Perception node started — listening on /camera/image_raw')

    # ------------------------------------------------------------------ #
    #  Parameter management
    # ------------------------------------------------------------------ #

    def _cache_parameters(self):
        """Read all HSV parameters once and store as numpy arrays."""
        self.green_lower = np.array(
            self.get_parameter('green_lower').value, dtype=np.uint8
        )
        self.green_upper = np.array(
            self.get_parameter('green_upper').value, dtype=np.uint8
        )
        self.yellow_lower = np.array(
            self.get_parameter('yellow_lower').value, dtype=np.uint8
        )
        self.yellow_upper = np.array(
            self.get_parameter('yellow_upper').value, dtype=np.uint8
        )
        self.red_lower_1 = np.array(
            self.get_parameter('red_lower_1').value, dtype=np.uint8
        )
        self.red_upper_1 = np.array(
            self.get_parameter('red_upper_1').value, dtype=np.uint8
        )
        self.red_lower_2 = np.array(
            self.get_parameter('red_lower_2').value, dtype=np.uint8
        )
        self.red_upper_2 = np.array(
            self.get_parameter('red_upper_2').value, dtype=np.uint8
        )
        self.blue_lower = np.array(
            self.get_parameter('blue_lower').value, dtype=np.uint8
        )
        self.blue_upper = np.array(
            self.get_parameter('blue_upper').value, dtype=np.uint8
        )
        self.min_ratio = self.get_parameter('min_detection_ratio').value
        self.floor_crop = self.get_parameter('floor_crop_ratio').value

    def _on_param_change(self, params):
        """Re-cache when parameters are changed at runtime."""
        self._cache_parameters()
        self.get_logger().info('Parameters updated — HSV thresholds re-cached')
        return rclpy.parameter.SetParametersResult(successful=True)

    # ------------------------------------------------------------------ #
    #  Image processing
    # ------------------------------------------------------------------ #

    def image_callback(self, msg):
        """Process camera frame, detect floor color, publish zone on change."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')
            return

        # Crop to the floor region (bottom portion of the image)
        height = cv_image.shape[0]
        floor_region = cv_image[int(height * self.floor_crop):, :]

        # Convert to HSV
        hsv = cv2.cvtColor(floor_region, cv2.COLOR_BGR2HSV)
        total_pixels = hsv.shape[0] * hsv.shape[1]

        if total_pixels == 0:
            return

        # Detect each color using cached thresholds
        detections = {}

        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        detections['green'] = cv2.countNonZero(green_mask) / total_pixels

        yellow_mask = cv2.inRange(hsv, self.yellow_lower, self.yellow_upper)
        detections['yellow'] = cv2.countNonZero(yellow_mask) / total_pixels

        red_mask = (
            cv2.inRange(hsv, self.red_lower_1, self.red_upper_1)
            | cv2.inRange(hsv, self.red_lower_2, self.red_upper_2)
        )
        detections['red'] = cv2.countNonZero(red_mask) / total_pixels

        blue_mask = cv2.inRange(hsv, self.blue_lower, self.blue_upper)
        detections['blue'] = cv2.countNonZero(blue_mask) / total_pixels

        # Pick the color with the highest ratio above the threshold
        frame_zone = 'unknown'
        max_ratio = 0.0

        for color, ratio in detections.items():
            if ratio > self.min_ratio and ratio > max_ratio:
                max_ratio = ratio
                frame_zone = color

        # Add to debounce buffer
        self.detection_buffer.append(frame_zone)

        # Majority vote over the buffer
        stable_zone = self._majority_vote()

        # Only publish when the zone actually changes
        if stable_zone != self.last_published_zone:
            msg_out = String()
            msg_out.data = stable_zone
            self.zone_pub.publish(msg_out)

            if stable_zone != 'unknown':
                self.get_logger().info(
                    f'Zone changed: {self.last_published_zone} -> {stable_zone} '
                    f'({max_ratio:.1%} of floor pixels)'
                )

            self.last_published_zone = stable_zone

    def _majority_vote(self):
        """Return the most common detection in the buffer.

        If the buffer isn't full yet, return the latest detection.
        Ties are broken in favor of the current zone (avoids flicker).
        """
        if len(self.detection_buffer) == 0:
            return 'unknown'

        # Count occurrences
        counts = {}
        for zone in self.detection_buffer:
            counts[zone] = counts.get(zone, 0) + 1

        # Find the max count
        max_count = max(counts.values())
        candidates = [z for z, c in counts.items() if c == max_count]

        # If current zone is among the tied candidates, keep it (stability)
        if self.last_published_zone in candidates:
            return self.last_published_zone

        return candidates[0]


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()