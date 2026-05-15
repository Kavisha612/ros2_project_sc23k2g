#!/usr/bin/env python3

import threading
import time
import signal
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.exceptions import ROSInterruptException
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from nav2_msgs.action import NavigateToPose
from cv_bridge import CvBridge, CvBridgeError
from math import sin, cos


# Three waypoints covering each compartment of the map
ZONE_A = (4.16, -1.76, -0.00143)
ZONE_B = (-0.5, -4.36,  0.00247)
ZONE_C = (3.91, -9.03,  0.00247)
SEARCH_ZONES = [ZONE_A, ZONE_B, ZONE_C]

# Area thresholds used to decide when to act on a blue detection
MIN_BLOB_AREA    = 500
NAV_CANCEL_AREA  = 5000
SCAN_CANCEL_AREA = 2000
STOP_AREA        = 270000
CENTRE_TOLERANCE = 30
SWEEP_SPEED      = 0.35


class Robot(Node):

    def __init__(self):
        super().__init__('robot')

        self.bridge      = CvBridge()
        self.sensitivity = 10

        self.sub = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.nav = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        # tracks which colours the robot has seen across all frames
        self.seen = {'red': False, 'green': False, 'blue': False}

        # blue target info, reset every frame
        self.blue_visible = False
        self.blue_area    = 0.0
        self.blue_cx      = None
        self.img_width    = 640

        self.done = False

    def get_masks(self, hsv):
        s      = self.sensitivity
        kernel = np.ones((5, 5), np.uint8)

        blue  = cv2.inRange(hsv, np.array([120-s, 100, 100]), np.array([120+s, 255, 255]))
        green = cv2.inRange(hsv, np.array([60-s,  100, 100]), np.array([60+s,  255, 255]))
        r1    = cv2.inRange(hsv, np.array([0,      100, 100]), np.array([s,     255, 255]))
        r2    = cv2.inRange(hsv, np.array([180-s,  100, 100]), np.array([180,   255, 255]))
        # red wraps around 180 in HSV so we need two ranges combined
        red   = cv2.bitwise_or(r1, r2)

        result = {}
        for name, m in [('blue', blue), ('green', green), ('red', red)]:
            # open removes small noise, close fills small gaps
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN,  kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)
            result[name] = m

        return result

    def biggest_blob(self, mask):
        # find all contours and return info about the largest one
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c    = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < MIN_BLOB_AREA:
            return None
        M = cv2.moments(c)
        if M['m00'] == 0:
            return None
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        x, y, w, h = cv2.boundingRect(c)
        return area, cx, cy, x, y, w, h

    def callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return

        self.img_width    = frame.shape[1]

        # reset blue state before processing this frame
        self.blue_visible = False
        self.blue_area    = 0.0
        self.blue_cx      = None

        hsv   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        masks = self.get_masks(hsv)

        combined = cv2.bitwise_or(cv2.bitwise_or(masks['red'], masks['blue']), masks['green'])
        filtered = cv2.bitwise_and(frame, frame, mask=combined)

        colour_style = {
            'blue':  ((255, 0,   0),   'Blue'),
            'green': ((0,   255, 0),   'Green'),
            'red':   ((0,   0,   255), 'Red'),
        }

        for colour, mask in masks.items():
            blob = self.biggest_blob(mask)
            if blob is None:
                continue
            self.seen[colour] = True
            area, cx, cy, x, y, w, h = blob
            bgr, label = colour_style[colour]

            # draw bounding box and centre dot on the raw frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), bgr, 2)
            cv2.circle(frame, (cx, cy), 6, bgr, -1)
            cv2.putText(frame, f'{label}: {int(area)}',
                        (x, max(y-8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr, 2)

            if colour == 'blue':
                self.blue_visible = True
                self.blue_area    = area
                self.blue_cx      = cx

        # show which colours have been spotted so far
        cv2.putText(frame,
                    f"R={self.seen['red']} G={self.seen['green']} B={self.seen['blue']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow('Camera',   frame)
        cv2.imshow('Filtered', filtered)
        cv2.waitKey(3)

    def stop(self):
        self.pub.publish(Twist())

    def go_to_waypoint(self, x, y, yaw=0.0):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id    = 'map'
        goal.pose.header.stamp       = self.get_clock().now().to_msg()
        goal.pose.pose.position.x    = float(x)
        goal.pose.pose.position.y    = float(y)
        goal.pose.pose.position.z    = 0.0
        goal.pose.pose.orientation.z = sin(yaw / 2)
        goal.pose.pose.orientation.w = cos(yaw / 2)

        self.get_logger().info(f'Going to ({x:.2f}, {y:.2f})')
        self.nav.wait_for_server()

        send_future = self.nav.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)

        handle = send_future.result()
        if handle is None or not handle.accepted:
            self.get_logger().warn('Goal rejected.')
            return False

        result_future = handle.get_result_async()

        # keep spinning so camera callbacks stay active while travelling
        while rclpy.ok() and not result_future.done():
            rclpy.spin_once(self, timeout_sec=0.1)
            if self.blue_visible and self.blue_area > NAV_CANCEL_AREA:
                self.get_logger().info('Blue spotted, cancelling nav goal.')
                handle.cancel_goal_async()
                self.stop()
                return True

        return False

    def scan_at_waypoint(self):
        # rotate left then right to check for blue before moving on
        self.get_logger().info('Scanning from current position.')

        sweep = [
            ( SWEEP_SPEED, 2.0, 'Turning left'),
            ( 0.0,         0.4, 'Pausing'),
            (-SWEEP_SPEED, 4.0, 'Turning right'),
            ( 0.0,         0.4, 'Pausing'),
            ( SWEEP_SPEED, 2.0, 'Returning to centre'),
            ( 0.0,         0.4, 'Pausing'),
        ]

        for angular, duration, msg in sweep:
            self.get_logger().info(msg)
            cmd = Twist()
            cmd.angular.z = angular
            start = self.get_clock().now().nanoseconds / 1e9

            while rclpy.ok() and (self.get_clock().now().nanoseconds / 1e9 - start) < duration:
                rclpy.spin_once(self, timeout_sec=0.05)
                if self.blue_visible and self.blue_area > SCAN_CANCEL_AREA:
                    self.stop()
                    self.get_logger().info('Blue found during scan.')
                    return True
                self.pub.publish(cmd)

            self.stop()

        return False

    def approach_blue(self):
        self.get_logger().info('Moving towards blue box.')

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            cmd = Twist()

            # if blue is lost, spin slowly to find it again
            if not self.blue_visible or self.blue_cx is None:
                cmd.angular.z = 0.3
                self.pub.publish(cmd)
                continue

            error   = (self.img_width // 2) - self.blue_cx
            aligned = abs(error) < CENTRE_TOLERANCE

            # stop when close enough and lined up
            if self.blue_area >= STOP_AREA and aligned:
                self.get_logger().info('Stopped at blue box.')
                self.stop()
                self.done = True
                return

            # slow down as we get closer
            cmd.linear.x  = 0.1 if self.blue_area < STOP_AREA else 0.02
            cmd.angular.z = float(error) * 0.006
            cmd.linear.x  = max(-0.15, min(0.15, cmd.linear.x))
            cmd.angular.z = max(-0.5,  min(0.5,  cmd.angular.z))
            self.pub.publish(cmd)


def main():
    def signal_handler(sig, frame):
        robot.stop()
        rclpy.shutdown()

    rclpy.init(args=None)
    robot = Robot()

    signal.signal(signal.SIGINT, signal_handler)

    # spin in background so camera callbacks keep running during nav
    thread = threading.Thread(target=rclpy.spin, args=(robot,), daemon=True)
    thread.start()

    try:
        # give camera time to start publishing before we move
        time.sleep(1.0)

        for x, y, yaw in SEARCH_ZONES:
            if not rclpy.ok() or robot.done:
                break

            found = robot.go_to_waypoint(x, y, yaw)
            if found or robot.blue_visible:
                robot.approach_blue()
                break

            found = robot.scan_at_waypoint()
            if found or robot.blue_visible:
                robot.approach_blue()
                break

        if not robot.done:
            robot.get_logger().info('Search complete, blue box not found.')

    except ROSInterruptException:
        pass

    finally:
        robot.stop()
        cv2.destroyAllWindows()
        robot.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()