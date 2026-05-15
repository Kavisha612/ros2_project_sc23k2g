# Exercise 4 - following a colour (green) and stopping upon sight of another (blue).

#from __future__ import division
import threading
import sys, time
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Vector3
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav2_msgs.action import NavigateToPose
from rclpy.action import ActionClient
from math import sin, cos
from rclpy.exceptions import ROSInterruptException
import signal


class Robot(Node):
    def __init__(self):
        super().__init__('robot')
        # translates ROS2 image messages into OpenCV images
        self.bridge = CvBridge()
        self.sensitivity = 10
        # Subscribe to camera topic - triggers callback() every new frame
        self.subscription = self.create_subscription(Image, '/camera/image_raw', self.callback, 10)
        # Publisher to send movement commands
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        

        # Initialise any flags that signal a colour has been detected (default to false)
        self.blue_detected = False
        self.task_complete = False
        self.blue_contour_detected = False
        
        # Action client to send navigation goals to Nav2
        self.nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
    
    
    
    def callback(self, data):

        try: #incase image conversion is corrupted
            # Convert the ROS2 image to OpenCV format
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            self.get_logger().error(str(e))
            return
        
        # COLOUR RANGES
        
        # Greem hue = 60
        hsv_green_lower = np.array([60 - self.sensitivity, 100, 100])
        hsv_green_upper = np.array([60 + self.sensitivity, 255, 255])
        
        # Blue hue = 120
        hsv_blue_lower = np.array([120 - self.sensitivity, 100, 100])
        hsv_blue_upper = np.array([120 + self.sensitivity, 255, 255])
        
        # Red wraps around so two ranges needed
        hsv_red1_lower = np.array([0, 100, 100])
        hsv_red1_upper = np.array([0 + self.sensitivity, 255, 255])
        
        hsv_red2_lower = np.array([180 - self.sensitivity, 100, 100])
        hsv_red2_upper = np.array([180, 255, 255])
        
        # Convert the rgb image into a hsv image
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Masks
        # inRange creates a black/white image where white = colour detected
        green_mask = cv2.inRange(hsv_image, hsv_green_lower, hsv_green_upper)
        blue_mask = cv2.inRange(hsv_image, hsv_blue_lower, hsv_blue_upper)
        red1_mask = cv2.inRange(hsv_image, hsv_red1_lower, hsv_red1_upper)
        red2_mask = cv2.inRange(hsv_image, hsv_red2_lower, hsv_red2_upper)
        
        # combine red masks
        red_mask = cv2.bitwise_or(red1_mask, red2_mask)
        
        # add in blue
        combined1 = cv2.bitwise_or(red_mask, blue_mask)
        
        # add in green
        combined = cv2.bitwise_or(combined1, green_mask)
        # bitwise_or means: white if pixel matches ANY of the colours


        # Apply combined mask to original image
        # Result: only red, green, blue pixels visible - everything else bla
        filtered_image = cv2.bitwise_and(image, image, mask = combined)
        
        # Show both windows for the video recording
        # Camera window shows raw feed, Filtered shows colour detection working
        cv2.namedWindow('camera_Feed',cv2.WINDOW_NORMAL)
        cv2.imshow('Camera', image)
        cv2.imshow('Filtered', filtered_image)
        cv2.resizeWindow('camera_Feed',320,240)
        cv2.waitKey(3)
        

        

        # Find the contours that appear within the certain colour mask using the cv2.findContours() method
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # Loop over the contours
        if len(contours)>0:
            # Get the biggest contour
            c = max(contours, key=cv2.contourArea)
            # Only count it if area is big enough to be a real object not noise
            if cv2.contourArea(c) > 500: #<What do you think is a suitable area?>
                # Alter the value of the flag
                self.blue_detected = True
                
                # Store contour so approach logic can use it to steer and judge distance
                self.blue_contour = c
                
                # Draw bounding box around blue object for video evidence
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        # Red contours
        red_contours, _ = cv2.findContours(red_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(red_contours) > 0:
            rc = max(red_contours, key=cv2.contourArea)
            if cv2.contourArea(rc) > 500:
                x, y, w, h = cv2.boundingRect(rc)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
        # Green contours
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(green_contours) > 0:
            gc = max(green_contours, key=cv2.contourArea)
            if cv2.contourArea(gc) > 500:
                x, y, w, h = cv2.boundingRect(gc)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

       

   

    def stop(self):
        # Use what you learnt in lab 3 to make the robot stop
        desired_velocity = Twist()  # all zeros by default = stop
        self.publisher.publish(desired_velocity)
        
    def go_to_waypoint(self, x, y, yaw=0.0):
        # Build the goal message
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = 'map'
        goal.pose.pose.position.x = float(x)
        goal.pose.pose.position.y = float(y)
        goal.pose.pose.orientation.z = sin(yaw / 2)
        goal.pose.pose.orientation.w = cos(yaw / 2)
        
        # Wait for Nav2 to be ready then send goal
        self.nav_client.wait_for_server()
        self.nav_client.send_goal_async(goal)

# Create a node of your class in the main and ensure it stays up and running
# handling exceptions and such
def main():
    def signal_handler(sig, frame):
        robot.stop()
        rclpy.shutdown()

    # Instantiate your class
    # And rclpy.init the entire node
    rclpy.init(args=None)
    robot = Robot()
    


    signal.signal(signal.SIGINT, signal_handler)
    thread = threading.Thread(target=rclpy.spin, args=(robot,), daemon=True)
    thread.start()

    try:
        while rclpy.ok():
            pass

    except ROSInterruptException:
        pass

    # Remember to destroy all image windows before closing node
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
