#!/usr/bin/env python3
# Author: Amy Phung

# Tell python where to find apriltags_rgbd and utils code
import sys
import os
fpath = os.path.join(os.path.dirname(__file__), "utils")
sys.path.append(fpath)

# Python Imports
import numpy as np
import cv2
from threading import Thread, Lock
mutex = Lock()

# ROS Imports
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped, Vector3
from sensor_msgs.msg import Image, CameraInfo, PointCloud
from apriltag_ros.msg import AprilTagDetectionArray
from message_filters import Subscriber, ApproximateTimeSynchronizer
from cv_bridge import CvBridge, CvBridgeError

class ApriltagsRgbdNode():
    def __init__(self):
        rospy.init_node("apriltags_rgbd")
        self.rate = rospy.Rate(60) # 10 Hz

        # CV bridge
        self.bridge = CvBridge()

        # Subscribers
        tss = ApproximateTimeSynchronizer([
            Subscriber("/kinect2/hd/camera_info", CameraInfo),
            Subscriber("/kinect2/hd/image_color_rect", Image),
            Subscriber("/kinect2/hd/image_depth_rect", Image),
            Subscriber("/tag_detections", AprilTagDetectionArray)], 1 ,0.5)
        tss.registerCallback(self.tagCallback)

        # ROS tf
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Publishers
        self.tag_tf_pub = rospy.Publisher("/apriltags_rgbd/tag_tfs", TransformStamped, queue_size=10)

    def tagCallback(self, camera_info_data, rgb_data, depth_data, tag_data):
        with mutex:
            camera_info_data = camera_info_data
            rgb_data = rgb_data
            depth_data = depth_data
            tag_data = tag_data

            # Convert ROS images to OpenCV frames
            try:
                rgb_image = self.bridge.imgmsg_to_cv2(rgb_data, "bgr8")
                depth_image = self.bridge.imgmsg_to_cv2(depth_data, "16UC1")
            except CvBridgeError as e:
                print(e)
                return

            # Extract metadata
            header = camera_info_data.header
            k_mtx = np.array(camera_info_data.K).reshape(3,3)

            # Create messages for point info
            tag_pts = PointCloud()
            tag_pts.header = header

            # Estimate pose of each tag
            for tag_idx, tag in enumerate(tag_data.detections):
                tag_id, image_pts = self.parseTag(tag)

                depth_pts = self.extractDepthPoints(depth_image, image_pts, k_mtx)

                # Update tf tree
                output_tf = TransformStamped()
                output_tf.header = header
                output_tf.child_frame_id = str(tag_id)

                # Estimate tag position based on average depth measurement
                if len(depth_pts) == 0:
                    rospy.logwarn_throttle(2, "No depth info found for tag " + str(tag_id))
                    continue
                output_tf.transform.translation = Vector3(*np.mean(depth_pts, axis=0))

                # Estimate tag orientation based on apriltag detection and camera intrinsics 
                output_tf.transform.rotation = tag.pose.pose.pose.orientation

                self.tf_broadcaster.sendTransform(output_tf)
                self.tag_tf_pub.publish(output_tf)

            self.rate.sleep()

    def extractDepthPoints(self, depth_image, image_pts, K):
        ## Generate the depth samples from the depth image
        fx = K[0][0]
        fy = K[1][1]
        px = K[0][2]
        py = K[1][2]
        rows, cols = depth_image.shape
        hull_pts = image_pts.reshape(4,1,2).astype(int)
        rect = cv2.convexHull(hull_pts)
        all_pts = []
        xcoord = image_pts[:, 0]
        ycoord = image_pts[: ,1]
        xmin = int(np.amin(xcoord))
        xmax = int(np.amax(xcoord))
        ymin = int(np.amin(ycoord))
        ymax = int(np.amax(ycoord))
        for j in range(ymin, ymax):
            for i in range(xmin, xmax):
                if (cv2.pointPolygonTest(rect, (i,j), False) > 0):
                    depth = depth_image[j,i] / 1000.0
                    if(depth != 0):
                        x = (i - px) * depth / fx
                        y = (j - py) * depth / fy
                        all_pts.append([x,y,depth])
        samples_depth = np.array(all_pts)
        return samples_depth

    def parseTag(self, tag):
        id = str(tag.id[0])

        im_pt0 = tag.pix_tl
        im_pt1 = tag.pix_tr
        im_pt2 = tag.pix_br
        im_pt3 = tag.pix_bl

        im_pts = im_pt0 + im_pt1 + im_pt2 + im_pt3
        image_pts = np.array(im_pts).reshape(4,2)

        return id, image_pts

    def loop(self):
        self.rate.sleep()

if __name__ == '__main__':
    node = ApriltagsRgbdNode()

    while not rospy.is_shutdown():
        with mutex:
            node.loop()
