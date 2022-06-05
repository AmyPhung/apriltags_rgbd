#!/usr/bin/env python3
# Author: Amy Phung

# Tell python where to find utils code
import sys
import os
fpath = os.path.join(os.path.dirname(__file__), "utils")
sys.path.append(fpath)

# Python Imports
import numpy as np

# ROS Imports
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import PointCloud
import tf.transformations as tr

# Custom Imports
import tf_utils

TF_TIMEOUT = 2 # Maximum age of tfs to use in estimate (in seconds)

class BodyPoseEstimationNode():
    def __init__(self):
        rospy.init_node("body_pose_estimator")
        self.rate = rospy.Rate(10)

        cfg_param = rospy.get_param("~apriltags_rbgd_config")
        self.cfg = self.parseConfig(cfg_param)
        if not self.cfg:
            sys.exit("Invalid configuration")

        self.num_tags = len(self.cfg['tags'])
        self.state = self.initState(self.num_tags)
        self.broadcastGroundTruthTfs()

        self.tf_broadcaster = tf2_ros.TransformBroadcaster()

        # Subscribers
        self.tag_sub = rospy.Subscriber("/apriltags_rgbd/tag_tfs", TransformStamped, self.tagCallback)

        # Private vars
        self.frame_id = ""
        self.bodies = list(set(self.cfg['bodies']))
        self.prev_ts = np.zeros(len(self.bodies))

    def broadcastGroundTruthTfs(self):
        broadcaster = tf2_ros.StaticTransformBroadcaster()
        static_tfs = []
        for i, tag in enumerate(self.cfg['tags']):
            if self.cfg['bodies'][i] == "base":
                se3 = self.cfg["tf_body2tag"][i]
                
                tf_msg = TransformStamped()
                tf_msg.header.stamp = rospy.Time.now()
                tf_msg.header.frame_id = "Base"
                tf_msg.child_frame_id = "ground_truth_" + str(tag)

                tf_msg.transform = tf_utils.se3_to_msg(se3)
                static_tfs.append(tf_msg)

        broadcaster.sendTransform(static_tfs)


    def parseConfig(self, cfg):
        formatted_cfg = {
            "tags": [],
            "sizes": [],
            "bodies": [],
            "tf_tag2body": [],
            "tf_body2tag": [],
            "corner_pts": [], # In body frame
        }
        for body in cfg['bodies']:
            name = list(body.keys())[0]
            if name in formatted_cfg["bodies"]:
                rospy.logerror("Duplicate body " + name + "found")
                return False
            for tag in body[name]['tags']:
                if tag['id'] in formatted_cfg["tags"]:
                    rospy.logerror("Duplicate tag ID " + str(tag['id']) + "found")
                    return False

                formatted_cfg["tags"].append(str(tag['id']))
                formatted_cfg["sizes"].append(tag['size'])
                formatted_cfg["bodies"].append(name)

                # Position of body to tag
                p_b_t = [tag['pose']['position']['x'],
                         tag['pose']['position']['y'],
                         tag['pose']['position']['z']]

                # Rotation from body to tag
                a_b_t = [tag['pose']['rotation']['x'],
                         tag['pose']['rotation']['y'],
                         tag['pose']['rotation']['z']]

                # Compute transformation between tag to body
                X_b_t = tr.compose_matrix(angles=a_b_t, translate=p_b_t)
                formatted_cfg["tf_body2tag"].append(X_b_t)
                X_t_b = tr.inverse_matrix(X_b_t)
                formatted_cfg["tf_tag2body"].append(X_t_b)

                # Compute corners of tag in body frame
                corner_pts = self.computeBodyPts(tag, tag['size'], X_t_b)
                formatted_cfg["corner_pts"].append(corner_pts)

        # Use np array for compute efficiency
        formatted_cfg['tags'] = np.array(formatted_cfg['tags'])
        formatted_cfg['bodies'] = np.array(formatted_cfg['bodies'])
        formatted_cfg['tf_tag2body'] = np.array(formatted_cfg['tf_tag2body'])

        return formatted_cfg

    def initState(self, num_tags):
        state = {
            "timestamps": np.zeros(num_tags),
            "tf_camera2tag": np.empty(num_tags, dtype=object)
        }
        return state

    def getCornerPoints(self, tag_size, use_3d=True):
        """Compute position for apriltag corners in tag frame

        @param:
        - tag_size: width of tag in meters
        - use_3d: whether to return 2d or 3d points. 3d points will set
        z value to 0
        @return corners: list of corners in 0,1,2,3 order according to this
        layout:

           +y
            ^
            |
        3___|___2
        |       |
        |       | ----> +x
        |_______|
        0       1
        """
        tag_radius = tag_size / 2

        if use_3d:
            corners = [[-tag_radius, -tag_radius, 0.0],
                       [ tag_radius, -tag_radius, 0.0],
                       [ tag_radius,  tag_radius, 0.0],
                       [-tag_radius,  tag_radius, 0.0]]
        else:
            corners = [[-tag_radius, -tag_radius],
                       [ tag_radius, -tag_radius],
                       [ tag_radius,  tag_radius],
                       [-tag_radius,  tag_radius]]
        return corners

    def computeBodyPts(self, tag, size, tf_tag2body):
        """Compute theoretical 3d points of apriltag corners in body frame

        @param
            - tag: name of tag to compute points for
            - size: size of tag
            - tf_tag2body: se3 matrix describing transform between tag to body
        @return pts: points of corners for apriltag relative to body based
        on config, as a numpy array
        """
        # Initialize output
        pts = []

        # Compute position of each corner in tag relative to body
        corners = self.getCornerPoints(size)

        X_b_t = tr.inverse_matrix(tf_tag2body)

        for c in corners:
             X_t_p = tr.translation_matrix(c)
             X_b_p = tr.concatenate_matrices(X_b_t, X_t_p)
             _, _, _, pos, _ = tr.decompose_matrix(X_b_p)
             pts.append(pos)

        return np.array(pts)

    def tagCallback(self, tf_data):
        # Extract position and quaternion for camera to tag
        p_c_t, q_c_t = tf_utils.transform_to_pq(tf_data.transform)
        a_c_t = tr.euler_from_quaternion(q_c_t)

        # Compute transformation between camera to tag
        X_c_t = tr.compose_matrix(angles=a_c_t, translate=p_c_t)

        # Save transform
        if tf_data.child_frame_id not in self.cfg['tags']:
            # rospy.loginfo("Unused tf " + tf_data.child_frame_id)
            return

        idx = np.where(self.cfg['tags'] == tf_data.child_frame_id)[0][0]
        self.state['tf_camera2tag'][idx] = X_c_t
        self.state['timestamps'][idx] = tf_data.header.stamp.to_sec()

        # TODO: make this more robust
        self.frame_id = tf_data.header.frame_id

    def computeInitialPose(self, body, time_mask):
        """Compute body pose estimate based on first body tag
        (chosen arbitrarily)
        """
        # Create bool array for body
        b_arr_body = [self.cfg['bodies'] == body]

        # Get tags with valid timestamps for this body
        b_arr = np.logical_and(time_mask, b_arr_body)[0]

        # Use first index
        idx = np.where(b_arr)[0][0]

        # Get tag info
        X_c_t = self.state['tf_camera2tag'][idx]
        X_t_b = self.cfg['tf_tag2body'][idx]
        tag_ts = self.state['timestamps'][idx]
        tag_id = self.cfg['tags'][idx]

        # Compute estimate of camera to body tf
        X_c_b = tr.concatenate_matrices(X_c_t, X_t_b)

        return X_c_b

    def constructOutputMsg(self, body, se3):
        output_msg = TransformStamped()

        # Use most recent timestamp for body
        ts = max(self.state['timestamps'][self.cfg['bodies'] == body])
        output_msg.header.stamp = rospy.Time.from_sec(ts)
        output_msg.header.frame_id = self.frame_id
        output_msg.child_frame_id = "detected_" + body

        output_msg.transform = tf_utils.se3_to_msg(se3)
        return output_msg

    def checkNewData(self, body):
        ts = max(self.state['timestamps'][self.cfg['bodies'] == body])
        b_idx = self.bodies.index(body)
        if self.prev_ts[b_idx] == ts:
            return False
        else:
            # Update ts
            self.prev_ts[b_idx] = ts
            return True

    def run(self):
        while not rospy.is_shutdown():
            # Create bool array of valid timestamps
            b_arr_time = [rospy.Time.now().to_sec() - self.state['timestamps'] < TF_TIMEOUT]

            # Get bodies with valid timestamps
            bodies_list = self.cfg['bodies'][tuple(b_arr_time)]
            bodies = set(bodies_list)

            # Estimate tf of each body
            for body in bodies:
                # Check if we have new data
                if not self.checkNewData(body):
                    continue

                # Use first detected tag for body pose
                X_c_b = self.computeInitialPose(body, b_arr_time)

                # Publish to ROS
                tf_msg = self.constructOutputMsg(body, X_c_b)
                self.tf_broadcaster.sendTransform(tf_msg)

            self.rate.sleep()

if __name__=="__main__":
    est = BodyPoseEstimationNode()
    est.run()
