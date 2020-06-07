#!/usr/bin/env python

import gym
import rospy
import time
import numpy as np
import tf
import time
from gym import utils, spaces
from geometry_msgs.msg import Twist, Vector3Stamped, Pose, Transform
#from mav_msgs.msg import CommandTrajectory
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint 
#from hector_uav_msgs.msg import Altimeter
from sensor_msgs.msg import Imu
from std_msgs.msg import Header
from std_msgs.msg import Empty as EmptyTopicMsg
from gym.utils import seeding
from gym.envs.registration import register
from gazebo_connection import GazeboConnection

#register the training environment in the gym as an available one
reg = register(
    id='QuadcopterLiveShow-v0',
    entry_point='myquadcopter_env:QuadCopterEnv'
    )


class QuadCopterEnv(gym.Env):

    def __init__(self):
        
        # We assume that a ROS node has already been created
        # before initialising the environment
        
        self.vel_pub = rospy.Publisher('/firefly/command/trajectory', MultiDOFJointTrajectory, queue_size=5)
        
        # gets training parameters from param server
        self.speed_value = rospy.get_param("/drone_hex/pos_step")
        self.desired_pose = Pose()
        self.desired_pose.position.z = rospy.get_param("/drone_hex/desired_pose/z")
        self.desired_pose.position.x = rospy.get_param("/drone_hex/desired_pose/x")
        self.desired_pose.position.y = rospy.get_param("/drone_hex/desired_pose/y")
        self.running_step = rospy.get_param("/drone_hex/running_step")
        self.max_incl = rospy.get_param("/drone_hex/max_incl")
        self.max_altitude = rospy.get_param("/drone_hex/max_altitude")
        
        # stablishes connection with simulator
        self.gazebo = GazeboConnection()
        
        self.action_space = spaces.Discrete(4) #Forward,Left,Right,Up,Down
        self.reward_range = (-np.inf, np.inf)

        self._seed()

    # A function to initialize the random generator
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
        
    # Resets the state of the environment and returns an initial observation.
    def _reset(self):
        
        # 1st: resets the simulation to initial values
        #self.gazebo.resetSim()

        # 2nd: Unpauses simulation
        self.gazebo.unpauseSim()

        # 3rd: resets the robot to initial conditions
        self.check_topic_publishers_connection()
        self.takeoff_sequence()
        self.init_desired_pose()
	rospy.Time(10.0)
        # 4th: takes an observation of the initial condition of the robot
        data_pose, data_imu = self.take_observation()
        observation = [data_pose.position.x,data_pose.position.y]
        
        # 5th: pauses simulation
        self.gazebo.pauseSim()

        return observation

    def _step(self, action):

        # Given the action selected by the learning algorithm,
        # we perform the corresponding movement of the robot
        self.gazebo.unpauseSim()  
	rospy.Time(5.0)      
	curr_pose, curr_imu = self.take_observation()
        self.gazebo.pauseSim()

        # 1st, we decide which velocity command corresponds
	vel_cmd = MultiDOFJointTrajectory()
	
	header = Header()
	header.frame_id='frame'
	header.stamp=rospy.Time()
	vel_cmd.joint_names.append('/firefly/base_link')
	vel_cmd.header = header
	transforms = Transform()
	transforms.translation.z = 1.0
	velocities = Twist()
	accelaration = Twist()

        if action == 0: #FORWARD
            transforms.translation.x = curr_pose.position.x + self.speed_value
            transforms.translation.y = curr_pose.position.y
            velocities.angular.z = 0.0
        elif action == 1: #BACKWARD
            transforms.translation.x = curr_pose.position.x - self.speed_value
            transforms.translation.y = curr_pose.position.y
            velocities.angular.z = 0.0
        elif action == 2: #LEFT
            transforms.translation.x = curr_pose.position.x
            transforms.translation.y = curr_pose.position.y + self.speed_value
            velocities.angular.z = 0.0
        elif action == 3: #RIGHT
            transforms.translation.x = curr_pose.position.x
            transforms.translation.y = curr_pose.position.y - self.speed_value
            velocities.angular.z = 0.0
        #elif action == 5: #NOTHING
        #    transforms.translation.x = curr_pose.position.x
        #    transforms.translation.y = curr_pose.position.y

	point = MultiDOFJointTrajectoryPoint([transforms],[velocities],[accelaration],rospy.Time(1.0))
	vel_cmd.points.append(point)
        # Then we send the command to the robot and let it go
        # for running_step seconds
        self.gazebo.unpauseSim()
        self.vel_pub.publish(vel_cmd)
        time.sleep(self.running_step)
        data_pose, data_imu = self.take_observation()
        self.gazebo.pauseSim()

        # finally we get an evaluation based on what happened in the sim
        reward,done = self.process_data(data_pose, data_imu)

        # Promote going forwards instead of turning
        if action == 1 or action == 3:
            reward -= 100
        else:
            reward += 100

        state = [data_pose.position.x,data_pose.position.y]
        return state, reward, done, {}


    def take_observation (self):
        data_pose = None
        while data_pose is None:
            try:
                data_pose = rospy.wait_for_message('/firefly/ground_truth/pose', Pose, timeout=5)
            except:
                rospy.loginfo("Current drone pose not ready yet, retrying for getting robot pose")

        data_imu = None
        while data_imu is None:
            try:
                data_imu = rospy.wait_for_message('/firefly/ground_truth/imu', Imu, timeout=5)
            except:
                rospy.loginfo("Current drone imu not ready yet, retrying for getting robot imu")
        
        return data_pose, data_imu

    def calculate_dist_between_two_Points(self,p_init,p_end):
        a = np.array((p_init.x ,p_init.y))
        b = np.array((p_end.x ,p_end.y))
        
        dist = np.linalg.norm(a-b)
        
        return dist


    def init_desired_pose(self):
        
        current_init_pose, imu = self.take_observation()
        
        self.best_dist = self.calculate_dist_between_two_Points(current_init_pose.position, self.desired_pose.position)
    

    def check_topic_publishers_connection(self):
        
        rate = rospy.Rate(10) # 10hz

        while(self.vel_pub.get_num_connections() == 0):
            rospy.loginfo("No susbribers to Cmd_vel yet so we wait and try again")
            rate.sleep();
        rospy.loginfo("Cmd_vel Publisher Connected")
        

    def reset_cmd_vel_commands(self):
        # We send an empty null Twist
	vel_cmd = MultiDOFJointTrajectory()
	
	header = Header()
	header.frame_id='frame'
	header.stamp=rospy.Time()
	vel_cmd.joint_names.append('/firefly/base_link')
	vel_cmd.header = header
	transforms = Transform()
	transforms.translation.x = 0.0
	transforms.translation.y = 0.0
	transforms.translation.z = 1.0
	velocities = Twist()
	velocities.linear.x = 0.0
	velocities.linear.y = 0.0
	accelaration = Twist()

	point = MultiDOFJointTrajectoryPoint([transforms],[velocities],[accelaration],rospy.Time(1.0))
	vel_cmd.points.append(point)
        self.vel_pub.publish(vel_cmd)


    def takeoff_sequence(self, seconds_taking_off=10):
        # Before taking off be sure that cmd_vel value there is is null to avoid drifts
        self.reset_cmd_vel_commands()
        
        rospy.loginfo( "Taking-Off Start")
        time.sleep(seconds_taking_off)
        rospy.loginfo( "Taking-Off sequence completed")
        

    def improved_distance_reward(self, current_pose):
        current_dist = self.calculate_dist_between_two_Points(current_pose.position, self.desired_pose.position)
	done = False
        rospy.loginfo("Best Distance = "+str(self.best_dist))
        rospy.loginfo("Calculated Distance = "+str(current_dist))
        #reward =  - 100*current_dist


        if current_dist < self.best_dist:
	    self.best_dist = current_dist
            reward = 50
	else:
	    reward = -50
            #print "Made Distance bigger= "+str(self.best_dist)
        
	if current_dist > 2.0:
		reward = -500
		done = True
	if current_dist < 0.5:
		self.best_dist = current_dist
		reward = 100
		done = True


        return reward, done
        
    def process_data(self, data_position, data_imu):

        done = False
        
        euler = tf.transformations.euler_from_quaternion([data_imu.orientation.x,data_imu.orientation.y,data_imu.orientation.z,data_imu.orientation.w])
        roll = euler[0]
        pitch = euler[1]
        yaw = euler[2]

        pitch_bad = not(-self.max_incl < pitch < self.max_incl)
        roll_bad = not(-self.max_incl < roll < self.max_incl)
        altitude_bad = data_position.position.z > self.max_altitude

        if altitude_bad or pitch_bad or roll_bad:
            rospy.loginfo ("(Drone flight status is wrong) >>> ("+str(altitude_bad)+","+str(pitch_bad)+","+str(roll_bad)+")")
            done = True
            reward = -400
        else:
            reward, done = self.improved_distance_reward(data_position)

        return reward,done
