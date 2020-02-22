/**********************************************
 * @file omni_wheel_robot_plant.h
 * @author Gareth Ellis
 * @date February 2nd, 2020
 * @brief Implementation of the OmniWheelRobotPlant class
 ***********************************************/

#ifndef OMNIWHEELROBOT_PLANT_H_
#define OMNIWHEELROBOT_PLANT_H_

#include "param_getter.h"

#include <autorally_control/ddp/util.h>
#include <autorally_msgs/runstop.h>

#include <autorally_msgs/pathIntegralStatus.h>
#include <autorally_msgs/pathIntegralTiming.h>
#include <autorally_msgs/wheelCommands.h>
#include <autorally_control/OmniWheelRobotPathIntegralParamsConfig.h>

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/transform_datatypes.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>

#include <geometry_msgs/Quaternion.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Point.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>

#include <eigen3/Eigen/Dense>

#include <boost/thread/thread.hpp>
#include <boost/thread/mutex.hpp>
#include <atomic>

namespace autorally_control {

/**
 * This enum denotes what controller was used for an operation
 */
enum class ControllerType {
  // The controller starting from the actual state
  ACTUAL_STATE,
  // The controller starting from it's internal estimation of the robot state
  PREDICTED_STATE,
  // No controller (for example if no trajectory has been generated yet)
  NONE
};


// TODO: better jdoc 
// TODO: better name
// TODO: we probably want controller arbitration (predicted state vs. actual)
//       in it's own class
/**
* @class OmniWheelRobotPlant 
* @brief Publishers and subscribers for the autorally control system.
* 
* This class is treated as the plant for the MPPI controller. When the MPPI
* controller has a control it sends to a function in this class to get
* send to the actuators. Likewise it calls functions in this class to receive
* state feedback. This class also publishes trajectory and spray information
* and status information for both the controller and the OCS.
*/

class OmniWheelRobotPlant
{
public:
  static const int STATE_DIM = 9;
  static const int CONTROL_DIM = 4;

  using StateVector = Eigen::Matrix<float, STATE_DIM, 1>;
  using ControlVector = Eigen::Matrix<float, CONTROL_DIM, 1>;

  // Not sure why these are needed, it's carry-over from legacy, looks like it 
  // has *something* to do with preventing heading wrapping
  float last_heading_ = 0.0;
  int heading_multiplier_ = 0;

  boost::mutex access_guard_;
  std::string nodeNamespace_;

  // The most recent debug image that we've received
  cv::Mat debugImg_;

  // Whether or not we've received the first solution
  bool solutionReceived_ = false;
  
  // Whether or not this is running in a ROS nodelet
  bool is_nodelet_;
  
  // Variables for interfacing with the CUDA code
  std::vector<float> controlSequence_;
  std::vector<float> stateSequence_;
  util::EigenAlignedVector<float, CONTROL_DIM, STATE_DIM> feedback_gains_;

  // TODO: what is this used for? *is* this for interacting with cuda code? 
  //       or something else entirely?
  ros::Time solutionTs_;

  // The controller type used to compute the last received solution
  ControllerType controller_type_used_for_solution_ = ControllerType::NONE;

  // The id of the point we publish to indicate what controller was used where.
  // We need to save this so we can constantly increment it, otherwise every
  // new point we publish will have the same id as the previous one, and so
  // will overwrite it
  int controller_type_debug_point_id = 0;

  // The expected number of timestamps in each state and control sequence
  int numTimesteps_;
  
  // The step size for both the control and state sequences
  // TODO: why is this public? Make private if possible
  double deltaT_;

  // An estimate of how long the most recent solution took to compute
  double optimizationLoopTime_;

  /**
  * @brief Constructor for OmniWheelRobotPlant, takes the a ros node handle and 
  *        initalizes publishers and subscribers.
  *
  * @param global_node A nodehandle that will be used to interface with the
  *                    outside world (ex. getting robot state)
  * @param mppi_node A nodehandle that will be used to get the parameters for
  *                  mppi. As such, this must be in a specific namespace
  * @param debug_mode Enable debugging, including visualizations
  * @param hz The frequency of the control publisher.
  * @param nodelet Whether or not this node is running as part of a nodelet
  */
  OmniWheelRobotPlant(ros::NodeHandle global_node, ros::NodeHandle mppi_node, 
                 bool debug_mode, int hz, bool nodelet);

  /**
  * @brief Constructor for OmniWheelRobotPlant, takes the a ros node handle and 
  *        initalizes publishers and subscribers.
  *
  * This constructor assumes this node is not running as part of a nodelet
  *
  * @param global_and_mppi_node A nodehandle that will be used to interface 
  *                             with the outside world (ex. getting robot state)
  *                             and also get the MPPI parameters
  * @param debug_mode Enable debugging, including visualizations
  * @param hz The frequency of the control publisher.
  */
  OmniWheelRobotPlant(ros::NodeHandle global_node, bool debug_mode, int hz);

  /**
  * @brief Callback for /pose_estimate subscriber.
  */
  void poseCall(nav_msgs::Odometry pose_msg);

  /**
  * @brief Publishes the controller's nominal path.
  */
  void pubPath(const ros::TimerEvent&);

  /**
   * Set the control solution to be executed by this plant
   *
   * @param traj The trajectory to track
   * @param controls The controls to track the tracjectory
   * @param gains The gains to use to track the trajectory if using feedback
   * @param timestamp The time the controls/trajectory were computed at
   * @param loop_speed An estimate of how long this solution took to compute
   * @param controller_type_used The type of controller used to compute the
   *                             trajectory/controls
   */
  void setSolution(std::vector<float> traj, std::vector<float> controls, 
                   util::EigenAlignedVector<float, CONTROL_DIM, STATE_DIM> gains,
                   ros::Time timestamp, double loop_speed, ControllerType controller_type_used);

  /**
   * @brief Set the debug image to the given one
   *
   * @param img The new debug image
   */
  void setDebugImage(cv::Mat img);

  // TODO: bit nasty this, why can't we do timing internally to this class
  /**
   * @brief Set timing information about MPPI
   *
   * This should be called externally from this class
   */
  void setTimingInfo(double poseDiff, double tickTime, double sleepTime);

  /**
   * @brief Publishes timing information about the controller
   *
   * This should be setup to be triggered from a ROS timer at a fixed interval
   */
  void pubTimingData(const ros::TimerEvent&);

  /**
  * @brief Publishes a control input
  * @param wheel_commands The control to publish
  */
  void pubControl(ControlVector wheel_commands);

  /**
   * @brief Publishes status information about this plant
   *
   * This should be setup to be triggered from a ROS timer at a fixed interval
   */
  void pubStatus(const ros::TimerEvent&);

  /**
   * @brief Publishes what controller was used to execute the trajectory this
   *        plant is currently executing
   * This should be setup to be triggered from a ROS timer at a fixed interval
   */
  void pubControllerTypeDebug(const ros::TimerEvent&);

  /**
   * @brief Return if the robot speed is being capped to a safe value
   * @return True if the robot speed is being capped to a safe value, false
   *        otherwise
   */
  bool getRunstop();

  /**
  * @brief Returns the timestamp of the last pose callback.
  */
  ros::Time getLastPoseTime();

  /**
  * @brief Checks the system status.
  * @return An integer specifying the status. 0 means the system is operating
  * nominally, 1 means something is wrong but no action needs to be taken,
  * 2 means that the vehicle should stop immediately.
  */
  int checkStatus();

  /**
   * @brief The dynamic reconfigure callback for this plant
   *
   * This should be configured in the constructor so that it is called when
   * a dynamic reconfig setting is changed
   *
   * @param config The new config
   * @param lvl The "level", which is the OR or all the level values of all
   *            the parameters that changed
   */
  void dynRcfgCall(autorally_control::OmniWheelRobotPathIntegralParamsConfig &config, int lvl);

  /**
   * @brief Check if this plant has a new dynamic reconfigure config
   *
   * This will return true until the dynamic reconfigure config is grabbed
   * from this plant via `getDynRcfgparams`
   *
   * @return True if there is a new dynamic reconfigure config, false otherwise
   */
  bool hasNewDynRcfg();

  /**
   * @brief Get the most recently received dynamic reconfigure config
   *
   * @return The most recently received dynamic reconfigure config
   */
  autorally_control::OmniWheelRobotPathIntegralParamsConfig getDynRcfgParams();

  /**
   * @brief Displays the debug image last set on this plant in a window
   *
   * This should be setup to be triggered from a ROS timer at a fixed interval
   */
  virtual void displayDebugImage(const ros::TimerEvent&);

  /**
  * @brief Returns the current state of this plant
  * @return A vector containing the state of this plant
  */
  StateVector getStateVector();

  /**
   * Set the state of this plant from the state represented by the given vector
   *
   * @param new_state The new state for this plant
   */
  void setStateFromVector(StateVector new_state);

  /**
   * @brief This is effectively the destructor for this class
   *
   * This will stop all pubs, subs, and timers.
   * Unfortunately for legacy reasons it has to be here (it's called in the
   * MPPI stuff), but this is really just an indicator of a poor concept of
   * ownership, it really should just be the class destructor.
   */
  virtual void shutdown();

protected:

  typedef struct
  { 
    // Actual robot state
    float x_pos;
    float y_pos;
    float yaw;
    float x_vel;
    float y_vel;
    float angular_vel;
    float x_accel;
    float y_accel;
    float angular_accel;

    

    // TODO: deleteme
    // //X-Y-theta position
    // float x_pos;
    // float y_pos;
    // float z_pos;
    // //1-2-3 Euler angles
    // float roll;
    // float pitch;
    // float yaw;
    // //Quaternions
    // float q0;
    // float q1;
    // float q2;
    // float q3;
    // //X-Y-Z velocity.
    // float x_vel;
    // float y_vel;
    // float z_vel;
    // //Body frame velocity
    // float u_x;
    // float u_y;
    // //Euler angle derivatives
    // float yaw_mder;
    // //Current servo commands
    // float steering;
    // float throttle;
  } FullState;
  
  /**
   * Get the current state of this plant
   * @return The current state of this plant
   */
  FullState getState();

  // The number of poses that we've received 
  int poseCount_ = 0;

  // Whether or not we should use the feedback gains given. If false, we 
  // will just execute the control solution given. 
  bool useFeedbackGains_ = false;

  // Whether or not we've received a debug image
  std::atomic<bool> receivedDebugImg_;

  // The parameters that make up the cost function for this plant
  // TODO: we're probably going to need to change this to our own config type
  autorally_control::OmniWheelRobotPathIntegralParamsConfig costParams_;

  // Whether or not we have a set of new cost parameters, in the form of a
  // new dynamic configuration. Should be set true after a new config is 
  // received until that config is retrieved via a getter.
  std::atomic<bool> hasNewCostParams_;

  //Time before declaring pose/controls stale
  const double TIMEOUT = 0.5;

  // Full state of the autorally vehicle.
  FullState full_state_;

  // The frequency of the control publisher.
  int hz_; 

  // TODO: change this to an enum. 
  int status_; ///< System status
  std::string ocs_msg_; ///< Message to send to the ocs.

  bool safe_speed_zero_; ///< Current value of safe speed.
  bool debug_mode_; ///< Whether or not the system is in debug/simulation mode.
  bool activated_; ///< Whether or not we've received an initial pose message.

  ros::Time last_pose_call_; ///< Timestamp of the last pose callback.

  ros::Publisher control_pub_; ///< Publisher of autorally_msgs::chassisCommand type on topic servoCommand.
  ros::Publisher status_pub_; ///< Publishes the status (0 good, 1 neutral, 2 bad) of the controller
  ros::Publisher subscribed_pose_pub_; ///< Publisher of the subscribed pose
  ros::Publisher path_pub_; ///< Publisher of nav_mags::Path on topic nominalPath.
  ros::Publisher timing_data_pub_;
  ros::Publisher debug_controller_type_pub_; ///< Publishes points indicating what controller was used where
  ros::Subscriber pose_sub_; ///< Subscriber to /pose_estimate.
  ros::Subscriber servo_sub_;
  ros::Timer pathTimer_;
  ros::Timer statusTimer_;
  ros::Timer debugImgTimer_;
  ros::Timer debugControllerTypeTimer_;
  ros::Timer timingInfoTimer_;

  nav_msgs::Path path_msg_; ///< Path message for publishing the planned path.
  geometry_msgs::Point time_delay_msg_; ///< Point message for publishing the observed delay.
  autorally_msgs::pathIntegralStatus status_msg_; ///<pathIntegralStatus message for publishing mppi status
  autorally_msgs::pathIntegralTiming timingData_; ///<pathIntegralStatus message for publishing mppi status
};

}

#endif /* OMNIWHEELROBOT_PLANT_H_ */
