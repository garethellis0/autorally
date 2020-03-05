/**********************************************
 * @file omni_wheel_robot_plant.cpp
 * @author Gareth Ellis
 * @date February 2nd, 2020
 * @brief Implementation of the OmniWheelRobotPlant class
 ***********************************************/
#include <autorally_control/path_integral/omni_wheel_robot_plant.h>

#include <visualization_msgs/Marker.h>

#include <stdio.h>
#include <stdlib.h>

namespace autorally_control {

OmniWheelRobotPlant::OmniWheelRobotPlant(
    std::string robot_state_ipc_queue_name,
    std::string robot_wheel_commands_ipc_queue_name,
    ros::NodeHandle global_and_mppi_node, bool debug_mode, int hz)
  : OmniWheelRobotPlant(
      robot_state_ipc_queue_name,
      robot_wheel_commands_ipc_queue_name,
      global_and_mppi_node, global_and_mppi_node, debug_mode, hz, false){};

OmniWheelRobotPlant::OmniWheelRobotPlant(
    std::string robot_state_ipc_queue_name,
    std::string robot_wheel_commands_ipc_queue_name,
    ros::NodeHandle global_node, ros::NodeHandle mppi_node, 
                               bool debug_mode, int hz, bool nodelet) :
    robot_state_message_queue_name(robot_state_ipc_queue_name),
    robot_wheel_commands_message_queue_name(robot_wheel_commands_ipc_queue_name),
    robot_state_message_queue(boost::interprocess::open_or_create,
                              robot_state_ipc_queue_name.c_str(), MAX_QUEUE_SIZE,
                              sizeof(RobotStateMsgQueueEntry)),
    robot_wheel_commands_message_queue(boost::interprocess::open_or_create,
                                       robot_wheel_commands_ipc_queue_name.c_str(),
                                       MAX_QUEUE_SIZE, sizeof(RobotWheelCommandsMsgQueueEntry)),
    _in_destructor(false)
{
  nodeNamespace_ = mppi_node.getNamespace(); 
  debug_mode_ = getRosParam<bool>("debug_mode", mppi_node);
  numTimesteps_ = getRosParam<int>("num_timesteps", mppi_node);
  useFeedbackGains_ = getRosParam<bool>("use_feedback_gains", mppi_node);
  deltaT_ = 1.0/hz;

  controlSequence_.resize(CONTROL_DIM*numTimesteps_);
  stateSequence_.resize(STATE_DIM*numTimesteps_);

  //Initialize the publishers.
  path_pub_ = mppi_node.advertise<nav_msgs::Path>("nominalPath", 1);
  status_pub_ = mppi_node.advertise<autorally_msgs::pathIntegralStatus>("mppiStatus", 1);
  timing_data_pub_ = mppi_node.advertise<autorally_msgs::pathIntegralTiming>("timingInfo", 1);
  debug_controller_type_pub_ = mppi_node.advertise<visualization_msgs::Marker>("controllerTypeDebug", 1);
  
  //Initialize the subscribers.
  //Timer callback for path publisher
  pathTimer_ = mppi_node.createTimer(ros::Duration(0.033), &OmniWheelRobotPlant::pubPath, this);
  statusTimer_ = mppi_node.createTimer(ros::Duration(0.033), &OmniWheelRobotPlant::pubStatus, this);
  debugImgTimer_ = mppi_node.createTimer(ros::Duration(0.033), &OmniWheelRobotPlant::displayDebugImage, this);
  debugControllerTypeTimer_ = mppi_node.createTimer(ros::Duration(0.033), &OmniWheelRobotPlant::pubControllerTypeDebug, this);
  timingInfoTimer_ = mppi_node.createTimer(ros::Duration(0.033), &OmniWheelRobotPlant::pubTimingData, this);

  //Initialize auxiliary variables.
  safe_speed_zero_ = false;
  debug_mode_ = debug_mode;
  activated_ = false;
  last_pose_call_ = ros::Time::now();

  // Initialize state
  full_state_.angular_vel = 0.0;
  
  //Initialize yaw derivative to zero
  status_ = 1;
  if (debug_mode_){
    ocs_msg_ = "Debug Mode";
  }
  else {
    ocs_msg_ = "";
  }
  std::string info = "MPPI Controller";
  std::string hardwareID = "";
  std::string portPath = "";

  //Debug image display signaller
  receivedDebugImg_ = false;
  is_nodelet_ = nodelet;

  hasNewCostParams_ = false;

  if (!debug_mode_){
    ROS_INFO("DEBUG MODE is set to FALSE, waiting to receive first pose estimate...  ");
  }
  else{
    ROS_WARN("DEBUG MODE is set to TRUE. DEBUG MODE must be FALSE in order to be launched from a remote machine. \n");
  }

  // Start thread to receive new robot states
  receive_robot_state_thread =
      boost::thread([this]() { return receiveRobotStatesLoop(); });
}

OmniWheelRobotPlant::~OmniWheelRobotPlant(){
  // Stop the thread continuously receiving robot states
  _in_destructor = true;
  receive_robot_state_thread.join();

  // Close message queues
  boost::interprocess::message_queue::remove(
      robot_state_message_queue_name.c_str());
  boost::interprocess::message_queue::remove(
      robot_wheel_commands_message_queue_name.c_str());
}

void OmniWheelRobotPlant::setSolution(std::vector<float> traj, std::vector<float> controls, 
                                util::EigenAlignedVector<float, CONTROL_DIM, STATE_DIM> gains,
                                ros::Time ts, double loop_speed, 
                                ControllerType controller_type_used)
{
  boost::mutex::scoped_lock lock(access_guard_);
  optimizationLoopTime_ = loop_speed;
  solutionTs_ = ts;

  if (traj.size() != numTimesteps_*STATE_DIM){
    ROS_INFO("Received a trajectory with the incorrect number of values.");
    ros::shutdown();
  }
  if (controls.size() != numTimesteps_*CONTROL_DIM){
    ROS_INFO("Received a controls with the incorrect number of values.");
    ros::shutdown();
  }

  for (int t = 0; t < numTimesteps_; t++){
    for (int i = 0; i < STATE_DIM; i++){
      stateSequence_[STATE_DIM*t + i] = traj[STATE_DIM*t + i];
    }
    for (int i = 0; i < CONTROL_DIM; i++){
      controlSequence_[CONTROL_DIM*t + i] = controls[CONTROL_DIM*t + i];
    }
  }
  feedback_gains_ = gains;
  solutionReceived_ = true;
  controller_type_used_for_solution_ = controller_type_used;
}

void OmniWheelRobotPlant::setTimingInfo(double poseDiff, double tickTime, double sleepTime)
{
  boost::mutex::scoped_lock lock(access_guard_);
  timingData_.averageTimeBetweenPoses = poseDiff;//.clear();
  timingData_.averageOptimizationCycleTime = tickTime;
  timingData_.averageSleepTime = sleepTime;
}

void OmniWheelRobotPlant::pubTimingData(const ros::TimerEvent&)
{
  boost::mutex::scoped_lock lock(access_guard_);
  timingData_.header.stamp = ros::Time::now();
  timing_data_pub_.publish(timingData_);
}

void OmniWheelRobotPlant::setDebugImages(cv::Mat actual_state_controller_img, 
      cv::Mat predicted_state_controller_img)
{
  receivedDebugImg_ = true;
  boost::mutex::scoped_lock lock(access_guard_);
  debugPredictedStateControllerImg_ = predicted_state_controller_img;
  debugActualStateControllerImg_ = actual_state_controller_img;
}

void OmniWheelRobotPlant::displayDebugImage(const ros::TimerEvent&)
{
  if (receivedDebugImg_.load() && !is_nodelet_) {
    {
      boost::mutex::scoped_lock lock(access_guard_);

      std::string actual_state_controller_window_name = 
        nodeNamespace_ + "/ActualStateController";
      cv::namedWindow(actual_state_controller_window_name, cv::WINDOW_AUTOSIZE);
      cv::imshow(actual_state_controller_window_name, debugActualStateControllerImg_);

      std::string predicted_state_controller_window_name = 
        nodeNamespace_ + "/PredictedStateController";
      cv::namedWindow(predicted_state_controller_window_name, cv::WINDOW_AUTOSIZE);
      cv::imshow(predicted_state_controller_window_name, debugPredictedStateControllerImg_);
    } 
  }
  if (receivedDebugImg_.load() && !is_nodelet_){
    cv::waitKey(1);
  }
}

void OmniWheelRobotPlant::newStateCallback(RobotStateMsgQueueEntry new_state)
{
  if (poseCount_ == 0){
    ROS_INFO(" First pose estimate received. \n");
  }

  // TODO: this should be it's own function
  {
    boost::mutex::scoped_lock lock(access_guard_);
    // Update the timestamp
    last_pose_call_ = ros::Time(
        new_state.timestamp_secs, 
        new_state.timestamp_nano_secs_correction);
    poseCount_++;
    //Set activated to true --> we are receiving state messages.
    activated_ = true;
    //Update position
    full_state_.x_pos = new_state.x_pos_mm / 1000.0;
    full_state_.y_pos = new_state.y_pos_mm / 1000.0;
    full_state_.yaw = new_state.yaw_milli_rad / 1000.0;

    //Don't allow heading to wrap around
    if (last_heading_ > 3.0 && full_state_.yaw < -3.0){
      heading_multiplier_ += 1;
    }
    else if (last_heading_ < -3.0 && full_state_.yaw > 3.0){
      heading_multiplier_ -= 1;
    }
    last_heading_ = full_state_.yaw;
    full_state_.yaw = full_state_.yaw + heading_multiplier_*2*3.14159265359;

    //Update the world frame velocity
    full_state_.x_vel = new_state.x_vel_mm_per_s / 1000.0;
    full_state_.y_vel = new_state.y_vel_mm_per_s / 1000.0;
    full_state_.angular_vel = new_state.angular_vel_milli_rad_per_s / 1000.0;

    //Update the body frame longitudenal and lateral velocity
    full_state_.x_vel_body_frame = cos(full_state_.yaw)*full_state_.x_vel + 
                                   sin(full_state_.yaw)*full_state_.y_vel;
    full_state_.y_vel_body_frame = -sin(full_state_.yaw)*full_state_.x_vel + 
                                   cos(full_state_.yaw)*full_state_.y_vel;
  }

  //Interpolate and publish the current control
  double timeFromLastOpt = (last_pose_call_ - solutionTs_).toSec();

  ControlVector ff_terms;
  if (solutionReceived_ && timeFromLastOpt > 0 && timeFromLastOpt < (numTimesteps_-1)*deltaT_){
    int lowerIdx = (int)(timeFromLastOpt/deltaT_);
    int upperIdx = lowerIdx + 1;
    double alpha = (timeFromLastOpt - lowerIdx*deltaT_)/deltaT_;

    for (size_t i = 0; i < CONTROL_DIM; i++){
      ff_terms(i) = (1 - alpha)*controlSequence_[CONTROL_DIM*lowerIdx +i] + 
                    alpha*controlSequence_[CONTROL_DIM*upperIdx + i];
    }

    // Default to just using feed forward terms (ie. open loop control)
    ControlVector final_controls = ff_terms;
    if (useFeedbackGains_) { 
      //Compute the error between the current and actual state and apply feedback gains
      StateVector current_state = getStateVector();
      StateVector desired_state;
      StateVector deltaU;
      for (int i = 0; i < STATE_DIM; i++){
        desired_state(i) = (1 - alpha)*stateSequence_[STATE_DIM*lowerIdx + i] 
          + alpha*stateSequence_[STATE_DIM*upperIdx + i];
      }
      
      ControlVector fb_terms = ((1-alpha)*feedback_gains_[lowerIdx] + 
          alpha*feedback_gains_[upperIdx])*(current_state - desired_state);

      if (!fb_terms.hasNaN()){
        // TODO: probably want to enforce some sort of constraints here
        //steering = fmin(0.99, fmax(-0.99, steering_ff + steering_fb));
        //throttle = fmin(throttleMax_, fmax(-0.99, throttle_ff + throttle_fb));
        final_controls = ff_terms + fb_terms;
      }
    }
    pubControl(final_controls);
  }
}

void OmniWheelRobotPlant::receiveRobotStatesLoop(){
  while (!_in_destructor)
  {
    // We use the version of receive with a timeout so that we can
    // regularly check if this class was destructed, and stop the loop
    // if so
    const boost::posix_time::ptime t_timeout =
        boost::posix_time::microsec_clock::universal_time() +
        boost::posix_time::milliseconds(IPC_QUEUE_TIMEOUT_MS);
    RobotStateMsgQueueEntry new_state;
    size_t received_size  = 0;
    unsigned int priority = 0;

    // TODO
    // For some reason boost will only allow a minimum max message
    // size of 100, even though the queue is configured to have a 
    // max message size equal to the struct size. 
    const bool data_available = robot_state_message_queue.timed_receive(
        &new_state, 100, received_size, priority,
        t_timeout);

    if (data_available && received_size == sizeof(new_state))
    {
        std::cout << "Received State: (" << new_state << ")" << std::endl;
        newStateCallback(new_state);
    }
    else if (data_available && received_size != sizeof(new_state))
    {
        ROS_WARN_STREAM("Received message of size " << received_size
                     << " which is different from the expected size "
                     << sizeof(new_state));
    }
  }
}

void OmniWheelRobotPlant::pubPath(const ros::TimerEvent&)
{
  boost::mutex::scoped_lock lock(access_guard_);
  path_msg_.poses.clear();
  nav_msgs::Odometry subscribed_state;
  ros::Time begin = solutionTs_;
  for (int i = 0; i < numTimesteps_; i++) {
    geometry_msgs::PoseStamped pose;
    pose.pose.position.x = stateSequence_[i*(STATE_DIM)];
    pose.pose.position.y = stateSequence_[i*(STATE_DIM) + 1];
    pose.pose.position.z = 0;

    tf2::Quaternion quat;
    float yaw = stateSequence_[i*(STATE_DIM) + 2];
    quat.setRPY(0, 0, yaw);
    pose.pose.orientation = tf2::toMsg(quat);

    pose.header.stamp = begin + ros::Duration(i*deltaT_);
    pose.header.frame_id = "odom";
    path_msg_.poses.push_back(pose);
    if (i == 0){
      subscribed_state.pose.pose = pose.pose;
      subscribed_state.twist.twist.linear.x = stateSequence_[3];
      subscribed_state.twist.twist.linear.y = stateSequence_[4];
      subscribed_state.twist.twist.angular.z = stateSequence_[5];
    }
  }
  subscribed_state.header.stamp = begin;
  subscribed_state.header.frame_id = "odom";
  path_msg_.header.stamp = begin;
  path_msg_.header.frame_id = "odom";
  path_pub_.publish(path_msg_);
}

void OmniWheelRobotPlant::pubControl(OmniWheelRobotPlant::ControlVector wheel_commands)
{
  RobotWheelCommandsMsgQueueEntry wheel_commands_queue_entry;
  ros::Time curr_time = ros::Time::now();
  wheel_commands_queue_entry.timestamp_secs = curr_time.sec;
  wheel_commands_queue_entry.timestamp_nano_secs_correction = curr_time.nsec;

  const boost::posix_time::ptime t_timeout =
      boost::posix_time::microsec_clock::universal_time() +
      boost::posix_time::milliseconds(IPC_QUEUE_TIMEOUT_MS);

  if (wheel_commands.hasNaN()){ 
    ROS_INFO("NaN Control Input Detected");
    wheel_commands_queue_entry.front_left_milli_newton = 0;
    wheel_commands_queue_entry.front_right_milli_newton = 0;
    wheel_commands_queue_entry.back_left_milli_newton = 0;
    wheel_commands_queue_entry.back_right_milli_newton = 0;

    robot_wheel_commands_message_queue.timed_send(
        &wheel_commands_queue_entry, sizeof(wheel_commands_queue_entry), 0, t_timeout);

    ros::shutdown(); //No use trying to recover, quitting is the best option.
  } else {
    wheel_commands_queue_entry.front_left_milli_newton = wheel_commands(0) * 1000;
    wheel_commands_queue_entry.front_right_milli_newton = wheel_commands(1) * 1000;
    wheel_commands_queue_entry.back_left_milli_newton = wheel_commands(2) * 1000;
    wheel_commands_queue_entry.back_right_milli_newton = wheel_commands(3) * 1000;

    ROS_INFO_STREAM("Transmitting Commands: " << wheel_commands_queue_entry);
    bool send_succeeded = robot_wheel_commands_message_queue.timed_send(
        &wheel_commands_queue_entry, sizeof(wheel_commands_queue_entry), 0, t_timeout);
    if (!send_succeeded){
      ROS_WARN_STREAM("Failed to send wheel commands, queue probably full!");
    }
  }
}

void OmniWheelRobotPlant::pubStatus(const ros::TimerEvent&){
  boost::mutex::scoped_lock lock(access_guard_);
  status_msg_.info = ocs_msg_;
  status_msg_.status = status_;
  status_msg_.header.stamp = ros::Time::now();
  status_pub_.publish(status_msg_);
}

void OmniWheelRobotPlant::pubControllerTypeDebug(const ros::TimerEvent&){
  visualization_msgs::Marker points;
  points.id = controller_type_debug_point_id;
  controller_type_debug_point_id++;
  points.type = visualization_msgs::Marker::POINTS;
  points.scale.x = 0.2;
  points.scale.y = 0.2;

  // Color is based on controller type
  switch(controller_type_used_for_solution_){
    case ControllerType::NONE:
      // Fully transparent point
      points.color.a = 0.0;
      break;
    case ControllerType::ACTUAL_STATE:
      points.color.a = 1.0;
      points.color.g = 1.0;
      points.color.b = 0.0;
      points.color.r = 0.0;
      break;
    case ControllerType::PREDICTED_STATE:
      points.color.a = 1.0;
      points.color.g = 0.0;
      points.color.b = 0.0;
      points.color.r = 1.0;
      break;
  }

  // Location of the point is just the current robot position
  geometry_msgs::Point robot_location;
  robot_location.x = getState().x_pos;
  robot_location.y = getState().y_pos;
  points.points.push_back(robot_location);

  // TODO: get this frame from somewhere else? Should not be hardcoded
  points.header.frame_id = "odom";

  debug_controller_type_pub_.publish(points);
}

OmniWheelRobotPlant::FullState OmniWheelRobotPlant::getState()
{
  boost::mutex::scoped_lock lock(access_guard_);
  return full_state_;
}

OmniWheelRobotPlant::StateVector OmniWheelRobotPlant::getStateVector()
{
  StateVector state_vector;
  boost::mutex::scoped_lock lock(access_guard_);
  state_vector << full_state_.x_pos, full_state_.y_pos, full_state_.yaw, 
                  full_state_.x_vel, full_state_.y_vel, full_state_.angular_vel;
  return state_vector;
}

void OmniWheelRobotPlant::setStateFromVector(StateVector new_state){
  full_state_.x_pos = new_state(0);
  full_state_.y_pos = new_state(1);
  full_state_.yaw = new_state(2);
  full_state_.x_vel = new_state(3);
  full_state_.y_vel = new_state(4);
  full_state_.angular_vel = new_state(5);
}

ros::Time OmniWheelRobotPlant::getLastPoseTime()
{
  boost::mutex::scoped_lock lock(access_guard_);
  return last_pose_call_;
}

int OmniWheelRobotPlant::checkStatus()
{
  boost::mutex::scoped_lock lock(access_guard_);
  if (!activated_) {
    status_ = 1;
    ocs_msg_ = "No pose estimates received.";
  }
  else if (safe_speed_zero_){
    status_ = 1;
    ocs_msg_ = "Safe speed zero.";
  }
  else {
    ocs_msg_ = "Controller OK";
    status_ = 0; //Everything is good.
  }
  return status_;
}

void OmniWheelRobotPlant::dynRcfgCall(autorally_control::OmniWheelRobotPathIntegralParamsConfig &config, int lvl)
{
  boost::mutex::scoped_lock lock(access_guard_);
  costParams_.max_wheel_speed = config.max_wheel_speed;
  costParams_.desired_speed = config.desired_speed;
  costParams_.speed_coefficient = config.speed_coefficient;
  costParams_.track_coefficient = config.track_coefficient;
  costParams_.crash_coefficient = config.crash_coefficient;
  costParams_.track_slop = config.track_slop;
  hasNewCostParams_ = true;
}

bool OmniWheelRobotPlant::hasNewDynRcfg()
{
  return hasNewCostParams_;
}

autorally_control::OmniWheelRobotPathIntegralParamsConfig OmniWheelRobotPlant::getDynRcfgParams()
{
  boost::mutex::scoped_lock lock(access_guard_);
  hasNewCostParams_ = false;
  return costParams_;
}

void OmniWheelRobotPlant::shutdown()
{
  //Shutdown timers, subscribers, and dynamic reconfigure
  boost::mutex::scoped_lock lock(access_guard_);
  path_pub_.shutdown();
  pathTimer_.stop();
  statusTimer_.stop();
  debugImgTimer_.stop();
  timingInfoTimer_.stop();
  //server_.clearCallback();
}

} //namespace autorally_control
