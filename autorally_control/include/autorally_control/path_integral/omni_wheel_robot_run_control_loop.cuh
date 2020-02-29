/*
* Software License Agreement (BSD License)
* Copyright (c) 2013, Georgia Institute of Technology
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*
* 1. Redistributions of source code must retain the above copyright notice, this
* list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright notice,
* this list of conditions and the following disclaimer in the documentation
* and/or other materials provided with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
* AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
* IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifndef OMNI_WHEEL_ROBOT_RUN_CONTROLLER_CUH_
#define OMNI_WHEEL_ROBOT_RUN_CONTROLLER_CUH_

#include "omni_wheel_robot_plant.h"
#include "param_getter.h"

#include <autorally_control/OmniWheelRobotPathIntegralParamsConfig.h>
#include <autorally_control/ddp/ddp_model_wrapper.h>
#include <autorally_control/ddp/ddp_tracking_costs.h>
#include <autorally_control/ddp/ddp.h>

#include <opencv2/core/core.hpp>
#include <atomic>

#include <boost/thread/thread.hpp>
#include <unistd.h>
#include <chrono>

#include <ros/ros.h>

namespace autorally_control {


/**
 * @brief Run the control loop
 *
 * This will run both a controller that computes a trajectory based on the 
 * current state of the robot, and a seperate controller that will use it's
 * own internally predicted state as the starting point for trajectory 
 * generation. It will compare the outputs of these two controllers and 
 * decide which to use based on the score of each trajectory. 
 *
 * If the score of the trajectory computed by the controller working from the
 * actual state is better then that of the trajectory generated by the 
 * controller working entirely from it's own internal prediction of the 
 * robot state, the trajectory starting from the actual robot state will be
 * used and the controller working from the predicted state will have it's
 * state reset to the actual robot state.
 * 
 * @param predicted_state_controller The controller to use to compute a
 *                                   trajectory based on it's prediction for 
 *                                   what the robot state should be
 * @param actual_state_controller This controller will continuously compute
 *                                a new trajectory from the actual current 
 *                                state of the robot
 * @param robot The robot to control
 * @param param General parameters
 * @param mppi_node The nodehandle for the ROS node this is running in
 * @param is_alive ???????????
 */
template <class CONTROLLER_T, class PLANT_T> 
void omniWheelRobotRunControlLoop(
    CONTROLLER_T* predicted_state_controller, 
    CONTROLLER_T* actual_state_controller,
    PLANT_T* robot, 
    SystemParams* params, 
    ros::NodeHandle* mppi_node, 
    std::atomic<bool>* is_alive
    )
{  
  //Initial condition of the robot
  Eigen::MatrixXf state = Eigen::Matrix<float, PLANT_T::STATE_DIM, 1>::Zero();
  // TODO: we really should just have the entire state as a list param in the
  //       launch file and pass it directly through to the plant, as that is
  //       more general
  state << params->x_pos, params->y_pos, params->heading, 0, 0, 0;
  robot->setStateFromVector(state);
  
  //Initial control value
  Eigen::MatrixXf u = Eigen::Matrix<float, PLANT_T::CONTROL_DIM, 1>::Zero();

  std::vector<float> controlSolution;
  std::vector<float> stateSolution;

  //Obstacle and map parameters
  std::vector<int> obstacleDescription;
  std::vector<float> obstacleData;
  std::vector<int> costmapDescription;
  std::vector<float> costmapData;
  std::vector<int> modelDescription;
  std::vector<float> modelData;

  //Counter, timing, and stride variables.
  int num_iter = 0;
  int status = 1;
  int optimization_stride = getRosParam<int>("optimization_stride", *mppi_node);
  bool use_feedback_gains = getRosParam<bool>("use_feedback_gains", *mppi_node);
  double avgOptimizationLoopTime = 0; //Average time between pose estimates
  double avgOptimizationTickTime = 0; //Avg. time it takes to get to the sleep at end of loop
  double avgSleepTime = 0; //Average time spent sleeping
  ros::Time last_pose_update = robot->getLastPoseTime();
  ros::Duration optimizationLoopTime(optimization_stride/(1.0*params->controls_frequency));

  //Set the loop rate
  std::chrono::milliseconds ms{(int)(optimization_stride*1000.0/params->controls_frequency)};

  if (!params->debug_mode){
    while(last_pose_update == robot->getLastPoseTime() && is_alive->load()){ //Wait until we receive a pose estimate
      usleep(50);
    }
  }

  state = robot->getStateVector();
  actual_state_controller->setState(state);
  predicted_state_controller->setState(state);

  actual_state_controller->resetControls();
  actual_state_controller->computeFeedbackGains(state);
  predicted_state_controller->resetControls();
  predicted_state_controller->computeFeedbackGains(state);

  //Start the control loop.
  while (is_alive->load()) {
    std::chrono::steady_clock::time_point loop_start = std::chrono::steady_clock::now();
    robot->setTimingInfo(avgOptimizationLoopTime, avgOptimizationTickTime, avgSleepTime);
    num_iter ++;

    if (params->debug_mode){ //Display the debug window.
      // TODO: need two debug windows, (actual and predicted states)

     // display costs around actual robot state
     //cv::Mat debug_img = actual_state_controller->costs_->getDebugDisplay(state(0), state(1), state(2));
     //robot->setDebugImage(debug_img);

     // display costs around predicted robot state
     std::vector<float> controller_state_sequence = predicted_state_controller->getStateSeq();
     cv::Mat debug_img = predicted_state_controller->costs_->getDebugDisplay(
         controller_state_sequence[0], controller_state_sequence[1], controller_state_sequence[2]);
     robot->setDebugImage(debug_img);
    }

    //Update the state estimate
    if (last_pose_update != robot->getLastPoseTime()){
      optimizationLoopTime = robot->getLastPoseTime() - last_pose_update;
      last_pose_update = robot->getLastPoseTime();
      state = robot->getStateVector();
    }

    //Update the cost parameters
    if (robot->hasNewDynRcfg()){
      actual_state_controller->costs_->updateParams_dcfg(robot->getDynRcfgParams());
      predicted_state_controller->costs_->updateParams_dcfg(robot->getDynRcfgParams());
    }
  
    // Figure out how many controls have been published since we were last here 
    // and slide the control and state sequence by that much.
    int stride = round(optimizationLoopTime.toSec()*params->controls_frequency);
    if (status != 0){
      stride = optimization_stride;
    }
    if (stride >= 0 && stride < params->num_timesteps){
      actual_state_controller->slideControlAndStateSeq(stride);
      predicted_state_controller->slideControlAndStateSeq(stride);
    }

    //Compute a new control sequence
    actual_state_controller->computeControl(state);
    predicted_state_controller->computeControl();
    if (use_feedback_gains){
      // TODO: only need to compute feedback gains from one of the controllers 
      //       after we decide which one we want to take the trajectory from
      actual_state_controller->computeFeedbackGains(state);
      predicted_state_controller->computeFeedbackGains(state);
    }

    // TODO: shouldn't neeed to call this here....
    auto feedback_gain = predicted_state_controller->getFeedbackGains().feedback_gain;

    //Decide what control sequence to use
    // TODO: using "NONE" to indicate what is really "either" is deceptive, fix it
    ControllerType controller_to_use = ControllerType::NONE;
    if (params->use_only_actual_state_controller && 
        params->use_only_predicted_state_controller){
      ROS_WARN_STREAM("use_only_actual_state_controller and"  <<
          "use_only_actual_predicted_controller both set to true, so ignoring both!");
    } else if (params->use_only_actual_state_controller && 
        !params->use_only_predicted_state_controller){
      controller_to_use = ControllerType::ACTUAL_STATE;
    } else if (!params->use_only_actual_state_controller && 
        params->use_only_predicted_state_controller) {
      controller_to_use = ControllerType::PREDICTED_STATE;
    }

    // TODO: clean this up, gross amount of duplication
    ControllerType controller_used = ControllerType::NONE;

    // TODO: delete this
    controller_to_use = ControllerType::ACTUAL_STATE;

    switch(controller_to_use){
      case ControllerType::NONE:
        if(actual_state_controller->getComputedTrajectoryCost() < 
            predicted_state_controller->getComputedTrajectoryCost()){
          controlSolution = actual_state_controller->getControlSeq();
          stateSolution = actual_state_controller->getStateSeq();
          feedback_gain = actual_state_controller->getFeedbackGains().feedback_gain;

          // If the actual state controller came up with a cheaper trajectory,
          // We update the predicted state controller with the information from
          // the actual state controller
          predicted_state_controller->setStateSequence(stateSolution);
          predicted_state_controller->setControlSequence(controlSolution);

          controller_used = ControllerType::ACTUAL_STATE;
          ROS_INFO_STREAM("Using actual state controller");
        } else {
          controlSolution = predicted_state_controller->getControlSeq();
          stateSolution = predicted_state_controller->getStateSeq();
          feedback_gain = predicted_state_controller->getFeedbackGains().feedback_gain;
          controller_used = ControllerType::PREDICTED_STATE;
          ROS_INFO_STREAM("Using predicted state controller");
        }
        break;
      case ControllerType::ACTUAL_STATE:
        controlSolution = actual_state_controller->getControlSeq();
        stateSolution = actual_state_controller->getStateSeq();
        feedback_gain = actual_state_controller->getFeedbackGains().feedback_gain;
        controller_used = ControllerType::ACTUAL_STATE;
        ROS_INFO_STREAM("Using actual state controller");
        break;
      case ControllerType::PREDICTED_STATE:
        controlSolution = predicted_state_controller->getControlSeq();
        stateSolution = predicted_state_controller->getStateSeq();
        feedback_gain = predicted_state_controller->getFeedbackGains().feedback_gain;
        controller_used = ControllerType::PREDICTED_STATE;
        ROS_INFO_STREAM("Using predicted state controller");
        break;
    }

    // TODO: remove this printout
    std::cout << "Control Solution: ";
    for (auto elem : controlSolution) {
      std::cout << elem << ", ";
    }
    std::cout << std::endl;
    std::cout << "State Solution: ";
    for (auto elem : stateSolution) {
      std::cout << elem << ", ";
    }
    std::cout << std::endl;

    //Set the updated solution for execution
    robot->setSolution(stateSolution, controlSolution, feedback_gain, 
        last_pose_update, avgOptimizationLoopTime, controller_used);
    
    //Check the robots status
    status = robot->checkStatus();

    //Increment the state if debug mode is set to true
    if (status != 0 && params->debug_mode){
      for (int t = 0; t < optimization_stride; t++){
        u << controlSolution[2*t], controlSolution[2*t + 1];
        actual_state_controller->model_->updateState(state, u); 
        predicted_state_controller->model_->updateState(state, u); 
      }
    }
    
    //Sleep for any leftover time in the control loop
    std::chrono::duration<double, std::milli> fp_ms = std::chrono::steady_clock::now() - loop_start;
    double optimizationTickTime = fp_ms.count();
    int count = 0;
    while(is_alive->load() && (fp_ms < ms || ((robot->getLastPoseTime() - last_pose_update).toSec() < (1.0/params->controls_frequency - 0.0025) && status == 0))) {
      usleep(50);
      fp_ms = std::chrono::steady_clock::now() - loop_start;
      count++;
    }
    double sleepTime = fp_ms.count() - optimizationTickTime;

    //Update the average loop time data
    avgOptimizationLoopTime = (num_iter - 1.0)/num_iter*avgOptimizationLoopTime + 1000.0*optimizationLoopTime.toSec()/num_iter; 
    avgOptimizationTickTime = (num_iter - 1.0)/num_iter*avgOptimizationTickTime + optimizationTickTime/num_iter;
    avgSleepTime = (num_iter - 1.0)/num_iter*avgSleepTime + sleepTime/num_iter;
  }
}

}

#endif /*OMNI_WHEEL_ROBOT_RUN_CONTROLLER_CUH_*/
