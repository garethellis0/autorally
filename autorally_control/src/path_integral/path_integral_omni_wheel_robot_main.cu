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
/**********************************************
 * @file path_integral_omni_wheel_robot_main.cpp
 * @author Gareth Ellis 
 * @date February 9, 2020
 * @brief Main file for the model predictive control systems for a 4-wheeled
 *        omni-wheel robot
 ***********************************************/

//Some versions of boost require __CUDACC_VER__, which is no longer defined in CUDA 9. This is
//the old expression for how it was defined, so should work for CUDA 9 and under.
#define __CUDACC_VER__ __CUDACC_VER_MAJOR__ * 10000 + __CUDACC_VER_MINOR__ * 100 + __CUDACC_VER_BUILD__

#include <autorally_control/path_integral/meta_math.h>
#include <autorally_control/path_integral/param_getter.h>
#include <autorally_control/path_integral/omni_wheel_robot_plant.h>
#include <autorally_control/OmniWheelRobotPathIntegralParamsConfig.h>
#include <autorally_control/path_integral/omni_wheel_robot_costs.cuh>

#include <autorally_control/path_integral/omni_wheel_robot_dynamics_model.cuh>
#include <autorally_control/path_integral/mppi_controller.cuh>
#include <autorally_control/path_integral/run_control_loop.cuh>

#include <ros/ros.h>
#include <atomic>

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

using namespace autorally_control;

typedef OmniWheelRobotDynamicsModel DynamicsModel;
typedef OmniWheelRobotCosts Costs;

// Parameters taken from the autorally basis functions dynamics model
const int MPPI_NUM_ROLLOUTS__ = 2560;
const int BLOCKSIZE_X = 16;
const int BLOCKSIZE_Y = 4;
// Parameters taken from the neural network dynamics model
//const int MPPI_NUM_ROLLOUTS__ = 1920;
//const int BLOCKSIZE_X = 8;
//const int BLOCKSIZE_Y = 16;

//Convenience typedef for the MPPI Controller.
typedef MPPIController<DynamicsModel, Costs, MPPI_NUM_ROLLOUTS__, 
                       BLOCKSIZE_X, BLOCKSIZE_Y> Controller;

int main(int argc, char** argv) {
  //Ros node initialization
  ros::init(argc, argv, "mppi_controller");

  ros::NodeHandle mppi_node("~");

  Costs* costs = new Costs(mppi_node);

  const float max_wheel_speed = getRosParam<float>("max_wheel_speed", mppi_node);
  const float controls_frequency = 
    getRosParam<float>("controls_frequency", mppi_node);
  DynamicsModel* model = 
    new DynamicsModel(1.0/controls_frequency, max_wheel_speed);


  //Define the controller
  const float init_u[4] = {0, 0, 0, 0};
  const float wheel_speed_exploration_variance = 
    getRosParam<float>("wheel_speed_exploration_variance", mppi_node);
  const float controls_variance[4] = {
    wheel_speed_exploration_variance,
    wheel_speed_exploration_variance,
    wheel_speed_exploration_variance,
    wheel_speed_exploration_variance
  };
  const int num_timesteps = getRosParam<int>("num_timesteps", mppi_node);
  const float norm_exp_kernel_gamma = 
    getRosParam<float>("norm_exp_kernel_gamma", mppi_node);
  const int num_optimization_iters = 
    getRosParam<int>("num_optimization_iters", mppi_node);
  const int optimization_stride = 
    getRosParam<int>("optimization_stride", mppi_node);
  Controller* actual_state_controller = 
    new Controller(model, costs, num_timesteps, controls_frequency, 
        norm_exp_kernel_gamma, wheel_speed_exploration_standard_deviation, 
        init_u, num_optimization_iters, optimization_stride);
  Controller* predicted_state_controller = 
    new Controller(model, costs, num_timesteps, controls_frequency, 
        norm_exp_kernel_gamma, wheel_speed_exploration_standard_deviation, 
        init_u, num_optimization_iters, optimization_stride);

  AutorallyPlant* robot = new AutorallyPlant(mppi_node, mppi_node, params.debug_mode, params.hz, false);

  //Setup dynamic reconfigure callback
  dynamic_reconfigure::Server<PathIntegralParamsConfig> server;
  dynamic_reconfigure::Server<PathIntegralParamsConfig>::CallbackType callback_f;
  callback_f = boost::bind(&AutorallyPlant::dynRcfgCall, robot, _1, _2);
  server.setCallback(callback_f);

  boost::thread optimizer;

  std::atomic<bool> is_alive(true);
  optimizer = boost::thread(
      &runControlLoop<Controller>, predicted_state_controller, 
      actual_state_controller, robot, &params, &mppi_node, &is_alive);

  ros::spin();

  //Shutdown procedure
  is_alive.store(false);
  optimizer.join();
  robot->shutdown();
  actual_state_controller->deallocateCudaMem();
  predicted_state_controller->deallocateCudaMem();
  delete robot;
  delete actual_state_controller;
  delete predicted_state_controller;
  delete costs;
  delete model;
}
