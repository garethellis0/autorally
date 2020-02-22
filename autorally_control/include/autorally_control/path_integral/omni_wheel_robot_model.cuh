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
 * @author Gareth Ellis
 * @date February 11, 2020
 * @brief Class definition for a 4 Omni-Wheeled Robot With Wheel Slip
 ***********************************************/

#ifndef OMNI_WHEEL_ROBOT_MODEL_CUH_
#define OMNI_WHEEL_ROBOT_MODEL_CUH_

#include "managed.cuh"
#include "meta_math.h"
#include "gpu_err_chk.h"
#include "cnpy.h"

#include <Eigen/Dense>

namespace autorally_control {

class OmniWheelRobotModel: public Managed
{
public:

  static const int STATE_DIM = 9;
  static const int CONTROL_DIM = 4;

  OmniWheelRobotModel() = delete;

  /**
   * Create an instance of this model

   * @param dt The timestep for a single state update step.
   * @param max_abs_wheel_speed The maximum absolute permissable wheel speed
   */
  OmniWheelRobotModel(double dt, double max_abs_wheel_speed);

  /**
   * Performs CUDA memory copies to move anything relevant to this model into
   * GPU memory
   */
  void paramsToDevice();

  /**
   * Free any CUDA memory allocated by this model
   */
  void freeCudaMem();

  /**
   * Modifies the control and state in-place to obey this models constraints
   *
   * @param state The state to enforce constraints on
   * @param control The controls to enforce constraints on
   */
  void enforceConstraints(Eigen::MatrixXf &state, Eigen::MatrixXf &control);

  /**
   * Updates the given state and controls after a single timestep
   * @param state The current state, this will be updated in-place to the state
   *              values after a single timestep. Constraints will be 
   *              enforced on this before updating.
   * @param control The set of controls to apply for a single timestep to 
   *                compute the next state. Constraints will be enforced on 
   *                this before updating.
   */
  void updateState(Eigen::MatrixXf &state, Eigen::MatrixXf &control);

  /**
   * Update the state_der_ components that can be determined solely 
   * from the current robot state
   *
   * @param state The current robot state (TODO: WHAT COORDINATE FRAME?)
   */
  void computeKinematics(Eigen::MatrixXf &state);

  /**
   * Update the state_der_ components that cannot be determined solely from
   * robot state
   *
   * @param state The current robot state
   * @param control The set of controls being applied to the robot
   */
  void computeDynamics(Eigen::MatrixXf &state, Eigen::MatrixXf &control);

  /**
   * Does nothing, but is required because of templating
   */
  void updateModel(std::vector<int> description, std::vector<float> data){};

  /**
   * Initialize shared CUDA memory
   *
   * @param theta_s A pointer to a block of shared CUDA memory. Must be of size
   *                SHARED_MEM_REQUEST_GRD + SHARED_MEM_REQUEST_BLK * # blocks 
   */
  __device__ void cudaInit(float* theta_s);

  /**
   * Modifies the control and state in-place to obey this models constraints
   *
   * @param state The state to enforce constraints on
   * @param control The controls to enforce constraints on
   */
  __device__ void enforceConstraints(float* state, float* control);

  /**
   * Update the state_der_ components that can be determined solely 
   * from the current robot state
   *
   * @param state The current robot state
   */
  __device__ void computeKinematics(float* state, float* state_der);

  /**
   * Update the state_der_ components that require the robot state and controls
   * 
   * @param state An array of length STATE_DIM holding the current state of the 
   *              robot
   * @param control An array of length CONTROL_DIM holding the controls 
   *                currently being given to the robot
   * @param state_der An array of length STATE_DIM that will be set to the 
   *                  derivative of the current state of the robot
   * @param theta_s The shared memory allocated for this model and setup via 
   *                `cudaInit` 
   */
  __device__ void computeDynamics(float* state, float* control, float* state_der, float* theta_s);

  /**
   * Computes the state derivative for the model
   * 
   * @param state An array of length STATE_DIM holding the current state of the 
   *              robot
   * @param control An array of length CONTROL_DIM holding the controls 
   *                currently being given to the robot
   * @param state_der An array of length STATE_DIM that will be set to the 
   *                  derivative of the current state of the robot
   * @param theta_s The shared memory allocated for this model and setup via 
   *                `cudaInit` 
   */
  __device__ void computeStateDeriv(float* state, float* control, float* state_der, float* theta_s);

  /**
   * Increments the given state by a single timestep
   * 
   * @param state The current state, this will be updated in-place to the state
   *              after a single timestep
   * @param state_der The derivative of the current state. This will be set to 
   *                  zero.
   */
  __device__ void incrementState(float* state, float* state_der);



  // TODO: populate and use both of these arrays
  // An array of length CONTROL_DIM of pairs of [min_value, max_value] for 
  // each control signal
  float2* control_rngs_;
  // The same as "control_rngs_", but for the GPU
  float2* control_rngs_d_;

  // The base amount of CUDA memory this model requires, in bytes
  // NOTE: We don't actually need any memory, but cuda won't allow allocation
  //       of zero-sized arrays
  static const int SHARED_MEM_REQUEST_GRD = 1;

  // The per-block amount of CUDA memory this model requires, in bytes
  // NOTE: We don't actually need any memory, but cuda won't allow allocation
  //       of zero-sized arrays
  static const int SHARED_MEM_REQUEST_BLK = 1; 

  /**
   * The most recently computed state derivative
   */
  Eigen::Matrix<float, STATE_DIM, 1> state_der_;

private:

  double dt_;
  double max_abs_wheel_speed_;
};

}

#include "omni_wheel_robot_model.cu"

#endif /*OMNI_WHEEL_ROBOT_MODEL_CUH_*/
