// TODO: the duplication of work between cuda-specific and non-cuda-specific
//       things here is *insane*. Revise if possible.

#include "vector_math.cuh"

namespace autorally_control {


OmniWheelRobotModel::OmniWheelRobotModel(double dt, double max_abs_wheel_speed) :
  dt_(dt), 
  max_abs_wheel_speed_(max_abs_wheel_speed),
  ROBOT_MOMENT_OF_INERTIA(0.5 * ROBOT_MASS_KG * pow(ROBOT_RADIUS_M, 2.0))
  {
    // TODO: this was copied from `generlized_linear`. Looks like like only the
    //       first two elements are used by the `mppi_controller` though. 
    //       This is probably wrong.
    control_rngs_ = new float2[CONTROL_DIM];
    for (int i = 0; i < CONTROL_DIM; i++){
      control_rngs_[i].x = -FLT_MAX;
      control_rngs_[i].y = FLT_MAX;
    }
  }

void OmniWheelRobotModel::paramsToDevice(){}

void OmniWheelRobotModel::freeCudaMem(){}

void OmniWheelRobotModel::enforceConstraints(Eigen::MatrixXf &state, Eigen::MatrixXf &control){}

void OmniWheelRobotModel::updateState(Eigen::MatrixXf &state, Eigen::MatrixXf &control){

  enforceConstraints(state, control);
  computeKinematics(state);
  computeDynamics(state, control);

  state += state_der_*dt_;
  state_der_ *= 0;
}

void OmniWheelRobotModel::computeKinematics(Eigen::MatrixXf &state) {
  // Here we compute the derivative by rotating the higher order
  // state variables (velocity and acceleration) into the robot frame
  static_assert(KINEMATICS_DIM == 3, "");
  static_assert(STATE_DIM == 6, "");

  // The change in the 0th derivative terms (x, y, yaw) is from the 1st 
  // derivative terms (v_x, v_y, v_angular)
  state_der_(0) = cosf(state(2))*state(3) - sinf(state(2))*state(4);
  state_der_(1) = sinf(state(2))*state(3) + cosf(state(2))*state(4);
  state_der_(2) = state(5); 

  // TODO: delete if unused
  //// The change in the 1st derivative terms (x, y, yaw) is from the 2nd 
  //// derivative terms (a_x, a_y, a_angular)
  //state_der_(3) = cosf(state(2))*state(6) - sinf(state(2))*state(7);
  //state_der_(4) = sinf(state(2))*state(6) + cosf(state(2))*state(7);
  //state_der_(5) = state(8); 
}

void OmniWheelRobotModel::computeDynamics(Eigen::MatrixXf &state, Eigen::MatrixXf &control){
  // We delegate to the version of this function built for CUDA. This is 
  // probably slightly less efficient overall (as compared to using native
  // Eigen functions to do the same thing), but seriously reduces the 
  // potential for bugs from re-implementing an algorithm twice, with no
  // tests on either implementation.....
  static_assert(KINEMATICS_DIM == 3, "");
  static_assert(STATE_DIM == 6, "");

  float state_array[KINEMATICS_DIM];
  for (int i = 0; i < KINEMATICS_DIM; i++){
    state_array[i] = state(i);
  }
  float control_array[CONTROL_DIM];
  for (int i = 0; i < CONTROL_DIM; i++){
    control_array[i] = control(i);
  }

  float state_der_array[6];
  computeDynamics(state_array, control_array, state_der_array, NULL);

  state_der_(3) = state_der_array[3];
  state_der_(4) = state_der_array[4];
  state_der_(5) = state_der_array[5];
}

__device__ void OmniWheelRobotModel::cudaInit(float* theta_s){}

__device__ void OmniWheelRobotModel::enforceConstraints(float* state, float* control){
  // TODO?
}

__device__ void OmniWheelRobotModel::computeKinematics(float* state, float* state_der){
  // Here we compute the derivative by rotating the higher order
  // state variables (velocity and acceleration) into the robot frame
  static_assert(KINEMATICS_DIM == 3, "");
  static_assert(STATE_DIM == 6, "");

  // The change in the 0th derivative terms (x, y, yaw) is from the 1st 
  // derivative terms (v_x, v_y, v_angular)
  state_der_[0] = cosf(state[2])*state[3] - sinf(state[2])*state[4];
  state_der_[1] = sinf(state[2])*state[3] + cosf(state[2])*state[4];
  state_der_[2] = state[5]; 

  // TODO: delete if unused
  //// The change in the 1st derivative terms (x, y, yaw) is from the 2nd 
  //// derivative terms (a_x, a_y, a_angular)
  //state_der_[3] = cosf(state[2])*state[6] - sinf(state[2])*state[7];
  //state_der_[4] = sinf(state[2])*state[6] + cosf(state[2])*state[7];
  //state_der_[5] = state[8]; 
}

__host__ __device__ void OmniWheelRobotModel::computeDynamics(
    float* state, float* control, float* state_der, float* theta_s){ 

  // TODO: if performance is an issue, there are several things here that
  //       we can cache and not recompute, like the wheel vectors

  // TODO: Many names are copy-pasted from the python model, we really should
  //       go through and clean them up

  // This function assumes there are 4 control inputs, one for each wheel
  static_assert(CONTROL_DIM == 4, "");
  // This function assumes that the dynamics output is of the form
  // [a_x, a_y, a_angular], and that these are the elements 7,8,9 of
  // the state vector
  static_assert(KINEMATICS_DIM == 3, "");
  static_assert(DYNAMICS_DIM == 3, "");

  // NOTE: unless otherwise specified, all arrays of wheel specific values 
  //       go [front left, back left, back right, front right]

  // Setup wheel position vectors, vectors from the center of the robot
  // to the center of each wheel
  float wheel_position_vectors[4][3];
  const float positive_x_vec[3] = {ROBOT_RADIUS_M, 0, 0};
  rotateVector3AboutZAxis(positive_x_vec, FRONT_WHEEL_ANGLE_RAD, wheel_position_vectors[0]);
  rotateVector3AboutZAxis(positive_x_vec, M_PI - REAR_WHEEL_ANGLE_RAD, wheel_position_vectors[1]);
  rotateVector3AboutZAxis(positive_x_vec, M_PI + REAR_WHEEL_ANGLE_RAD, wheel_position_vectors[2]);
  rotateVector3AboutZAxis(positive_x_vec, -FRONT_WHEEL_ANGLE_RAD, wheel_position_vectors[3]);
  float unit_wheel_position_vectors[4][3];
  for (int i = 0; i < 4; i++){
    getUnitVectorInDirection(wheel_position_vectors[i], 
        unit_wheel_position_vectors[i]);
  }

  // Setup wheel orientation vectors, vectors pointing in the direction that
  // each wheel is oriented
  float wheel_orientation_vectors[4][3];
  const float positive_y_vec[3] = {0, 1, 0};
  rotateVector3AboutZAxis(positive_y_vec, FRONT_WHEEL_ANGLE_RAD, wheel_orientation_vectors[0]);
  rotateVector3AboutZAxis(positive_y_vec, M_PI - REAR_WHEEL_ANGLE_RAD, wheel_orientation_vectors[1]);
  rotateVector3AboutZAxis(positive_y_vec, M_PI + REAR_WHEEL_ANGLE_RAD, wheel_orientation_vectors[2]);
  rotateVector3AboutZAxis(positive_y_vec, -FRONT_WHEEL_ANGLE_RAD, wheel_orientation_vectors[3]);
  float unit_wheel_orientation_vectors[4][3];
  for (int i = 0; i < 4; i++){
    getUnitVectorInDirection(wheel_orientation_vectors[i], 
        unit_wheel_orientation_vectors[i]);
  }

  float curr_orientation = state[2];
  // Rotation matrix from global coordinates to robot-relative coordinates
  float R_0_to_M[3][3];
  createRotationMatrixAboutZAxis(curr_orientation, R_0_to_M);
  // Rotation matrix from robot-relative coordinates to global coordinates
  float R_M_to_0[3][3];
  createRotationMatrixAboutZAxis(-curr_orientation, R_0_to_M);

  // Velocity vector of the robot in the global frame
  float v_G[3] = {state[3], state[4], 0};

  // Angular velocity vector
  float w[3] = {0, 0, state[5]};

  // Scale the wheelspeeds
  // TODO: not sure what this is needed for... dbl check in python model
  float scaled_wheel_speeds[4];
  for (int i = 0; i < 4; i++){
    scaled_wheel_speeds[i] = 4 * pow(M_PI, 2.0) / control[i];
  }

  // For each wheel, compute the component of it's velocity in the direction
  // that it's pointing, as well as in the direction tangent to the direction
  // that it's pointing. Then compute the force resulting from that wheel
  float wheel_forces[4][3];
  float force_angular = 0;
  for (int i = 0; i < 4; i++){
    float transformed_wheel_orientation_vec[3];
    float transformed_wheel_position_vec[3];
    multiplyVector3By3x3Matrix(R_M_to_0, unit_wheel_position_vectors[i], 
        transformed_wheel_position_vec);
    multiplyVector3By3x3Matrix(R_M_to_0, unit_wheel_orientation_vectors[i], 
        transformed_wheel_position_vec);

    // TODO: Better name for this. I mean *REALLY*.
    float cross_result[3];
    crossProductVector3(w, transformed_wheel_position_vec, cross_result);

    float v_W = dotProductVector3(v_G, transformed_wheel_orientation_vec) +
      dotProductVector3(cross_result, transformed_wheel_orientation_vec) +
      WHEEL_RADIUS_M*scaled_wheel_speeds[i];
    float v_T = dotProductVector3(v_G, transformed_wheel_position_vec) +
      dotProductVector3(cross_result, transformed_wheel_position_vec);

    // TODO: better names for these?
    float scaled_transformed_wheel_orientation_vec[3];
    multiplyVector3ByScalar(transformed_wheel_orientation_vec, 
                            computeWheelFrictionCoeffInWheelDir(v_W),
                            scaled_transformed_wheel_orientation_vec);
    float scaled_transformed_wheel_position_vec[3];
    multiplyVector3ByScalar(transformed_wheel_position_vec, 
                            computeWheelFrictionCoeffInTransverseDir(v_T),
                            scaled_transformed_wheel_position_vec);

    float summed_acceleration[3];
    addVector3(scaled_transformed_wheel_orientation_vec, 
        scaled_transformed_wheel_position_vec, summed_acceleration);

    multiplyVector3ByScalar(summed_acceleration, -ROBOT_MASS_KG*9.8/4.0, wheel_forces[i]);

    // TODO: better name for this
    float cross_result1[3];
    crossProductVector3(transformed_wheel_position_vec, wheel_forces[i], cross_result1);

    const float positive_z_vec[3] = {0, 0, 1};
    force_angular += dotProductVector3(positive_z_vec, cross_result1);
  }

  float total_force[3] = {0, 0, 0};
  for (int i = 0; i < 4; i++){
    addVector3(total_force, wheel_forces[i], total_force);
  }

  float force_x = dotProductVector3(positive_x_vec, total_force);
  float force_y = dotProductVector3(positive_y_vec, total_force);

  const float a_x = force_x / ROBOT_MASS_KG;
  const float a_y = force_y / ROBOT_MASS_KG;
  const float a_angular = -force_angular / ROBOT_MOMENT_OF_INERTIA;

  state_der[3] = a_x;
  state_der[4] = a_y;
  state_der[5] = a_angular;
}

__device__ void OmniWheelRobotModel::computeStateDeriv(
    float* state, float* control, float* state_der, float* theta_s){
  // We only compute the kinematics on the first thread because the other
  // threads only differ in the controls given, not the state
  if (threadIdx.y == 0){
    computeKinematics(state, state_der);
  }

  computeDynamics(state, control, state_der, theta_s);
}

__device__ void OmniWheelRobotModel::incrementState(float* state, float* state_der){

  // TODO: not sure if we should be starting at threadIdx.y or stepping by 
  //       blockDim.y here.... this is carry over from the generalized 
  //       linear model and the neural net model..

  int i;
  int tdy = threadIdx.y;

  //Add the state derivative time dt to the current state.
  for (i = tdy; i < STATE_DIM; i+=blockDim.y){
    state[i] += state_der[i]*dt_;

    // It was indicated in "GeneralizedLinear" that it's important that the 
    // state deriv. get reset to zero. But no indication was given as to *why* 
    // it is important. Just copying over for the sake of time, but we should 
    // really figure out why this is required......
    state_der[i] = 0; 
  }
}

__host__ __device__ float OmniWheelRobotModel::computeWheelFrictionCoeffInWheelDir(float wheel_sliding_speed){
  return WHEEL_FRICTION_COEFF_IN_WHEEL_DIR * 2.0 / M_PI * atan(
      FRICTION_COEFF_TRANSITION_COEFF * wheel_sliding_speed);
}

__host__ __device__ float OmniWheelRobotModel::computeWheelFrictionCoeffInTransverseDir(float wheel_transverse_speed){
        return WHEEL_FRICTION_COEFF_IN_TRANSVERSE_DIR * 2.0 / M_PI * atan(
            FRICTION_COEFF_TRANSITION_COEFF * wheel_transverse_speed);
}

}

