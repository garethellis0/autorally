#include "vector_math.cuh"

namespace autorally_control {

OmniWheelRobotModel::OmniWheelRobotModel(double dt, double max_abs_wheel_speed) :
  dt_(dt),
  max_abs_wheel_speed_(max_abs_wheel_speed),
  ROBOT_MOMENT_OF_INERTIA(0.5 * ROBOT_MASS_KG * pow(ROBOT_RADIUS_M, 2.0)) {

  control_rngs_ = new float2[CONTROL_DIM];
  for (int i = 0; i < CONTROL_DIM; i++){
    control_rngs_[i].x = -max_abs_wheel_speed;
    control_rngs_[i].y = max_abs_wheel_speed;
  }

  for (int i = 0; i < STATE_DIM; i++){
    state_der_(i) = 0;
  }
}

void OmniWheelRobotModel::enforceConstraints(Eigen::MatrixXf &state, Eigen::MatrixXf &control){
  for (int i = 0; i < CONTROL_DIM; i++){
    if (control(i) < control_rngs_[i].x){
      control(i) = control_rngs_[i].x;
    }
    else if (control(i) > control_rngs_[i].y){
      control(i) = control_rngs_[i].y;
    }
  }
}

void OmniWheelRobotModel::updateState(Eigen::MatrixXf &state, Eigen::MatrixXf &control){
  enforceConstraints(state, control);
  computeKinematics(state);
  computeDynamics(state, control);

  std::cout << "Control:          " <<  std::endl << control << std::endl;
  std::cout << "State:            " <<  std::endl << state << std::endl;
  std::cout << "State Derivative: " <<  std::endl << state_der_ << std::endl;
  state += state_der_*dt_;
  state_der_ *= 0;
}

void OmniWheelRobotModel::computeKinematics(Eigen::MatrixXf &state) {
  // Here we compute the derivative by rotating the higher order
  // state variables (velocity and acceleration) into the robot frame
  static_assert(KINEMATICS_DIM == 3, "");
  static_assert(STATE_DIM == 6, "");

  float state_array[STATE_DIM];
  for (int i = 0; i < STATE_DIM; i++){
    state_array[i] = state(i);
  }
  float state_der_array[STATE_DIM];

  computeKinematics(state_array, state_der_array);

  for (int i = 0; i < KINEMATICS_DIM; i++){
    state_der_(i) = state_der_array[i];
  }
}


void OmniWheelRobotModel::computeDynamics(Eigen::MatrixXf &state, Eigen::MatrixXf &control){
  // We delegate to the version of this function built for CUDA. This is
  // probably slightly less efficient overall (as compared to using native
  // Eigen functions to do the same thing), but seriously reduces the
  // potential for bugs from re-implementing an algorithm twice
  static_assert(KINEMATICS_DIM == 3, "");
  static_assert(STATE_DIM == 6, "");

  float state_array[STATE_DIM];
  for (int i = 0; i < STATE_DIM; i++){
    state_array[i] = state(i);
  }
  float control_array[CONTROL_DIM];
  for (int i = 0; i < CONTROL_DIM; i++){
    control_array[i] = control(i);
  }

  float state_der_array[6] = {0, 0, 0, 0, 0, 0};

  computeDynamics(state_array, control_array, state_der_array, NULL);

  state_der_(3) = state_der_array[3];
  state_der_(4) = state_der_array[4];
  state_der_(5) = state_der_array[5];
}

CUDA_HOSTDEV void OmniWheelRobotModel::computeKinematics(float* state, float* state_der){
  // Here we compute the derivative by rotating the higher order
  // state variables (velocity and acceleration) into the robot frame
  static_assert(KINEMATICS_DIM == 3, "");
  static_assert(STATE_DIM == 6, "");

  // TODO: not sure which of these is correct...

  // The change in the 0th derivative terms (x, y, yaw) is from the 1st
  // derivative terms (v_x, v_y, v_angular)
  //state_der[0] = cosf(-state[2])*state[3] - sinf(-state[2])*state[4];
  //state_der[1] = sinf(-state[2])*state[3] + cosf(-state[2])*state[4];
  //state_der[2] = state[5];
  state_der[0] = state[3];
  state_der[1] = state[4];
  state_der[2] = state[5];

  // TODO: delete this
  //state_der[0] = min(max(state_der[0], -1.0), 1.0);
  //state_der[1] = min(max(state_der[1], -1.0), 1.0);
  //state_der[2] = min(max(state_der[2], -1.0), 1.0);
}

CUDA_HOSTDEV void OmniWheelRobotModel::computeDynamics(
    float* state, float* control, float* state_der, float* theta_s){
  // TODO: if performance is an issue, there are several things here that
  //       we can cache and not recompute, like the wheel vectors

  // TODO: Many names are copy-pasted from the python model, we really should
  //       go through and clean them up

  // This function assumes there are 4 control inputs, one for each wheel
  static_assert(CONTROL_DIM == 4, "");

  static_assert(KINEMATICS_DIM == 3, "");
  static_assert(DYNAMICS_DIM == 3, "");

  // NOTE: unless otherwise specified, all arrays of wheel specific values
  //       go [front left, back left, back right, front right]

  // Setup wheel position vectors, vectors from the center of the robot
  // to the center of each wheel
  float wheel_position_vectors[4][3];
  const float positive_x_vec_robot_radius[3] = {ROBOT_RADIUS_M, 0, 0};
  rotateVector3AboutZAxis(positive_x_vec_robot_radius, FRONT_WHEEL_ANGLE_RAD, wheel_position_vectors[0]);
  rotateVector3AboutZAxis(positive_x_vec_robot_radius, M_PI - REAR_WHEEL_ANGLE_RAD, wheel_position_vectors[1]);
  rotateVector3AboutZAxis(positive_x_vec_robot_radius, M_PI + REAR_WHEEL_ANGLE_RAD, wheel_position_vectors[2]);
  rotateVector3AboutZAxis(positive_x_vec_robot_radius, -FRONT_WHEEL_ANGLE_RAD, wheel_position_vectors[3]);
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
  createRotationMatrixAboutZAxis(-curr_orientation, R_0_to_M);
  // Rotation matrix from robot-relative coordinates to global coordinates
  float R_M_to_0[3][3];
  createRotationMatrixAboutZAxis(curr_orientation, R_M_to_0);

  // Velocity vector of the robot in the global frame
  float v_G[3] = {state[3], state[4], 0};

  // Angular velocity vector
  float w[3] = {0, 0, state[5]};

  // Flip the controls, because the rest of this function assumes positive rotation
  // is cw rotation of the wheel from the perspective of the robot, but the
  // tests we wrote assume the opposite
  for (int i = 0; i < 4; i++){
      control[i] *= -1;
  }

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
        transformed_wheel_orientation_vec);

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


  float total_linear_force[3] = {0, 0, 0};
  for (int i = 0; i < 4; i++){
    addVector3(total_linear_force, wheel_forces[i], total_linear_force);
  }

  float positive_x_vec[3] = {1, 0, 0};
  float force_x = dotProductVector3(positive_x_vec, total_linear_force);
  float force_y = dotProductVector3(positive_y_vec, total_linear_force);

  float acceleration[3] = {
    force_x / ROBOT_MASS_KG,
    force_y / ROBOT_MASS_KG,
    -force_angular / ROBOT_MOMENT_OF_INERTIA
  };

  state_der[3] = acceleration[0];
  state_der[4] = acceleration[1];
  state_der[5] = acceleration[2];
}

CUDA_HOSTDEV float OmniWheelRobotModel::computeWheelFrictionCoeffInWheelDir(float wheel_sliding_speed){
  return WHEEL_FRICTION_COEFF_IN_WHEEL_DIR * 2.0 / M_PI * atan(
      FRICTION_COEFF_TRANSITION_COEFF * wheel_sliding_speed);
}

CUDA_HOSTDEV float OmniWheelRobotModel::computeWheelFrictionCoeffInTransverseDir(float wheel_transverse_speed){
  return WHEEL_FRICTION_COEFF_IN_TRANSVERSE_DIR * 2.0 / M_PI * atan(
      FRICTION_COEFF_TRANSITION_COEFF * wheel_transverse_speed);
}

}
