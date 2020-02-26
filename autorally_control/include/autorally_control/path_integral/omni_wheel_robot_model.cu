// TODO: the duplication of work between cuda-specific and non-cuda-specific
//       things here is *insane*. Revise if possible.

#include "vector_math.cuh"

namespace autorally_control {


OmniWheelRobotModel::OmniWheelRobotModel(double dt, double max_abs_wheel_speed) :
  dt_(dt), 
  max_abs_wheel_speed_(max_abs_wheel_speed)
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
  state_der_ *= 0;

  enforceConstraints(state, control);
  computeKinematics(state);
  computeDynamics(state, control);

  state += state_der_*dt_;
}

void OmniWheelRobotModel::computeKinematics(Eigen::MatrixXf &state) {
  state_der_(0) = cosf(state(2))*state(3) - sinf(state(2))*state(4);
  state_der_(1) = sinf(state(2))*state(3) + cosf(state(2))*state(4);
  state_der_(2) = state(5); 
}

void OmniWheelRobotModel::computeDynamics(Eigen::MatrixXf &state, Eigen::MatrixXf &control){
  // TODO: implement me, this is the "meat" of the model
  auto deleteme = Eigen::VectorXf::Constant(DYNAMICS_DIM, 1);
  state_der_.block(STATE_DIM - DYNAMICS_DIM, 0, DYNAMICS_DIM, 1) = deleteme;
}

__device__ void OmniWheelRobotModel::cudaInit(float* theta_s){}

__device__ void OmniWheelRobotModel::enforceConstraints(float* state, float* control){
  // TODO?
}

__device__ void OmniWheelRobotModel::computeKinematics(float* state, float* state_der){
  state_der[0] = cosf(state[2])*state[3] - sinf(state[2])*state[4];
  state_der[1] = sinf(state[2])*state[3] + cosf(state[2])*state[4];
  state_der[2] = state[5];
}

__device__ void OmniWheelRobotModel::computeDynamics(
    float* state, float* control, float* state_der, float* theta_s){ 
  // TODO: if performance is an issue, there are several things here that
  //       we can cache and not recompute, like the wheel vectors

  // TODO: Many names are copy-pasted from the python model, we really should
  //       go through and clean them up

  // This function assumes there are 4 control inputs, one for each wheel
  static_assert(CONTROL_DIM == 4);

  // NOTE: unless otherwise specified, all arrays of wheel specific values 
  //       go [front left, back left, back right, front right]

  // Setup wheel position vectors, vectors from the center of the robot
  // to the center of each wheel
  float[4][3] wheel_position_vectors;
  float[3] pos_x_vec = {ROBOT_RADIUS, 0, 0};
  rotateVector3AboutZAxis(pos_x_vec, FRONT_WHEEL_ANGLE_RAD, wheel_orientation_vectors[0]);
  rotateVector3AboutZAxis(pos_x_vec, M_PI - REAR_WHEEL_ANGLE_RAD, wheel_orientation_vectors[1]);
  rotateVector3AboutZAxis(pos_x_vec, M_PI + REAR_WHEEL_ANGLE_RAD, wheel_orientation_vectors[2]);
  rotateVector3AboutZAxis(pos_x_vec, -FRONT_WHEEL_ANGLE_RAD, wheel_orientation_vectors[3]);
  float[4][3] unit_wheel_position_vectors;
  for (int i = 0; i < 4; i++){
    getUnitVectorInDirection(wheel_position_vectors[i], 
        unit_wheel_position_vectors[i]);
  }

  // Setup wheel orientation vectors, vectors pointing in the direction that
  // each wheel is oriented
  float[4][3] wheel_orientation_vectors;
  float[3] pos_y_vec = {0, 1, 0};
  rotateVector3AboutZAxis(neg_y_vec, FRONT_WHEEL_ANGLE_RAD, wheel_orientation_vectors[0]);
  rotateVector3AboutZAxis(neg_y_vec, M_PI - REAR_WHEEL_ANGLE_RAD, wheel_orientation_vectors[1]);
  rotateVector3AboutZAxis(neg_y_vec, M_PI + REAR_WHEEL_ANGLE_RAD, wheel_orientation_vectors[2]);
  rotateVector3AboutZAxis(neg_y_vec, -FRONT_WHEEL_ANGLE_RAD, wheel_orientation_vectors[3]);
  float[4][3] unit_wheel_orientation_vectors;
  for (int i = 0; i < 4; i++){
    getUnitVectorInDirection(wheel_orientation_vectors[i], 
        unit_wheel_orientation_vectors[i]);
  }

  float curr_orientation = state[2];
  // Rotation matrix from global coordinates to robot-relative coordinates
  float[3][3] R_0_to_M;
  createrotationMatrixAboutZAxis(curr_orientation, R_0_to_M);
  // Rotation matrix from robot-relative coordinates to global coordinates
  float[3][3] R_M_to_0;
  createrotationMatrixAboutZAxis(-curr_orientation, R_0_to_M);

  // Velocity vector of the robot in the global frame
  float[3] v_G = {state[3], state[4], 0};

  // Angular velocity vector
  float[3] w = {0, 0, state[5]};

  // Scale the wheelspeeds
  // TODO: not sure what this is needed for... dbl check in python model
  float[4] scaled_wheel_speeds;
  for (int i = 0; i < 4; i++){
    scaled_wheel_speeds[i] = [4 * pow(M_PI, 2.0) / control[i]]
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
    crossProductVector3(w, transformed_wheel_position_vec);

    float v_W = dotProductVector3(v_G, transformed_wheel_orientation_vec) +
      dotProductVector3(cross_result, transformed_wheel_orientation_vec) +
      WHEEL_RADIUS*wheel_speeds_scaled[i];
    float v_T = dotProductVector3(v_G, transformed_wheel_position_vec) +
      dotProductVector3(cross_result, transformed_wheel_position_vec);

    // TODO: better names for these?
    float[3] scaled_transformed_wheel_orientation_vec;
    multiplyVector3ByScalar(computeWheelFrictionCoeffInWheelDir(v_W), 
        transformed_wheel_orientation_vec, 
        scaled_transformed_wheel_orientation_vec);
    float[3] scaled_transformed_wheel_position_vec;
    multiplyVector3ByScalar(computeWheelFrictionCoeffInTransverseDir(v_T), 
        transformed_wheel_position_vec, 
        scaled_transformed_wheel_position_vec);

    float[3] summed_acceleration;
    addVector3(scaled_transformed_wheel_orientation_vec, 
        scaled_transformed_wheel_position_vec, summed_acceleration)

    multiplyVector3ByScalar(summed_acceleration, -MASS_KG*9.8/4.0, wheel_forces[i]);

    // TODO: better name for this
    float cross_result1[3];
    crossProductVector3(transformed_wheel_position_vec, wheel_forces[i]);

    const pos_z_vec[3] = {0, 0, 1};
    force_angular += dotProductVector3(pos_z_vec, cross_result1);
  }

  float total_force[3] = {0, 0, 0};
  for (int i = 0; i < 4; i++){
    addVector3(total_force, wheel_forces[i], total_force);
  }

  float force_x[3];
  float force_y[3];
  dotProductVector3(pos_x_vec, total_force, force_x);
  dotProductVector3(pos_y_vec, total_force, force_y);

  // TODO: YOU ARE HERE -  need to compute accelerations, then probably
  //       just fill out state_der? double check.

  // TODO: finish the implementation of this, what's below is a placeholder!
  for (int i = 0; i < DYNAMICS_DIM; i++){
    state_der[i+(STATE_DIM - DYNAMICS_DIM)] = control[i];
  }
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

}

