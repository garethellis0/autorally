// TODO: the duplication of work between cuda-specific and non-cuda-specific
//       things here is *insane*. Revise if possible.

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
  // TODO: implement me, this is the "meat" of the model
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

