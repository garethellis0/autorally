#include "vector_math.cuh"

namespace autorally_control {


OmniWheelRobotAutorallyModel::OmniWheelRobotAutorallyModel(double dt, double max_abs_wheel_speed) : OmniWheelRobotModel(dt, max_abs_wheel_speed)
{
  HANDLE_ERROR( cudaMalloc((void**)&control_rngs_d_, CONTROL_DIM*sizeof(float2)) );
}

void OmniWheelRobotAutorallyModel::paramsToDevice(){
  HANDLE_ERROR( cudaMemcpy(control_rngs_d_, control_rngs_, CONTROL_DIM*sizeof(float2), cudaMemcpyHostToDevice) );
}

void OmniWheelRobotAutorallyModel::freeCudaMem(){}

__device__ void OmniWheelRobotAutorallyModel::cudaInit(float* theta_s){}

__device__ void OmniWheelRobotAutorallyModel::enforceConstraints(float* state, float* control){
  for (int i = 0; i < CONTROL_DIM; i++){
    if (control[i] < control_rngs_d_[i].x){
      control[i] = control_rngs_d_[i].x;
    }
    else if (control[i] > control_rngs_d_[i].y){
      control[i] = control_rngs_d_[i].y;
    }
  }
}

__device__ void OmniWheelRobotAutorallyModel::computeStateDeriv(
    float* state, float* control, float* state_der, float* theta_s){
  // We only compute the kinematics on the first thread because the other
  // threads only differ in the controls given, not the state
  if (threadIdx.y == 0){
    computeKinematics(state, state_der);
  }

  computeDynamics(state, control, state_der, theta_s);
}

__device__ void OmniWheelRobotAutorallyModel::incrementState(float* state, float* state_der){

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

