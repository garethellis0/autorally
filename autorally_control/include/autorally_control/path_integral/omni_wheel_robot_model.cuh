#ifndef OMNI_WHEEL_ROBOT_MODEL_CUH_
#define OMNI_WHEEL_ROBOT_MODEL_CUH_

/**********************************************
 * @author Gareth Ellis
 * @date February 11, 2020
 * @brief Class definition for a 4 Omni-Wheeled Robot With Wheel Slip. This 
 *        file can be compiled for x86 for testing purposes.
 ***********************************************/

#include <Eigen/Dense>

// This macro allows us to compile functions without nvcc for unit testing
// purposes
#ifdef __CUDACC__
#define CUDA_HOSTDEV __host__ __device__
#define CUDA_HOST __host__ 
#define CUDA_DEV __host__ 
#else
#define CUDA_HOSTDEV
#define CUDA_HOST
#define CUDA_DEV
struct float2 {
  float x;
  float y;
};
using namespace std;
#endif

namespace autorally_control {

class OmniWheelRobotModel
{
    // We declare the test as a friend class so that we can access protected/private functions
    friend class OmniWheelRobotModelTest;
    //friend class GetWheelSpeedsTest;
public:

  // The size of the entire state
  static const int STATE_DIM = 6;
  // The number of kinematic variables in the state
  static const int KINEMATICS_DIM = 3;
  // The number of variables in the control input
  static const int CONTROL_DIM = 4;
  // The dynamics dimension
  // TODO: better comment here
  static const int DYNAMICS_DIM = STATE_DIM - KINEMATICS_DIM;

  OmniWheelRobotModel() = delete;

  /**
   * Create an instance of this model

   * @param dt The timestep for a single state update step.
   * @param max_abs_wheel_force The maximum absolute permissable wheel speed
   * @param front_wheel_angle_rad The angle, measured from a vector pointing
   *                              directly forwards, that the front wheels are
   *                              oriented at (rad)
   * @param rear_wheel_angle_rad The angle, measured from a vector pointing
   *                             directly backwards, that the rear wheels are
   *                             oriented at (rad)
   */
  OmniWheelRobotModel(double dt, double max_abs_wheel_force,
      float front_wheel_angle_rad, float rear_wheel_angle_rad);

  /**
   * Create an instance of this model

   * @param dt The timestep for a single state update step.
   * @param max_abs_wheel_force The maximum absolute permissable wheel speed
   */
  OmniWheelRobotModel(double dt, double max_abs_wheel_force);

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
   * @param state The current robot state in the global coordinate frame
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
   * Update the state_der_ components that can be determined solely
   * from the current robot state
   *
   * @param state The current robot state
   */
  CUDA_HOSTDEV void computeKinematics(float* state, float* state_der);

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
  CUDA_HOSTDEV void computeDynamics(float* state, float* control, float* state_der, float* theta_s);

  /**
   * The most recently computed state derivative
   */
  Eigen::Matrix<float, STATE_DIM, 1> state_der_;

  // An array of length CONTROL_DIM of pairs of [min_value, max_value] for
  // each control signal
  float2* control_rngs_;

  // TODO: these should _really_ be protected members
public:

  /**
   * Scale the controls using a simple model of wheel force
   *
   * @param state The state of the robot [x, y, yaw, v_x, v_y, v_angular], in global
   *              coordinates
   * @param controls The original controls
   * @param scaled_controls An array that will be set to the scaled controls
   */
  CUDA_HOSTDEV void scaleControls(float state[6], float controls[4], float scaled_controls[4]);

  /**
   * Scale the wheel force that's actually applied based on the wheel speed
   * to model wheel slip
   *
   * Positive wheel force and positive wheel speed must be in the same direction
   *
   * @param wheel_speed The speed of the wheel
   * @param wheel_force The force we're attempting to apply to the wheel
   */
  CUDA_HOSTDEV float scaleWheelForce(float wheel_speed, float wheel_force);

  /**
   * Get the speeds for the wheels in the direction they are oriented,
   * that is, tangent to the robot body.
   *
   * All values are in global coordinates
   *
   * @param yaw The current yaw of the robot (rad)
   * @param v_x The x-component of the robot's linear velocity (m/s)
   * @param v_y The y-component of the robot's linear velocity (m/s)
   * @param v_angular The angular speed of the robot (rad/s)
   * @param wheel_speeds An array that will be set to the speeds of the
   *                     individual wheels, in the direction the wheel is
   *                     oriented (m/s)
   */
  CUDA_HOSTDEV void getWheelSpeeds(float yaw, float v_x, float v_y,
                                   float v_angular, float wheel_speeds[4]);

  /**
   * Create a set of unit vectors for the wheels, with each pointing the
   * direction a wheel is facing.
   *
   * @param wheel_orientation_vectors An array that the vectors will be
   *        returned in.
   */
  CUDA_HOSTDEV void createUnitWheelOrientationVectors(float wheel_orientation_vectors[4][3]);

  /**
   * Create a set of vectors from the center of the robot to each wheel
   *
   * @param wheel_position_vectors An array that the vectors will be
   *        returned in.
   */
  CUDA_HOSTDEV void createWheelPositionVectors(float wheel_position_vectors[4][3]);

protected:
  /**
   * Computes the wheel coefficient of friction for a given speed, in the
   * direction the wheel is oriented
   * @param wheel_sliding_speed The speed of the wheel, in the direction that
   *                            it is orientated
   * @return The coefficient of friction for the given speed in the direction
   *         the wheel is oriented
   */
  CUDA_HOSTDEV float computeWheelFrictionCoeffInWheelDir(float wheel_sliding_speed);

  /**
   * Computes the wheel coefficient of friction for a given speed, tranverse to
   * the direction the wheel is oriented
   * @param wheel_sliding_speed The speed of the wheel, in the direction
   *                            transverse to it is orientated
   * @return The coefficient of friction for the given speed in the direction
   *         transverse to the direction the wheel is oriented
   */
  CUDA_HOSTDEV float computeWheelFrictionCoeffInTransverseDir(float wheel_transverse_speed);

  // The angle of the front two wheels, measured relative to a vector
  // pointing directly forward (+x) on the robot
  const float FRONT_WHEEL_ANGLE_RAD;
  // The angle of the front two wheels, measured relative to a vector
  // pointing directly backwards (-x) on the robot
  const float REAR_WHEEL_ANGLE_RAD;
  // The radius of the robot, the center of each wheel is assumed to be
  // displaced from the robot center by this amount
  static constexpr float ROBOT_RADIUS_M = 0.088;
  // The mass of the robot
  static constexpr float ROBOT_MASS_KG = 1.4;
  // The moment of inertia of the robot. We init this in the constructor
  // because it uses pow(radius), but pow is not a constepxr function
  const float ROBOT_MOMENT_OF_INERTIA;
  // The radius of each wheel
  static constexpr float WHEEL_RADIUS_M = 0.025;
  // The coefficient of friction for the wheels when the wheel is being
  // translated parallel to it's direction of rotation
  static constexpr float WHEEL_FRICTION_COEFF_IN_WHEEL_DIR = 0.25;
  //static constexpr float WHEEL_FRICTION_COEFF_IN_WHEEL_DIR = 0.9;
  // The coefficient of friction for the wheels when the wheel is
  // travelling in the direction perpendicular to it's direction of
  // rotation
  static constexpr float WHEEL_FRICTION_COEFF_IN_TRANSVERSE_DIR = 0.09;
  //static constexpr float WHEEL_FRICTION_COEFF_IN_TRANSVERSE_DIR = 0.9;
  // The constant used to govern the steepness of the slope between the
  // friction coefficient in the wheel direction and the tangent direction
  // about zero sliding velocity
  //static constexpr float FRICTION_COEFF_TRANSITION_COEFF = 1000;
  static constexpr float FRICTION_COEFF_TRANSITION_COEFF = 100;
  // The maximum linear speed of the robot
  static constexpr float MAX_LINEAR_SPEED_M_PER_S = 2.0;
  // The maximum angular speed of the robot
  static constexpr float MAX_ANGULAR_SPEED_RAD_PER_S = 2.0;
  // The maximum linear acceleration of the robot
  static constexpr float MAX_LINEAR_ACCELERATION_M_PER_S_PER_S = 3.0;
  // The maximum angular acceleration of the robot
  static constexpr float MAX_ANGULAR_ACCELERATION_RAD_PER_S_PER_S = 5.0;

  double dt_;
  double max_abs_wheel_force_;
};

}

#include "omni_wheel_robot_model.cu"

#endif /*OMNI_WHEEL_ROBOT_MODEL_CUH_*/
