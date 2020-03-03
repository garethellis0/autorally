#include <gtest/gtest.h>
#include "autorally_control/path_integral/omni_wheel_robot_model.cuh"

#include <Eigen/Dense>
#include <memory>

using namespace autorally_control;

class OmniWheelRobotModelTest : public ::testing::Test {
protected:
  void SetUp(){
    model = std::make_shared<OmniWheelRobotModel>(0.1, 2.0);
  }

  void TearDown(){
  }

  static void checkRawArrayEq(int n, float* expected, float* actual, float tol = 1e-4){
    for (int i = 0; i < n; i++){
      EXPECT_NEAR(expected[i], actual[i], tol);
    }
  }

  std::shared_ptr<OmniWheelRobotModel> model;
};


// TODO: test constructor

TEST_F(OmniWheelRobotModelTest, enforce_constraints){
  Eigen::MatrixXf control(4,1);
  control << 5, -5, 5, -5;
  Eigen::MatrixXf state(6,1);
  state << 123, 456, 789, 12, 12, 12;

  Eigen::MatrixXf expected_state = state;
  Eigen::MatrixXf expected_control(4,1);
  expected_control << 2, -2, 2, -2;

  model->enforceConstraints(state, control);

  EXPECT_EQ(expected_state, state);
  EXPECT_EQ(expected_control, control);
}

TEST_F(OmniWheelRobotModelTest, updateState){
  // TODO
}

TEST_F(OmniWheelRobotModelTest, computeKinematics_CPU_no_rotation_at_origin){
  Eigen::MatrixXf state(6,1);
  state << 0, 0, 0, 0.1, 0.2, 0.3;

  model->computeKinematics(state);

  Eigen::MatrixXf expected_state_der(6,1);
  expected_state_der << 0.1, 0.2, 0.3, 0, 0, 0;

  EXPECT_EQ(expected_state_der, model->state_der_);
}

TEST_F(OmniWheelRobotModelTest, computeKinematics_CPU_no_rotation_first_quadrant){
  Eigen::MatrixXf state(6,1);
  state << 1, 1, 0, 0.1, 0.2, 0.3;

  model->computeKinematics(state);

  Eigen::MatrixXf expected_state_der(6,1);
  expected_state_der << 0.1, 0.2, 0.3, 0, 0, 0;

  EXPECT_EQ(expected_state_der, model->state_der_);
}
 
TEST_F(OmniWheelRobotModelTest, computeKinematics_CPU_with_rotation_at_origin){
  Eigen::MatrixXf state(6,1);
  state << 0, 0, 2.31, 0.1, 0.2, 0.3;

  model->computeKinematics(state);

  Eigen::MatrixXf expected_state_der(6,1);
  expected_state_der << 0.1, 0.2, 0.3, 0, 0, 0;

  EXPECT_EQ(expected_state_der, model->state_der_);
}

TEST_F(OmniWheelRobotModelTest, computeKinematics_CPU_with_rotation_first_quadrant){
  Eigen::MatrixXf state(6,1);
  state << 1, 1, 2.33, 0.1, 0.2, 0.3;

  model->computeKinematics(state);

  Eigen::MatrixXf expected_state_der(6,1);
  expected_state_der << 0.1, 0.2, 0.3, 0, 0, 0;

  EXPECT_EQ(expected_state_der, model->state_der_);
}

TEST_F(OmniWheelRobotModelTest, computeKinematics_GPU_no_rotation_at_origin){
  float state[6] = {0, 0, 0, 0.1, 0.2, 0.3};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeKinematics(state, state_der);

  float expected_state_der[6] = {0.1, 0.2, 0.3, 0, 0, 0};

  checkRawArrayEq(6, expected_state_der, state_der);
}

TEST_F(OmniWheelRobotModelTest, computeKinematics_GPU_no_rotation_first_quadrant){
  float state[6] = {1, 1, 0, 0.1, 0.2, 0.3};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeKinematics(state, state_der);

  float expected_state_der[6] = {0.1, 0.2, 0.3, 0, 0, 0};

  checkRawArrayEq(6, expected_state_der, state_der);
}
 
TEST_F(OmniWheelRobotModelTest, computeKinematics_GPU_with_rotation_at_origin){
  float state[6] = {0, 0, 2.33, 0.1, 0.2, 0.3};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeKinematics(state, state_der);

  float expected_state_der[6] = {0.1, 0.2, 0.3, 0, 0, 0};

  checkRawArrayEq(6, expected_state_der, state_der);
}

TEST_F(OmniWheelRobotModelTest, computeKinematics_GPU_with_rotation_first_quadrant){
  float state[6] = {1, 1, 2.33, 0.1, 0.2, 0.3};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeKinematics(state, state_der);

  float expected_state_der[6] = {0.1, 0.2, 0.3, 0, 0, 0};

  checkRawArrayEq(6, expected_state_der, state_der);
}

// TODO: dynamics CPU TESTS

//  ------- At Origin, No Rotation ----------

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_forward_at_origin_no_rotation){
  float state[6] = {0, 0, 0, 0, 0, 0};
  float control[4] = {-1, -1, 1, 1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_GE(state_der[3], 0.1);
  EXPECT_NEAR(0, state_der[4], 1e-4);
  EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_backwards_at_origin_no_rotation){
  float state[6] = {0, 0, 0, 0, 0, 0};
  float control[4] = {1, 1, -1, -1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_LE(state_der[3], -0.1);
  EXPECT_NEAR(0, state_der[4], 1e-4);
  EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_right_at_origin_no_rotation){
  float state[6] = {0, 0, 0, 0, 0, 0};
  float control[4] = {-1, 1, 1, -1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_NEAR(0, state_der[3], 1e-4);
  EXPECT_LE(state_der[4], -0.1);
  EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_left_at_origin_no_rotation){
  float state[6] = {0, 0, 0, 0, 0, 0};
  float control[4] = {1, -1, -1, 1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_NEAR(0, state_der[3], 1e-4);
  EXPECT_GE(state_der[4], 0.1);
  EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_spinning_clockwise_at_origin_no_initial_rotation){
  float state[6] = {0, 0, 0, 0, 0, 0};
  float control[4] = {-1, -1, -1, -1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_NEAR(0, state_der[3], 1e-4);
  EXPECT_NEAR(0, state_der[4], 1e-4);
  EXPECT_GE(state_der[5], 0.05);
}

TEST_F(OmniWheelRobotModelTest, 
    computeDynamics_spinning_counterclockwise_at_origin_no_initial_rotation){
  float state[6] = {0, 0, 0, 0, 0, 0};
  float control[4] = {1, 1, 1, 1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_NEAR(0, state_der[3], 1e-4);
  EXPECT_NEAR(0, state_der[4], 1e-4);
  EXPECT_LE(state_der[5], -0.05);
}

//  ------- At Origin, Rotated CW ----------

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_forward_at_origin_rotated_ccw){
  float state[6] = {0, 0, M_PI/2.0, 0, 0, 0};
  float control[4] = {-1, -1, 1, 1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_NEAR(0, state_der[3], 1e-4);
  EXPECT_GE(state_der[4], 0.1);
  EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_backwards_at_origin_rotated_ccw){
  float state[6] = {0, 0, M_PI/2.0, 0, 0, 0};
  float control[4] = {1, 1, -1, -1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_NEAR(0, state_der[3], 1e-4);
  EXPECT_LE(state_der[4], -0.1);
  EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_right_at_origin_rotated_ccw){
  float state[6] = {0, 0, M_PI/2.0, 0, 0, 0};
  float control[4] = {-1, 1, 1, -1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_LE(-0.1, state_der[3]);
  EXPECT_NEAR(0, state_der[4], 1e-4);
  EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_left_at_origin_rotated_ccw){
  float state[6] = {0, 0, M_PI/2.0, 0, 0, 0};
  float control[4] = {-1, 1, 1, -1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_GE(state_der[3], 0.1);
  EXPECT_NEAR(0, state_der[4], 1e-4);
  EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_spinning_clockwise_at_origin_rotated_ccw){
  float state[6] = {0, 0, M_PI/2.0, 0, 0, 0};
  float control[4] = {-1, -1, -1, -1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_NEAR(0, state_der[3], 1e-4);
  EXPECT_NEAR(0, state_der[4], 1e-4);
  EXPECT_GE(state_der[5], 0.05);
}

TEST_F(OmniWheelRobotModelTest, 
    computeDynamics_spinning_counterclockwise_at_origin_rotated_ccw){
  float state[6] = {0, 0, M_PI/2.0, 0, 0, 0};
  float control[4] = {1, 1, 1, 1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_NEAR(0, state_der[3], 1e-4);
  EXPECT_NEAR(0, state_der[4], 1e-4);
  EXPECT_LE(state_der[5], -0.05);
}

//  ------- In +x/+y Quadrant, No Rotation ----------

TEST_F(OmniWheelRobotModelTest, 
    computeDynamics_moving_forward_in_first_quadrant_no_rotation){
  float state[6] = {1, 1, 0, 0, 0, 0};
  float control[4] = {-1, -1, 1, 1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_GE(state_der[3], 0.1);
  EXPECT_NEAR(0, state_der[4], 1e-4);
  EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_backwards_in_first_quadrant_no_rotation){
  float state[6] = {1, 1, 0, 0, 0, 0};
  float control[4] = {1, 1, -1, -1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_LE(state_der[3], -0.1);
  EXPECT_NEAR(0, state_der[4], 1e-4);
  EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_right_in_first_quadrant_no_rotation){
  float state[6] = {1, 1, 0, 0, 0, 0};
  float control[4] = {-1, 1, 1, -1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_NEAR(0, state_der[3], 1e-4);
  EXPECT_LE(state_der[4], -0.1);
  EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_left_in_first_quadrant_no_rotation){
  float state[6] = {1, 1, 0, 0, 0, 0};
  float control[4] = {-1, 1, 1, -1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_NEAR(0, state_der[3], 1e-4);
  EXPECT_GE(0.1, state_der[4]);
  EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_spinning_clockwise_in_first_quadrant_no_initial_rotation){
  float state[6] = {1, 1, 0, 0, 0, 0};
  float control[4] = {1, 1, 1, 1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_NEAR(0, state_der[3], 1e-4);
  EXPECT_NEAR(0, state_der[4], 1e-4);
  EXPECT_GE(0.1, state_der[5]);
}

TEST_F(OmniWheelRobotModelTest, 
    computeDynamics_spinning_counterclockwise_in_first_quadrant_no_initial_rotation){
  float state[6] = {1, 1, 0, 0, 0, 0};
  float control[4] = {1, 1, 1, 1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_NEAR(0, state_der[3], 1e-4);
  EXPECT_NEAR(0, state_der[4], 1e-4);
  EXPECT_LE(state_der[5], -0.05);
}

//  ------- At +x/+y Quadrant, Rotated CW ----------

TEST_F(OmniWheelRobotModelTest, 
    computeDynamics_moving_forward_in_first_quadrant_rotated_ccw){
  float state[6] = {1, 1, M_PI/2.0, 0, 0, 0};
  float control[4] = {-1, -1, 1, 1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_NEAR(0, state_der[3], 1e-4);
  EXPECT_GE(state_der[4], 0.1);
  EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, 
    computeDynamics_moving_backwards_in_first_quadrant_rotated_ccw){
  float state[6] = {0, 0, M_PI/2.0, 0, 0, 0};
  float control[4] = {1, 1, -1, -1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_NEAR(0, state_der[3], 1e-4);
  EXPECT_LE(state_der[4], -0.1);
  EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, 
    computeDynamics_moving_right_in_first_quadrant_rotated_ccw){
  float state[6] = {1, 1, M_PI/2.0, 0, 0, 0};
  float control[4] = {-1, 1, 1, -1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_GE(state_der[3], 0.1);
  EXPECT_NEAR(0, state_der[4], 1e-4);
  EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, 
    computeDynamics_moving_left_in_first_quadrant_rotated_ccw){
  float state[6] = {1, 1, M_PI/2.0, 0, 0, 0};
  float control[4] = {1, -1, -1, 1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_LE(state_der[3], -0.1);
  EXPECT_NEAR(0, state_der[4], 1e-4);
  EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, 
    computeDynamics_spinning_clockwise_in_first_quadrant_rotated_ccw){
  float state[6] = {1, 1, M_PI/2.0, 0, 0, 0};
  float control[4] = {1, 1, 1, 1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_NEAR(0, state_der[3], 1e-4);
  EXPECT_NEAR(0, state_der[4], 1e-4);
  EXPECT_GE(0.1, state_der[5]);
}

TEST_F(OmniWheelRobotModelTest, 
    computeDynamics_spinning_counterclockwise_in_first_quadrant_rotated_ccw){
  float state[6] = {1, 1, M_PI/2.0, 0, 0, 0};
  float control[4] = {1, 1, 1, 1};
  float state_der[6] = {0, 0, 0, 0, 0, 0};

  model->computeDynamics(state, control, state_der, NULL);

  EXPECT_NEAR(0, state_der[3], 1e-4);
  EXPECT_NEAR(0, state_der[4], 1e-4);
  EXPECT_LE(state_der[5], -0.05);
}

//  ------- At Origin, No Rotation, Moving Forwards ----------

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_forward_at_origin_with_initial_velocity_no_rotation_all_zero_control){
    float state[6] = {0, 0, 0, 1, -1, -1};
    float control[4] = {0, 0, 0, 0};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_forward_at_origin_with_initial_velocity_no_rotation){
    float state[6] = {0, 0, 0, 1, -1, -1};
    float control[4] = {-1, -1, 1, 1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_GE(state_der[3], 0.1);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_backwards_at_origin_with_initial_velocity_no_rotation){
    float state[6] = {0, 0, 0, 1, -1, -1};
    float control[4] = {1, 1, -1, -1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_LE(state_der[3], -0.1);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_right_at_origin_with_initial_velocity_no_rotation){
    float state[6] = {0, 0, 0, 1, -1, -1};
    float control[4] = {-1, 1, 1, -1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_LE(state_der[4], -0.1);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_left_at_origin_with_initial_velocity_no_rotation){
    float state[6] = {0, 0, 0, 1, -1, -1};
    float control[4] = {1, -1, -1, 1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_GE(state_der[4], 0.1);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_spinning_clockwise_at_origin_with_initial_velocity_no_initial_rotation){
    float state[6] = {0, 0, 0, 1, -1, -1};
    float control[4] = {-1, -1, -1, -1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_GE(state_der[5], 0.05);
}

TEST_F(OmniWheelRobotModelTest,
       computeDynamics_spinning_counterclockwise_at_origin_with_initial_velocity_no_initial_rotation){
    float state[6] = {0, 0, 0, 1, -1, -1};
    float control[4] = {1, 1, 1, 1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_LE(state_der[5], -0.05);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
