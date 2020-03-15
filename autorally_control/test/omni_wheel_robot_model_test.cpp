#include <gtest/gtest.h>
#include "autorally_control/path_integral/omni_wheel_robot_model.cuh"

#include <Eigen/Dense>
#include <memory>

using namespace autorally_control;

class OmniWheelRobotModelTest : public ::testing::Test {
public:
    void SetUp() {
        model = std::make_shared<OmniWheelRobotModel>(0.1, 2.0, 0.785, 0.785);
    }

    void TearDown() {
    }

    static void checkRawArrayEq(int n, float *expected, float *actual, float tol = 1e-4) {
        for (int i = 0; i < n; i++) {
            EXPECT_NEAR(expected[i], actual[i], tol);
        }
    }

    std::shared_ptr<OmniWheelRobotModel> model;
};


// TODO: test constructor

TEST_F(OmniWheelRobotModelTest, enforce_constraints) {
    Eigen::MatrixXf control(4, 1);
    control << 5, -5, 5, -5;
    Eigen::MatrixXf state(6, 1);
    state << 123, 456, 789, 12, 12, 12;

    Eigen::MatrixXf expected_state = state;
    Eigen::MatrixXf expected_control(4, 1);
    expected_control << 2, -2, 2, -2;

    model->enforceConstraints(state, control);

    EXPECT_EQ(expected_state, state);
    EXPECT_EQ(expected_control, control);
}

TEST_F(OmniWheelRobotModelTest, updateState) {
    // TODO
}

TEST_F(OmniWheelRobotModelTest, computeKinematics_CPU_no_rotation_at_origin) {
    Eigen::MatrixXf state(6, 1);
    state << 0, 0, 0, 0.1, 0.2, 0.3;

    model->computeKinematics(state);

    Eigen::MatrixXf expected_state_der(6, 1);
    expected_state_der << 0.1, 0.2, 0.3, 0, 0, 0;

    EXPECT_EQ(expected_state_der, model->state_der_);
}

TEST_F(OmniWheelRobotModelTest, computeKinematics_CPU_no_rotation_first_quadrant) {
    Eigen::MatrixXf state(6, 1);
    state << 1, 1, 0, 0.1, 0.2, 0.3;

    model->computeKinematics(state);

    Eigen::MatrixXf expected_state_der(6, 1);
    expected_state_der << 0.1, 0.2, 0.3, 0, 0, 0;

    EXPECT_EQ(expected_state_der, model->state_der_);
}

TEST_F(OmniWheelRobotModelTest, computeKinematics_CPU_with_rotation_at_origin) {
    Eigen::MatrixXf state(6, 1);
    state << 0, 0, 2.31, 0.1, 0.2, 0.3;

    model->computeKinematics(state);

    Eigen::MatrixXf expected_state_der(6, 1);
    expected_state_der << 0.1, 0.2, 0.3, 0, 0, 0;

    EXPECT_EQ(expected_state_der, model->state_der_);
}

TEST_F(OmniWheelRobotModelTest, computeKinematics_CPU_with_rotation_first_quadrant) {
    Eigen::MatrixXf state(6, 1);
    state << 1, 1, 2.33, 0.1, 0.2, 0.3;

    model->computeKinematics(state);

    Eigen::MatrixXf expected_state_der(6, 1);
    expected_state_der << 0.1, 0.2, 0.3, 0, 0, 0;

    EXPECT_EQ(expected_state_der, model->state_der_);
}

TEST_F(OmniWheelRobotModelTest, computeKinematics_GPU_no_rotation_at_origin) {
    float state[6] = {0, 0, 0, 0.1, 0.2, 0.3};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeKinematics(state, state_der);

    float expected_state_der[6] = {0.1, 0.2, 0.3, 0, 0, 0};

    checkRawArrayEq(6, expected_state_der, state_der);
}

TEST_F(OmniWheelRobotModelTest, computeKinematics_GPU_no_rotation_first_quadrant) {
    float state[6] = {1, 1, 0, 0.1, 0.2, 0.3};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeKinematics(state, state_der);

    float expected_state_der[6] = {0.1, 0.2, 0.3, 0, 0, 0};

    checkRawArrayEq(6, expected_state_der, state_der);
}

TEST_F(OmniWheelRobotModelTest, computeKinematics_GPU_with_rotation_at_origin) {
    float state[6] = {0, 0, 2.33, 0.1, 0.2, 0.3};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeKinematics(state, state_der);

    float expected_state_der[6] = {0.1, 0.2, 0.3, 0, 0, 0};

    checkRawArrayEq(6, expected_state_der, state_der);
}

TEST_F(OmniWheelRobotModelTest, computeKinematics_GPU_with_rotation_first_quadrant) {
    float state[6] = {1, 1, 2.33, 0.1, 0.2, 0.3};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeKinematics(state, state_der);

    float expected_state_der[6] = {0.1, 0.2, 0.3, 0, 0, 0};

    checkRawArrayEq(6, expected_state_der, state_der);
}

// TODO: dynamics CPU TESTS

//  ------- At Origin, No Rotation ----------

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_forward_at_origin_no_rotation) {
    float state[6] = {0, 0, 0, 0, 0, 0};
    float control[4] = {-1, -1, 1, 1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_GE(state_der[3], 0.1);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_backwards_at_origin_no_rotation) {
    float state[6] = {0, 0, 0, 0, 0, 0};
    float control[4] = {1, 1, -1, -1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_LE(state_der[3], -0.1);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_right_at_origin_no_rotation) {
    float state[6] = {0, 0, 0, 0, 0, 0};
    float control[4] = {-1, 1, 1, -1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_LE(state_der[4], -0.1);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_left_at_origin_no_rotation) {
    float state[6] = {0, 0, 0, 0, 0, 0};
    float control[4] = {1, -1, -1, 1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_GE(state_der[4], 0.1);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_spinning_clockwise_at_origin_no_initial_rotation) {
    float state[6] = {0, 0, 0, 0, 0, 0};
    float control[4] = {-1, -1, -1, -1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_LE(state_der[5], -0.05);
}

TEST_F(OmniWheelRobotModelTest,
       computeDynamics_spinning_counterclockwise_at_origin_no_initial_rotation) {
    float state[6] = {0, 0, 0, 0, 0, 0};
    float control[4] = {1, 1, 1, 1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_GE(state_der[5], 0.05);
}

//  ------- At Origin, Rotated CW ----------

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_forward_at_origin_rotated_ccw) {
    float state[6] = {0, 0, M_PI / 2.0, 0, 0, 0};
    float control[4] = {-1, -1, 1, 1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_GE(state_der[4], 0.1);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_backwards_at_origin_rotated_ccw) {
    float state[6] = {0, 0, M_PI / 2.0, 0, 0, 0};
    float control[4] = {1, 1, -1, -1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_LE(state_der[4], -0.1);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_right_at_origin_rotated_ccw) {
    float state[6] = {0, 0, M_PI / 2.0, 0, 0, 0};
    float control[4] = {-1, 1, 1, -1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_LE(-0.1, state_der[3]);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_left_at_origin_rotated_ccw) {
    float state[6] = {0, 0, M_PI / 2.0, 0, 0, 0};
    float control[4] = {-1, 1, 1, -1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_GE(state_der[3], 0.1);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_spinning_clockwise_at_origin_rotated_ccw) {
    float state[6] = {0, 0, M_PI / 2.0, 0, 0, 0};
    float control[4] = {-1, -1, -1, -1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_LE(state_der[5], -0.05);
}

TEST_F(OmniWheelRobotModelTest,
       computeDynamics_spinning_counterclockwise_at_origin_rotated_ccw) {
    float state[6] = {0, 0, M_PI / 2.0, 0, 0, 0};
    float control[4] = {1, 1, 1, 1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_GE(state_der[5], 0.05);
}

//  ------- In +x/+y Quadrant, No Rotation ----------

TEST_F(OmniWheelRobotModelTest,
       computeDynamics_moving_forward_in_first_quadrant_no_rotation) {
    float state[6] = {1, 1, 0, 0, 0, 0};
    float control[4] = {-1, -1, 1, 1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_GE(state_der[3], 0.1);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_backwards_in_first_quadrant_no_rotation) {
    float state[6] = {1, 1, 0, 0, 0, 0};
    float control[4] = {1, 1, -1, -1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_LE(state_der[3], -0.1);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_right_in_first_quadrant_no_rotation) {
    float state[6] = {1, 1, 0, 0, 0, 0};
    float control[4] = {-1, 1, 1, -1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_LE(state_der[4], -0.1);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_left_in_first_quadrant_no_rotation) {
    float state[6] = {1, 1, 0, 0, 0, 0};
    float control[4] = {-1, 1, 1, -1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_GE(0.1, state_der[4]);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_spinning_clockwise_in_first_quadrant_no_initial_rotation) {
    float state[6] = {1, 1, 0, 0, 0, 0};
    float control[4] = {1, 1, 1, 1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_LE(-0.1, state_der[5]);
}

TEST_F(OmniWheelRobotModelTest,
       computeDynamics_spinning_counterclockwise_in_first_quadrant_no_initial_rotation) {
    float state[6] = {1, 1, 0, 0, 0, 0};
    float control[4] = {1, 1, 1, 1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_GE(state_der[5], 0.05);
}

//  ------- At +x/+y Quadrant, Rotated CW ----------

TEST_F(OmniWheelRobotModelTest,
       computeDynamics_moving_forward_in_first_quadrant_rotated_ccw) {
    float state[6] = {1, 1, M_PI / 2.0, 0, 0, 0};
    float control[4] = {-1, -1, 1, 1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_GE(state_der[4], 0.1);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest,
       computeDynamics_moving_backwards_in_first_quadrant_rotated_ccw) {
    float state[6] = {0, 0, M_PI / 2.0, 0, 0, 0};
    float control[4] = {1, 1, -1, -1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_LE(state_der[4], -0.1);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest,
       computeDynamics_moving_right_in_first_quadrant_rotated_ccw) {
    float state[6] = {1, 1, M_PI / 2.0, 0, 0, 0};
    float control[4] = {-1, 1, 1, -1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_GE(state_der[3], 0.1);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest,
       computeDynamics_moving_left_in_first_quadrant_rotated_ccw) {
    float state[6] = {1, 1, M_PI / 2.0, 0, 0, 0};
    float control[4] = {1, -1, -1, 1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_LE(state_der[3], -0.1);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest,
       computeDynamics_spinning_clockwise_in_first_quadrant_rotated_ccw) {
    float state[6] = {1, 1, M_PI / 2.0, 0, 0, 0};
    float control[4] = {1, 1, 1, 1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_LE(-0.1, state_der[5]);
}

TEST_F(OmniWheelRobotModelTest,
       computeDynamics_spinning_counterclockwise_in_first_quadrant_rotated_ccw) {
    float state[6] = {1, 1, M_PI / 2.0, 0, 0, 0};
    float control[4] = {1, 1, 1, 1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_GE(state_der[5], 0.05);
}

//  ------- At Origin, No Rotation, Moving Forwards ----------

TEST_F(OmniWheelRobotModelTest,
       computeDynamics_moving_forward_at_origin_with_initial_velocity_no_rotation_all_zero_control) {
    float state[6] = {0, 0, 0, 1, -1, -1};
    float control[4] = {0, 0, 0, 0};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_forward_at_origin_with_initial_velocity_no_rotation) {
    float state[6] = {0, 0, 0, 1, -1, -1};
    float control[4] = {-1, -1, 1, 1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_GE(state_der[3], 0.1);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_backwards_at_origin_with_initial_velocity_no_rotation) {
    float state[6] = {0, 0, 0, 1, -1, -1};
    float control[4] = {1, 1, -1, -1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_LE(state_der[3], -0.1);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_right_at_origin_with_initial_velocity_no_rotation) {
    float state[6] = {0, 0, 0, 1, -1, -1};
    float control[4] = {-1, 1, 1, -1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_LE(state_der[4], -0.1);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest, computeDynamics_moving_left_at_origin_with_initial_velocity_no_rotation) {
    float state[6] = {0, 0, 0, 1, -1, -1};
    float control[4] = {1, -1, -1, 1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_GE(state_der[4], 0.1);
    EXPECT_NEAR(0, state_der[5], 1e-4);
}

TEST_F(OmniWheelRobotModelTest,
       computeDynamics_spinning_clockwise_at_origin_with_initial_velocity_no_initial_rotation) {
    float state[6] = {0, 0, 0, 1, -1, -1};
    float control[4] = {-1, -1, -1, -1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_LE(state_der[5], -0.05);
}

TEST_F(OmniWheelRobotModelTest,
       computeDynamics_spinning_counterclockwise_at_origin_with_initial_velocity_no_initial_rotation) {
    float state[6] = {0, 0, 0, 1, -1, -1};
    float control[4] = {1, 1, 1, 1};
    float state_der[6] = {0, 0, 0, 0, 0, 0};

    model->computeDynamics(state, control, state_der, NULL);

    EXPECT_NEAR(0, state_der[3], 1e-4);
    EXPECT_NEAR(0, state_der[4], 1e-4);
    EXPECT_GE(state_der[5], 0.05);
}

// Test that wheel orientation vectors are created correctly for two pseudo-random wheel angles
TEST_F(OmniWheelRobotModelTest,
       createWheelOrientationVectors) {
    model = std::make_shared<OmniWheelRobotModel>(0.1, 2.0, 0.6, 0.3);

    float wheel_orientation_vectors[4][3];
    model->createUnitWheelOrientationVectors(wheel_orientation_vectors);

    // Front left
    EXPECT_NEAR(wheel_orientation_vectors[0][0], -0.56464, 0.0001);
    EXPECT_NEAR(wheel_orientation_vectors[0][1], 0.82534, 0.0001);
    EXPECT_NEAR(wheel_orientation_vectors[0][2], 0.0, 0.0001);

    // Front right
    EXPECT_NEAR(wheel_orientation_vectors[3][0], 0.56464, 0.0001);
    EXPECT_NEAR(wheel_orientation_vectors[3][1], 0.82534, 0.0001);
    EXPECT_NEAR(wheel_orientation_vectors[3][2], 0.0, 0.0001);

    // Back left
    EXPECT_NEAR(wheel_orientation_vectors[1][0], -0.29552, 0.0001);
    EXPECT_NEAR(wheel_orientation_vectors[1][1], -0.95534, 0.0001);
    EXPECT_NEAR(wheel_orientation_vectors[1][2], 0.0, 0.0001);

    // Back right
    EXPECT_NEAR(wheel_orientation_vectors[2][0], 0.29552, 0.0001);
    EXPECT_NEAR(wheel_orientation_vectors[2][1], -0.95534, 0.0001);
    EXPECT_NEAR(wheel_orientation_vectors[2][2], 0.0, 0.0001);
}

// Test that wheel position vectors are created correctly for two pseudo-random wheel angles
TEST_F(OmniWheelRobotModelTest,
       createWheelPositionVectors) {
    model = std::make_shared<OmniWheelRobotModel>(0.1, 2.0, 0.6, 0.3);

    float wheel_position_vectors[4][3];
    model->createWheelPositionVectors(wheel_position_vectors);

    // Front left
    EXPECT_NEAR(wheel_position_vectors[0][0], 0.072630, 0.0001);
    EXPECT_NEAR(wheel_position_vectors[0][1], 0.049689, 0.0001);
    EXPECT_NEAR(wheel_position_vectors[0][2], 0.0, 0.0001);

    // Front right
    EXPECT_NEAR(wheel_position_vectors[3][0], 0.072630, 0.0001);
    EXPECT_NEAR(wheel_position_vectors[3][1], -0.049689, 0.0001);
    EXPECT_NEAR(wheel_position_vectors[3][2], 0.0, 0.0001);

    // Back left
    EXPECT_NEAR(wheel_position_vectors[1][0], -0.084070, 0.0001);
    EXPECT_NEAR(wheel_position_vectors[1][1], 0.026006, 0.0001);
    EXPECT_NEAR(wheel_position_vectors[1][2], 0.0, 0.0001);

    // Back right
    EXPECT_NEAR(wheel_position_vectors[2][0], -0.084070, 0.0001);
    EXPECT_NEAR(wheel_position_vectors[2][1], -0.026006, 0.0001);
    EXPECT_NEAR(wheel_position_vectors[2][2], 0.0, 0.0001);
}

enum WheelSpeedDirection {
    CW,
    CCW,
    ZERO,
};
struct GetWheelSpeedsTestParams {
    std::string test_name;
    float yaw;
    float v_x;
    float v_y;
    float v_angular;
    WheelSpeedDirection wheel_directions[4];
};

class GetWheelSpeedsTest : public testing::TestWithParam<GetWheelSpeedsTestParams> {
    void SetUp() {
        model = std::make_shared<OmniWheelRobotModel>(0.1, 2.0, 0.785, 0.785);
    }

    void TearDown() {
    }

    static void checkRawArrayEq(int n, float *expected, float *actual, float tol = 1e-4) {
        for (int i = 0; i < n; i++) {
            EXPECT_NEAR(expected[i], actual[i], tol);
        }
    }

protected:
    std::shared_ptr<OmniWheelRobotModel> model;
};

TEST_P(GetWheelSpeedsTest, checkWheelDirections) {
    GetWheelSpeedsTestParams params = GetParam();

    float wheel_speeds[4] = {0, 0, 0, 0};
    model->getWheelSpeeds(params.yaw, params.v_x, params.v_y, params.v_angular, wheel_speeds);

    for (int i = 0; i < 4; i++) {
        switch (params.wheel_directions[i]) {
            case WheelSpeedDirection::CW:
                EXPECT_LE(wheel_speeds[i], -0.1)
                                    << " Test Name: " << params.test_name << ", i=" << i;
                break;
            case WheelSpeedDirection::CCW:
                EXPECT_GE(wheel_speeds[i], 0.1)
                                    << " Test Name: " << params.test_name << ", i=" << i;
                break;
            case WheelSpeedDirection::ZERO:
                EXPECT_NEAR(0.0, wheel_speeds[i], 0.01)
                                    << " Test Name: " << params.test_name << ", i=" << i;
                break;
        }
    }
}

static std::string getNameForWheelSpeedsTest(GetWheelSpeedsTestParams params) {
    return params.test_name;
}

INSTANTIATE_TEST_CASE_P(checkWheelDirections, GetWheelSpeedsTest, testing::Values(
        (GetWheelSpeedsTestParams) {
                .test_name = "no velocity, no yaw",
                .yaw = 0,
                .v_x = 0,
                .v_y = 0,
                .v_angular = 0,
                .wheel_directions = {ZERO, ZERO, ZERO, ZERO},
        },
        (GetWheelSpeedsTestParams) {
                .test_name = "moving forwards, no yaw",
                .yaw = 0,
                .v_x = 1,
                .v_y = 0,
                .v_angular = 0,
                .wheel_directions = {CW, CW, CCW, CCW},
        },
        (GetWheelSpeedsTestParams) {
                .test_name = "moving backwards, no yaw",
                .yaw = 0,
                .v_x = -1,
                .v_y = 0,
                .v_angular = 0,
                .wheel_directions = {CCW, CCW, CW, CW},
        },
        (GetWheelSpeedsTestParams) {
                .test_name = "moving left, no yaw",
                .yaw = 0,
                .v_x = 0,
                .v_y = 1,
                .v_angular = 0,
                .wheel_directions = {CCW, CW, CW, CCW},
        },
        (GetWheelSpeedsTestParams) {
                .test_name = "moving right, no yaw",
                .yaw = 0,
                .v_x = 0,
                .v_y = -1,
                .v_angular = 0,
                .wheel_directions = {CW, CCW, CCW, CW},
        },
        (GetWheelSpeedsTestParams) {
                .test_name = "no velocity, facing backwards",
                .yaw = 0,
                .v_x = 0,
                .v_y = 0,
                .v_angular = 0,
                .wheel_directions = {ZERO, ZERO, ZERO, ZERO},
        },
        (GetWheelSpeedsTestParams) {
                .test_name = "moving +x, yaw = pi",
                .yaw = M_PI,
                .v_x = 1,
                .v_y = 0,
                .v_angular = 0,
                .wheel_directions = {CCW, CCW, CW, CW},
        },
        (GetWheelSpeedsTestParams) {
                .test_name = "moving -x, yaw = pi",
                .yaw = M_PI,
                .v_x = -1,
                .v_y = 0,
                .v_angular = 0,
                .wheel_directions = {CW, CW, CCW, CCW},
        },
        (GetWheelSpeedsTestParams) {
                .test_name = "moving +y, yaw = pi",
                .yaw = M_PI,
                .v_x = 0,
                .v_y = 1,
                .v_angular = 0,
                .wheel_directions = {CW, CCW, CCW, CW},
        },
        (GetWheelSpeedsTestParams) {
                .test_name = "moving -y, yaw = pi",
                .yaw = M_PI,
                .v_x = 0,
                .v_y = -1,
                .v_angular = 0,
                .wheel_directions = {CCW, CW, CW, CCW},
        },
        (GetWheelSpeedsTestParams) {
                .test_name = "spinning clockwise, yaw = pi",
                .yaw = M_PI,
                .v_x = 0,
                .v_y = 0,
                .v_angular = -2,
                .wheel_directions = {CW, CW, CW, CW},
        },
        (GetWheelSpeedsTestParams) {
                .test_name = "spinning counter-clockwise, yaw = pi",
                .yaw = M_PI,
                .v_x = 0,
                .v_y = 0,
                .v_angular = 2,
                .wheel_directions = {CCW, CCW, CCW, CCW},
        },
        (GetWheelSpeedsTestParams) {
                .test_name = "moving +x, facing +y",
                .yaw = M_PI / 2,
                .v_x = 1,
                .v_y = 0,
                .v_angular = 0,
                .wheel_directions = {CW, CCW, CCW, CW},
        },
        (GetWheelSpeedsTestParams) {
                .test_name = "moving -x, facing +y",
                .yaw = M_PI / 2,
                .v_x = -1,
                .v_y = 0,
                .v_angular = 0,
                .wheel_directions = {CCW, CW, CW, CCW},
        },
        (GetWheelSpeedsTestParams) {
                .test_name = "moving +y, facing +y",
                .yaw = M_PI / 2,
                .v_x = 0,
                .v_y = 1,
                .v_angular = 0,
                .wheel_directions = {CW, CW, CCW, CCW},
        },
        (GetWheelSpeedsTestParams) {
                .test_name = "moving -y, facing +y",
                .yaw = M_PI / 2,
                .v_x = 0,
                .v_y = -1,
                .v_angular = 0,
                .wheel_directions = {CCW, CCW, CW, CW},
        }
));

// Zero Wheel Speed Tests
TEST_F(OmniWheelRobotModelTest, scaleWheelForce_zero_wheel_speed_zero_force){
    float scaled_force = model->scaleWheelForce(0, 0);
    EXPECT_EQ(0, scaled_force);
}

// Small Wheel Speed Tests
TEST_F(OmniWheelRobotModelTest, scaleWheelForce_small_pos_wheel_speed_small_pos_force){
    float scaled_force = model->scaleWheelForce(0.1, 0.1);
    EXPECT_GE(scaled_force, 0.095);
}
TEST_F(OmniWheelRobotModelTest, scaleWheelForce_small_pos_wheel_speed_small_neg_force){
    float scaled_force = model->scaleWheelForce(0.1, -0.1);
    EXPECT_LE(scaled_force, -0.095);
}
TEST_F(OmniWheelRobotModelTest, scaleWheelForce_small_neg_wheel_speed_small_neg_force){
    float scaled_force = model->scaleWheelForce(-0.1, -0.1);
    EXPECT_LE(scaled_force, -0.095);
}
TEST_F(OmniWheelRobotModelTest, scaleWheelForce_small_neg_wheel_speed_small_pos_force){
    float scaled_force = model->scaleWheelForce(-0.1, 0.1);
    EXPECT_GE(scaled_force, 0.095);
}
TEST_F(OmniWheelRobotModelTest, scaleWheelForce_small_pos_wheel_speed_large_pos_force){
    float scaled_force = model->scaleWheelForce(0.1, 1.5);
    EXPECT_NEAR(scaled_force, 0.01, 0.01);
}
TEST_F(OmniWheelRobotModelTest, scaleWheelForce_small_pos_wheel_speed_large_neg_force){
    float scaled_force = model->scaleWheelForce(0.1, -1.5);
    EXPECT_NEAR(scaled_force, -0.01, 0.01);
}

// Large wheel speed tests
TEST_F(OmniWheelRobotModelTest, scaleWheelForce_large_pos_wheel_speed_small_pos_force){
    float scaled_force = model->scaleWheelForce(3, 0.1);
    EXPECT_GE(scaled_force, 0.095);
}
TEST_F(OmniWheelRobotModelTest, scaleWheelForce_large_pos_wheel_speed_small_neg_force){
    float scaled_force = model->scaleWheelForce(3, -0.1);
    EXPECT_LE(scaled_force, -0.095);
}
TEST_F(OmniWheelRobotModelTest, scaleWheelForce_large_neg_wheel_speed_small_neg_force){
    float scaled_force = model->scaleWheelForce(-3, -0.1);
    EXPECT_LE(scaled_force, -0.095);
}
TEST_F(OmniWheelRobotModelTest, scaleWheelForce_large_neg_wheel_speed_small_pos_force){
    float scaled_force = model->scaleWheelForce(-3, 0.1);
    EXPECT_GE(scaled_force, 0.095);
}
TEST_F(OmniWheelRobotModelTest, scaleWheelForce_large_pos_wheel_speed_large_pos_force){
    float scaled_force = model->scaleWheelForce(3, 1.5);
    EXPECT_NEAR(scaled_force, 1.45, 0.051);
}
TEST_F(OmniWheelRobotModelTest, scaleWheelForce_large_pos_wheel_speed_large_neg_force){
    float scaled_force = model->scaleWheelForce(3, -1.5);
    EXPECT_NEAR(scaled_force, 0.01, 0.011);
}
TEST_F(OmniWheelRobotModelTest, scaleWheelForce_large_neg_wheel_speed_large_pos_force){
    float scaled_force = model->scaleWheelForce(-3, 1.5);
    EXPECT_NEAR(scaled_force, 0.01, 0.01);
}
TEST_F(OmniWheelRobotModelTest, scaleWheelForce_large_neg_wheel_speed_large_neg_force){
    float scaled_force = model->scaleWheelForce(-3, -1.5);
    EXPECT_NEAR(scaled_force, -1.45, 0.051);
}

TEST_F(OmniWheelRobotModelTest, scaleControls_not_moving_small_wheel_force){
    float state[6] = {0, 0, 0, 0, 0, 0};
    float controls[4] = {0.1, 0.1, 0.1, 0.1};
    float scaled_controls[4] = {0, 0, 0, 0};

    model->scaleControls(state, controls, scaled_controls);

    EXPECT_NEAR(0.095, scaled_controls[0], 0.005);
    EXPECT_NEAR(0.095, scaled_controls[1], 0.005);
    EXPECT_NEAR(0.095, scaled_controls[2], 0.005);
    EXPECT_NEAR(0.095, scaled_controls[3], 0.005);
}

TEST_F(OmniWheelRobotModelTest, scaleControls_not_moving_large_wheel_force){
    float state[6] = {0, 0, 0, 0, 0, 0};
    float controls[4] = {1.5, 1.5, 1.5, 1.5};
    float scaled_controls[4] = {0, 0, 0, 0};

    model->scaleControls(state, controls, scaled_controls);

    EXPECT_NEAR(0.01, scaled_controls[0], 0.01);
    EXPECT_NEAR(0.01, scaled_controls[1], 0.01);
    EXPECT_NEAR(0.01, scaled_controls[2], 0.01);
    EXPECT_NEAR(0.01, scaled_controls[3], 0.01);
}

TEST_F(OmniWheelRobotModelTest, scaleControls_not_moving_some_wheels_large_force_others_small){
    float state[6] = {0, 0, 0, 0, 0, 0};
    float controls[4] = {1.5, 0.1, 1.5, 0.1};
    float scaled_controls[4] = {0, 0, 0, 0};

    model->scaleControls(state, controls, scaled_controls);

    EXPECT_NEAR(0.01, scaled_controls[0], 0.01);
    EXPECT_NEAR(0.095, scaled_controls[1], 0.005);
    EXPECT_NEAR(0.01, scaled_controls[0], 0.01);
    EXPECT_NEAR(0.095, scaled_controls[3], 0.005);
}

TEST_F(OmniWheelRobotModelTest, scaleControls_moving_forwards_large_forward_force){
    float state[6] = {0, 0, 0, 1, 0, 0};
    float controls[4] = {-1, -1, 1, 1};
    float scaled_controls[4] = {0, 0, 0, 0};

    model->scaleControls(state, controls, scaled_controls);

    EXPECT_NEAR(-0.95, scaled_controls[0], 0.05);
    EXPECT_NEAR(-0.95, scaled_controls[1], 0.05);
    EXPECT_NEAR(0.95, scaled_controls[2], 0.05);
    EXPECT_NEAR(0.95, scaled_controls[3], 0.05);
}

TEST_F(OmniWheelRobotModelTest, scaleControls_moving_forwards_large_backwards_force){
    float state[6] = {0, 0, 0, -1, 0, 0};
    float controls[4] = {1, 1, -1, -1};
    float scaled_controls[4] = {0, 0, 0, 0};

    model->scaleControls(state, controls, scaled_controls);

    EXPECT_NEAR(0.95, scaled_controls[0], 0.05);
    EXPECT_NEAR(0.95, scaled_controls[1], 0.05);
    EXPECT_NEAR(-0.95, scaled_controls[2], 0.05);
    EXPECT_NEAR(-0.95, scaled_controls[3], 0.05);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
