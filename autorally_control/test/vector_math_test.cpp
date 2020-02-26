#include <gtest/gtest.h>
#include "autorally_control/path_integral/vector_math.cuh"

TEST(OmniWheelRobotModelTest, test_vector3_addition){
  float v1[3] = {1, 2, 3};
  float v2[3] = {4, -5.5, 6};
  float result[3] = {0, 0, 0};
  addVector3(v1, v2, result);

  EXPECT_DOUBLE_EQ(5, result[0]);
  EXPECT_DOUBLE_EQ(-3.5, result[1]);
  EXPECT_DOUBLE_EQ(9, result[2]);
}

TEST(OmniWheelRobotModelTest, test_matrix3x3_vector3_multiplication){
  float v[3] = {1, 2, 3};
  float M[3][3] = {
    {4, -2, 5.5},
    {1, 2, 3},
    {3.7, 2.33, 4.1},
  };
  float result[3] = {0, 0, 0};
  multiplyVector3By3x3Matrix(M, v, result);

  EXPECT_NEAR(16.5, result[0], 1e-5);
  EXPECT_NEAR(14.0, result[1], 1e-5);
  EXPECT_NEAR(20.66, result[2], 1e-5);
}

TEST(OmniWheelRobotModelTest, test_dot_product_vector3){
  float v1[3] = {1, 2, 3};
  float v2[3] = {3.4, 5.3, 29};
  
  EXPECT_DOUBLE_EQ(101, dotProductVector3(v1, v2));
}

TEST(OmniWheelRobotModelTest, test_cross_product_vector3){
  float v1[3] = {1, 2, 3};
  float v2[3] = {3.4, 5.3, 29};
  float result[3] = {0, 0, 0};
  crossProductVector3(v1, v2, result);
  
  EXPECT_NEAR(42.1, result[0], 1e-5);
  EXPECT_NEAR(-18.8, result[1], 1e-5);
  EXPECT_NEAR(-1.5, result[2], 1e-5);
}

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
    return 0;
}
