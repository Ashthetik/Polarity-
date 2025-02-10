#include <gtest/gtest.h>
#include <vector>
#include <cstdint>
#include "bhs.h"
#include "Matrix.h"

class CleanTableDataTest : public ::testing::Test {
protected:
    void SetUp() override {
    }

    void TearDown() override {
    }
};

TEST_F(CleanTableDataTest, HandlesEmptyDataTable) {
    DataTable emptyTable;
    ASSERT_NO_THROW(clean_table_data(emptyTable));
}

TEST_F(CleanTableDataTest, ProcessesSingleBehaviourDataEntry) {
    DataTable table;
    BehaviourTable behaviour;
    
    // Set up test data
    behaviour.movement_occurrences = {Mat3D{1, 2, 3}, Mat3D{4, 5, 6}};
    behaviour.bpm_fluctuation = {70.5f, 80.2f, 75.8f};
    behaviour.voice_fluctation = {0.1f, 0.2f, 0.3f};
    table.behaviour_data.push_back(behaviour);

    // Call the function
    clean_table_data(table);

    // Since the function doesn't return or modify the input, we can't directly test the output.
    // However, we can verify that it doesn't throw any exceptions and completes successfully.
    SUCCEED();
}

TEST_F(CleanTableDataTest, HandlesMaxSizeInputVectors) {
    DataTable table;
    BehaviourTable behaviour;
    
    // Create vectors at their maximum size
    std::vector<Mat3D> max_movement(std::vector<Mat3D>::max_size(), Mat3D{1, 1, 1});
    std::vector<float> max_bpm(std::vector<float>::max_size(), 70.0f);
    std::vector<float> max_voice(std::vector<float>::max_size(), 0.1f);
    
    behaviour.movement_occurrences = max_movement;
    behaviour.bpm_fluctuation = max_bpm;
    behaviour.voice_fluctation = max_voice;
    table.behaviour_data.push_back(behaviour);

    // Expect no exceptions when processing maximum size vectors
    ASSERT_NO_THROW(clean_table_data(table));
}

TEST_F(CleanTableDataTest, HandlesZeroSizeInputVectors) {
    DataTable table;
    BehaviourTable behaviour;
    
    // Set up test data with zero-size vectors
    behaviour.movement_occurrences = {};
    behaviour.bpm_fluctuation = {};
    behaviour.voice_fluctation = {};
    table.behaviour_data.push_back(behaviour);

    // Call the function
    ASSERT_NO_THROW(clean_table_data(table));

    // Since the function doesn't return or modify the input, we can't directly test the output.
    // However, we can verify that it completes successfully without throwing exceptions.
    SUCCEED();
}

TEST_F(CleanTableDataTest, UsesDBLEpsilonInLogTransformations) {
    DataTable table;
    BehaviourTable behaviour;

    // Set up test data
    behaviour.movement_occurrences = {Mat3D{1, 2, 3}};
    behaviour.bpm_fluctuation = {70.5f};
    behaviour.voice_fluctation = {0.1f};
    table.behaviour_data.push_back(behaviour);

    // Mock the convert_mat and convert_float functions
    EXPECT_CALL(*this, convert_mat(testing::_))
        .WillOnce(testing::Return(std::vector<int64_t>{100}));
    EXPECT_CALL(*this, convert_float(testing::_))
        .WillRepeatedly(testing::Return(std::vector<int64_t>{200}));

    // Mock the log_transform function to check its input
    EXPECT_CALL(*this, log_transform(testing::_, testing::_, testing::_, testing::_))
        .Times(3)
        .WillRepeatedly(testing::Invoke([](const std::vector<int64_t>& x, int32_t a, int64_t h, int64_t k) {
            EXPECT_EQ(k, static_cast<int64_t>(__DBL_EPSILON__));
            return std::vector<int64_t>{};
        }));

    clean_table_data(table);
}

TEST_F(CleanTableDataTest, DoesNotModifyOriginalDataTable) {
    DataTable originalTable;
    BehaviourTable behaviour;
    
    // Set up test data
    behaviour.movement_occurrences = {Mat3D{1, 2, 3}, Mat3D{4, 5, 6}};
    behaviour.bpm_fluctuation = {70.5f, 80.2f, 75.8f};
    behaviour.voice_fluctation = {0.1f, 0.2f, 0.3f};
    originalTable.behaviour_data.push_back(behaviour);

    // Create a copy of the original table
    DataTable tableCopy = originalTable;

    // Call the function
    clean_table_data(tableCopy);

    // Verify that the original table remains unchanged
    ASSERT_EQ(originalTable.behaviour_data.size(), tableCopy.behaviour_data.size());
    ASSERT_EQ(originalTable.behaviour_data[0].movement_occurrences, tableCopy.behaviour_data[0].movement_occurrences);
    ASSERT_EQ(originalTable.behaviour_data[0].bpm_fluctuation, tableCopy.behaviour_data[0].bpm_fluctuation);
    ASSERT_EQ(originalTable.behaviour_data[0].voice_fluctation, tableCopy.behaviour_data[0].voice_fluctation);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
