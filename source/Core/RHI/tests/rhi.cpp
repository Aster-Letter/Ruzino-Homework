#include "RHI/rhi.hpp"

#include <gtest/gtest.h>

TEST(CreateRHI, create_rhi)
{
    EXPECT_EQ(Ruzino::RHI::init(), 0);
    EXPECT_TRUE(Ruzino::RHI::get_device() != nullptr);
    EXPECT_EQ(Ruzino::RHI::shutdown(), 0);
}

TEST(CreateRHI, create_rhi_with_window)
{
    EXPECT_EQ(Ruzino::RHI::init(true), 0);
    EXPECT_TRUE(Ruzino::RHI::get_device() != nullptr);
    EXPECT_TRUE(Ruzino::RHI::internal::get_device_manager() != nullptr);
    EXPECT_EQ(Ruzino::RHI::shutdown(), 0);
}

#ifndef __linux__
TEST(CreateRHI, create_rhi_with_dx12)
{
    EXPECT_EQ(Ruzino::RHI::init(false, true), 0);
    EXPECT_TRUE(Ruzino::RHI::get_device() != nullptr);
    EXPECT_TRUE(Ruzino::RHI::internal::get_device_manager() != nullptr);
    EXPECT_EQ(Ruzino::RHI::shutdown(), 0);
}

TEST(CreateRHI, create_rhi_with_window_and_dx12)
{
    EXPECT_EQ(Ruzino::RHI::init(true, true), 0);
    EXPECT_TRUE(Ruzino::RHI::get_device() != nullptr);
    EXPECT_TRUE(Ruzino::RHI::internal::get_device_manager() != nullptr);
    EXPECT_EQ(Ruzino::RHI::shutdown(), 0);
}
#endif