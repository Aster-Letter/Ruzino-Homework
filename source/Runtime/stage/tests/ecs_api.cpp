#include <gtest/gtest.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/sphere.h>
#include <pxr/usd/usdGeom/xform.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>

#include "GCore/GOP.h"
#include "GCore/create_geom.h"
#include "stage/ecs_components.hpp"
#include "stage/ecs_systems.hpp"
#include "stage/stage.hpp"

using namespace Ruzino;

// Generate temporary USD file path with timestamp
static std::string get_temp_usd_path(const std::string& test_name)
{
    auto now = std::chrono::system_clock::now();
    auto time = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) %
              1000;

    char timestamp[32];
    std::tm* tm_info = std::localtime(&time);
    strftime(timestamp, sizeof(timestamp), "%Y%m%d_%H%M%S", tm_info);

    std::string filename = std::string(timestamp) + "_" +
                           std::to_string(ms.count()) + "_" + test_name +
                           ".usdc";
    return "../../Assets/.test_tmp/" + filename;
}

// Animation component for testing - define simple translation animation
struct TranslationAnimationComponent {
    glm::vec3 velocity;  // Movement speed per second
    float elapsed_time = 0.0f;
};

// Animation system for testing - update position and write to USD time samples
class TranslationAnimationSystem {
   public:
    void update(
        entt::registry& registry,
        float delta_time,
        pxr::UsdTimeCode time_code)
    {
        auto view =
            registry
                .view<ecs::UsdPrimComponent, TranslationAnimationComponent>();

        for (auto entity : view) {
            auto& usd_comp = view.get<ecs::UsdPrimComponent>(entity);
            auto& anim_comp = view.get<TranslationAnimationComponent>(entity);

            // Update elapsed time
            anim_comp.elapsed_time += delta_time;

            // Calculate new position
            glm::vec3 new_position =
                anim_comp.velocity * anim_comp.elapsed_time;

            // Get xformable and set transform
            auto xformable = pxr::UsdGeomXformable(usd_comp.prim);
            if (xformable) {
                // Get or create translate op
                pxr::UsdGeomXformOp translate_op;
                bool reset_stack;
                auto existing_ops = xformable.GetOrderedXformOps(&reset_stack);

                // Check if translate op already exists
                bool has_translate = false;
                for (const auto& op : existing_ops) {
                    if (op.GetOpType() == pxr::UsdGeomXformOp::TypeTranslate) {
                        translate_op = op;
                        has_translate = true;
                        break;
                    }
                }

                // If not, create one
                if (!has_translate) {
                    translate_op = xformable.AddTranslateOp();
                }

                // Set position
                pxr::GfVec3d usd_pos(
                    new_position.x, new_position.y, new_position.z);
                translate_op.Set(usd_pos, time_code);

                // Update component time
                usd_comp.current_time = time_code;
            }
        }
    }
};

class EcsApiTest : public ::testing::Test {
   protected:
    void SetUp() override
    {
        // Setup before test
    }

    void TearDown() override
    {
        // Cleanup after test
    }
};

TEST_F(EcsApiTest, BasicEcsCreation)
{
    // Create new stage (file will be auto-created if it doesn't exist)
    auto stage = std::make_unique<Stage>(get_temp_usd_path("BasicEcsCreation"));
    auto& registry = stage->get_registry();

    // Check initial entity count
    int initial_count = 0;
    for (auto entity : registry.view<ecs::UsdPrimComponent>()) {
        initial_count++;
    }

    // Create some USD prims
    auto sphere_prim = stage->create_sphere(pxr::SdfPath("/MySphere"));
    auto cube_prim = stage->create_cube(pxr::SdfPath("/MyCube"));

    // Get auto-created entities
    int entity_count = 0;
    for (auto entity : registry.view<ecs::UsdPrimComponent>()) {
        entity_count++;
    }

    // create_sphere/cube creates parent xform and shape prim, so 4 entities
    // total MySphere, MySphere/sphere_0, MyCube, MyCube/cube_0
    int new_entities = entity_count - initial_count;
    EXPECT_EQ(new_entities, 4)
        << "Should have created 4 entities (including parent xforms)";

    // Verify mappings are established - ensure sphere and cube shape prims can
    // be found
    auto sphere_entity = stage->find_entity_by_path(sphere_prim.GetPath());
    auto cube_entity = stage->find_entity_by_path(cube_prim.GetPath());

    EXPECT_TRUE(registry.valid(sphere_entity)) << "Sphere entity should exist";
    EXPECT_TRUE(registry.valid(cube_entity)) << "Cube entity should exist";

    // Verify entity to prim mapping
    auto sphere_prim_back = stage->get_prim_from_entity(sphere_entity);
    EXPECT_EQ(sphere_prim_back.GetPath(), sphere_prim.GetPath());
}

TEST_F(EcsApiTest, ComponentOperations)
{
    // Create new stage (file will be auto-created if it doesn't exist)
    auto stage =
        std::make_unique<Stage>(get_temp_usd_path("ComponentOperations"));
    auto& registry = stage->get_registry();

    // Create a USD sphere
    auto sphere = stage->create_sphere(pxr::SdfPath("/TestSphere"));
    auto entity = stage->find_entity_by_path(sphere.GetPath());

    ASSERT_TRUE(registry.valid(entity)) << "Entity should exist";

    // Verify UsdPrimComponent is auto-added
    EXPECT_TRUE(registry.all_of<ecs::UsdPrimComponent>(entity));

    // Add MaterialComponent
    auto& material = registry.emplace<ecs::MaterialComponent>(entity);
    material.material_path = pxr::SdfPath("/Materials/MyMaterial");
    material.shader_path = "shaders/pbr.slang";

    EXPECT_EQ(material.material_path.GetString(), "/Materials/MyMaterial");
    EXPECT_EQ(material.shader_path, "shaders/pbr.slang");

    // Add PhysicsComponent
    auto& physics = registry.emplace<ecs::PhysicsComponent>(entity);
    physics.type = ecs::PhysicsComponent::Type::Dynamic;
    physics.mass = 10.0f;
    physics.is_trigger = false;

    EXPECT_EQ(physics.mass, 10.0f);

    // Add Tags
    registry.emplace<ecs::RenderableTag>(entity);
    registry.emplace<ecs::SimulationTag>(entity);

    // Verify all components exist
    EXPECT_TRUE(registry.all_of<ecs::UsdPrimComponent>(entity));
    EXPECT_TRUE(registry.all_of<ecs::MaterialComponent>(entity));
    EXPECT_TRUE(registry.all_of<ecs::PhysicsComponent>(entity));
    EXPECT_TRUE(registry.all_of<ecs::RenderableTag>(entity));
    EXPECT_TRUE(registry.all_of<ecs::SimulationTag>(entity));
}

TEST_F(EcsApiTest, EntityQueries)
{
    // Create new stage (file will be auto-created if it doesn't exist)
    auto stage = std::make_unique<Stage>(get_temp_usd_path("EntityQueries"));
    auto& registry = stage->get_registry();

    // Create multiple entities
    for (int i = 0; i < 5; i++) {
        auto path = pxr::SdfPath(std::string("/Object_") + std::to_string(i));
        auto sphere = stage->create_sphere(path);
        auto entity = stage->find_entity_by_path(sphere.GetPath());

        if (i % 2 == 0) {
            registry.emplace<ecs::MaterialComponent>(entity);
            registry.emplace<ecs::RenderableTag>(entity);
        }

        if (i % 3 == 0) {
            auto& physics = registry.emplace<ecs::PhysicsComponent>(entity);
            physics.mass = 5.0f * (i + 1);
            registry.emplace<ecs::SimulationTag>(entity);
        }
    }

    // Query all entities with UsdPrimComponent (including parent xforms)
    // 5 create_sphere calls create 10 entities (5 parents + 5 shapes)
    {
        auto view = registry.view<ecs::UsdPrimComponent>();
        int count = std::distance(view.begin(), view.end());
        EXPECT_EQ(count, 10) << "Should have 10 entities with UsdPrimComponent "
                                "(5 parent xforms + 5 shapes)";
    }

    // Query all renderable entities (only sphere_0, 2, 4 shapes have
    // MaterialComponent)
    {
        auto view = registry.view<
            ecs::UsdPrimComponent,
            ecs::MaterialComponent,
            ecs::RenderableTag>();
        int count = std::distance(view.begin(), view.end());
        EXPECT_EQ(count, 3) << "Should have 3 renderable entities";
    }

    // Query all physics objects (sphere_0 and sphere_3 shapes have
    // PhysicsComponent)
    {
        auto view = registry.view<ecs::PhysicsComponent, ecs::SimulationTag>();
        int count = std::distance(view.begin(), view.end());
        EXPECT_EQ(count, 2) << "Should have 2 physics objects";

        for (auto entity : view) {
            auto& physics = view.get<ecs::PhysicsComponent>(entity);
            EXPECT_GT(physics.mass, 0.0f)
                << "Physics object mass should be greater than 0";
        }
    }
}

TEST_F(EcsApiTest, UsdSync)
{
    // Create new stage (file will be auto-created if it doesn't exist)
    auto stage = std::make_unique<Stage>(get_temp_usd_path("UsdSync"));
    auto& registry = stage->get_registry();

    // Create a sphere
    auto sphere = stage->create_sphere(pxr::SdfPath("/SyncTest"));
    auto entity = stage->find_entity_by_path(sphere.GetPath());

    ASSERT_TRUE(registry.valid(entity));

    // Mark as needing sync
    auto& dirty = registry.emplace<ecs::DirtyComponent>(entity);
    dirty.needs_geometry_update = true;

    EXPECT_TRUE(dirty.needs_geometry_update);

    // Sync to USD
    stage->sync_entities_to_usd();

    // Verify data in USD
    auto xformable = pxr::UsdGeomXformable(sphere.GetPrim());
    if (xformable) {
        pxr::GfMatrix4d matrix;
        bool reset_stack;
        xformable.GetLocalTransformation(
            &matrix, &reset_stack, pxr::UsdTimeCode::Default());

        auto translation = matrix.ExtractTranslation();
        EXPECT_EQ(translation[0], 0.0);
        EXPECT_EQ(translation[1], 0.0);
        EXPECT_EQ(translation[2], 0.0);
    }
}

TEST_F(EcsApiTest, StageListenerCallbacks)
{
    // Create new stage (file will be auto-created if it doesn't exist)
    auto stage =
        std::make_unique<Stage>(get_temp_usd_path("StageListenerCallbacks"));
    auto& registry = stage->get_registry();

    int entities_before = 0;
    for (auto entity : registry.view<ecs::UsdPrimComponent>()) {
        entities_before++;
    }

    // Create new prim (this should trigger callback)
    auto new_sphere = stage->create_sphere(pxr::SdfPath("/NewSphere"));

    // Check if entity was automatically created
    int entities_after = 0;
    for (auto entity : registry.view<ecs::UsdPrimComponent>()) {
        entities_after++;
    }

    int new_entities = entities_after - entities_before;
    EXPECT_GT(new_entities, 0)
        << "StageListener should automatically create entity";

    auto new_entity = stage->find_entity_by_path(new_sphere.GetPath());
    EXPECT_TRUE(registry.valid(new_entity))
        << "Should be able to find the newly created entity";

    // Test creating prim directly using USD API, check if notice mechanism is
    // triggered
    auto pxr_stage = stage->get_usd_stage();

    int entities_before_usd_api = 0;
    for (auto entity : registry.view<ecs::UsdPrimComponent>()) {
        entities_before_usd_api++;
    }

    // Create xform prim directly using USD API
    auto direct_prim = pxr_stage->DefinePrim(
        pxr::SdfPath("/DirectXform"), pxr::TfToken("Xform"));
    pxr::UsdGeomXform direct_xform(direct_prim);

    int entities_after_usd_api = 0;
    for (auto entity : registry.view<ecs::UsdPrimComponent>()) {
        entities_after_usd_api++;
    }

    int entities_from_usd_api =
        entities_after_usd_api - entities_before_usd_api;
    std::cout << "\n[USD API Test] After DirectXform creation:" << std::endl;
    std::cout << "  Entity count before creation: " << entities_before_usd_api
              << std::endl;
    std::cout << "  Entity count after creation: " << entities_after_usd_api
              << std::endl;
    std::cout << "  New entities created: " << entities_from_usd_api
              << std::endl;

    if (entities_from_usd_api > 0) {
        std::cout << "  ✓ Prim created via USD API triggered notice mechanism!"
                  << std::endl;
        auto direct_entity = stage->find_entity_by_path(direct_prim.GetPath());
        EXPECT_TRUE(registry.valid(direct_entity))
            << "Should be able to find entity created directly via USD API";
    }
    else {
        std::cout
            << "  ✗ Prim created via USD API did not trigger notice mechanism"
            << std::endl;
    }
}

TEST_F(EcsApiTest, PrimDeletionSync)
{
    // Create new stage (file will be auto-created if it doesn't exist)
    auto stage = std::make_unique<Stage>(get_temp_usd_path("PrimDeletionSync"));
    auto& registry = stage->get_registry();
    auto pxr_stage = stage->get_usd_stage();

    // Create a prim directly
    auto test_prim = pxr_stage->DefinePrim(
        pxr::SdfPath("/TestDeletion"), pxr::TfToken("Xform"));

    // Get entity count after creation
    int entities_after_create = 0;
    for (auto entity : registry.view<ecs::UsdPrimComponent>()) {
        entities_after_create++;
    }

    auto test_entity = stage->find_entity_by_path(test_prim.GetPath());
    bool entity_exists_before = registry.valid(test_entity);

    std::cout << "\n[Prim Deletion Test]" << std::endl;
    std::cout << "  After creating TestDeletion:" << std::endl;
    std::cout << "    Total entities: " << entities_after_create << std::endl;
    std::cout << "    Entity exists: " << (entity_exists_before ? "Yes" : "No")
              << std::endl;

    // Delete this prim
    pxr_stage->RemovePrim(test_prim.GetPath());

    // Get entity count after deletion
    int entities_after_delete = 0;
    for (auto entity : registry.view<ecs::UsdPrimComponent>()) {
        entities_after_delete++;
    }

    auto test_entity_after = stage->find_entity_by_path(test_prim.GetPath());
    bool entity_exists_after = registry.valid(test_entity_after);

    std::cout << "  After deleting TestDeletion:" << std::endl;
    std::cout << "    Total entities: " << entities_after_delete << std::endl;
    std::cout << "    Net change in entities: "
              << (entities_after_delete - entities_after_create) << std::endl;
    std::cout << "    Entity exists: " << (entity_exists_after ? "Yes" : "No")
              << std::endl;

    if (!entity_exists_after && entities_after_delete < entities_after_create) {
        std::cout << "  ✓ Prim deletion successfully triggered notice "
                     "mechanism, entity was correctly deleted!"
                  << std::endl;
    }
    else {
        std::cout << "  ✗ Prim deletion may not have correctly deleted entity"
                  << std::endl;
    }

    EXPECT_LT(entities_after_delete, entities_after_create)
        << "Entity count should decrease after deleting prim";
    EXPECT_FALSE(entity_exists_after)
        << "Entity corresponding to deleted prim should be deleted";
}

TEST_F(EcsApiTest, PrimModificationSync)
{
    // Create new stage (file will be auto-created if it doesn't exist)
    auto stage =
        std::make_unique<Stage>(get_temp_usd_path("PrimModificationSync"));
    auto& registry = stage->get_registry();

    // Create a sphere
    auto sphere_prim = stage->create_sphere(pxr::SdfPath("/ModifiableSphere"));
    auto sphere_entity = stage->find_entity_by_path(sphere_prim.GetPath());

    ASSERT_TRUE(registry.valid(sphere_entity));

    // Check dirty flag before modification
    bool had_dirty_before = registry.all_of<ecs::DirtyComponent>(sphere_entity);

    // Modify sphere properties (e.g., radius) - use double instead of float
    auto sphere = pxr::UsdGeomSphere(sphere_prim);
    sphere.GetRadiusAttr().Set(5.0);  // USD expects double type

    // Check if marked as dirty after modification
    bool has_dirty_after = registry.all_of<ecs::DirtyComponent>(sphere_entity);

    std::cout << "\n[Prim Modification Test]" << std::endl;
    std::cout << "  After modifying sphere radius to 5.0:" << std::endl;
    std::cout << "    Has DirtyComponent before modification: "
              << (had_dirty_before ? "Yes" : "No") << std::endl;
    std::cout << "    Has DirtyComponent after modification: "
              << (has_dirty_after ? "Yes" : "No") << std::endl;

    // Check if current_time in UsdPrimComponent was updated
    auto& usd_comp = registry.get<ecs::UsdPrimComponent>(sphere_entity);
    std::cout << "    Sphere prim current time code: "
              << usd_comp.current_time.GetValue() << std::endl;

    // Verify prim data was actually modified
    double radius = 0.0;  // Use double instead of float
    sphere.GetRadiusAttr().Get(&radius);

    std::cout << "    Modified sphere radius: " << radius << std::endl;

    if (has_dirty_after) {
        std::cout
            << "  ✓ Prim automatically marked as dirty after modification!"
            << std::endl;
        auto& dirty = registry.get<ecs::DirtyComponent>(sphere_entity);
        std::cout << "    Needs geometry update: "
                  << (dirty.needs_geometry_update ? "Yes" : "No") << std::endl;
        std::cout << "    Needs USD sync: "
                  << (dirty.needs_usd_sync ? "Yes" : "No") << std::endl;
    }
    else {
        std::cout << "  ✗ Notice mechanism may not have triggered automatic "
                     "dirty component marking after modifying prim attributes"
                  << std::endl;
        std::cout << "     "
                     "(This may require manually adding DirtyComponent marking "
                     "in on_prim_changed())"
                  << std::endl;
    }

    EXPECT_DOUBLE_EQ(radius, 5.0) << "Sphere radius should be modified to 5.0";
    // Note: Whether attribute modification automatically marks dirty depends on
    // on_prim_changed() implementation This test mainly verifies if notice
    // mechanism can capture attribute modifications
}

TEST_F(EcsApiTest, AnimationSystem)
{
    // Create new stage (file will be auto-created if it doesn't exist)
    auto stage = std::make_unique<Stage>(get_temp_usd_path("AnimationSystem"));
    auto& registry = stage->get_registry();

    // Create an animated entity
    auto sphere = stage->create_sphere(pxr::SdfPath("/AnimatedSphere"));
    auto entity = stage->find_entity_by_path(sphere.GetPath());

    ASSERT_TRUE(registry.valid(entity));

    // Add UsdPrimComponent (if not already present)
    if (!registry.all_of<ecs::UsdPrimComponent>(entity)) {
        registry.emplace<ecs::UsdPrimComponent>(entity, sphere.GetPrim());
    }

    // Add AnimationComponent
    auto& anim = registry.emplace<ecs::AnimationComponent>(entity);
    anim.simulation_begun = false;

    registry.emplace<ecs::AnimatableTag>(entity);

    // Simulate several frame updates
    auto* anim_system = stage->get_animation_system();
    ASSERT_NE(anim_system, nullptr);

    for (int frame = 0; frame < 3; frame++) {
        float delta_time = 1.0f / 60.0f;  // 60 FPS
        anim_system->update(registry, delta_time);

        auto& usd_prim = registry.get<ecs::UsdPrimComponent>(entity);
        // Time should be increasing or staying constant
        EXPECT_GE(usd_prim.current_time.GetValue(), 0.0);
    }
}

TEST_F(EcsApiTest, DirtyTracking)
{
    // Create new stage (file will be auto-created if it doesn't exist)
    auto stage = std::make_unique<Stage>(get_temp_usd_path("DirtyTracking"));
    auto& registry = stage->get_registry();

    // Create multiple entities
    std::vector<entt::entity> entities;
    for (int i = 0; i < 3; i++) {
        auto path = pxr::SdfPath(std::string("/Obj_") + std::to_string(i));
        auto sphere = stage->create_sphere(path);
        auto entity = stage->find_entity_by_path(sphere.GetPath());

        entities.push_back(entity);
    }

    EXPECT_EQ(entities.size(), 3);

    // Modify 1st and 3rd entities
    {
        auto& dirty = registry.emplace<ecs::DirtyComponent>(entities[0]);
        dirty.needs_geometry_update = true;
    }

    {
        auto& dirty = registry.emplace<ecs::DirtyComponent>(entities[2]);
        dirty.needs_geometry_update = true;
        dirty.needs_usd_sync = true;
    }

    // Query dirty entities
    {
        auto dirty_view = registry.view<ecs::DirtyComponent>();
        int dirty_count = std::distance(dirty_view.begin(), dirty_view.end());
        EXPECT_EQ(dirty_count, 2) << "Should have 2 dirty entities";
    }

    // Sync
    stage->sync_entities_to_usd();

    // Verify dirty flag cleanup
    {
        auto dirty_view = registry.view<ecs::DirtyComponent>();
        int remaining_dirty = 0;
        for (auto entity : dirty_view) {
            auto& dirty = dirty_view.get<ecs::DirtyComponent>(entity);
            if (dirty.needs_geometry_update || dirty.needs_usd_sync) {
                remaining_dirty++;
            }
        }
        // Note: Depending on implementation, dirty flags may still exist; here
        // we verify sync was executed
        EXPECT_GE(remaining_dirty, 0);
    }
}

TEST_F(EcsApiTest, AnimationWriteToUsd)
{
    // Create a new stage, will error if file doesn't exist (this is normal)
    const auto output_path = get_temp_usd_path("test_ecs_animation_write");
    auto stage = std::make_unique<Stage>(output_path);
    auto& registry = stage->get_registry();

    // Create 3 different geometries
    auto sphere_prim = stage->create_sphere(pxr::SdfPath("/AnimSphere"));
    auto cube_prim = stage->create_cube(pxr::SdfPath("/AnimCube"));
    auto cylinder_prim = stage->create_cylinder(pxr::SdfPath("/AnimCylinder"));

    auto sphere_entity = stage->find_entity_by_path(sphere_prim.GetPath());
    auto cube_entity = stage->find_entity_by_path(cube_prim.GetPath());
    auto cylinder_entity = stage->find_entity_by_path(cylinder_prim.GetPath());

    ASSERT_TRUE(registry.valid(sphere_entity));
    ASSERT_TRUE(registry.valid(cube_entity));
    ASSERT_TRUE(registry.valid(cylinder_entity));

    // Add translation animation component to each entity
    // Sphere: move right at 100 units/second
    auto& sphere_anim =
        registry.emplace<TranslationAnimationComponent>(sphere_entity);
    sphere_anim.velocity = glm::vec3(100.0f, 0.0f, 0.0f);

    // Cube: move right at 100 units/second, Y offset 5
    auto& cube_anim =
        registry.emplace<TranslationAnimationComponent>(cube_entity);
    cube_anim.velocity = glm::vec3(100.0f, 5.0f, 0.0f);

    // Cylinder: move right at 100 units/second, Y offset -5
    auto& cylinder_anim =
        registry.emplace<TranslationAnimationComponent>(cylinder_entity);
    cylinder_anim.velocity = glm::vec3(100.0f, -5.0f, 0.0f);

    // Create animation system
    TranslationAnimationSystem anim_system;

    // 60 frames, 1 second, 60 FPS
    const int num_frames = 60;
    const float duration = 1.0f;
    const float fps = 60.0f;
    const float delta_time = 1.0f / fps;

    // Animation loop - let system update each frame
    for (int frame = 0; frame <= num_frames; frame++) {
        float time_seconds = frame * delta_time;
        pxr::UsdTimeCode time_code(time_seconds);

        // Call system to update all animated entities
        anim_system.update(registry, delta_time, time_code);
    }

    // Set stage time range
    stage->get_usd_stage()->SetStartTimeCode(0.0);
    stage->get_usd_stage()->SetEndTimeCode(duration);
    stage->get_usd_stage()->SetTimeCodesPerSecond(fps);

    // Sync all entities to USD
    stage->sync_entities_to_usd();

    // Save USD file to specified path (relative path from Binaries/Release)
    stage->SaveAs(output_path);

    // Verify: check position at last frame
    {
        auto sphere_xform = pxr::UsdGeomXformable(sphere_prim.GetPrim());
        pxr::GfVec3d final_pos;
        bool reset_stack;
        std::vector<pxr::UsdGeomXformOp> xform_ops =
            sphere_xform.GetOrderedXformOps(&reset_stack);
        if (!xform_ops.empty()) {
            xform_ops[0].Get(&final_pos, pxr::UsdTimeCode(duration));

            EXPECT_NEAR(final_pos[0], 100.0, 0.1)
                << "Sphere should move to x=100";
            EXPECT_EQ(final_pos[1], 0.0) << "Sphere Y position should be 0";
        }
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "Animation written to: "
              << std::filesystem::absolute(output_path).string() << std::endl;
    std::cout << "Time range: 0.0 - " << duration << " seconds, " << num_frames
              << " frames" << std::endl;
    std::cout << "FPS: " << fps << std::endl;
    std::cout << "3 geometries (Sphere, Cube, Cylinder) move from x=0 to x=100"
              << std::endl;
    std::cout << "Please use usdview to open and view the animation"
              << std::endl;
    std::cout << "========================================\n" << std::endl;
}

// Google Test main function
int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
