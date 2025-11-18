#include "TreeGen/TreeGrowth.h"
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <cmath>
#include <algorithm>

namespace TreeGen {

TreeGrowth::TreeGrowth(const TreeParameters& params) 
    : params_(params), normal_dist_(0.0f, 1.0f), uniform_dist_(0.0f, 1.0f) {
    init_random();
}

void TreeGrowth::init_random() {
    rng_.seed(params_.random_seed);
}

float TreeGrowth::random_normal(float mean, float stddev) {
    return mean + stddev * normal_dist_(rng_);
}

float TreeGrowth::random_uniform(float min, float max) {
    return min + (max - min) * uniform_dist_(rng_);
}

TreeStructure TreeGrowth::initialize_tree() {
    TreeStructure tree;
    tree.current_age = 0;
    
    // Create root branch (initial trunk segment)
    auto root = std::make_shared<TreeBranch>();
    root->start_position = glm::vec3(0.0f, 0.0f, 0.0f);
    root->end_position = glm::vec3(0.0f, params_.internode_base_length, 0.0f);
    root->direction = glm::normalize(root->end_position - root->start_position);
    root->length = params_.internode_base_length;
    root->radius = params_.initial_radius;
    root->level = 0;
    root->age = 0;
    root->parent = nullptr;
    
    // Create apical bud
    auto apical_bud = std::make_shared<TreeBud>();
    apical_bud->type = BudType::Apical;
    apical_bud->state = BudState::Active;
    apical_bud->position = root->end_position;
    apical_bud->direction = root->direction;
    apical_bud->level = 0;
    apical_bud->age = 0;
    apical_bud->illumination = 1.0f;
    apical_bud->parent_branch = root.get();
    
    root->apical_bud = apical_bud;
    
    tree.root = root;
    tree.all_branches.push_back(root);
    tree.all_buds.push_back(apical_bud);
    
    return tree;
}

void TreeGrowth::grow_tree(TreeStructure& tree, int cycles) {
    for (int i = 0; i < cycles; ++i) {
        grow_one_cycle(tree);
        tree.current_age++;
    }
}

void TreeGrowth::grow_one_cycle(TreeStructure& tree) {
    // Step 1: Update bud states (death, dormancy)
    update_bud_states(tree);
    
    // Step 2: Calculate illumination
    calculate_illumination(tree);
    
    // Step 3: Calculate auxin levels for lateral buds
    calculate_auxin_levels(tree);
    
    // Step 4: Determine which buds will flush
    determine_bud_flushing(tree);
    
    // Step 5: Grow shoots from flushing buds
    // Make a copy of active buds to avoid iterator invalidation
    std::vector<std::shared_ptr<TreeBud>> flushing_buds;
    for (auto& bud : tree.all_buds) {
        if (bud->state == BudState::Active) {
            flushing_buds.push_back(bud);
        }
    }
    
    for (auto& bud : flushing_buds) {
        grow_shoot_from_bud(tree, bud);
    }
    
    // Step 6: Update structural properties
    update_branch_radii(tree);
    apply_structural_bending(tree);
    prune_branches(tree);
    
    // Step 7: Rebuild branch and bud lists
    tree.all_branches.clear();
    tree.collect_branches(tree.root);
    tree.collect_active_buds();
    tree.collect_all_leaves();
}

void TreeGrowth::update_bud_states(TreeStructure& tree) {
    for (auto& bud : tree.all_buds) {
        if (bud->state == BudState::Dead) continue;
        
        // Determine death probability
        float death_prob = (bud->type == BudType::Apical) 
            ? params_.apical_bud_death 
            : params_.lateral_bud_death;
        
        if (random_uniform() < death_prob) {
            bud->state = BudState::Dead;
        }
        
        bud->age++;
    }
}

void TreeGrowth::calculate_illumination(TreeStructure& tree) {
    // Simplified illumination model
    // In a full implementation, this would do shadow casting
    // For now, we use a simple height-based approximation
    
    for (auto& bud : tree.all_buds) {
        if (bud->state == BudState::Dead) continue;
        
        // Higher buds get more light
        float height = bud->position.y;
        float max_height = 0.0f;
        for (auto& b : tree.all_buds) {
            max_height = std::max(max_height, b->position.y);
        }
        
        // Normalize illumination between 0.3 and 1.0
        if (max_height > 0.0f) {
            bud->illumination = 0.3f + 0.7f * (height / max_height);
        } else {
            bud->illumination = 1.0f;
        }
    }
}

void TreeGrowth::calculate_auxin_levels(TreeStructure& tree) {
    // Calculate auxin concentration for lateral buds
    // Auxin is produced by apical buds and inhibits lateral bud growth
    
    for (auto& bud : tree.all_buds) {
        if (bud->type != BudType::Lateral) continue;
        if (bud->state == BudState::Dead) continue;
        
        float auxin = 0.0f;
        
        // Find all buds above this one
        for (auto& other_bud : tree.all_buds) {
            if (other_bud->state == BudState::Dead) continue;
            
            // Check if other_bud is above this bud (approximate)
            if (other_bud->position.y > bud->position.y) {
                // Calculate branch-wise distance (simplified as Euclidean)
                float distance = glm::length(other_bud->position - bud->position);
                
                // Add auxin contribution
                float age_factor = std::pow(params_.apical_dominance_age, tree.current_age);
                auxin += std::exp(-distance * params_.apical_dominance_distance) 
                       * params_.apical_dominance_base * age_factor;
            }
        }
        
        bud->auxin_level = auxin;
    }
}

void TreeGrowth::determine_bud_flushing(TreeStructure& tree) {
    for (auto& bud : tree.all_buds) {
        if (bud->state != BudState::Active) continue;
        
        float flush_prob = (bud->type == BudType::Apical)
            ? calculate_apical_flush_probability(*bud)
            : calculate_lateral_flush_probability(*bud, tree);
        
        if (random_uniform() > flush_prob) {
            bud->state = BudState::Dormant;
        }
    }
}

float TreeGrowth::calculate_apical_flush_probability(const TreeBud& bud) {
    // P(F) = I^(light_factor)
    return std::pow(bud.illumination, params_.apical_light_factor);
}

float TreeGrowth::calculate_lateral_flush_probability(const TreeBud& bud, 
                                                       const TreeStructure& tree) {
    // P(F) = I^(light_factor) * exp(-auxin)
    float light_term = std::pow(bud.illumination, params_.lateral_light_factor);
    float auxin_term = std::exp(-bud.auxin_level);
    
    return light_term * auxin_term;
}

void TreeGrowth::grow_shoot_from_bud(TreeStructure& tree, std::shared_ptr<TreeBud> bud) {
    if (bud->state != BudState::Active) return;
    
    // Calculate number of internodes to grow
    float growth_rate = calculate_growth_rate(bud->level, tree.current_age);
    int num_internodes = static_cast<int>(std::round(growth_rate));
    
    if (num_internodes < 1) return;
    
    // Calculate growth direction with tropisms
    glm::vec3 growth_dir = calculate_growth_direction(
        bud->direction, bud->position, bud->illumination);
    
    // Find parent branch properly
    std::shared_ptr<TreeBranch> parent_branch = nullptr;
    if (bud->parent_branch) {
        // Search for the shared_ptr in all_branches
        for (auto& branch : tree.all_branches) {
            if (branch.get() == bud->parent_branch) {
                parent_branch = branch;
                break;
            }
        }
    }
    
    // Create internodes
    create_internodes(tree, parent_branch, 
                      bud->position, growth_dir, 
                      num_internodes, bud->level);
    
    // Mark bud as flushed (it created a shoot)
    bud->state = BudState::Dormant;
}

float TreeGrowth::calculate_growth_rate(int branch_level, int tree_age) {
    // Apply apical control: higher level branches grow slower
    float age_factor = std::pow(params_.apical_control_age_factor, tree_age);
    float control = params_.apical_control * age_factor;
    
    if (control > 1.0f) {
        return params_.growth_rate / std::pow(control, branch_level);
    } else {
        return params_.growth_rate / std::pow(control, 
                                             std::max(0, branch_level - 1));
    }
}

float TreeGrowth::calculate_internode_length(int tree_age) {
    // Length decreases with age
    return params_.internode_base_length * 
           std::pow(params_.internode_length_age_factor, tree_age);
}

void TreeGrowth::create_internodes(TreeStructure& tree,
                                   std::shared_ptr<TreeBranch> parent,
                                   const glm::vec3& start_pos,
                                   const glm::vec3& initial_dir,
                                   int num_internodes,
                                   int branch_level) {
    glm::vec3 current_pos = start_pos;
    glm::vec3 current_dir = glm::normalize(initial_dir);
    std::shared_ptr<TreeBranch> current_parent = parent;
    
    float internode_len = calculate_internode_length(tree.current_age);
    
    for (int i = 0; i < num_internodes; ++i) {
        // Create new branch (internode)
        auto branch = std::make_shared<TreeBranch>();
        branch->start_position = current_pos;
        branch->direction = current_dir;
        branch->length = internode_len;
        branch->end_position = current_pos + current_dir * internode_len;
        branch->level = branch_level;
        branch->age = 0;
        branch->parent = current_parent.get();
        
        // Set initial small radius (will be updated by pipe model)
        branch->radius = params_.initial_radius * 0.1f;
        
        // Add to parent's children
        if (current_parent) {
            current_parent->children.push_back(branch);
        }
        
        // Create lateral buds along this internode
        create_lateral_buds(branch);
        
        // Create leaves if enabled and at appropriate level
        if (params_.generate_leaves && branch_level >= params_.min_leaf_level) {
            create_leaves(branch);
        }
        
        // Create apical bud at the end
        auto apical_bud = std::make_shared<TreeBud>();
        apical_bud->type = BudType::Apical;
        apical_bud->state = BudState::Active;
        apical_bud->position = branch->end_position;
        
        // Add some randomness to apical direction
        float angle_variance = glm::radians(params_.apical_angle_variance);
        float theta = random_normal(0.0f, angle_variance);
        float phi = random_uniform(0.0f, 2.0f * 3.14159f);
        
        glm::vec3 perturbed_dir = current_dir;
        glm::vec3 perp = get_perpendicular(current_dir);
        perturbed_dir = glm::rotate(perturbed_dir, theta, perp);
        perturbed_dir = glm::rotate(perturbed_dir, phi, current_dir);
        
        apical_bud->direction = glm::normalize(perturbed_dir);
        apical_bud->level = branch_level;
        apical_bud->age = 0;
        apical_bud->illumination = 1.0f;
        apical_bud->parent_branch = branch.get();
        
        branch->apical_bud = apical_bud;
        
        // Update for next iteration
        current_pos = branch->end_position;
        current_dir = apical_bud->direction;
        current_parent = branch;
        
        // If this is the first branch and no parent, set as root
        if (i == 0 && !parent) {
            tree.root = branch;
        }
    }
}

glm::vec3 TreeGrowth::calculate_growth_direction(const glm::vec3& bud_direction,
                                                 const glm::vec3& position,
                                                 float illumination) {
    glm::vec3 dir = bud_direction;
    
    // Apply phototropism
    if (params_.phototropism > 0.0f) {
        dir = apply_phototropism(dir, params_.phototropism);
    }
    
    // Apply gravitropism
    if (params_.gravitropism > 0.0f) {
        dir = apply_gravitropism(dir, params_.gravitropism);
    }
    
    return glm::normalize(dir);
}

glm::vec3 TreeGrowth::apply_phototropism(const glm::vec3& direction, float strength) {
    // Bend towards light direction
    glm::vec3 result = direction + params_.light_direction * strength;
    return glm::normalize(result);
}

glm::vec3 TreeGrowth::apply_gravitropism(const glm::vec3& direction, float strength) {
    // Bend away from gravity (upward)
    glm::vec3 result = direction - params_.gravity_direction * strength;
    return glm::normalize(result);
}

void TreeGrowth::create_lateral_buds(std::shared_ptr<TreeBranch> branch) {
    int num_buds = params_.num_lateral_buds;
    
    for (int i = 0; i < num_buds; ++i) {
        auto lateral_bud = std::make_shared<TreeBud>();
        lateral_bud->type = BudType::Lateral;
        lateral_bud->state = BudState::Active;
        
        // Position along the branch
        float t = (i + 1.0f) / (num_buds + 1.0f);
        lateral_bud->position = branch->start_position + 
                               (branch->end_position - branch->start_position) * t;
        
        // Calculate direction
        lateral_bud->direction = calculate_lateral_direction(
            branch->direction, i, num_buds);
        
        lateral_bud->level = branch->level + 1;
        lateral_bud->age = 0;
        lateral_bud->illumination = 1.0f;
        lateral_bud->auxin_level = 0.0f;
        lateral_bud->parent_branch = branch.get();
        
        branch->lateral_buds.push_back(lateral_bud);
    }
}

glm::vec3 TreeGrowth::calculate_lateral_direction(const glm::vec3& parent_dir,
                                                  int bud_index,
                                                  int total_buds) {
    // Calculate branching angle from parent direction
    float branch_angle = glm::radians(
        random_normal(params_.branching_angle_mean, 
                     params_.branching_angle_variance));
    
    // Calculate roll angle (phyllotaxis) - golden angle distribution
    // Each bud is rotated around the parent axis by approximately 137.5 degrees
    float base_roll = params_.roll_angle_mean;  // Default: 137.5 degrees (golden angle)
    float roll_angle = glm::radians(
        random_normal(base_roll * (bud_index + 1.0f), 
                     params_.roll_angle_variance));
    
    // Get perpendicular vector to parent direction
    glm::vec3 perp = get_perpendicular(parent_dir);
    
    // Rotate perpendicular by roll angle around parent direction
    glm::vec3 radial = glm::rotate(perp, roll_angle, parent_dir);
    
    // Rotate parent direction by branch angle towards radial direction
    glm::vec3 axis = glm::normalize(glm::cross(parent_dir, radial));
    glm::vec3 lateral_dir = glm::rotate(parent_dir, branch_angle, axis);
    
    return glm::normalize(lateral_dir);
}

void TreeGrowth::update_branch_radii(TreeStructure& tree) {
    // Pipe model: parent cross-sectional area = sum of child cross-sectional areas
    // Process branches from tips to root
    
    std::function<void(std::shared_ptr<TreeBranch>)> update_radius;
    update_radius = [&](std::shared_ptr<TreeBranch> branch) {
        if (!branch) return;
        
        // Recursively update children first
        for (auto& child : branch->children) {
            update_radius(child);
        }
        
        // Calculate this branch's radius based on children
        if (!branch->children.empty()) {
            // Sum of child cross-sectional areas
            float child_area_sum = 0.0f;
            for (auto& child : branch->children) {
                child_area_sum += child->radius * child->radius;
            }
            // Parent radius from total area: r = sqrt(sum(r_child^2))
            branch->radius = std::sqrt(child_area_sum);
        } else {
            // Terminal branch (no children) - use base radius scaled by level
            branch->radius = params_.initial_radius * 
                           std::pow(params_.thickness_ratio, branch->level);
        }
        
        // Ensure minimum radius
        branch->radius = std::max(branch->radius, params_.initial_radius * 0.05f);
    };
    
    if (tree.root) {
        update_radius(tree.root);
    }
}

void TreeGrowth::apply_structural_bending(TreeStructure& tree) {
    // Simplified structural bending
    // In full implementation, this would simulate weight-induced sagging
    // For now, we skip this for simplicity
}

void TreeGrowth::prune_branches(TreeStructure& tree) {
    // Remove branches based on illumination and height
    // Simplified: mark low branches and poorly lit branches for removal
    
    for (auto& branch : tree.all_branches) {
        // Low branch pruning
        if (branch->start_position.y < params_.low_branch_pruning_factor) {
            // Mark for removal (simplified, just reduce radius)
            branch->radius *= 0.5f;
        }
        
        // Light-based pruning
        if (branch->apical_bud && branch->apical_bud->illumination < params_.pruning_factor) {
            branch->radius *= 0.7f;
        }
    }
}

float TreeGrowth::calculate_branch_distance(const TreeBud& bud1, const TreeBud& bud2) {
    // Simplified as Euclidean distance
    // Full implementation would traverse branch hierarchy
    return glm::length(bud2.position - bud1.position);
}

glm::vec3 TreeGrowth::rotate_vector(const glm::vec3& vec, 
                                   const glm::vec3& axis, 
                                   float angle) {
    return glm::rotate(vec, angle, axis);
}

glm::vec3 TreeGrowth::get_perpendicular(const glm::vec3& vec) {
    // Find a vector perpendicular to input
    glm::vec3 arbitrary = (std::abs(vec.x) < 0.9f) 
        ? glm::vec3(1.0f, 0.0f, 0.0f) 
        : glm::vec3(0.0f, 1.0f, 0.0f);
    
    return glm::normalize(glm::cross(vec, arbitrary));
}

void TreeGrowth::create_leaves(std::shared_ptr<TreeBranch> branch) {
    if (!params_.generate_leaves) return;
    if (branch->level < params_.min_leaf_level) return;
    
    // Calculate tree's maximum depth from this branch
    int max_depth = branch->max_depth();
    
    // Check if this branch should have leaves (terminal branches only if specified)
    if (!should_have_leaves(branch, max_depth)) return;
    
    int num_leaves = params_.leaves_per_internode;
    
    // Calculate tree height for normalization
    float tree_height = 0.0f;
    std::shared_ptr<TreeBranch> current = branch;
    while (current) {
        tree_height = std::max(tree_height, current->end_position.y);
        current = (current->parent) ? std::shared_ptr<TreeBranch>(current->parent, [](TreeBranch*){}) : nullptr;
    }
    
    for (int i = 0; i < num_leaves; ++i) {
        auto leaf = std::make_shared<TreeLeaf>();
        
        // Position along the branch (avoid exact start and end)
        float t = (i + 0.5f) / num_leaves;
        t = random_normal(t, 0.1f);
        t = glm::clamp(t, 0.1f, 0.9f);
        
        leaf->position = branch->start_position + 
                        (branch->end_position - branch->start_position) * t;
        
        // Calculate leaf orientation with improved phyllotaxis
        glm::vec3 branch_dir = glm::normalize(branch->direction);
        
        // Calculate normal with phototropism
        leaf->normal = calculate_leaf_normal(branch_dir, leaf->position, i);
        
        // Calculate tangent and binormal
        calculate_leaf_orientation(leaf->normal, branch_dir, 
                                  leaf->tangent, leaf->binormal);
        
        // Leaf size decreases with branch level but increases in terminal branches
        float level_factor = std::pow(0.85f, branch->level - params_.min_leaf_level);
        
        // Terminal branches have larger leaves
        float terminal_bonus = branch->is_terminal() ? 1.3f : 1.0f;
        
        leaf->size = random_normal(params_.leaf_size_base * level_factor * terminal_bonus, 
                                   params_.leaf_size_variance);
        leaf->size = std::max(0.01f, leaf->size);
        
        // Apply aspect ratio
        leaf->length = leaf->size * params_.leaf_aspect_ratio;
        leaf->width = leaf->size;
        
        // Calculate inclination (angle from horizontal)
        float height_factor = (tree_height > 0.0f) ? (leaf->position.y / tree_height) : 0.5f;
        leaf->inclination = calculate_leaf_inclination(leaf->position, height_factor);
        
        // Add rotation variation around the normal
        leaf->rotation = random_normal(0.0f, 
                                      glm::radians(params_.leaf_rotation_variance));
        
        // Curvature
        leaf->curvature = random_normal(params_.leaf_curvature, 0.1f);
        leaf->curvature = glm::clamp(leaf->curvature, 0.0f, 1.0f);
        
        leaf->age = 0;
        leaf->parent_level = branch->level;
        leaf->parent_branch = branch.get();
        leaf->attachment_direction = branch_dir;
        
        branch->leaves.push_back(leaf);
    }
}

bool TreeGrowth::should_have_leaves(std::shared_ptr<TreeBranch> branch, int max_depth) {
    if (!params_.leaves_on_terminal_only) {
        return true;  // All branches can have leaves
    }
    
    // Only terminal branches (within certain depth from tips)
    return max_depth <= params_.leaf_terminal_levels;
}

glm::vec3 TreeGrowth::calculate_leaf_normal(const glm::vec3& branch_dir, 
                                           const glm::vec3& position,
                                           int leaf_index) {
    // Use golden angle phyllotaxis for spiral arrangement
    float phyllotaxis = glm::radians(params_.leaf_phyllotaxis_angle);
    float angle = phyllotaxis * (leaf_index + 1.0f);
    
    // Get perpendicular to branch
    glm::vec3 perp = get_perpendicular(branch_dir);
    
    // Rotate around branch direction to get radial direction
    glm::vec3 radial = glm::rotate(perp, angle, branch_dir);
    
    // Start with radial direction (pointing outward from branch)
    glm::vec3 normal = radial;
    
    // Apply phototropism - bend towards light
    if (params_.leaf_phototropism > 0.0f) {
        glm::vec3 to_light = glm::normalize(params_.light_direction);
        normal = glm::normalize(normal + to_light * params_.leaf_phototropism);
    }
    
    // Add slight upward bias to avoid pointing down (most leaves face upward)
    glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
    float dot_up = glm::dot(normal, up);
    if (dot_up < 0.2f) {
        // If pointing too far down, add more upward component
        normal = glm::normalize(normal + up * 0.3f);
    }
    
    return glm::normalize(normal);
}

void TreeGrowth::calculate_leaf_orientation(const glm::vec3& normal,
                                           const glm::vec3& branch_dir,
                                           glm::vec3& tangent,
                                           glm::vec3& binormal) {
    // Tangent should be perpendicular to normal and roughly aligned with branch direction
    // This represents the "length" direction of the leaf
    
    // Project branch direction onto plane perpendicular to normal
    glm::vec3 branch_projected = branch_dir - normal * glm::dot(branch_dir, normal);
    
    if (glm::length(branch_projected) < 0.01f) {
        // Branch is parallel to normal, use arbitrary perpendicular
        tangent = get_perpendicular(normal);
    } else {
        tangent = glm::normalize(branch_projected);
    }
    
    // Binormal is perpendicular to both (represents leaf "width" direction)
    binormal = glm::normalize(glm::cross(normal, tangent));
    
    // Ensure right-handed coordinate system
    tangent = glm::normalize(glm::cross(binormal, normal));
}

float TreeGrowth::calculate_leaf_inclination(const glm::vec3& position, float height_factor) {
    // Leaves lower in the tree tend to be more horizontal to catch light
    // Leaves higher up can be more angled
    
    float base_inclination = params_.leaf_inclination_mean;
    
    // Lower leaves are more horizontal
    float height_adjustment = -20.0f * (1.0f - height_factor);
    
    float inclination = random_normal(
        base_inclination + height_adjustment,
        params_.leaf_inclination_variance
    );
    
    // Clamp to reasonable range (10° to 80°)
    inclination = glm::clamp(inclination, 10.0f, 80.0f);
    
    return glm::radians(inclination);
}

} // namespace TreeGen
