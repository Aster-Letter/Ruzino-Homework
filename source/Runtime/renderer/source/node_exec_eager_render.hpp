#pragma once

#include <set>

#include "hd_RUZINO/render_global_payload.hpp"
#include "node_exec_eager_render.hpp"
#include "nodes/core/node_exec.hpp"
#include "nodes/core/node_exec_eager.hpp"

RUZINO_NAMESPACE_OPEN_SCOPE

class EagerNodeTreeExecutorRender : public EagerNodeTreeExecutor {
   protected:
    bool execute_node(NodeTree* tree, Node* node) override;

    void try_storage() override;
    void remove_storage(const std::set<std::string>::value_type& key) override;

   public:
    void execute_tree(NodeTree* tree) override;
    void finalize(NodeTree* tree) override;

    virtual void reset_allocator();

   private:
    ResourceAllocator& resource_allocator()
    {
        return global_payload.cast<RenderGlobalPayload&>().resource_allocator;
    }
};

RUZINO_NAMESPACE_CLOSE_SCOPE