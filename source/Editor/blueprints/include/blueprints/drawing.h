#pragma once
#include <imgui.h>

#include "blueprints/api.h"

RUZINO_NAMESPACE_OPEN_SCOPE
namespace ax {
namespace Drawing {
    enum class IconType : ImU32 {
        Flow,
        Circle,
        Square,
        Grid,
        RoundSquare,
        Diamond
    };

    void DrawIcon(
        ImDrawList* drawList,
        const ImVec2& a,
        const ImVec2& b,
        IconType type,
        bool filled,
        ImU32 color,
        ImU32 innerColor);
}  // namespace Drawing
}  // namespace ax

RUZINO_NAMESPACE_CLOSE_SCOPE
