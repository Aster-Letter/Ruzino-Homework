#pragma once
#include "api.h"
#include "api.h"
#include "pxr/base/gf/vec3f.h"
#include "pxr/imaging/garch/glApi.h"
#include "pxr/imaging/hd/light.h"
#include "pxr/imaging/hd/material.h"
#include "pxr/imaging/hio/image.h"
#include "pxr/pxr.h"
#include "pxr/usd/sdf/assetPath.h"
#include "internal/memory/DeviceMemoryPool.hpp"
#include "DescriptorTableManager.h"

USTC_CG_NAMESPACE_OPEN_SCOPE

using namespace pxr;
// Forward declarations
struct LightData;
// Base light class
class HD_USTC_CG_API Hd_USTC_CG_Light : public HdLight {
   public:
    explicit Hd_USTC_CG_Light(const SdfPath& id, const TfToken& lightType)
        : HdLight(id),
          _lightType(lightType)
    {
    }

    virtual ~Hd_USTC_CG_Light() = default;

    void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits)
        override;
    HdDirtyBits GetInitialDirtyBitsMask() const override;

    VtValue Get(TfToken const& token) const;

    [[nodiscard]] TfToken GetLightType() const
    {
        return _lightType;
    }

    [[nodiscard]] typename DeviceMemoryPool<LightData>::MemoryHandle GetLightBuffer() const
    {
        return light_buffer;
    }

   protected:
    // Stores the internal light type of this light.
    TfToken _lightType;
    // Cached states.
    TfHashMap<TfToken, VtValue, TfToken::HashFunctor> _params;
    // GPU buffer for light data
    typename DeviceMemoryPool<LightData>::MemoryHandle light_buffer;
};

// Simple light (directional, point light)
class HD_USTC_CG_API Hd_USTC_CG_Simple_Light : public Hd_USTC_CG_Light {
   public:
    Hd_USTC_CG_Simple_Light(const SdfPath& id, const TfToken& lightType)
        : Hd_USTC_CG_Light(id, lightType)
    {
    }

    void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;
};

// Distant light (sun light)
class HD_USTC_CG_API Hd_USTC_CG_Distant_Light : public Hd_USTC_CG_Light {
   public:
    Hd_USTC_CG_Distant_Light(const SdfPath& id, const TfToken& lightType)
        : Hd_USTC_CG_Light(id, lightType)
    {
    }

    void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

    // Accessors
    GfVec3f GetDirection() const { return _direction; }
    float GetAngle() const { return _angle; }

   private:
    GfVec3f _direction;
    float _angle = 0.53f; // Default sun angle
};

// Sphere light
class HD_USTC_CG_API Hd_USTC_CG_Sphere_Light : public Hd_USTC_CG_Light {
   public:
    Hd_USTC_CG_Sphere_Light(const SdfPath& id, const TfToken& lightType)
        : Hd_USTC_CG_Light(id, lightType)
    {
    }

    void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

    // Accessor
    float GetRadius() const { return _radius; }

   private:
    float _radius = 1.0f;
};

// Rectangle light
class HD_USTC_CG_API Hd_USTC_CG_Rect_Light : public Hd_USTC_CG_Light {
   public:
    Hd_USTC_CG_Rect_Light(const SdfPath& id, const TfToken& lightType)
        : Hd_USTC_CG_Light(id, lightType)
    {
    }

    void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

    // Accessors
    float GetWidth() const { return _width; }
    float GetHeight() const { return _height; }

   private:
    float _width = 1.0f;
    float _height = 1.0f;
};

// Disk light
class HD_USTC_CG_API Hd_USTC_CG_Disk_Light : public Hd_USTC_CG_Light {
   public:
    Hd_USTC_CG_Disk_Light(const SdfPath& id, const TfToken& lightType)
        : Hd_USTC_CG_Light(id, lightType)
    {
    }

    void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

    // Accessor
    float GetRadius() const { return _radius; }

   private:
    float _radius = 1.0f;
};

// Cylinder light
class HD_USTC_CG_API Hd_USTC_CG_Cylinder_Light : public Hd_USTC_CG_Light {
   public:
    Hd_USTC_CG_Cylinder_Light(const SdfPath& id, const TfToken& lightType)
        : Hd_USTC_CG_Light(id, lightType)
    {
    }

    void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits) override;

    // Accessors
    float GetRadius() const { return _radius; }
    float GetLength() const { return _length; }

   private:
    float _radius = 1.0f;
    float _length = 1.0f;
};

class HD_USTC_CG_API Hd_USTC_CG_Dome_Light : public Hd_USTC_CG_Light {
   public:
    struct HD_USTC_CG_API InputDescriptor {
        HioImageSharedPtr image = nullptr;

        TfToken wrapS;
        TfToken wrapT;

        TfToken uv_primvar_name;

        VtValue value;

        GLuint glTexture = 0;
        TfToken input_name;
        
        nvrhi::TextureHandle texture;
        DescriptorHandle descriptor;
    } ;

    Hd_USTC_CG_Dome_Light(const SdfPath& id, const TfToken& lightType)
        : Hd_USTC_CG_Light(id, lightType)
    {
    }


    void _PrepareDomeLight(SdfPath const& id, HdSceneDelegate* scene_delegate);
    void Sync(HdSceneDelegate* sceneDelegate, HdRenderParam* renderParam, HdDirtyBits* dirtyBits)
        override;

    void Finalize(HdRenderParam* renderParam) override;

   private:
    pxr::SdfAssetPath textureFileName;
    GfVec3f radiance;

    InputDescriptor env_texture;
};

USTC_CG_NAMESPACE_CLOSE_SCOPE
