//
// Copyright 2020 Pixar
//
// Licensed under the Apache License, Version 2.0 (the "Apache License")
// with the following modification; you may not use this file except in
// compliance with the Apache License and the following modification to it:
// Section 6. Trademarks. is deleted and replaced with:
//
// 6. Trademarks. This License does not grant permission to use the trade
//    names, trademarks, service marks, or product names of the Licensor
//    and its affiliates, except as required to comply with Section 4(c) of
//    the License and to reproduce the content of the NOTICE file.
//
// You may obtain a copy of the Apache License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the Apache License with the above modification is
// distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the Apache License for the specific
// language governing permissions and limitations under the Apache License.
//
#ifndef Hd_RUZINO_FIELD_H
#define Hd_RUZINO_FIELD_H

#include <memory>
#include <string>

#include "../api.h"
#include "pxr/imaging/hd/field.h"
#include "pxr/imaging/hf/perfLog.h"
#include "pxr/pxr.h"

RUZINO_NAMESPACE_OPEN_SCOPE
class Hd_RUZINO_RenderParam;
using namespace pxr;

/// \class Hd_RUZINO_Field
///
/// A placeholder implementation of HdField for the RUZINO renderer.
/// This serves as a basic field primitive that can be extended later
/// with actual field data handling and GPU resource management.
///
class HD_RUZINO_API Hd_RUZINO_Field final : public HdField {
   public:
    HF_MALLOC_TAG_NEW("new Hd_RUZINO_Field");

    Hd_RUZINO_Field(const SdfPath& id);
    ~Hd_RUZINO_Field()
        override;  /// Return the initial dirty bits for the primitive
    HdDirtyBits GetInitialDirtyBitsMask() const override;

    /// Synchronize the field data
    void Sync(
        HdSceneDelegate* sceneDelegate,
        HdRenderParam* renderParam,
        HdDirtyBits* dirtyBits) override;

    /// Finalize and clean up resources
    void Finalize(HdRenderParam* renderParam) override;

    // Placeholder accessors for future use
    const std::string& GetFilePath() const
    {
        return _filePath;
    }
    const std::string& GetFieldName() const
    {
        return _fieldName;
    }

   private:
    // Field properties
    std::string _filePath;
    std::string _fieldName;

    // State tracking
    bool _isLoaded;

    // This class does not support copying
    Hd_RUZINO_Field(const Hd_RUZINO_Field&) = delete;
    Hd_RUZINO_Field& operator=(const Hd_RUZINO_Field&) = delete;
};

RUZINO_NAMESPACE_CLOSE_SCOPE

#endif  // Hd_RUZINO_FIELD_H
