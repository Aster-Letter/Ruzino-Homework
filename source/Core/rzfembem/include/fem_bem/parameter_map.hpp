#pragma once
#include <atomic>
#include <cassert>
#include <cstring>
#include <initializer_list>
#include <utility>

#include "api.h"

// Compiler optimization hints
#ifdef __GNUC__
#define LIKELY(x)    __builtin_expect(!!(x), 1)
#define UNLIKELY(x)  __builtin_expect(!!(x), 0)
#define RESTRICT     __restrict__
#define FORCE_INLINE __attribute__((always_inline)) inline
#elif defined(_MSC_VER)
#define LIKELY(x)    (x)
#define UNLIKELY(x)  (x)
#define RESTRICT     __restrict
#define FORCE_INLINE __forceinline
#else
#define LIKELY(x)   (x)
#define UNLIKELY(x) (x)
#define RESTRICT
#define FORCE_INLINE inline
#endif

namespace Ruzino {
namespace fem_bem {

    // Global counters for profiling
    RZFEMBEM_API extern std::size_t g_insert_or_assign_calls;
    RZFEMBEM_API extern std::size_t g_insert_unchecked_calls;
    RZFEMBEM_API extern std::size_t g_evaluate_calls;

    // Stack-based parameter map using fixed-size array
    // Ultra-lightweight optimized for small parameter lists with no dynamic
    // memory allocation
    template<
        typename T,
        std::size_t MaxSize = 16,
        std::size_t NameBufferSize = 16>
    class ParameterMap {
       private:
        // Ultra-compact entry layout optimized for cache lines
        struct alignas(alignof(T) > 8 ? alignof(T) : 8) Entry {
            char name[NameBufferSize];
            T value;

            // Default constructor - zero-initialize for fastest construction
            Entry() = default;

            FORCE_INLINE bool is_empty() const noexcept
            {
                return name[0] == '\0';
            }
            FORCE_INLINE void mark_empty() noexcept
            {
                name[0] = '\0';
            }

            // Fast initialization without constructor overhead
            FORCE_INLINE void init_empty() noexcept
            {
                name[0] = '\0';
            }
        };

        // Pre-aligned storage for maximum cache efficiency
        alignas(64) Entry entries_[MaxSize];  // 64-byte cache line alignment
        std::size_t size_;

       public:
        // Ultra-fast constructor with pre-initialized storage
        ParameterMap() noexcept : size_(0)
        {
            // Pre-initialize first entry name for hot path optimization
            entries_[0].name[0] = '\0';
        }

        // Constructor from initializer list
        ParameterMap(
            std::initializer_list<std::pair<const char*, T>> init) noexcept
            : size_(0)
        {
            entries_[0].name[0] = '\0';
            for (const auto& pair : init) {
                insert_unchecked(pair.first, pair.second);
            }
        }

        ParameterMap<T>& operator=(const ParameterMap<T>& other) noexcept
        {
            if (!(this->size() == other.size() || this->empty() ||
                  other.empty())) {
                assert(false && "Incompatible ParameterMap assignment");
            }

            if (this != &other) {
                size_ = other.size_;
                for (std::size_t i = 0; i < size_; ++i) {
                    entries_[i] = other.entries_[i];
                }
            }
            return *this;
        }

        FORCE_INLINE void insert_unchecked(
            const char* RESTRICT name,
            const T& value) noexcept
        {
            ++g_insert_unchecked_calls;
            Entry& entry = entries_[size_];
            std::memcpy(entry.name, name, NameBufferSize);
            entry.value = value;
            ++size_;
        }

        // Ultra-fast insert or update - optimized for hot path
        FORCE_INLINE void insert_or_assign(
            const char* RESTRICT name,
            const T& value) noexcept
        {
            ++g_insert_or_assign_calls;
            // Fast path for single character names (most common case)
            if (LIKELY(name[1] == '\0')) {
                const char ch = name[0];
                // Check existing entries first (likely to be small number)
                for (std::size_t i = 0; LIKELY(i < size_); ++i) {
                    Entry& entry = entries_[i];
                    if (LIKELY(entry.name[0] == ch && entry.name[1] == '\0')) {
                        entry.value = value;
                        return;
                    }
                }

                // Insert new single-char entry
                if (LIKELY(size_ < MaxSize)) {
                    Entry& entry = entries_[size_];
                    entry.name[0] = ch;
                    entry.name[1] = '\0';
                    entry.value = value;
                    ++size_;
                    return;
                }
            }
            else if (name[2] == '\0') {
                // Fast path for two-character names
                const char ch1 = name[0];
                const char ch2 = name[1];
                for (std::size_t i = 0; LIKELY(i < size_); ++i) {
                    Entry& entry = entries_[i];
                    if (LIKELY(
                            entry.name[0] == ch1 && entry.name[1] == ch2 &&
                            entry.name[2] == '\0')) {
                        entry.value = value;
                        return;
                    }
                }

                // Insert new two-char entry
                if (LIKELY(size_ < MaxSize)) {
                    Entry& entry = entries_[size_];
                    entry.name[0] = ch1;
                    entry.name[1] = ch2;
                    entry.name[2] = '\0';
                    entry.value = value;
                    ++size_;
                    return;
                }
            }

            // Multi-character path with efficient string comparison
            const std::size_t name_len = std::strlen(name);
            if (UNLIKELY(name_len >= NameBufferSize))
                return;  // Name too long

            // Check existing entries
            for (std::size_t i = 0; i < size_; ++i) {
                Entry& entry = entries_[i];
                if (entry.name[0] == name[0]) {
                    // Quick memcmp for exact match (faster than strcmp for
                    // short strings)
                    if (std::memcmp(entry.name, name, name_len + 1) == 0) {
                        entry.value = value;
                        return;
                    }
                }
            }

            // Insert new entry
            if (LIKELY(size_ < MaxSize)) {
                Entry& entry = entries_[size_];
                std::memcpy(entry.name, name, name_len + 1);
                entry.value = value;
                ++size_;
            }
        }

        // Ultra-fast find for const access
        FORCE_INLINE const T* find(const char* RESTRICT name) const noexcept
        {
            // Fast path for single character names
            if (LIKELY(name[1] == '\0')) {
                const char ch = name[0];
                for (std::size_t i = 0; LIKELY(i < size_); ++i) {
                    const Entry& entry = entries_[i];
                    if (LIKELY(entry.name[0] == ch && entry.name[1] == '\0')) {
                        return &entry.value;
                    }
                }
                return nullptr;
            }

            // Multi-character search with optimized comparison
            const std::size_t name_len = std::strlen(name);
            for (std::size_t i = 0; i < size_; ++i) {
                const Entry& entry = entries_[i];
                if (entry.name[0] == name[0]) {
                    // Quick memcmp for exact match
                    if (std::memcmp(entry.name, name, name_len + 1) == 0) {
                        return &entry.value;
                    }
                }
            }
            return nullptr;
        }

        // Ultra-fast find for non-const access
        FORCE_INLINE T* find(const char* RESTRICT name) noexcept
        {
            // Fast path for single character names
            if (LIKELY(name[1] == '\0')) {
                const char ch = name[0];
                for (std::size_t i = 0; LIKELY(i < size_); ++i) {
                    Entry& entry = entries_[i];
                    if (LIKELY(entry.name[0] == ch && entry.name[1] == '\0')) {
                        return &entry.value;
                    }
                }
                return nullptr;
            }

            // Multi-character search with optimized comparison
            const std::size_t name_len = std::strlen(name);
            for (std::size_t i = 0; i < size_; ++i) {
                Entry& entry = entries_[i];
                if (entry.name[0] == name[0]) {
                    // Quick memcmp for exact match
                    if (std::memcmp(entry.name, name, name_len + 1) == 0) {
                        return &entry.value;
                    }
                }
            }
            return nullptr;
        }

        // Check if key exists
        bool contains(const char* name) const
        {
            return find(name) != nullptr;
        }

        // Get size
        std::size_t size() const
        {
            return size_;
        }

        // Check if empty
        bool empty() const
        {
            return size_ == 0;
        }

        // Clear all entries
        void clear()
        {
            size_ = 0;
            // Simply reset size, entries will be overwritten
        }

        // Index-based access for performance (replaces iterator interface)
        const char* get_name_at(std::size_t index) const
        {
            return index < size_ ? entries_[index].name : nullptr;
        }

        const T& get_value_at(std::size_t index) const
        {
            return entries_[index].value;
        }

        T& get_value_at(std::size_t index)
        {
            return entries_[index].value;
        }
    };

    // Type aliases for common use cases
    using ParameterMapD = ParameterMap<double>;
    using ParameterMapF = ParameterMap<float>;

}  // namespace fem_bem
}  // namespace Ruzino
