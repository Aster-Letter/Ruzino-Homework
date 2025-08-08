#pragma once
#include <cstring>
#include <vector>


namespace USTC_CG {
namespace fem_bem {

    // Efficient parameter map using vector with pre-allocated capacity
    // Optimized for small parameter lists with minimal string construction
    // overhead
    template<typename T, std::size_t ReserveSize = 8, std::size_t NameBufferSize = 16>
    class ParameterMap {
       private:
        struct Entry {
            char name[NameBufferSize];
            T value;

            Entry() : value{}
            {
                name[0] = '\0';
            }
            
            Entry(const char* n, const T& v) : value(v)
            {
                std::strncpy(name, n, NameBufferSize - 1);
                name[NameBufferSize - 1] = '\0';
            }
        };

        std::vector<Entry> entries_;

       public:
        ParameterMap()
        {
            entries_.reserve(
                ReserveSize);  // Pre-allocate capacity to avoid reallocations
        }

        // Constructor from initializer list
        ParameterMap(std::initializer_list<std::pair<const char*, T>> init)
        {
            entries_.reserve(std::max(ReserveSize, init.size()));
            for (const auto& pair : init) {
                insert_or_assign(pair.first, pair.second);
            }
        }

        // Insert or update a value
        void insert_or_assign(const char* name, const T& value)
        {
            // First check if key already exists
            for (auto& entry : entries_) {
                if (std::strcmp(entry.name, name) == 0) {
                    entry.value = value;
                    return;
                }
            }

            // Insert new entry
            entries_.emplace_back(name, value);
        }

        // Access by key (const version)
        const T* find(const char* name) const
        {
            for (const auto& entry : entries_) {
                if (std::strcmp(entry.name, name) == 0) {
                    return &entry.value;
                }
            }
            return nullptr;
        }

        // Access by key (non-const version)
        T* find(const char* name)
        {
            for (auto& entry : entries_) {
                if (std::strcmp(entry.name, name) == 0) {
                    return &entry.value;
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
            return entries_.size();
        }

        // Check if empty
        bool empty() const
        {
            return entries_.empty();
        }

        // Clear all entries
        void clear()
        {
            entries_.clear();
        }

        // Iterator support for range-based loops
        class const_iterator {
           private:
            typename std::vector<Entry>::const_iterator it_;

           public:
            const_iterator(typename std::vector<Entry>::const_iterator it)
                : it_(it)
            {
            }

            const std::pair<const char*, const T&> operator*() const
            {
                return { it_->name, it_->value };
            }

            const_iterator& operator++()
            {
                ++it_;
                return *this;
            }

            bool operator!=(const const_iterator& other) const
            {
                return it_ != other.it_;
            }

            bool operator==(const const_iterator& other) const
            {
                return it_ == other.it_;
            }
        };

        const_iterator begin() const
        {
            return const_iterator(entries_.begin());
        }

        const_iterator end() const
        {
            return const_iterator(entries_.end());
        }
    };

    // Type aliases for common use cases
    using ParameterMapD = ParameterMap<double>;
    using ParameterMapF = ParameterMap<float>;

}  // namespace fem_bem
}  // namespace USTC_CG