#pragma once

#include "duckdb.hpp"

namespace duckdb {

class InferaExtension : public Extension {
public:
    void Load(ExtensionLoader &loader) override;
    std::string Name() override;
};

} // namespace duckdb
