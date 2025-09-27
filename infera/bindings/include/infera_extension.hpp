#pragma once

#include "duckdb.hpp"
#include "duckdb/main/extension/extension_loader.hpp"

namespace duckdb {

class InferaExtension : public Extension {
public:
  void Load(ExtensionLoader &loader) override;
  std::string Name() override;
  std::string Version() const override;
  // std::string Description() const override;
};

} // namespace duckdb
