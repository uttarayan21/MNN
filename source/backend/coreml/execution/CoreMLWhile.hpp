//
// Created by Amit Sheokand on 06/10/2024.
//

#ifndef COREML_WHILE_HPP
#define COREML_WHILE_HPP

#include "CoreMLBackend.hpp"
#include "CoreMLCommonExecution.hpp"

namespace MNN {
class CoreMLWhile : public CoreMLCommonExecution {
public:
  CoreMLWhile(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
              const std::vector<Tensor *> &outputs);
  virtual ~CoreMLWhile() = default;
  virtual ErrorCode onResize(const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs) override;

private:
  const Op *mOp;
};

} // namespace MNN

#endif // CoreMLGatherV2_hpp
