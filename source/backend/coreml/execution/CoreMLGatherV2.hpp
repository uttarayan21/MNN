//
// Created by Amit Sheokand on 05/10/2024.
//

#ifndef CoreMLGatherV2_hpp
#define CoreMLGatherV2_hpp

#include "CoreMLBackend.hpp"
#include "CoreMLCommonExecution.hpp"

namespace MNN {
class CoreMLGatherV2 : public CoreMLCommonExecution {
public:
  CoreMLGatherV2(Backend *b, const Op *op, const std::vector<Tensor *> &inputs,
                 const std::vector<Tensor *> &outputs);
  virtual ~CoreMLGatherV2() = default;
  virtual ErrorCode onResize(const std::vector<Tensor *> &inputs,
                             const std::vector<Tensor *> &outputs) override;
};

} // namespace MNN

#endif // CoreMLGatherV2_hpp
