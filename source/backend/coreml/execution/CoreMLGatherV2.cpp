//
// Created by Amit Sheokand on 05/10/2024.
//

#include <MNN_generated.h>

#include "CoreMLGatherV2.hpp"

namespace MNN {
CoreMLGatherV2::CoreMLGatherV2(Backend *b, const Op *op,
                               const std::vector<Tensor *> &inputs,
                               const std::vector<Tensor *> &outputs)
    : CoreMLCommonExecution(b, op) {
  initLayer();
}

ErrorCode CoreMLGatherV2::onResize(const std::vector<Tensor *> &inputs,
                                   const std::vector<Tensor *> &outputs) {
  MNN_PRINT("CoreMLGatherV2::onResize\n");

  mLayer_->layer_case =
      CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_GATHER_ND;
  mLayer_->gathernd =
      mCoreMLBackend->create<CoreML__Specification__GatherNDLayerParams>();
  core_ml__specification__gather_ndlayer_params__init(mLayer_->gathernd);
  setLayerInputsAndOutputs(mLayer_, {mCoreMLBackend->getTensorName(inputs[0]), mCoreMLBackend->getTensorName(inputs[1])},
                           {mCoreMLBackend->getTensorName(outputs[0])});
    mCoreMLBackend->setIO
  mCoreMLBackend->addLayer(mLayer_);

  MNN_PRINT("CoreMLGatherV2::onResize complete\n");
  return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLGatherV2, OpType_GatherV2);

} // namespace MNN
