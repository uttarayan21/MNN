//
// Created by Amit Sheokand on 06/10/2024.
//

// CoreMLWhile.cpp
#include "CoreMLWhile.hpp"

namespace MNN {
CoreMLWhile::CoreMLWhile(Backend *b, const Op *op,
                         const std::vector<Tensor *> &inputs,
                         const std::vector<Tensor *> &outputs)
    : CoreMLCommonExecution(b, op) {
  initLayer();
}

ErrorCode CoreMLWhile::onResize(const std::vector<Tensor *> &inputs,
                                const std::vector<Tensor *> &outputs) {
  MNN_PRINT("CoreMLWhile::onResize\n");

  MNN_ASSERT(inputs.size() == 1 && outputs.size() == 1);

  auto whileLoopParam = mOp->main_as_LoopParam();
  mLayer_->layer_case =
      CORE_ML__SPECIFICATION__NEURAL_NETWORK_LAYER__LAYER_LOOP;
  mLayer_->loop =
      mCoreMLBackend->create<CoreML__Specification__LoopLayerParams>();
  core_ml__specification__loop_layer_params__init(mLayer_->loop);

  if (whileLoopParam->loopNumber() > 0) {
    mLayer_->loop->maxloopiterations = whileLoopParam->loopNumber();
  }

  if (whileLoopParam->tensorNumber() > 0) {
    if (inputs[whileLoopParam->tensorNumber()] != nullptr) {
      const char *conditionVarName =
          mCoreMLBackend->getTensorName(inputs[whileLoopParam->tensorNumber()])
              .c_str();
      mLayer_->loop->conditionvar =
          mCoreMLBackend->create<char>(strlen(conditionVarName) + 1);
      strcpy(mLayer_->loop->conditionvar, conditionVarName);
    }
  }

  mLayer_->loop->conditionnetwork =
      mCoreMLBackend->create<CoreML__Specification__NeuralNetwork>();
  core_ml__specification__neural_network__init(mLayer_->loop->conditionnetwork);
  mLayer_->loop->bodynetwork =
      mCoreMLBackend->create<CoreML__Specification__NeuralNetwork>();
  core_ml__specification__neural_network__init(mLayer_->loop->bodynetwork);

  setLayerInputsAndOutputs(mLayer_, {mCoreMLBackend->getTensorName(inputs[0])},
                           {mCoreMLBackend->getTensorName(outputs[0])});

  mCoreMLBackend->addLayer(mLayer_);
  return NO_ERROR;
}

REGISTER_COREML_OP_CREATOR(CoreMLWhile, OpType_While);

} // namespace MNN
