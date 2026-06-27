#include "MaskedOpsToLLVM.h"

#include "TritonAMDGPUToLLVM/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;
namespace AMD = mlir::triton::AMD;

namespace mlir::triton {
#define GEN_PASS_DEF_TRITONAMDGPUFORMMASKEDREGIONS
#include "TritonAMDGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace {

constexpr int kMaxAggregateMaskResolutionDepth = 32;

static Value getMaskedOpMask(Operation *op) {
  if (auto load = dyn_cast<triton::amdgpu::MaskedLoadOp>(op)) {
    if (load.getMulticastMask())
      return {};
    return load.getMask();
  }
  if (auto store = dyn_cast<triton::amdgpu::MaskedStoreOp>(op))
    return store.getMask();
  return {};
}

static bool isMaskedMemoryOp(Operation *op) {
  return static_cast<bool>(getMaskedOpMask(op));
}

static bool aggregatePositionsOverlap(ArrayRef<int64_t> lhs,
                                      ArrayRef<int64_t> rhs) {
  size_t commonSize = lhs.size() < rhs.size() ? lhs.size() : rhs.size();
  for (size_t i = 0; i < commonSize; ++i) {
    if (lhs[i] != rhs[i])
      return false;
  }
  return true;
}

static Value resolveAggregateMask(Value mask) {
  auto extract = mask.getDefiningOp<LLVM::ExtractValueOp>();
  if (!extract)
    return mask;

  // Keep this intentionally narrow: only fold an extractvalue that reads a
  // value inserted at the same aggregate position.
  Value aggregate = extract.getContainer();
  ArrayRef<int64_t> position = extract.getPosition();
  for (int depth = 0; depth < kMaxAggregateMaskResolutionDepth; ++depth) {
    auto insert = aggregate.getDefiningOp<LLVM::InsertValueOp>();
    if (!insert)
      break;
    ArrayRef<int64_t> insertPosition = insert.getPosition();
    if (insertPosition == position)
      return resolveAggregateMask(insert.getValue());
    if (aggregatePositionsOverlap(insertPosition, position))
      break;
    aggregate = insert.getContainer();
  }

  return mask;
}

static bool hasUseOutside(Value value,
                          const llvm::SmallPtrSetImpl<Operation *> &ops) {
  return llvm::any_of(value.getUses(), [&](OpOperand &use) {
    return !ops.contains(use.getOwner());
  });
}

static bool hasResultUseOutside(Operation *op,
                                const llvm::SmallPtrSetImpl<Operation *> &ops) {
  for (Value result : op->getResults()) {
    if (hasUseOutside(result, ops))
      return true;
  }
  return false;
}

static bool isMovablePureOp(Operation *op) {
  return op->getNumRegions() == 0 && isMemoryEffectFree(op) &&
         isSpeculatable(op);
}

static bool appendDependency(Value value,
                             const llvm::SmallPtrSetImpl<Operation *> &interval,
                             llvm::SmallPtrSetImpl<Operation *> &moveSet,
                             SmallVectorImpl<Operation *> &worklist) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp || !interval.contains(defOp))
    return true;
  if (isMaskedMemoryOp(defOp))
    return moveSet.contains(defOp);
  if (!isMovablePureOp(defOp))
    return false;
  if (moveSet.insert(defOp).second)
    worklist.push_back(defOp);
  return true;
}

static bool
appendHoistDependency(Value value,
                      const llvm::SmallPtrSetImpl<Operation *> &interval,
                      const llvm::SmallPtrSetImpl<Operation *> &moveSet,
                      llvm::SmallPtrSetImpl<Operation *> &hoistSet,
                      SmallVectorImpl<Operation *> &worklist) {
  Operation *defOp = value.getDefiningOp();
  if (!defOp || !interval.contains(defOp))
    return true;
  if (moveSet.contains(defOp) || isMaskedMemoryOp(defOp) ||
      !isMovablePureOp(defOp))
    return false;
  if (hoistSet.insert(defOp).second)
    worklist.push_back(defOp);
  return true;
}

static bool
isEscapedExtractOfMovedLoad(Operation *op,
                            const llvm::SmallPtrSetImpl<Operation *> &moveSet) {
  if (!isa<LLVM::ExtractElementOp>(op))
    return false;
  Operation *defOp = op->getOperand(0).getDefiningOp();
  return isa_and_nonnull<triton::amdgpu::MaskedLoadOp>(defOp) &&
         moveSet.contains(defOp);
}

static bool isMaskedLoadFalseValueUse(Operation *owner, OpOperand &use) {
  auto load = dyn_cast<triton::amdgpu::MaskedLoadOp>(owner);
  return load && use.getOperandNumber() == 2;
}

static bool hasResultUseInMoveSetOutsideFalseValues(
    Operation *op, const llvm::SmallPtrSetImpl<Operation *> &moveSet) {
  for (Value result : op->getResults()) {
    for (OpOperand &use : result.getUses()) {
      Operation *owner = use.getOwner();
      if (!moveSet.contains(owner))
        continue;
      if (isMaskedLoadFalseValueUse(owner, use))
        continue;
      return true;
    }
  }
  return false;
}

struct ClusterPlan {
  SmallVector<Operation *> opsToMove;
  SmallVector<Operation *> opsToHoist;
};

static FailureOr<ClusterPlan>
computeClusterPlan(ArrayRef<Operation *> intervalOps, Value mask) {
  llvm::SmallPtrSet<Operation *, 16> intervalSet(intervalOps.begin(),
                                                 intervalOps.end());
  Value canonicalMask = resolveAggregateMask(mask);
  llvm::SmallPtrSet<Operation *, 16> moveSet;
  llvm::SmallPtrSet<Operation *, 16> hoistSet;
  SmallVector<Operation *> worklist;
  SmallVector<Operation *> hoistWorklist;

  for (Operation *op : intervalOps) {
    if (Value opMask = getMaskedOpMask(op)) {
      if (resolveAggregateMask(opMask) != canonicalMask)
        return failure();
      moveSet.insert(op);
      if (auto load = dyn_cast<triton::amdgpu::MaskedLoadOp>(op)) {
        if (!appendHoistDependency(load.getFalseVal(), intervalSet, moveSet,
                                   hoistSet, hoistWorklist))
          return failure();
        if (!appendDependency(load.getPtr(), intervalSet, moveSet, worklist))
          return failure();
        continue;
      }
      auto store = cast<triton::amdgpu::MaskedStoreOp>(op);
      if (!appendDependency(store.getPtr(), intervalSet, moveSet, worklist))
        return failure();
      if (!appendDependency(store.getValue(), intervalSet, moveSet, worklist))
        return failure();
      continue;
    }

    if (!isMovablePureOp(op))
      return failure();
  }

  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    for (Value operand : op->getOperands()) {
      if (!appendDependency(operand, intervalSet, moveSet, worklist))
        return failure();
    }
  }

  while (!hoistWorklist.empty()) {
    Operation *op = hoistWorklist.pop_back_val();
    if (moveSet.contains(op))
      return failure();
    for (Value operand : op->getOperands()) {
      if (!appendHoistDependency(operand, intervalSet, moveSet, hoistSet,
                                 hoistWorklist))
        return failure();
    }
  }

  SmallVector<Operation *> opsToMove;
  SmallVector<Operation *> opsToHoist;
  for (Operation *op : intervalOps) {
    if (moveSet.contains(op))
      opsToMove.push_back(op);
    if (hoistSet.contains(op))
      opsToHoist.push_back(op);
  }

  for (Operation *op : intervalOps) {
    if (hoistSet.contains(op))
      continue;

    if (moveSet.contains(op)) {
      if (!isMaskedMemoryOp(op) && hasResultUseOutside(op, moveSet))
        return failure();
      continue;
    }

    if (hasResultUseInMoveSetOutsideFalseValues(op, moveSet))
      return failure();

    bool hasMovedOperand = false;
    for (Value operand : op->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp || !moveSet.contains(defOp))
        continue;
      if (!isa<triton::amdgpu::MaskedLoadOp>(defOp))
        return failure();
      hasMovedOperand = true;
    }

    if (hasResultUseOutside(op, intervalSet)) {
      if (!hasMovedOperand || !isEscapedExtractOfMovedLoad(op, moveSet))
        return failure();
    }
  }

  return ClusterPlan{std::move(opsToMove), std::move(opsToHoist)};
}

static SmallVector<Operation *> getInterval(Operation *first, Operation *last) {
  SmallVector<Operation *> interval;
  for (Operation *op = first;; op = op->getNextNode()) {
    interval.push_back(op);
    if (op == last)
      break;
  }
  return interval;
}

static FailureOr<SmallVector<Operation *>> findCluster(Operation *first) {
  Value mask = getMaskedOpMask(first);
  if (!mask)
    return failure();

  unsigned maskedOpCount = 1;
  Operation *lastMaskedOp = first;
  Value canonicalMask = resolveAggregateMask(mask);
  for (Operation *op = first->getNextNode(); op != nullptr;
       op = op->getNextNode()) {
    if (op->hasTrait<OpTrait::IsTerminator>())
      break;

    if (Value opMask = getMaskedOpMask(op)) {
      if (resolveAggregateMask(opMask) != canonicalMask)
        break;
      ++maskedOpCount;
      lastMaskedOp = op;
      continue;
    }

    if (!isMovablePureOp(op))
      break;
  }

  if (maskedOpCount < 2)
    return failure();

  SmallVector<Operation *> intervalOps = getInterval(first, lastMaskedOp);
  if (failed(computeClusterPlan(intervalOps, mask)))
    return failure();

  return intervalOps;
}

static triton::amdgpu::MaskedRegionOp
createMaskedRegionOp(IRRewriter &rewriter, Location loc, Value mask,
                     ValueRange falseValues, TypeRange resultTypes) {
  OperationState state(loc, triton::amdgpu::MaskedRegionOp::getOperationName());
  state.addOperands(mask);
  state.addOperands(falseValues);
  state.addTypes(resultTypes);
  state.addRegion();
  return cast<triton::amdgpu::MaskedRegionOp>(rewriter.create(state));
}

static void replaceGroupedLoadUses(
    triton::amdgpu::MaskedLoadOp load, Value trueValue, Value outsideValue,
    const llvm::SmallPtrSetImpl<Operation *> &moveSet, IRRewriter &rewriter) {
  SmallVector<OpOperand *> uses;
  for (OpOperand &use : load.getResult().getUses())
    uses.push_back(&use);

  for (OpOperand *use : uses) {
    Operation *owner = use->getOwner();
    bool useMovedIntoRegion = moveSet.contains(owner);
    if (!useMovedIntoRegion)
      assert(outsideValue && "expected region result for outside use");
    Value replacement = useMovedIntoRegion ? trueValue : outsideValue;
    rewriter.modifyOpInPlace(owner, [&]() { use->set(replacement); });
  }
}

static void formMaskedRegion(ArrayRef<Operation *> intervalOps,
                             IRRewriter &rewriter) {
  Operation *first = intervalOps.front();
  Location loc = first->getLoc();
  Value mask = getMaskedOpMask(first);
  FailureOr<ClusterPlan> plan = computeClusterPlan(intervalOps, mask);
  assert(succeeded(plan) && "expected legal masked region cluster");
  llvm::SmallPtrSet<Operation *, 16> moveSet(plan->opsToMove.begin(),
                                             plan->opsToMove.end());

  SmallVector<triton::amdgpu::MaskedLoadOp> loads;
  SmallVector<Value> falseValues;
  SmallVector<Type> resultTypes;
  DenseMap<Operation *, unsigned> loadToResultIndex;
  for (Operation *op : plan->opsToMove) {
    auto load = dyn_cast<triton::amdgpu::MaskedLoadOp>(op);
    if (!load)
      continue;
    if (!hasUseOutside(load.getResult(), moveSet))
      continue;
    loadToResultIndex[op] = loads.size();
    loads.push_back(load);
    falseValues.push_back(load.getFalseVal());
    resultTypes.push_back(load.getResult().getType());
  }

  for (Operation *op : plan->opsToHoist)
    rewriter.moveOpBefore(op, first);

  rewriter.setInsertionPoint(first);
  auto regionOp = createMaskedRegionOp(rewriter, loc, mask, falseValues,
                                       TypeRange(resultTypes));
  Block *body = rewriter.createBlock(&regionOp.getBody());

  for (Operation *op : plan->opsToMove)
    rewriter.moveOpBefore(op, body, body->end());

  SmallVector<Value> yieldValues(loads.size());
  for (Operation &op : llvm::make_early_inc_range(*body)) {
    if (auto load = dyn_cast<triton::amdgpu::MaskedLoadOp>(&op)) {
      rewriter.setInsertionPoint(load);
      Value trueValue =
          AMD::createRegularLoadFromMaskedOp(rewriter, load.getLoc(), load);
      Value outsideValue;
      auto it = loadToResultIndex.find(load);
      if (it != loadToResultIndex.end()) {
        unsigned resultIndex = it->second;
        yieldValues[resultIndex] = trueValue;
        outsideValue = regionOp.getResult(resultIndex);
      }
      replaceGroupedLoadUses(load, trueValue, outsideValue, moveSet, rewriter);
      rewriter.eraseOp(load);
      continue;
    }

    if (auto store = dyn_cast<triton::amdgpu::MaskedStoreOp>(&op)) {
      rewriter.setInsertionPoint(store);
      AMD::createUnmaskedStoreFromMaskedOp(rewriter, store.getLoc(), store);
      rewriter.eraseOp(store);
    }
  }

  rewriter.setInsertionPointToEnd(body);
  triton::amdgpu::MaskedYieldOp::create(rewriter, loc, yieldValues);
}

static bool runOnBlock(Block &block, IRRewriter &rewriter) {
  for (Operation *parent = block.getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (isa<triton::amdgpu::MaskedRegionOp>(parent))
      return false;
  }

  SmallVector<Operation *> ops;
  for (Operation &op : block.without_terminator())
    ops.push_back(&op);

  for (Operation *op : ops) {
    if (op->getBlock() != &block || !isMaskedMemoryOp(op))
      continue;

    FailureOr<SmallVector<Operation *>> cluster = findCluster(op);
    if (failed(cluster))
      continue;

    formMaskedRegion(*cluster, rewriter);
    return true;
  }

  return false;
}

struct TritonAMDGPUFormMaskedRegionsPass
    : public triton::impl::TritonAMDGPUFormMaskedRegionsBase<
          TritonAMDGPUFormMaskedRegionsPass> {
  explicit TritonAMDGPUFormMaskedRegionsPass(StringRef gfxArch) {
    this->gfxArch = gfxArch.str();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    AMD::formMaskedRegions(module);
  }
};

} // namespace

namespace mlir::triton::AMD {

void formMaskedRegions(ModuleOp module) {
  IRRewriter rewriter(module.getContext());

  bool changed = true;
  while (changed) {
    changed = false;
    module.walk([&](Operation *op) {
      for (Region &region : op->getRegions()) {
        for (Block &block : region.getBlocks()) {
          if (runOnBlock(block, rewriter)) {
            changed = true;
            return WalkResult::interrupt();
          }
        }
      }
      return WalkResult::advance();
    });
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
createTritonAMDGPUFormMaskedRegionsPass(StringRef gfxArch) {
  return std::make_unique<TritonAMDGPUFormMaskedRegionsPass>(gfxArch);
}

} // namespace mlir::triton::AMD
