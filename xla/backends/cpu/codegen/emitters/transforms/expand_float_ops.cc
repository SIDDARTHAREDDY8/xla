/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <cassert>
#include <cstdint>
#include <memory>  // IWYU pragma: keep
#include <utility>

#include "llvm/ADT/APFloat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/IR/ArithAttributes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"  // IWYU pragma: keep
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"  // IWYU pragma: keep
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "xla/codegen/intrinsic/cpp/cpp_gen_intrinsics.h"

namespace xla::cpu {

#define GEN_PASS_DECL_EXPANDFLOATOPSPASS
#define GEN_PASS_DEF_EXPANDFLOATOPSPASS
#include "xla/backends/cpu/codegen/emitters/transforms/passes.h.inc"

namespace {

namespace ma = ::mlir::arith;

// Returns `element_type`, or a vector of `element_type` matching `like_type`'s
// shape if `like_type` is a `mlir::VectorType`.
mlir::Type GetSameShapeType(mlir::Type like_type, mlir::Type element_type) {
  if (auto vector_type = mlir::dyn_cast<mlir::VectorType>(like_type)) {
    return vector_type.clone(element_type);
  }
  return element_type;
}

// Get a constant value, if the type is a vector, splat the value to the vector
// type.
mlir::Value GetConst(mlir::ImplicitLocOpBuilder& b, mlir::Type type,
                     mlir::TypedAttr value) {
  if (auto vector_type = mlir::dyn_cast<mlir::VectorType>(type)) {
    value =
        mlir::SplatElementsAttr::get(mlir::cast<mlir::ShapedType>(type), value);
  }
  return mlir::arith::ConstantOp::create(b, type, value);
}

mlir::Value GetI32Const(mlir::ImplicitLocOpBuilder& b, mlir::Type i32_type,
                        uint32_t value) {
  return GetConst(b, i32_type,
                  b.getI32IntegerAttr(static_cast<int32_t>(value)));
}

bool IsSupportedByteFloat(mlir::Type type) {
  return mlir::isa<mlir::Float8E5M2Type, mlir::Float8E4M3FNType,
                   mlir::Float8E4M3Type, mlir::Float8E3M4Type,
                   mlir::Float8E4M3FNUZType, mlir::Float8E5M2FNUZType,
                   mlir::Float8E4M3B11FNUZType, mlir::Float8E8M0FNUType>(type);
}

bool IsFNUZ(mlir::FloatType ty) {
  return mlir::isa<mlir::Float8E4M3B11FNUZType, mlir::Float8E4M3FNUZType,
                   mlir::Float8E5M2FNUZType>(ty);
}

mlir::Value EmitBF16ToF32(mlir::Type dst_ty, mlir::Value in,
                          mlir::ImplicitLocOpBuilder& b) {
  mlir::Type i16_type = GetSameShapeType(in.getType(), b.getI16Type());
  mlir::Type i32_type = GetSameShapeType(in.getType(), b.getI32Type());

  mlir::Value i16 = ma::BitcastOp::create(b, i16_type, in);
  mlir::Value i32 = ma::ExtUIOp::create(b, i32_type, i16);

  mlir::TypedAttr shift_value = b.getI32IntegerAttr(16);
  mlir::Value shift_const = GetConst(b, i32_type, shift_value);

  mlir::Value i32_shl = mlir::arith::ShLIOp::create(b, i32, shift_const);
  return ma::BitcastOp::create(b, dst_ty, i32_shl);
}

// Branchless integer bit conversion from an 8-bit float (`f8e4m3fn`, `f8e5m2`,
// `f8e4m3`, `f8e3m4`, `f8e4m3fnuz`, `f8e5m2fnuz`, `f8e4m3b11fnuz`,
// `f8e8m0fnu`) to `f32`, supporting both scalar and vector types.
mlir::Value EmitByteFloatToF32(mlir::Type dst_ty, mlir::Value in,
                               mlir::ImplicitLocOpBuilder& b) {
  auto src_elem =
      mlir::cast<mlir::FloatType>(mlir::getElementTypeOrSelf(in.getType()));
  const llvm::fltSemantics& sem = src_elem.getFloatSemantics();

  mlir::Type i8_type = GetSameShapeType(in.getType(), b.getI8Type());
  mlir::Type i32_type = GetSameShapeType(in.getType(), b.getI32Type());
  auto c32 = [&](uint32_t v) { return GetI32Const(b, i32_type, v); };

  mlir::Value u =
      ma::ExtUIOp::create(b, i32_type, ma::BitcastOp::create(b, i8_type, in));

  if (mlir::isa<mlir::Float8E8M0FNUType>(src_elem)) {
    // 0x00 -> 2^-127 (0x00400000), 0x01..0xFE -> u << 23, 0xFF -> qNaN.
    mlir::Value res = ma::ShLIOp::create(b, u, c32(23));
    mlir::Value is_min =
        ma::CmpIOp::create(b, ma::CmpIPredicate::eq, u, c32(0));
    res = ma::SelectOp::create(b, is_min, c32(0x00400000u), res);
    mlir::Value is_nan =
        ma::CmpIOp::create(b, ma::CmpIPredicate::eq, u, c32(0xFFu));
    res = ma::SelectOp::create(b, is_nan, c32(0x7FC00000u), res);
    return ma::BitcastOp::create(b, dst_ty, res);
  }

  const int mant_bits = llvm::APFloat::semanticsPrecision(sem) - 1;
  const int bias = 1 - llvm::APFloat::semanticsMinExponent(sem);
  const int mant_shift = 23 - mant_bits;
  const uint32_t bias_diff = static_cast<uint32_t>(127 - bias) << 23;
  const bool is_fnuz = IsFNUZ(src_elem);
  const bool has_inf = !is_fnuz && !mlir::isa<mlir::Float8E4M3FNType>(src_elem);

  mlir::Value abs_u = ma::AndIOp::create(b, u, c32(0x7Fu));
  mlir::Value sign =
      ma::ShLIOp::create(b, ma::AndIOp::create(b, u, c32(0x80u)), c32(24));

  // Normals: re-bias exponent and shift mantissa into position.
  mlir::Value norm = ma::AddIOp::create(
      b, ma::ShLIOp::create(b, abs_u, c32(mant_shift)), c32(bias_diff));

  // Subnormals and zero: abs_u in [0, 2^mant_bits - 1] multiplied by the exact
  // power-of-two subnormal quantum 2^(1 - bias - mant_bits) produces exact f32
  // values (including +0.0) in two vector instructions with no clz or spills.
  mlir::Type f32_type = GetSameShapeType(in.getType(), b.getF32Type());
  const uint32_t sub_scale_bits = static_cast<uint32_t>(128 - bias - mant_bits)
                                  << 23;
  mlir::Value sub_scale =
      ma::BitcastOp::create(b, f32_type, c32(sub_scale_bits));
  mlir::Value sub_f32 = ma::MulFOp::create(
      b, ma::UIToFPOp::create(b, f32_type, abs_u), sub_scale);
  mlir::Value sub = ma::BitcastOp::create(b, i32_type, sub_f32);

  mlir::Value is_normal = ma::CmpIOp::create(b, ma::CmpIPredicate::uge, abs_u,
                                             c32(1u << mant_bits));
  mlir::Value val = ma::SelectOp::create(b, is_normal, norm, sub);

  if (has_inf) {
    const uint32_t inf_abs =
        llvm::APFloat::getInf(sem).bitcastToAPInt().getZExtValue();
    mlir::Value is_inf =
        ma::CmpIOp::create(b, ma::CmpIPredicate::eq, abs_u, c32(inf_abs));
    val = ma::SelectOp::create(b, is_inf, c32(0x7F800000u), val);
    mlir::Value is_nan =
        ma::CmpIOp::create(b, ma::CmpIPredicate::ugt, abs_u, c32(inf_abs));
    val = ma::SelectOp::create(b, is_nan, c32(0x7FC00000u), val);
    return ma::BitcastOp::create(b, dst_ty, ma::OrIOp::create(b, val, sign));
  }

  if (!is_fnuz) {
    mlir::Value is_nan =
        ma::CmpIOp::create(b, ma::CmpIPredicate::eq, abs_u, c32(0x7Fu));
    val = ma::SelectOp::create(b, is_nan, c32(0x7FC00000u), val);
    return ma::BitcastOp::create(b, dst_ty, ma::OrIOp::create(b, val, sign));
  }

  // FNUZ: 0x80 is NaN (with sign cleared in f32), 0x00 is +0.0, 0x81..0xFF are
  // negative finite values.
  mlir::Value res = ma::OrIOp::create(b, val, sign);
  mlir::Value is_nan =
      ma::CmpIOp::create(b, ma::CmpIPredicate::eq, u, c32(0x80u));
  res = ma::SelectOp::create(b, is_nan, c32(0x7FC00000u), res);
  return ma::BitcastOp::create(b, dst_ty, res);
}

// Branchless integer bit conversion (RTNE) from `f32` to an 8-bit float,
// supporting both scalar and vector types.
mlir::Value EmitF32ToByteFloat(mlir::Type dst_ty, mlir::Value in,
                               mlir::ImplicitLocOpBuilder& b) {
  auto dst_elem =
      mlir::cast<mlir::FloatType>(mlir::getElementTypeOrSelf(dst_ty));
  const llvm::fltSemantics& sem = dst_elem.getFloatSemantics();

  mlir::Type i8_type = GetSameShapeType(in.getType(), b.getI8Type());
  mlir::Type i32_type = GetSameShapeType(in.getType(), b.getI32Type());
  auto c32 = [&](uint32_t v) { return GetI32Const(b, i32_type, v); };

  mlir::Value u = ma::BitcastOp::create(b, i32_type, in);
  mlir::Value sign_bit = ma::ShRUIOp::create(b, u, c32(31));
  mlir::Value abs_u = ma::AndIOp::create(b, u, c32(0x7FFFFFFFu));

  const bool is_fnuz = IsFNUZ(dst_elem);
  const bool is_e8m0 = mlir::isa<mlir::Float8E8M0FNUType>(dst_elem);
  const uint32_t max_finite =
      llvm::APFloat::getLargest(sem).bitcastToAPInt().getZExtValue();
  const uint32_t pos_inf =
      llvm::APFloat::getInf(sem).bitcastToAPInt().getZExtValue();
  const uint32_t pos_nan =
      llvm::APFloat::getNaN(sem).bitcastToAPInt().getZExtValue();

  if (is_e8m0) {
    mlir::Value normal_res = ma::ShRUIOp::create(
        b, ma::AddIOp::create(b, abs_u, c32(0x00400000u)), c32(23));
    mlir::Value subnorm_res = ma::SelectOp::create(
        b,
        ma::CmpIOp::create(b, ma::CmpIPredicate::uge, abs_u, c32(0x00600000u)),
        c32(1), c32(0));
    mlir::Value res = ma::SelectOp::create(
        b,
        ma::CmpIOp::create(b, ma::CmpIPredicate::uge, abs_u, c32(0x00800000u)),
        normal_res, subnorm_res);
    res = ma::SelectOp::create(
        b, ma::CmpIOp::create(b, ma::CmpIPredicate::ugt, res, c32(max_finite)),
        c32(pos_nan), res);
    mlir::Value is_neg_or_zero = ma::OrIOp::create(
        b, ma::CmpIOp::create(b, ma::CmpIPredicate::ne, sign_bit, c32(0)),
        ma::CmpIOp::create(b, ma::CmpIPredicate::eq, abs_u, c32(0)));
    res = ma::SelectOp::create(b, is_neg_or_zero, c32(pos_nan), res);
    return ma::BitcastOp::create(b, dst_ty,
                                 ma::TruncIOp::create(b, i8_type, res));
  }

  const int mant_bits = llvm::APFloat::semanticsPrecision(sem) - 1;
  const int bias = 1 - llvm::APFloat::semanticsMinExponent(sem);
  const int mant_shift = 23 - mant_bits;
  const uint32_t bias_diff = static_cast<uint32_t>(127 - bias) << 23;
  const uint32_t half_m1 = (1u << (mant_shift - 1)) - 1u;
  const uint32_t min_normal = static_cast<uint32_t>(128 - bias) << 23;
  const bool has_inf = !is_fnuz && !mlir::isa<mlir::Float8E4M3FNType>(dst_elem);
  // Subnormal RTNE magic constant: 2^(24 - bias - mant_bits), whose f32 ULP is
  // the exact subnormal quantum 2^(1 - bias - mant_bits). For any
  // 0 <= |x| < 2^(1 - bias), fadd_rn(|x|, magic) lies in [magic, 2*magic) and
  // performs a single exact RTNE rounding, yielding the subnormal code (or
  // smallest normal code 2^mant_bits on carry) in the low bits.
  const uint32_t magic_bits = static_cast<uint32_t>(151 - bias - mant_bits)
                              << 23;

  // Normal path: abs_u >= min_normal >= bias_diff, and max abs_u (0x7FFFFFFF)
  // minus bias_diff + half_m1 + 1 never overflows uint32.
  mlir::Value n_sub = ma::SubIOp::create(b, abs_u, c32(bias_diff));
  mlir::Value n_lsb = ma::AndIOp::create(
      b, ma::ShRUIOp::create(b, n_sub, c32(mant_shift)), c32(1));
  mlir::Value n_raw = ma::ShRUIOp::create(
      b,
      ma::AddIOp::create(b, ma::AddIOp::create(b, n_sub, c32(half_m1)), n_lsb),
      c32(mant_shift));
  mlir::Value n_res = (pos_inf == max_finite + 1)
                          ? static_cast<mlir::Value>(
                                ma::MinUIOp::create(b, n_raw, c32(pos_inf)))
                          : static_cast<mlir::Value>(ma::SelectOp::create(
                                b,
                                ma::CmpIOp::create(b, ma::CmpIPredicate::ugt,
                                                   n_raw, c32(max_finite)),
                                c32(pos_inf), n_raw));

  // Subnormal path: clamp to min_normal so the input to fadd is always finite
  // and non-negative (avoiding poison under fast-math nnan/ninf and blocking
  // FMA contraction with upstream ops), then do a single hardware fadd_rn.
  mlir::Type f32_type = GetSameShapeType(in.getType(), b.getF32Type());
  mlir::Value s_abs_u = ma::MinUIOp::create(b, abs_u, c32(min_normal));
  mlir::Value abs_f32 = ma::BitcastOp::create(b, f32_type, s_abs_u);
  mlir::Value magic_f32 = ma::BitcastOp::create(b, f32_type, c32(magic_bits));
  mlir::Value s_res = ma::SubIOp::create(
      b,
      ma::BitcastOp::create(b, i32_type,
                            ma::AddFOp::create(b, abs_f32, magic_f32)),
      c32(magic_bits));

  mlir::Value res = ma::SelectOp::create(
      b, ma::CmpIOp::create(b, ma::CmpIPredicate::uge, abs_u, c32(min_normal)),
      n_res, s_res);

  mlir::Value sign = ma::ShLIOp::create(b, sign_bit, c32(7));
  mlir::Value signed_res = ma::OrIOp::create(b, sign, res);
  if (is_fnuz) {
    res = ma::SelectOp::create(
        b, ma::CmpIOp::create(b, ma::CmpIPredicate::eq, res, c32(0)), c32(0),
        signed_res);
  } else {
    res = signed_res;
  }

  if (has_inf) {
    // Inf already maps through n_res -> pos_inf | sign. Only NaN needs an
    // explicit override because pos_nan != pos_inf.
    mlir::Value nan_val = ma::OrIOp::create(b, sign, c32(pos_nan));
    res = ma::SelectOp::create(
        b,
        ma::CmpIOp::create(b, ma::CmpIPredicate::ugt, abs_u, c32(0x7F800000u)),
        nan_val, res);
  }
  // For non-Inf formats (f8e4m3fn and FNUZ), pos_inf == pos_nan and
  // (pos_nan | sign) == neg_nan, so n_res already produces the exact Inf and
  // NaN encoding without an extra select.

  return ma::BitcastOp::create(b, dst_ty,
                               ma::TruncIOp::create(b, i8_type, res));
}

struct RewriteExtFPattern : public mlir::OpRewritePattern<ma::ExtFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ma::ExtFOp op, mlir::PatternRewriter& rewriter) const override {
    auto src = op.getOperand();
    auto dst_ty = op.getType();
    mlir::Type src_elem = mlir::getElementTypeOrSelf(src.getType());
    mlir::Type dst_elem = mlir::getElementTypeOrSelf(dst_ty);

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);

    if (mlir::isa<mlir::BFloat16Type>(src_elem) &&
        mlir::isa<mlir::Float32Type>(dst_elem)) {
      rewriter.replaceOp(op, EmitBF16ToF32(dst_ty, src, builder));
      return mlir::success();
    }

    if (IsSupportedByteFloat(src_elem)) {
      if (mlir::isa<mlir::Float8E5M2Type>(src_elem) &&
          mlir::isa<mlir::Float16Type>(dst_elem)) {
        mlir::Type i8_type =
            GetSameShapeType(src.getType(), builder.getI8Type());
        mlir::Type i16_type =
            GetSameShapeType(src.getType(), builder.getI16Type());
        mlir::Value i8 = ma::BitcastOp::create(builder, i8_type, src);
        mlir::Value i16 = ma::ExtUIOp::create(builder, i16_type, i8);
        mlir::Value shift =
            GetConst(builder, i16_type, builder.getI16IntegerAttr(8));
        mlir::Value shl = ma::ShLIOp::create(builder, i16, shift);
        rewriter.replaceOp(op, ma::BitcastOp::create(builder, dst_ty, shl));
        return mlir::success();
      }
      if (mlir::isa<mlir::Float32Type>(dst_elem)) {
        rewriter.replaceOp(op, EmitByteFloatToF32(dst_ty, src, builder));
        return mlir::success();
      }
      if (mlir::isa<mlir::Float64Type>(dst_elem)) {
        mlir::Type f32_type =
            GetSameShapeType(src.getType(), builder.getF32Type());
        mlir::Value f32 = EmitByteFloatToF32(f32_type, src, builder);
        rewriter.replaceOpWithNewOp<ma::ExtFOp>(op, dst_ty, f32);
        return mlir::success();
      }
      if (mlir::isa<mlir::BFloat16Type>(dst_elem)) {
        // Every value of all 8-bit float formats (finite, Inf, and canonical
        // qNaN 0x7FC00000) is exact in bf16, so the low 16 bits of the f32
        // result are always zero and truncating is a simple right shift.
        mlir::Type f32_type =
            GetSameShapeType(src.getType(), builder.getF32Type());
        mlir::Type i32_type =
            GetSameShapeType(src.getType(), builder.getI32Type());
        mlir::Type i16_type =
            GetSameShapeType(src.getType(), builder.getI16Type());
        mlir::Value f32 = EmitByteFloatToF32(f32_type, src, builder);
        mlir::Value i32 = ma::BitcastOp::create(builder, i32_type, f32);
        mlir::Value hi16 = ma::TruncIOp::create(
            builder, i16_type,
            ma::ShRUIOp::create(builder, i32,
                                GetI32Const(builder, i32_type, 16)));
        rewriter.replaceOp(op, ma::BitcastOp::create(builder, dst_ty, hi16));
        return mlir::success();
      }
      if (mlir::isa<mlir::Float16Type>(dst_elem)) {
        mlir::Type f32_type =
            GetSameShapeType(src.getType(), builder.getF32Type());
        mlir::Value f32 = EmitByteFloatToF32(f32_type, src, builder);
        rewriter.replaceOpWithNewOp<ma::TruncFOp>(op, dst_ty, f32);
        return mlir::success();
      }
    }

    return rewriter.notifyMatchFailure(op, "Not supported extf");
  }
};

struct RewriteTruncFPattern : public mlir::OpRewritePattern<ma::TruncFOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      ma::TruncFOp op, mlir::PatternRewriter& rewriter) const override {
    auto src = op.getOperand();
    auto dst_ty = op.getType();
    mlir::Type src_elem = mlir::getElementTypeOrSelf(src.getType());
    mlir::Type dst_elem = mlir::getElementTypeOrSelf(dst_ty);

    if (!IsSupportedByteFloat(dst_elem)) {
      return rewriter.notifyMatchFailure(op, "Not an 8-bit float truncf");
    }

    mlir::ImplicitLocOpBuilder builder(op.getLoc(), rewriter);
    if (mlir::isa<mlir::Float32Type>(src_elem)) {
      rewriter.replaceOp(op, EmitF32ToByteFloat(dst_ty, src, builder));
      return mlir::success();
    }
    if (mlir::isa<mlir::BFloat16Type>(src_elem)) {
      // bf16 -> f32 is lossless, so truncating via f32 avoids double rounding.
      mlir::Type f32_type =
          GetSameShapeType(src.getType(), builder.getF32Type());
      mlir::Value f32 = EmitBF16ToF32(f32_type, src, builder);
      rewriter.replaceOp(op, EmitF32ToByteFloat(dst_ty, f32, builder));
      return mlir::success();
    }
    if (mlir::isa<mlir::Float16Type>(src_elem)) {
      // f16 -> f32 is lossless, so truncating via f32 avoids double rounding.
      mlir::Type f32_type =
          GetSameShapeType(src.getType(), builder.getF32Type());
      mlir::Value f32 = ma::ExtFOp::create(builder, f32_type, src);
      rewriter.replaceOp(op, EmitF32ToByteFloat(dst_ty, f32, builder));
      return mlir::success();
    }

    return rewriter.notifyMatchFailure(op, "Unsupported source type for f8");
  }
};

// Fallback lowering for expm1(x) on platforms (such as Windows) where CppGen
// bitcode intrinsics are unavailable:
// |x| > 0.5: exp(x) - 1
// |x| <= 0.5: tanh(x / 2) * (exp(x) + 1)
class RewriteExpm1Pattern : public mlir::OpRewritePattern<mlir::math::ExpM1Op> {
 public:
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult matchAndRewrite(
      mlir::math::ExpM1Op op, mlir::PatternRewriter& rewriter) const override {
    mlir::ImplicitLocOpBuilder b(op.getLoc(), rewriter);

    mlir::Type type = op.getType();
    mlir::Type element_type = mlir::getElementTypeOrSelf(type);
    mlir::Value one = GetConst(b, type, b.getFloatAttr(element_type, 1.0));
    mlir::Value half = GetConst(b, type, b.getFloatAttr(element_type, 0.5));
    mlir::Value zero = GetConst(b, type, b.getFloatAttr(element_type, 0.0));
    mlir::Value x = op.getOperand();

    mlir::arith::FastMathFlagsAttr fastmath = op.getFastmathAttr();

    mlir::Value exp_x = mlir::math::ExpOp::create(b, x, fastmath);

    mlir::Value exp_x_minus_1 =
        mlir::arith::SubFOp::create(b, exp_x, one, fastmath);

    mlir::Value half_x = mlir::arith::MulFOp::create(b, x, half, fastmath);
    mlir::Value tanh_half_x = mlir::math::TanhOp::create(b, half_x, fastmath);
    mlir::Value exp_x_plus_1 =
        mlir::arith::AddFOp::create(b, exp_x, one, fastmath);
    mlir::Value small_result =
        mlir::arith::MulFOp::create(b, tanh_half_x, exp_x_plus_1, fastmath);

    mlir::Value abs_x = mlir::math::AbsFOp::create(b, x, fastmath);
    mlir::Value x_is_large = mlir::arith::CmpFOp::create(
        b, mlir::arith::CmpFPredicate::OGT, abs_x, half);
    mlir::Value normal_result = mlir::arith::SelectOp::create(
        b, x_is_large, exp_x_minus_1, small_result);

    // half_x can underflow resulting in zero.
    // TODO(willfroom): Do we actually need this check? tanh(0) == 0.
    mlir::Value half_x_is_zero = mlir::arith::CmpFOp::create(
        b, mlir::arith::CmpFPredicate::OEQ, half_x, zero);
    mlir::Value result =
        mlir::arith::SelectOp::create(b, half_x_is_zero, x, normal_result);

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

class ExpandFloatOpsPass
    : public impl::ExpandFloatOpsPassBase<ExpandFloatOpsPass> {
 public:
  using ExpandFloatOpsPassBase::ExpandFloatOpsPassBase;

  void runOnOperation() override {
    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<RewriteExtFPattern, RewriteTruncFPattern>(&getContext());
    if (!xla::codegen::AreCppGenIntrinsicsAvailable()) {
      patterns.add<RewriteExpm1Pattern>(&getContext());
    }

    if (mlir::failed(
            mlir::applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

}  // namespace
}  // namespace xla::cpu
