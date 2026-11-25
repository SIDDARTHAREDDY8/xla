// Copyright 2026 The OpenXLA Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// ==============================================================================
// RUN: emitters_opt %s -split-input-file -xla-cpu-expand-float-ops | FileCheck %s

func.func @extend(%input: bf16) -> f32 {
  // CHECK-NOT: arith.extf
  %truncated = arith.extf %input : bf16 to f32
  func.return %truncated : f32
}

// -----

func.func @extend_vector(%input: vector<8xbf16>) -> vector<8xf32> {
  // CHECK-NOT: arith.extf
  %truncated = arith.extf %input : vector<8xbf16> to vector<8xf32>
  func.return %truncated : vector<8xf32>
}

// -----

func.func @extend_f8e4m3fn(%input: f8E4M3FN) -> f32 {
  // CHECK-NOT: arith.extf
  // CHECK-NOT: math.ctlz
  %res = arith.extf %input : f8E4M3FN to f32
  func.return %res : f32
}

// -----

func.func @extend_f8e4m3fn_vector(%input: vector<8xf8E4M3FN>) -> vector<8xf32> {
  // CHECK-NOT: arith.extf
  // CHECK-NOT: math.ctlz
  %res = arith.extf %input : vector<8xf8E4M3FN> to vector<8xf32>
  func.return %res : vector<8xf32>
}

// -----

func.func @trunc_f8e4m3fn(%input: f32) -> f8E4M3FN {
  // CHECK-NOT: arith.truncf
  %res = arith.truncf %input : f32 to f8E4M3FN
  func.return %res : f8E4M3FN
}

// -----

func.func @trunc_f8e4m3fn_vector(%input: vector<8xf32>) -> vector<8xf8E4M3FN> {
  // CHECK-NOT: arith.truncf
  %res = arith.truncf %input : vector<8xf32> to vector<8xf8E4M3FN>
  func.return %res : vector<8xf8E4M3FN>
}

// -----

func.func @extend_f8e5m2_to_f16(%input: vector<8xf8E5M2>) -> vector<8xf16> {
  // CHECK-NOT: arith.extf
  // CHECK: arith.shli
  %res = arith.extf %input : vector<8xf8E5M2> to vector<8xf16>
  func.return %res : vector<8xf16>
}

// -----

func.func @trunc_bf16_to_f8e4m3fn(%input: bf16) -> f8E4M3FN {
  // CHECK-NOT: arith.truncf
  // CHECK-NOT: arith.extf
  %res = arith.truncf %input : bf16 to f8E4M3FN
  func.return %res : f8E4M3FN
}

// -----

func.func @extend_f8e8m0fnu(%input: f8E8M0FNU) -> f32 {
  // CHECK-NOT: arith.extf
  %res = arith.extf %input : f8E8M0FNU to f32
  func.return %res : f32
}

// -----

func.func @trunc_f8e8m0fnu(%input: f32) -> f8E8M0FNU {
  // CHECK-NOT: arith.truncf
  %res = arith.truncf %input : f32 to f8E8M0FNU
  func.return %res : f8E8M0FNU
}

// -----

func.func @extend_f8e5m2_vector(%input: vector<8xf8E5M2>) -> vector<8xf32> {
  // CHECK-NOT: arith.extf
  // CHECK-NOT: math.ctlz
  %res = arith.extf %input : vector<8xf8E5M2> to vector<8xf32>
  func.return %res : vector<8xf32>
}

// -----

func.func @trunc_f8e5m2_vector(%input: vector<8xf32>) -> vector<8xf8E5M2> {
  // CHECK-NOT: arith.truncf
  %res = arith.truncf %input : vector<8xf32> to vector<8xf8E5M2>
  func.return %res : vector<8xf8E5M2>
}

// -----

func.func @extend_f8e4m3fn_to_bf16(%input: f8E4M3FN) -> bf16 {
  // CHECK-NOT: arith.extf
  // CHECK-NOT: arith.truncf
  %res = arith.extf %input : f8E4M3FN to bf16
  func.return %res : bf16
}

// -----

func.func @extend_f8e4m3fn_to_f64(%input: f8E4M3FN) -> f64 {
  // CHECK: arith.extf %{{.*}} : f32 to f64
  %res = arith.extf %input : f8E4M3FN to f64
  func.return %res : f64
}

// -----

func.func @trunc_f16_to_f8e4m3fn(%input: f16) -> f8E4M3FN {
  // CHECK-NOT: arith.truncf
  // CHECK: arith.extf %{{.*}} : f16 to f32
  %res = arith.truncf %input : f16 to f8E4M3FN
  func.return %res : f8E4M3FN
}

// -----

func.func @extend_and_trunc_other_fp8(
    %a: f8E4M3, %b: f8E3M4, %c: f8E4M3FNUZ, %d: f8E5M2FNUZ,
    %e: f8E4M3B11FNUZ) -> (f8E4M3, f8E3M4, f8E4M3FNUZ, f8E5M2FNUZ, f8E4M3B11FNUZ) {
  // CHECK-NOT: arith.extf
  // CHECK-NOT: arith.truncf
  %af = arith.extf %a : f8E4M3 to f32
  %bf = arith.extf %b : f8E3M4 to f32
  %cf = arith.extf %c : f8E4M3FNUZ to f32
  %df = arith.extf %d : f8E5M2FNUZ to f32
  %ef = arith.extf %e : f8E4M3B11FNUZ to f32
  %ao = arith.truncf %af : f32 to f8E4M3
  %bo = arith.truncf %bf : f32 to f8E3M4
  %co = arith.truncf %cf : f32 to f8E4M3FNUZ
  %do = arith.truncf %df : f32 to f8E5M2FNUZ
  %eo = arith.truncf %ef : f32 to f8E4M3B11FNUZ
  func.return %ao, %bo, %co, %do, %eo
      : f8E4M3, f8E3M4, f8E4M3FNUZ, f8E5M2FNUZ, f8E4M3B11FNUZ
}
