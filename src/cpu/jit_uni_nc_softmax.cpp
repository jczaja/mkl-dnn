/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include "jit_uni_nc_softmax.hpp"

#include <math.h>

#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

template <cpu_isa_t isa>
jit_uni_nc_softmax_fwd_t<isa>::
jit_uni_nc_softmax_fwd_t(const pd_t *apd)
    : cpu_primitive_t(apd), ker_(nullptr)
{ /*ker_ = new jit_uni_i8i8_pooling_fwd_ker_t<isa>(pd()->jpp_);*/ }

template <cpu_isa_t isa>
void jit_uni_nc_softmax_fwd_t<isa>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src_nc = CTX_IN_MEM(const char *, MKLDNN_ARG_SRC);
    auto dst_nc = CTX_OUT_MEM(char *, MKLDNN_ARG_DST);

    //const memory_desc_wrapper src_d(pd()->src_md());
    //const memory_desc_wrapper dst_d(pd()->dst_md());

    (void)src_nc;
    (void)dst_nc;
    
/*
    const auto &jpp = pd()->jpp_;

    parallel_nd(jpp.mb, jpp.oh, jpp.ow,
            [&](int n, int oh, int ow) {
        const int ih = nstl::max(oh*jpp.stride_h - jpp.t_pad, 0);
        const int iw = nstl::max(ow*jpp.stride_w - jpp.l_pad, 0);

        const int kh_start = nstl::max(0, jpp.t_pad - oh * jpp.stride_h);
        const int kh_end = nstl::min(jpp.kh,
                jpp.ih + jpp.t_pad - oh * jpp.stride_h);
        const int kw_start = nstl::max(0, jpp.l_pad - ow * jpp.stride_w);
        const int kw_end = nstl::min(jpp.kw,
                jpp.iw + jpp.l_pad - ow * jpp.stride_w);

        auto p = typename jit_uni_i8i8_pooling_fwd_ker_t<isa>::call_params_t();
        p.src_i8 = &src_i8[
            src_d.blk_off(n, 0, ih, iw) * src_d.data_type_size()];
        p.dst_i8 = &dst_i8[
            dst_d.blk_off(n, 0, oh, ow) * dst_d.data_type_size()];
        p.kw_range = (size_t)(kw_end - kw_start);
        p.kh_range = (size_t)(kh_end - kh_start);
        p.idivider = 1.0f / ((jpp.alg == pooling_avg_exclude_padding) ?
            p.kh_range*p.kw_range : jpp.kw*jpp.kh);

        ker_->ker_(&p);
				
    });
  */
}

template <cpu_isa_t isa>
status_t jit_uni_nc_softmax_fwd_t<isa>::pd_t::jit_conf() {
//    return jit_uni_nc_softmax_fwd_ker_t<isa>::init_conf(jpp_, this);
   return status::success; //HACK
}

template <cpu_isa_t isa>
jit_uni_nc_softmax_fwd_t<isa>::~jit_uni_nc_softmax_fwd_t()
{
}

// Explicit instantiation only for supported <isa> values.
//
//template struct jit_uni_nc_softmax_fwd_ker_t<avx512_core>;
template struct jit_uni_nc_softmax_fwd_t<avx512_core>;

//template struct jit_uni_nc_softmax_fwd_ker_t<avx2>;
template struct jit_uni_nc_softmax_fwd_t<avx2>;
}
}
}
