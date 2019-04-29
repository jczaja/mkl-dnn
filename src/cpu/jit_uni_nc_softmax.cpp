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


// Kernel to compute softmax for given data 
template <cpu_isa_t isa>
struct jit_uni_nc_softmax_fwd_ker_t: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_nc_softmax_fwd_ker_t)

    struct call_params_t {
        const char *src_nc;
        const char *dst_nc;
        size_t channel_size;
        size_t axis;
    };

    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    Xmm xreg(int idx) const { return Xmm(idx); }
    Ymm yreg(int idx) const { return Ymm(xreg(idx).getIdx()); }
    Vmm vreg(int idx) const { return Vmm(xreg(idx).getIdx()); }

    Reg64 reg_ptr_src_nc = r8;
    Reg64 reg_ptr_dst_nc = r9;

    jit_softmax_conf_t jsp;
    void (*ker_)(const call_params_t *);

    void generate();
    static status_t init_conf(jit_softmax_conf_t &jsp, const softmax_pd_t *spd);
};

template <cpu_isa_t isa>
void jit_uni_nc_softmax_fwd_ker_t<isa>::generate() {
    preamble();

#   define READ_PARAM(reg, field) \
        mov(reg, ptr[reg_param + offsetof(call_params_t, field)])
//    READ_PARAM(reg_ptr_src_nc, src_nc);
//    READ_PARAM(reg_ptr_dst_nc, dst_nc);
//    READ_PARAM(reg_kw, channel_size);
//    READ_PARAM(reg_kh, axis);

#   undef READ_PARAM

    // 1. Get Maximum
    // 2. subtract maximum
    // 3. exponential values
    // 4. sum of exponential values
    // 5. scale exponential value with computed sum

    postamble();
}

template <cpu_isa_t isa>
status_t jit_uni_nc_softmax_fwd_ker_t<isa>::init_conf(jit_softmax_conf_t &jpp,
        const softmax_pd_t *spd) {
    if (!mayiuse(isa))
        return status::unimplemented;

//	const auto &sd = *spd->desc();
	const memory_desc_wrapper src_d(spd->src_md());
	const memory_desc_wrapper dst_d(spd->dst_md());

	jpp.mb =  src_d.dims()[0];
  jpp.c  =  src_d.dims()[1];
  jpp.ndims = 2;

	//TODO(jczaja): implement
    return status::success;
}

template <cpu_isa_t isa>
jit_uni_nc_softmax_fwd_t<isa>::
jit_uni_nc_softmax_fwd_t(const pd_t *apd)
    : cpu_primitive_t(apd), ker_(nullptr)
{ ker_ = new jit_uni_nc_softmax_fwd_ker_t<isa>(/*pd()->jsp_*/); }

template <cpu_isa_t isa>
void jit_uni_nc_softmax_fwd_t<isa>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src_nc = CTX_IN_MEM(const char *, MKLDNN_ARG_SRC);
    auto dst_nc = CTX_OUT_MEM(char *, MKLDNN_ARG_DST);

    //const memory_desc_wrapper src_d(pd()->src_md());
    //const memory_desc_wrapper dst_d(pd()->dst_md());

    (void)src_nc;
    (void)dst_nc;


    std::cout << "==> JIT SOFTMAX EXECUTION!" << std::endl;    
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
    return jit_uni_nc_softmax_fwd_ker_t<isa>::init_conf(jsp_, this);
}

template <cpu_isa_t isa>
jit_uni_nc_softmax_fwd_t<isa>::~jit_uni_nc_softmax_fwd_t()
{
}

// Explicit instantiation only for supported <isa> values.
//
template struct jit_uni_nc_softmax_fwd_ker_t<avx512_core>;
template struct jit_uni_nc_softmax_fwd_t<avx512_core>;

template struct jit_uni_nc_softmax_fwd_ker_t<avx2>;
template struct jit_uni_nc_softmax_fwd_t<avx2>;
}
}
}
