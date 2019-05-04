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

    typedef typename utils::conditional<isa == avx512_core,
            jit_uni_eltwise_injector_f32<avx512_common>,
            jit_uni_eltwise_injector_f32<isa>>::type injector_t;

    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    Xmm xreg(int idx) const { return Xmm(idx); }
    Ymm yreg(int idx) const { return Ymm(xreg(idx).getIdx()); }
    Vmm vreg(int idx) const { return Vmm(xreg(idx).getIdx()); }

    Reg64 reg_param         = rcx; // Our "unified abi_param1"
    Reg64 reg_ptr_src_nc    = r8;
    Reg64 reg_ptr_dst_nc    = r9;
    Reg64 reg_channel_size  = r10;
    Reg64 reg_axis          = r11;

    Reg64 reg_tmp = rdx;
    Reg64 reg_tmp2 = r12;
    Reg64 reg_tmp3 = r13;
    Reg64 reg_tmp4 = r14;

    jit_softmax_conf_t jsp;
    injector_t *vsexp_injector_;
    void (*ker_)(const call_params_t *);

    void generate();
    static status_t init_conf(jit_softmax_conf_t &jsp, const softmax_pd_t *spd);

    void compute_max();
    void compute_sub();

    jit_uni_nc_softmax_fwd_ker_t(const jit_softmax_conf_t &jsp_)
           : jsp(jsp_) {
        
        vsexp_injector_ = new injector_t(this,
                alg_kind::eltwise_exp, 0.0f, 0.0f, true, rax);
        generate();
        ker_ = reinterpret_cast<decltype(ker_)>(const_cast<uint8_t*>(
                       getCode()));
    }
};

template <cpu_isa_t isa>
void jit_uni_nc_softmax_fwd_ker_t<isa>::generate() {
    preamble();

#if !defined(_WIN32)
    // Always use rcx as abi_param1 regardless OS
    mov(rcx, rdi);
#endif

#   define READ_PARAM(reg, field) \
        mov(reg, ptr[reg_param + offsetof(call_params_t, field)])
    READ_PARAM(reg_ptr_src_nc, src_nc);
    READ_PARAM(reg_ptr_dst_nc, dst_nc);
    READ_PARAM(reg_channel_size,channel_size);
    READ_PARAM(reg_axis, axis);
#   undef READ_PARAM

    // 1. Get Maximum
    compute_max();
    // 2. subtract maximum
    compute_sub();
    // 3. exponential values
    // 4. sum of exponential values
    // 5. scale exponential value with computed sum

    postamble();
}


template <cpu_isa_t isa>
void jit_uni_nc_softmax_fwd_ker_t<isa>::compute_max(){

	Label for_i_label, tail_label, seq_label, done_label;

  //TODO(jczaja): compute number of 8*floats items offline
  mov (reg_tmp,reg_channel_size);	
  shr (reg_tmp,3);  // Divide by 8 (eight floats)
  shl (reg_tmp,2);  // num of Output elements * size of float (4)
  shl (reg_tmp,5);  // Trunc to 32 bytes 

	// Compute partial maximums
  vpbroadcastd(ymm0,ptr [reg_ptr_src_nc]);
  xor_(rax,rax);				// Move offset for next 8 floating point values
  L(for_i_label);
    cmp(rax,reg_tmp);
    jz(tail_label);
    vmovaps(ymm1,ptr [reg_ptr_src_nc + rax]);  // A
		add(rax,32);				// Move offset for next 8 floating point values
		vmaxps(ymm0,ymm0,ymm1);
    jmp(for_i_label);
  L(tail_label);
    sub(rdx,reg_tmp);
    cmp(rdx,16);  
    jb(seq_label);
    vmovaps(xmm2,ptr [reg_ptr_src_nc + rax]);  // A
		add(rax,16);				// Move offset for next 4 floating point values
    sub(rdx,16);
		vperm2f128(ymm2,ymm2,ymm2,0);
		vmaxps(ymm0,ymm0,ymm2);  //partial maxes in ymm0
  L(seq_label);
	  cmp(rdx,0);
    jz(done_label);	
		vpbroadcastd(ymm2,ptr [reg_ptr_src_nc + rax]);
		vmaxps(ymm0,ymm0,ymm2);  //partial maxes in ymm0
    sub(rdx,4);
    add(rax,4);
    jmp(seq_label);
  L(done_label);
  // Get within shortlisted buffer maximum
	vperm2f128(ymm1,ymm0,ymm0,1);
  vmaxps(ymm0,ymm0,ymm1);  //partial maxes in ymm0
  vpermilps(xmm1,xmm0,0x1B);
  vmaxps(ymm0,ymm0,ymm1);  //partial maxes in ymm0
  vpermilps(xmm1,xmm0,1);
  vmaxps(ymm0,ymm0,ymm1);  //ymm0[0:31] contains global maximum

	// Maximum is stored in XMM0
}

template <cpu_isa_t isa>
void jit_uni_nc_softmax_fwd_ker_t<isa>::compute_sub(){

	Label for_i_label, tail_label, seq_label, done_label;

  //TODO(jczaja): compute number of 8*floats items offline
  mov (reg_tmp,reg_channel_size);
  shr (reg_tmp,3);  // Divide by 8 (eight floats)
  shl (reg_tmp,2);  // num of Output elements * size of float (4)
  shl (reg_tmp,5);  // Trunc to 32 bytes 

	// Compute partial maximums
  vpbroadcastd(ymm0,ptr [reg_ptr_src_nc]);
  xor_(rax,rax);				// Move offset for next 8 floating point values
  L(for_i_label);
    cmp(rax,reg_tmp);
    jz(tail_label);
    vmovaps(ymm1,ptr [reg_ptr_src_nc + rax]);
		vsubps(ymm1,ymm1,ymm0);
		vmovaps(ptr[reg_ptr_dst_nc + rax],ymm1);
		add(rax,32);				// Move offset for next 8 floating point values
    jmp(for_i_label);
  L(tail_label);
    sub(rdx,reg_tmp);
    cmp(rdx,16);  
    jb(seq_label);
    vmovaps(xmm1, ptr [reg_ptr_src_nc + rax]);  // A
    sub(rdx,16);
		vsubps(xmm1, xmm1, xmm0);
		vmovaps(ptr[reg_ptr_dst_nc + rax],xmm1);
		add(rax,16);				// Move offset for next 4 floating point values
  L(seq_label);
	  cmp(rdx,0);
    jz(done_label);	
		vpbroadcastd(ymm1, ptr [reg_ptr_src_nc + rax]);
    sub(rdx,4);
		vsubps(ymm1,ymm1,ymm0);
    vmovss(ptr [reg_ptr_dst_nc + rax], xmm1);
    add(rax,4);
    jmp(seq_label);
  L(done_label);
}

template <cpu_isa_t isa>
status_t jit_uni_nc_softmax_fwd_ker_t<isa>::init_conf(jit_softmax_conf_t &jsp,
        const softmax_pd_t *spd) {
    if (!mayiuse(isa))
        return status::unimplemented;

	const auto &sd = *spd->desc();
	const memory_desc_wrapper src_d(spd->src_md());
	const memory_desc_wrapper dst_d(spd->dst_md());

	jsp.mb =  src_d.dims()[0];
  jsp.c  =  src_d.dims()[1];
  jsp.ndims = 2;
  jsp.axis = sd.softmax_axis;

  return status::success;
}

template <cpu_isa_t isa>
jit_uni_nc_softmax_fwd_t<isa>::
jit_uni_nc_softmax_fwd_t(const pd_t *apd)
    : cpu_primitive_t(apd), ker_(nullptr)
{ ker_ = new jit_uni_nc_softmax_fwd_ker_t<isa>(pd()->jsp_); }

template <cpu_isa_t isa>
void jit_uni_nc_softmax_fwd_t<isa>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src_nc = CTX_IN_MEM(const char *, MKLDNN_ARG_SRC);
    auto dst_nc = CTX_OUT_MEM(char *, MKLDNN_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    (void)src_nc;
    (void)dst_nc;


    std::cout << "==> JIT SOFTMAX EXECUTION!" << std::endl;
    const auto &jsp = pd()->jsp_;

/*
    parallel_nd(jpp.mb, jpp.oh, jpp.ow,
            [&](int n, int oh, int ow) {
        const int kh_start = nstl::max(0, jpp.t_pad - oh * jpp.stride_h);
        const int kh_end = nstl::min(jpp.kh,
                jpp.ih + jpp.t_pad - oh * jpp.stride_h);
        const int kw_start = nstl::max(0, jpp.l_pad - ow * jpp.stride_w);
        const int kw_end = nstl::min(jpp.kw,
                jpp.iw + jpp.l_pad - ow * jpp.stride_w);

  */
        auto p = typename jit_uni_nc_softmax_fwd_ker_t<isa>::call_params_t();
        p.src_nc = &src_nc[
            src_d.blk_off(jsp.mb, jsp.c) * src_d.data_type_size()];
        p.dst_nc = &dst_nc[
            dst_d.blk_off(jsp.mb, jsp.c) * dst_d.data_type_size()];

        p.channel_size = jsp.c;
        p.axis = jsp.axis;

        ker_->ker_(&p);
				
    //}//);
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
