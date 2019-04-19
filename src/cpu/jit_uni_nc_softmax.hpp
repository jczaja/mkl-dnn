/*******************************************************************************
* Copyright 2017-2019 Intel Corporation
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

#ifndef CPU_JIT_SOFTMAX_FWD_HPP
#define CPU_JIT_SOFTMAX_FWD_HPP

#include "c_types_map.hpp"

#include "cpu_softmax_pd.hpp"
#include "cpu_primitive.hpp"

#include "cpu_isa_traits.hpp"
#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
struct jit_uni_nc_softmax_fwd_ker_t;

template <cpu_isa_t isa>
struct jit_uni_nc_softmax_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_softmax_fwd_pd_t {
        using cpu_softmax_fwd_pd_t::cpu_softmax_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_uni_nc_softmax_fwd_t<isa>);

        status_t init() {
            bool ok = true
                && mayiuse(isa)
                && ndims() == 2 // PaddlePaddle is 2D
                && desc()->prop_kind == prop_kind::forward_scoring
								&& desc()->softmax_axis == 1  // Final axis 
                && utils::one_of(src_md()->data_type, data_type::f32)
                && src_md()->data_type == dst_md()->data_type
                && attr()->has_default_values()
                && memory_desc_matches_tag(*src_md(), format_tag::nc);
            if (!ok) return status::unimplemented;
						std::cout << "===> JIT SOFTMAX INITIALIZED" << std::endl;
            return jit_conf();
        }

        //jit_softmax_conf_t jsp_; // Do I need that?

    protected:
        status_t jit_conf();
    };

    jit_uni_nc_softmax_fwd_t(const pd_t *apd);
    ~jit_uni_nc_softmax_fwd_t();

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_uni_nc_softmax_fwd_ker_t<isa> *ker_;
};

}
}
}

#endif
