#pragma once

#include "common.hpp"
#include "ops/opsbase.hpp"

namespace EigenSinn
{
	template <int Rank>
	class TensorDescWrapper
	{
	public:
		TensorDescWrapper(DSizes<Index, Rank> dims) {
			tensor_desc = tensor4d(dims);
		}

		~TensorDescWrapper() {
			checkCudnnErrors(cudnnDestroyTensorDescriptor(tensor_desc));
		}

		operator cudnnTensorDescriptor_t() { 
			return tensor_desc; 
		}
	private:
		cudnnTensorDescriptor_t tensor_desc;
	};

	template<>
	class TensorDescWrapper<2> {

	public:
		TensorDescWrapper(DSizes<Index, 2> dims) {
			
		}
		operator cudnnTensorDescriptor_t() { return nullptr; }
	};
}