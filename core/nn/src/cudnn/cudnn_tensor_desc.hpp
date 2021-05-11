#pragma once

#include "common.hpp"
#include "ops/opsbase.hpp"

namespace EigenSinn
{
	class TensorDescWrapper
	{
	public:
		TensorDescWrapper(DSizes<Index, 4> dims) {
			tensor_desc = tensor4d(dims);
		}

		~TensorDescWrapper() {
			checkCudnnErrors(cudnnDestroyTensorDescriptor(tensor_desc));
		}

		operator cudnnTensorDescriptor_t() { return tensor_desc; }
	private:
		cudnnTensorDescriptor_t tensor_desc;
	};
}