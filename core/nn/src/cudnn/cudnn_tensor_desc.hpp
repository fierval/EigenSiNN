#pragma once

#include "common.hpp"
#include "ops/opsbase.hpp"

namespace EigenSinn
{
	template <int Rank>
	class TensorDescWrapper
	{
	public:
		TensorDescWrapper(DSizes<Index, Rank> _dims) : dims(_dims) {
			tensor_desc = tensor4d(dims);
		}

		~TensorDescWrapper() {
			if (tensor_desc != nullptr) {
				checkCudnnErrors(cudnnDestroyTensorDescriptor(tensor_desc));
			}
		}

		operator cudnnTensorDescriptor_t() { 
			return tensor_desc; 
		}

		TensorDescWrapper(TensorDescWrapper&& d) {
			if (this == &d) {
				return;
			}

			tensor_desc = d.tensor_desc;
			dims = d.dims;
			d.tensor_desc = nullptr;
		}

		TensorDescWrapper(TensorDescWrapper& d) {
			if (this == &d) {
				return;
			}

			tensor_desc = tensor4d(d.dims);
			dims = d.dims;
		}

	private:
		cudnnTensorDescriptor_t tensor_desc;
		DSizes<Index, Rank> dims;
	};

	template<>
	class TensorDescWrapper<2> {

	public:
		TensorDescWrapper(DSizes<Index, 2> dims) {
			
		}
		operator cudnnTensorDescriptor_t() { return nullptr; }
	};
}