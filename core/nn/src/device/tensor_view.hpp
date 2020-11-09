#pragma once

#include "device_tensor.hpp"

namespace EigenSinn
{
	template<typename Device_, typename Scalar, Index Rank, int Layout = ColMajor>
	class DeviceTensor {

	public:

		// allocation constructors
		explicit DeviceTensor() 
			: dispatcher(Dispatcher<Device_>::create()) {
			, device(dispatcher->get_device())
		}

		explicit DeviceTensor(const DSizes<Index, Rank> dims)
			: DeviceTensor()
			, tensor_view(create_device_ptr(device, dims)) {

		}

		template<typename... IndexTypes>
		explicit DeviceTensor(Index firstDimension, Index secondDimension, IndexTypes... otherDimensions) 
			: DeviceTensor(DSizes<Index, Rank>(firstDimension, otherDimensions...)) {
		}

		explicit DeviceTensor(const Index firstDimension)
			: DeviceTensor(DSizes<Index, Rank>(firstDimension)) {
		}

		explicit DeviceTensor(const array<Index, Rank>& dims)
			: DeviceTensor(DSizes<Index, Rank>(dims)) {

		}

		///<summary>
		/// Take ownership of the pointer already on the device
		///</summary
		explicit DeviceTensor(Scalar* data, const DSizes<Index, Rank> dims) 
			: tensor_view(new TensorView<Scalar, Rank, Layout>(data, dims))	{
		}

		explicit DeviceTensor(Scalar* data, const array<Index, Rank> dims)
			: tensor_view(new TensorView<Scalar, Rank, Layout>(data, dims)) {
		}


		/// <summary>
		/// Initialize from a tensor on the host
		/// </summary>
		/// <param name="data"></param>
		/// <param name="dims"></param>
		explicit DeviceTensor(const Tensor<Scalar, Rank, Layout> data, const DSizes<Index, Rank> dims)
			: DeviceTensor(dims) {
			move_to(*tensor_view, data, device);
		}

		explicit DeviceTensor(const Tensor<Scalar, Rank, Layout> data, const array<Index, Rank> dims)
			: DeviceTensor(dims) {
			move_to(*tensor_view, data, device);
		}

		template<typename... IndexTypes>
		explicit DeviceTensor(const Tensor<Scalar, Rank, Layout> data, Index firstDimension, Index secondDimension, IndexTypes... otherDimensions)
			: DeviceTensor(data, DSizes<Index, Rank>(firstDimension, otherDimensions...)) {
		}

		explicit DeviceTensor(const Tensor<Scalar, Rank, Layout> data, const Index firstDimension)
			: DeviceTensor(data, DSizes<Index, Rank>(firstDimension)) {
		}

		// Rule of 5 definitions
		// the destructor frees the data held by the tensor_view
		~DeviceTensor() {
			if (!tensor_view) { return; }
			free(*tensor_view, device);
		}

		// copy constructor: deep copy
		DeviceTensor(const DeviceTensor& d) 
			: DeviceTensor() {
			size_t alloc_size = sizeof(scalar) * d.tensor_view->dimensions().TotalSize();
			Scalar* data = device.allocate(alloc_size);

			device.memcpy(data, d.tensor_view->data(), alloc_size);
			tensor_view = make_unique<Scalar, Rank, Layout>(data, d.tensor_view->dimensions());
		}

		// copy assignment
		DeviceTensor& operator=(const DeviceTensor& d) {
			if (&d == this) {
				return *this;
			}

			dispatcher = Dispatcher<Device_>::create();
			device = dispatcher->get_device();

			if (tensor_view) {
				free(*tensor_view, device);
			}

			size_t alloc_size = sizeof(scalar) * d.tensor_view->dimensions().TotalSize();
			Scalar* data = device.allocate(alloc_size);

			device.memcpy(data, d.tensor_view->data(), alloc_size);
			tensor_view.reset(new TensorView<Scalar, Rank, Layout>(data, d.tensor_view->dimensions());
		}

		// move constructor
		DeviceTensor(DeviceTensor&& d) noexcept
			: DeviceTensor() 
			, tensor_view(std::move(d.tensor_view)) {

		}

		// move assignment
		DeviceTensor& operator=(DeviceTensor&& d) noexcept {
			if (&d == this) {
				return this;
			}

			dispatcher = Dispatcher<Device_>::create();
			device = dispatcher->get_device();

			tensor_view.reset(std::move(d.tensor_view));
		}

	private:
		PtrTensorView<Scalar, Rank, Layout> tensor_view;
		std::unique_ptr<Dispatcher<Device_>>& dispatcher;
		Device_& device;
	};
}