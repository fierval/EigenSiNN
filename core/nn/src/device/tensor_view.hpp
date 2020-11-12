#pragma once

#include "device_tensor.hpp"

namespace EigenSinn
{
	template<typename Device_, typename Scalar, Index Rank, int Layout = ColMajor>
	class DeviceTensor {

	public:

		// allocation constructors
		explicit DeviceTensor() 
			: dispatcher(Dispatcher<Device_>::create()) 
			, device(dispatcher.get_device()) {
		}

		explicit DeviceTensor(const DSizes<Index, Rank> dims)
			: DeviceTensor() {

			tensor_view = create_device_ptr<Device_, Scalar, Rank, Layout>(dims, device);

		}

		template<typename... IndexTypes>
		explicit DeviceTensor(Index firstDimension, Index secondDimension, IndexTypes... otherDimensions) 
			: DeviceTensor(DSizes<Index, Rank>(firstDimension, secondDimension, otherDimensions...)) {
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
			: DeviceTensor() {

			tensor_view.reset(new TensorView<Scalar, Rank, Layout>(data, dims));
		}

		explicit DeviceTensor(Scalar* data, const array<Index, Rank> dims)
			: DeviceTensor() {
			tensor_view.reset(new TensorView<Scalar, Rank, Layout>(data, dims));
		}


		/// <summary>
		/// Initialize from a tensor on the host
		/// </summary>
		/// <param name="data"></param>
		/// <param name="dims"></param>
		explicit DeviceTensor(const Tensor<Scalar, Rank, Layout>& data)
			: DeviceTensor(data.dimensions()) {

			move_to<Device_, Scalar, Rank, Layout>(*tensor_view, data.data(), device);
		}

		void set_data_from_device(const Scalar* data, const DSizes<Index, Rank>& dims) {
			create_device_tensor_if_needed(dims);
			device.memcpy(tensor_view->data(), data, tensor_view->dimensions().TotalSize() * sizeof(Scalar));
		}

		void set_data_from_host(const Scalar* data, const DSizes<Index, Rank>& dims) {
			create_device_tensor_if_needed(dims);
			move_to<Device_, Scalar, Rank, Layout>(*tensor_view, data, device);
		}

		void set_data_from_host(const Tensor<Scalar, Rank, Layout>& data) {
			set_data_from_host(data.data(), data.dimensions());
		}

		/// <summary>
		/// Create tensor on the host from the device tensor.
		/// </summary>
		/// <returns></returns>
		TensorView<Scalar, Rank, Layout> to_host() {

			DefaultDevice host;

			size_t alloc_size = size() * sizeof(Scalar);

			Scalar* data = static_cast<Scalar*>(host.allocate(alloc_size));
			device.memcpyDeviceToHost(data, tensor_view->data(), alloc_size);
			TensorView<Scalar, Rank, Layout> out(data, dimensions());
			return out;
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
			tensor_view.reset(new TensorView<Scalar, Rank, Layout>(data, d.tensor_view->dimensions()));
			return *this;
		}

		// move constructor
		DeviceTensor(DeviceTensor&& d) noexcept
			: DeviceTensor() {

			tensor_view = std::move(d.tensor_view);
		}

		// move assignment
		DeviceTensor& operator=(DeviceTensor&& d) noexcept {
			if (&d == this) {
				return this;
			}

			dispatcher = Dispatcher<Device_>::create();
			device = dispatcher->get_device();

			tensor_view.reset(std::move(d.tensor_view));
			return *this;
		}

		// access
		TensorView<Scalar, Rank, Layout>& operator* () {
			return *tensor_view;
		}

		TensorView<Scalar, Rank, Layout> * operator-> () {
			return tensor_view.operator->();
		}

		explicit operator bool() const {
			return tensor_view ? true : false;
		}

		Index size() {
			return tensor_view->dimensions().TotalSize();
		}

		const DSizes<Index, Rank>& dimensions() {
			return tensor_view->dimensions();
		}

		Device_& get_device() { return device; }

	private:
		void create_device_tensor_if_needed(const DSizes<Index, Rank>& dims) {
			if (tensor_view) {
				free(*tensor_view, device);
				tensor_view = create_device_ptr<Device_, Scalar, Rank, Layout>(dims, device);
			}
		}

		PtrTensorView<Scalar, Rank, Layout> tensor_view;
		Dispatcher<Device_>& dispatcher;
		Device_& device;
	};
}