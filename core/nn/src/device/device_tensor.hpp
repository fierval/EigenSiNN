#pragma once

#include "device_helpers.hpp"
#include "ops/conversions.hpp"
#include "tensor_adapter.hpp"

namespace EigenSinn
{
  template<typename Scalar, Index Rank, typename Device_= ThreadPoolDevice, int Layout = ColMajor>
  class DeviceTensor {

  public:

    // allocation constructors
    explicit DeviceTensor()
      : dispatcher(Dispatcher<Device_>::create())
      , device_(dispatcher.get_device()) {

    }

    explicit DeviceTensor(const DSizes<Index, Rank> dims)
      : DeviceTensor() {

      // already created
      create_device_tensor(dims);

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

      create_device_tensor(dims, data);
    }

    /// <summary>
    /// Initialize from a tensor on the host
    /// </summary>
    /// <param name="data"></param>
    /// <param name="dims"></param>
    explicit DeviceTensor(const Tensor<Scalar, Rank, Layout>& data)
      : DeviceTensor(data.dimensions()) {


      create_device_tensor(data.dimensions());
      move_to<Scalar, Rank, Device_, Layout>(*tensor_view, data.data(), device_);
    }

    void set_from_device(const Scalar* data, const DSizes<Index, Rank>& dims) {
      create_device_tensor(dims);
      device_.memcpy(tensor_view.data(), data, tensor_view->dimensions().TotalSize() * sizeof(Scalar));
    }

    void set_from_host(const Scalar* data, const DSizes<Index, Rank>& dims) {
      create_device_tensor(dims, data);
      move_to<Scalar, Rank, Device_, Layout>(*tensor_view, data, device_);
    }

    void set_from_host(const Tensor<Scalar, Rank, Layout>& data) {
      set_from_host(data.data(), data.dimensions());
    }

    /// <summary>
    /// Create tensor on the host from the device tensor.
    /// </summary>
    /// <returns></returns>
    Tensor<Scalar, Rank, Layout> to_host() const {

      DefaultDevice host;

      size_t alloc_size = size() * sizeof(Scalar);

      Scalar* data = static_cast<Scalar*>(host.allocate(alloc_size));
      device_.memcpyDeviceToHost(data, tensor_view->data(), alloc_size);
      Tensor<Scalar, Rank, Layout> out = TensorView<Scalar, Rank, Layout>(data, dimensions());
      
      return out;
    }

    // Rule of 5 definitions
    // the destructor frees the data held by the tensor_view
    //~DeviceTensor() {
    //  dispatcher.release();
    //}

    // copy constructor: we 
    DeviceTensor(const DeviceTensor& d) : DeviceTensor() {

      bool is_null_copy = !d.tensor_view.has_value();

      if (!is_null_copy) {
        tensor_adapter = d.tensor_adapter;
        tensor_view = d.tensor_view;
      }
    }

    // copy assignment
    DeviceTensor& operator=(const DeviceTensor& d) {
      if (&d == this) {
        return *this;
      }

      tensor_adapter = d.tensor_adapter;
      tensor_view = d.tensor_view;
      return *this;
    }

    // move constructor
    DeviceTensor(DeviceTensor&& d) noexcept
      : DeviceTensor() {

      tensor_adapter = std::move(d.tensor_adapter);
      tensor_view = std::move(d.tensor_view);
    }

    // move assignment
    DeviceTensor& operator=(DeviceTensor&& d) noexcept {
      if (&d == this) {
        return *this;
      }

      tensor_adapter = std::move(d.tensor_adapter);
      tensor_view = std::move(d.tensor_view);

      return *this;
    }

    // resizing, setting values
    void resize(DSizes<Index, Rank> dims) {
      tensor_adapter.reset();
      tensor_view.reset();
      create_device_tensor(DSizes<Index, Rank>(dims));
    }

    DeviceTensor& resize(array<Index, Rank> dims) {
      resize(DSizes<Index, Rank>(dims));
      return *this;
    }

    DeviceTensor& resize(Index dim) {
      resize(DSizes<Index, 1>{dim});
      return *this;
    }

    template<typename... IndexTypes>
    DeviceTensor& resize(Index firstDim, Index secondDimension, IndexTypes... otherDimensions) {
      resize(DSizes<Index, Rank>(firstDim, secondDimension, otherDimensions...));
      return *this;
    }

    DeviceTensor& setConstant(Scalar c) {
      EigenSinn::setConstant(*tensor_view, c, device_);
      return *this;
    }

    DeviceTensor& setZero() {
      EigenSinn::setZero(*tensor_view, device_);
      return *this;
    }

    DeviceTensor& setValues(
      const typename internal::Initializer<Tensor<Scalar, Rank, Layout>, Rank>::InitList& vals) {

      Tensor<Scalar, Rank, Layout> temp(dimensions());
      TensorEvaluator<Tensor<Scalar, Rank, Layout>, DefaultDevice> eval(*static_cast<const Tensor<Scalar, Rank, Layout>*>(&temp), DefaultDevice());
      internal::initialize_tensor<Tensor<Scalar, Rank, Layout>, Rank>(eval, vals);

      EigenSinn::setValues(*tensor_view, temp, device_);
      return *this;
    }

    DeviceTensor& setRandom() {

      assert(tensor_view);
      Tensor<Scalar, Rank, Layout> temp(dimensions());
      temp.setRandom();
      move_to(*tensor_view, temp.data(), device_);

      return *this;
    }

    template <typename RandomGenerator>
    DeviceTensor& setRandom() {

      assert(tensor_view);
      Tensor<Scalar, Rank, Layout> temp(dimensions());
      temp.setRandom<RandomGenerator>();
      move_to(*tensor_view, temp.data(), device_);

      return *this;
    }

    // access
    const TensorView<Scalar, Rank, Layout>& operator* () const {
      return *tensor_view;
    }

    const TensorView<Scalar, Rank, Layout>* operator-> () const {
      return tensor_view.operator->();
    }

    explicit operator bool() const {
      return tensor_view ? true : false;
    }

    Index size() const {
      return tensor_view->dimensions().TotalSize();
    }

    const DSizes<Index, Rank>& dimensions() const {
      return tensor_view->dimensions();
    }

    Index dimension(Index i) const {
      return tensor_view->dimension(i);
    }

    auto view() { return tensor_view->device(device_); }

    Device_& get_device() { return device_; }

    std::shared_ptr<TensorAdapter<Scalar, Device_>> raw() { return tensor_adapter; }

  private:

    void DeviceTensor::create_device_tensor(const DSizes<Index, Rank> dims, Scalar * data = nullptr) {

      tensor_adapter = std::make_shared<TensorAdapter<Scalar, Device_>>(TensorAdapter<Scalar, Device_>::dims2vec(dims), device_, data);
      tensor_view = OptionalTensorView<Scalar, Rank, Layout>(TensorView<Scalar, Rank, Layout>(tensor_adapter->data(), dims));
    }

    // for tensor ops
    OptionalTensorView<Scalar, Rank, Layout> tensor_view;

    // for passing itc
    std::shared_ptr<TensorAdapter<Scalar, Device_>> tensor_adapter;
    Dispatcher<Device_>& dispatcher;
    Device_ device_;
  };
}