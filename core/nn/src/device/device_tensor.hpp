#pragma once

#include "device_helpers.hpp"
#include "ops/conversions.hpp"
#include "tensor_adapter.hpp"

namespace EigenSinn
{
  template<typename Scalar, typename Device_>
  using PtrTensorAdapter = std::shared_ptr<TensorAdapter<Scalar, Device_>>;

  template<typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = ColMajor>
  class DeviceTensor {

  public:

    // allocation constructors
    explicit DeviceTensor() {
      create_device_tensor();
    }

    explicit DeviceTensor(const DSizes<Index, Rank> dims) {

      // already created
      create_device_tensor(dims, nullptr);

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

      create_device_tensor(data.dimensions(), nullptr);
      move_to<Scalar, Rank, Device_, Layout>(*tensor_view, data.data(), device());
    }

    explicit DeviceTensor(const TensorView<Scalar, Rank, Layout>& tv)
      : DeviceTensor(tv.dimensions()) {


      create_device_tensor(tv.dimensions(), tv.data());
    }

    DeviceTensor(const DeviceTensor& d) {
      create_device_tensor(d.tensor_view->dimensions(), d.tensor_adapter);
    }

    DeviceTensor(const DeviceTensor&& d) {
      create_device_tensor(d.dimensions(), d.tensor_adapter);
    }

    DeviceTensor& operator=(const DeviceTensor& d) {
      if (this == &d) { return *this; }

      create_device_tensor(d.dimensions(), d.tensor_adapter);
      return *this;
    }

    DeviceTensor& operator=(const DeviceTensor&& d) {
      if (this == &d) { return *this; }

      create_device_tensor(d.dimensions(), d.tensor_adapter);
      return *this;
    }

    explicit DeviceTensor(PtrTensorAdapter<Scalar, Device_>& adapter) {
      tensor_adapter = adapter;
      tensor_view = OptionalTensorView<Scalar, Rank, Layout>(TensorView<Scalar, Rank, Layout>(tensor_adapter->data(), TensorAdapter<Scalar, Device_>::template vec2dims<Rank>(adapter->get_dims())));
    }

    Tensor<Scalar, Rank, Layout> to_host() const {

      DefaultDevice host;

      size_t alloc_size = size() * sizeof(Scalar);

      Scalar* data = static_cast<Scalar*>(host.allocate(alloc_size));
      device().memcpyDeviceToHost(data, tensor_view->data(), alloc_size);
      Tensor<Scalar, Rank, Layout> out = TensorView<Scalar, Rank, Layout>(data, dimensions());

      return out;
    }

    // resizing, setting values
    void resize(DSizes<Index, Rank> dims) {
      tensor_adapter.reset();
      tensor_view.reset();
      create_device_tensor(DSizes<Index, Rank>(dims), nullptr);
    }

    DeviceTensor& resize(array<Index, Rank> dims) {
      resize(DSizes<Index, Rank>(dims));
      return *this;
    }

    template<typename... IndexTypes>
    DeviceTensor& resize(Index firstDim, IndexTypes... otherDimensions) {
      resize(DSizes<Index, Rank>(firstDim, otherDimensions...));
      return *this;
    }

    DeviceTensor& setConstant(Scalar c) {
      EigenSinn::setConstant(*tensor_view, c, device());
      return *this;
    }

    DeviceTensor& setZero() {
      EigenSinn::setZero(*tensor_view, device());
      return *this;
    }

    DeviceTensor& setValues(
      const typename internal::Initializer<Tensor<Scalar, Rank, Layout>, Rank>::InitList& vals) {

      Tensor<Scalar, Rank, Layout> temp(dimensions());
      TensorEvaluator<Tensor<Scalar, Rank, Layout>, DefaultDevice> eval(*static_cast<const Tensor<Scalar, Rank, Layout>*>(&temp), DefaultDevice());
      internal::initialize_tensor<Tensor<Scalar, Rank, Layout>, Rank>(eval, vals);

      EigenSinn::setValues(*tensor_view, temp, device());
      return *this;
    }

    DeviceTensor& setRandom() {

      assert(tensor_view);
      Tensor<Scalar, Rank, Layout> temp(dimensions());
      temp.setRandom();
      move_to(*tensor_view, temp.data(), device());

      return *this;
    }

    template <typename RandomGenerator>
    DeviceTensor& setRandom() {

      assert(tensor_view);
      Tensor<Scalar, Rank, Layout> temp(dimensions());
      temp.setRandom<RandomGenerator>();
      move_to(*tensor_view, temp.data(), device());

      return *this;
    }

    // access
    inline TensorView<Scalar, Rank, Layout>& operator*() const {
      return const_cast<TensorView<Scalar, Rank, Layout>&>(*tensor_view);
    }

    inline TensorView<Scalar, Rank, Layout>* operator->() const {
      return const_cast<TensorView<Scalar, Rank, Layout>*>(&tensor_view.value());
    }

    explicit operator bool() const {
      return tensor_view ? true : false;
    }

    Index size() const {
      return tensor_view.value().dimensions().TotalSize();
    }

    const DSizes<Index, Rank>& dimensions() const {
      return tensor_view.value().dimensions();
    }

    Index dimension(Index i) const {
      return tensor_view.value().dimension(i);
    }

    auto view() { return tensor_view->device(device()); }

    Device_& device() const { return tensor_adapter->get_device(); }

    PtrTensorAdapter<Scalar, Device_> raw() { return tensor_adapter; }

    void set_from_device(const Scalar* data, const DSizes<Index, Rank>& dims) {
      create_device_tensor(dims, nullptr);

      device().memcpy(tensor_view->data(), data, tensor_view->dimensions().TotalSize() * sizeof(Scalar));
    }

    void set_from_host(Scalar* data, const DSizes<Index, Rank>& dims) {
      create_device_tensor(dims, nullptr);
      move_to<Scalar, Rank, Device_, Layout>(*tensor_view, data, device());
    }

    void set_from_host(Tensor<Scalar, Rank, Layout>& data) {
      set_from_host(data.data(), data.dimensions());
    }

  private:

    void create_device_tensor() {
      tensor_adapter = std::make_shared<TensorAdapter<Scalar, Device_>>();
      tensor_view.reset();
    }

    void create_device_tensor(const DSizes<Index, Rank>& dims, Scalar* data) {

      tensor_adapter = std::make_shared<TensorAdapter<Scalar, Device_>>(TensorAdapter<Scalar, Device_>::dims2vec(dims), data);
      tensor_view.reset();
      tensor_view = OptionalTensorView<Scalar, Rank, Layout>(TensorView<Scalar, Rank, Layout>(tensor_adapter->data(), dims));
    }

    void create_device_tensor(const DSizes<Index, Rank>& dims, const PtrTensorAdapter<Scalar, Device_>& data) {

      tensor_adapter = data;
      tensor_view.reset();
      tensor_view = OptionalTensorView<Scalar, Rank, Layout>(TensorView<Scalar, Rank, Layout>(tensor_adapter->data(), dims));
    }


    // for tensor ops
    OptionalTensorView<Scalar, Rank, Layout> tensor_view;

    // for passing tensors around
    std::shared_ptr<TensorAdapter<Scalar, Device_>> tensor_adapter;
  };
}