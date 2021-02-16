#pragma once

#include "device_helpers.hpp"
#include "ops/conversions.hpp"

namespace EigenSinn
{
  template<typename Device_, typename Scalar, Index Rank, int Layout = ColMajor>
  class DeviceTensor {

  public:

    // allocation constructors
    explicit DeviceTensor()
      : dispatcher(Dispatcher<Device_>::create())
      , device_(dispatcher.get_device()) {

      if (Rank == 0) {
        tensor_view = create_device_ptr<Device_, Scalar, Rank, Layout>(DSizes<Index, Rank>(), device_);
      }
    }

    explicit DeviceTensor(const DSizes<Index, Rank> dims)
      : DeviceTensor() {

      tensor_view = create_device_ptr<Device_, Scalar, Rank, Layout>(dims, device_);

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

    explicit DeviceTensor(TensorView<Scalar, Rank, Layout>& tv)
      : DeviceTensor() {

      tensor_view.reset(new TensorView<Scalar, Rank, Layout>(tv.data(), tv.dimensions()));
    }

    DeviceTensor(std::any& any) 
      : DeviceTensor() {

      auto d = from_any(any);

      tensor_view = std::move(d.tensor_view);
    }

    /// <summary>
    /// Copy tensor from host
    /// </summary>
    /// <param name="t"></param>
    /// <returns></returns>
    DeviceTensor& operator=(const Tensor<Scalar, Rank, Layout>& t) {
      set_from_host(t);
      return *this;
    }

    /// <summary>
    /// Initialize from a tensor on the host
    /// </summary>
    /// <param name="data"></param>
    /// <param name="dims"></param>
    explicit DeviceTensor(const Tensor<Scalar, Rank, Layout>& data)
      : DeviceTensor(data.dimensions()) {

      move_to<Device_, Scalar, Rank, Layout>(*tensor_view, data.data(), device_);
    }

    void set_from_device(const Scalar* data, const DSizes<Index, Rank>& dims) {
      create_device_tensor_if_needed(dims);
      device_.memcpy(tensor_view->data(), data, tensor_view->dimensions().TotalSize() * sizeof(Scalar));
    }

    void set_from_host(const Scalar* data, const DSizes<Index, Rank>& dims) {
      create_device_tensor_if_needed(dims);
      move_to<Device_, Scalar, Rank, Layout>(*tensor_view, data, device_);
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
    ~DeviceTensor() {
      release(true);
    }

    // copy constructor: deep copy
    DeviceTensor(const DeviceTensor& d) : DeviceTensor() {

      bool is_null_copy = d.tensor_view.get() == nullptr;
      if (!is_null_copy) {

        size_t alloc_size = sizeof(Scalar) * d.tensor_view->dimensions().TotalSize();
        Scalar* data = static_cast<Scalar*>(device_.allocate(alloc_size));

        device_.memcpy(data, d.tensor_view->data(), alloc_size);
        tensor_view.reset(new TensorView<Scalar, Rank, Layout>(data, d.tensor_view->dimensions()));
      }
    }

    // copy assignment
    DeviceTensor& operator=(const DeviceTensor& d) {
      if (&d == this) {
        return *this;
      }

      release();

      size_t alloc_size = sizeof(Scalar) * d.tensor_view->dimensions().TotalSize();
      Scalar* data = static_cast<Scalar*>(device_.allocate(alloc_size));

      device_.memcpy(data, d.tensor_view->data(), alloc_size);
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
        return *this;
      }

      release();

      tensor_view = std::move(d.tensor_view);

      return *this;
    }

    // resizing, setting values
    void resize(DSizes<Index, Rank> dims) {
      release();
      tensor_view = create_device_ptr<Device_, Scalar, Rank, Layout>(DSizes<Index, Rank>(dims), device_);
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
    TensorView<Scalar, Rank, Layout>& operator* () const {
      return *tensor_view;
    }

    TensorView<Scalar, Rank, Layout>* operator-> () const {
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

    // Operators with device tensor
    DeviceTensor& operator+=(const DeviceTensor& d) {

      tensor_view->device(device_) = *tensor_view + *d.tensor_view;
      return *this;
    }

    friend DeviceTensor operator+(DeviceTensor lhs, DeviceTensor& rhs) {
      lhs += rhs;
      return lhs;
    }

    DeviceTensor& operator-=(const DeviceTensor& d) {

      tensor_view->device(device_) = *tensor_view - *d.tensor_view;
      return *this;
    }

    friend DeviceTensor operator-(DeviceTensor lhs, DeviceTensor& rhs) {
      lhs -= rhs;
      return lhs;
    }

    DeviceTensor& operator*=(const DeviceTensor& d) {

      tensor_view->device(device_) = *tensor_view * *d.tensor_view;
      return *this;
    }

    friend DeviceTensor operator*(DeviceTensor lhs, DeviceTensor& rhs) {
      lhs *= rhs;
      return lhs;
    }

    DeviceTensor& operator/=(const DeviceTensor& d) {

      tensor_view->device(device_) = *tensor_view / *d.tensor_view;
      return *this;
    }

    friend DeviceTensor operator/(DeviceTensor lhs, DeviceTensor& rhs) {
      lhs /= rhs;
      return lhs;
    }

    friend DeviceTensor<Device_, bool, Rank, Layout> operator==(DeviceTensor<Device_, Scalar, Rank, Layout>& lhs, Scalar& v) {
      DeviceTensor<Device_, bool, Rank, Layout> out(lhs.dimensions());
      out.view() = (lhs == v);
      return out;
    }

    friend DeviceTensor<Device_, bool, Rank, Layout> operator!=(DeviceTensor<Device_, Scalar, Rank, Layout>& lhs, Scalar& v) {
      DeviceTensor<Device_, bool, Rank, Layout> out(lhs.dimensions());
      out.view() = (lhs != v);
      return out;
    }

    friend DeviceTensor<Device_, bool, Rank, Layout> operator<=(DeviceTensor<Device_, Scalar, Rank, Layout>& lhs, Scalar& v) {
      DeviceTensor<Device_, bool, Rank, Layout> out(lhs.dimensions());
      out.view() = (lhs <= v);
      return out;
    }

    friend DeviceTensor<Device_, bool, Rank, Layout> operator>=(DeviceTensor<Device_, Scalar, Rank, Layout>& lhs, Scalar& v) {
      DeviceTensor<Device_, bool, Rank, Layout> out(lhs.dimensions());
      out.view() = (lhs >= v);
      return out;
    }

    friend DeviceTensor<Device_, bool, Rank, Layout> operator<(DeviceTensor<Device_, Scalar, Rank, Layout>& lhs, Scalar& v) {
      DeviceTensor<Device_, bool, Rank, Layout> out(lhs.dimensions());
      out.view() = (lhs < v);
      return out;
    }

    friend DeviceTensor<Device_, bool, Rank, Layout> operator>(DeviceTensor<Device_, Scalar, Rank, Layout>& lhs, Scalar& v) {
      DeviceTensor<Device_, bool, Rank, Layout> out(lhs.dimensions());
      out.view() = (lhs > v);
      return out;
    }

    // Operators with device const
    DeviceTensor& operator+=(Scalar rhs) {
      tensor_view->device(device_) = *tensor_view + rhs;
      return *this;
    }

    friend DeviceTensor operator+(DeviceTensor lhs, Scalar& rhs) {
      lhs += rhs;
      return lhs;
    }

    DeviceTensor& operator-=(Scalar rhs) {
      tensor_view->device(device_) = *tensor_view - rhs;
      return *this;
    }

    friend DeviceTensor operator-(DeviceTensor lhs, Scalar& rhs) {
      lhs -= rhs;
      return lhs;
    }

    DeviceTensor& operator*=(Scalar rhs) {
      tensor_view->device(device_) = *tensor_view * rhs;
      return *this;
    }

    friend DeviceTensor operator*(DeviceTensor lhs, Scalar& rhs) {
      lhs *= rhs;
      return lhs;
    }

    DeviceTensor& operator/=(Scalar rhs) {
      tensor_view->device(device_) = *tensor_view / rhs;
      return *this;
    }

    friend DeviceTensor operator/(DeviceTensor lhs, Scalar& rhs) {
      lhs /= rhs;
      return lhs;
    }

    friend DeviceTensor operator/(Scalar lhs, DeviceTensor rhs) {

      DeviceTensor<Device_, Scalar, Rank, Layout> res(rhs.dimensions());
      res->device(rhs.device_) = lhs / rhs;
      return res;
    }


    friend DeviceTensor operator*(Scalar lhs, DeviceTensor rhs) {
      rhs *= lhs;
      return rhs;
    }


    friend DeviceTensor operator-(Scalar lhs, DeviceTensor rhs) {
      rhs -= lhs;
      return rhs;
    }


    friend DeviceTensor operator+(Scalar lhs, DeviceTensor rhs) {
      rhs += lhs;
      return rhs;
    }

    // comparison operators
    friend DeviceTensor<Device_, bool, Rank, Layout> operator==(DeviceTensor& lhs, DeviceTensor& rhs) {
      return *lhs == *rhs;
    }

    friend DeviceTensor<Device_, bool, Rank, Layout> operator>(DeviceTensor& lhs, DeviceTensor& rhs) {
      return *lhs > *rhs;
    }

    friend DeviceTensor<Device_, bool, Rank, Layout> operator<(DeviceTensor& lhs, DeviceTensor& rhs) {
      return *lhs < *rhs;
    }

    friend DeviceTensor<Device_, bool, Rank, Layout> operator>=(DeviceTensor& lhs, DeviceTensor& rhs) {
      return *lhs >= *rhs;
    }

    friend DeviceTensor<Device_, bool, Rank, Layout> operator<=(DeviceTensor& lhs, DeviceTensor& rhs) {
      return *lhs <= *rhs;
    }

    Device_& get_device() { return device_; }
  private:
    void create_device_tensor_if_needed(const DSizes<Index, Rank>& dims) {
      release();
      tensor_view = create_device_ptr<Device_, Scalar, Rank, Layout>(dims, device_);
    }

    void release(bool release_device = false) {

      if (tensor_view) {
        free(*tensor_view, device_);
        tensor_view.release();
      }

      if (release_device) {
        dispatcher.release();
      }
    }

    PtrTensorView<Scalar, Rank, Layout> tensor_view;
    Dispatcher<Device_>& dispatcher;
    Device_& device_;

    DeviceTensor<Device_, Scalar, Rank, Layout> from_any(std::any t) {
      return std::any_cast<DeviceTensor<Device_, Scalar, Rank, Layout>&>(t);
    }

  };
}