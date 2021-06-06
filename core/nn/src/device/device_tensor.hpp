#pragma once

#include "device_helpers.hpp"
#include "ops/opsbase.hpp"
#include "tensor_adapter.hpp"
#include "onnx/model.h"

namespace EigenSinn
{
  template<typename Scalar, typename Device_>
  using PtrTensorAdapter = std::shared_ptr<TensorAdapter<Scalar, Device_>>;

  template<typename Scalar, Index Rank, typename Device_ = ThreadPoolDevice, int Layout = RowMajor>
  class DeviceTensor {

  public:

    // allocation constructors
    explicit DeviceTensor() {

      if (Rank == 0) {
        std::vector<Index> dims;
        create_device_tensor(dims, nullptr);
      }
      else {
        create_device_tensor();
      }
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
      if (!d.tensor_view) { return; }
      create_device_tensor(d.dimensions(), d.tensor_adapter);
    }

    DeviceTensor(const DeviceTensor&& d) {
      if (!d.tensor_view) { return; }
      tensor_adapter = std::move(d.tensor_adapter);
      reset_view();
      tensor_view = OptionalTensorView<Scalar, Rank, Layout>(TensorView<Scalar, Rank, Layout>(tensor_adapter->data(), d.dimensions()));
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
      tensor_view = OptionalTensorView<Scalar, Rank, Layout>(TensorView<Scalar, Rank, Layout>(tensor_adapter->data(), vec2dims<Rank>(adapter->get_dims())));
    }

    Tensor<Scalar, Rank, Layout> to_host() const {

#ifdef EIGEN_USE_GPU
      if (!std::is_same<Device_, GpuDevice>::value) {
#endif
        return *tensor_view;
#ifdef EIGEN_USE_GPU
      }

      DefaultDevice host;

      size_t alloc_size = size() * sizeof(Scalar);

      Scalar* data = static_cast<Scalar*>(host.allocate(alloc_size));
      device().memcpyDeviceToHost(data, tensor_view->data(), alloc_size);
      Tensor<Scalar, Rank, Layout> out = TensorView<Scalar, Rank, Layout>(data, dimensions());

      host.deallocate(data);
      return out;
#endif
    }

    // resizing, setting values
    void resize(DSizes<Index, Rank> dims) {
      tensor_adapter.reset();
      reset_view();
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
      return static_cast<bool>(tensor_view);
    }

    Index size() const {
      return tensor_view.value().dimensions().TotalSize();
    }

    const DSizes<Index, Rank>& dimensions() const {
      return tensor_view.value().dimensions();
    }

    const std::vector<Index> vec_dims() const {
      return dims2vec(dimensions());
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

    DeviceTensor clone() {
      DeviceTensor out;

      out.set_from_device(tensor_adapter->data(), tensor_view->dimensions());
      return out;
    }

    // switch layout
    inline DeviceTensor<Scalar, Rank, Device_, (Layout^ RowMajor)> swap_layout() const {

      DeviceTensor<Scalar, Rank, Device_, Layout^ RowMajor> out_t(dimensions());

      out_t.view() = tensor_view->swap_layout().reshape(dimensions());
      return out_t;
    }

    ~DeviceTensor() {
      assert(true);
    }

    // ONNX
    inline void save_onnx_initializer(EigenModel& model, const std::string& name) {

      const std::string data = get_data_row_major();
      auto graph = model.get_graph();
      onnx::TensorProto* initializer = graph->add_initializer();
      initializer->set_name(name);

      // Case of Rank = 0
      const int rank = Rank > 0 ? Rank : 1;

      for (Index i = 0; i < rank; i++) {
        initializer->add_dims(dimensions()[i]);
      }

      initializer->set_raw_data(data);
      initializer->set_data_type(onnx_data_type_from_scalar<Scalar>());

      // we may pass a fancy name instead of a generated number
      if (node_input_name.empty()) {
        node_input_name = name;
      }
    }

    inline void save_onnx_initializer(EigenModel& model) {
      save_onnx_initializer(model, get_onnx_input_name());
    }

    inline std::string get_onnx_input_name() {

      if (node_input_name.empty()) {
        node_input_name = EigenModel::get_tensor_value_name();
      }
      return node_input_name;
    }

  private:

    inline void reset_view() {
      if (tensor_view) {
        tensor_view.reset();
      }
    }
    void create_device_tensor() {
      tensor_adapter = std::make_shared<TensorAdapter<Scalar, Device_>>();
      reset_view();
    }

    void create_device_tensor(const DSizes<Index, Rank>& dims, Scalar* data) {

      tensor_adapter = std::make_shared<TensorAdapter<Scalar, Device_>>(dims2vec(dims), data);
      reset_view();
      tensor_view = OptionalTensorView<Scalar, Rank, Layout>(TensorView<Scalar, Rank, Layout>(tensor_adapter->data(), dims));
    }

    void create_device_tensor(const std::vector<Index>& dims, Scalar* data) {

      tensor_adapter = std::make_shared<TensorAdapter<Scalar, Device_>>(dims, data);
      reset_view();
      tensor_view = OptionalTensorView<Scalar, Rank, Layout>(TensorView<Scalar, Rank, Layout>(tensor_adapter->data(), vec2dims<Rank>(dims)));
    }

    void create_device_tensor(const DSizes<Index, Rank>& dims, const PtrTensorAdapter<Scalar, Device_>& data) {

      tensor_adapter = data;
      reset_view();
      tensor_view = OptionalTensorView<Scalar, Rank, Layout>(TensorView<Scalar, Rank, Layout>(tensor_adapter->data(), dims));
    }

    // ONNX
    // Given device tensor return its data on the host in RowMajor layout
    // For saving in ONNX format
    const std::string get_data_row_major() {

      // we want RowMajor layout
      Tensor<Scalar, Rank, Layout> host_t = to_host();
      Scalar* _data = host_t.data();

      std::vector<char> out_data;
      out_data.resize(dimensions().TotalSize() * sizeof(Scalar));

      Tensor<Scalar, Rank, RowMajor> out_t;
      DefaultDevice default_device;

      // we don't care about the layout if it's single-dimensional tensor
      if (Layout != RowMajor) {

        // swap_layout reverses dimensions, we'll need to shuffle them back
        DSizes<Index, Rank> shuffle_dims, reverse_dims;
        for (Index i = Rank - 1; i >= 0; i--) {
          shuffle_dims[Rank - 1 - i] = host_t.dimension(i);
          reverse_dims[Rank - 1 - i] = i;
        }

        TensorView<Scalar, Rank, RowMajor> out_view(_data, shuffle_dims);
        out_view = out_view.shuffle(reverse_dims);
        _data = out_view.data();
      }

      // everything is happening on the CPU
      default_device.memcpy(out_data.data(), (char*)_data, out_data.size());
      std::string out(out_data.begin(), out_data.end());
      return out;
    }

    // for tensor ops
    OptionalTensorView<Scalar, Rank, Layout> tensor_view;

    // for passing tensors around
    std::shared_ptr<TensorAdapter<Scalar, Device_>> tensor_adapter;

    // name for ONNX node input/initializer
    std::string node_input_name;
  };
}