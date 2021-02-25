#pragma once

#include "unsupported/Eigen/CXX11/Tensor"
#include <fstream>
#include <iostream>
#include <string>
#include <functional>
#include <sstream>
#include <stdexcept>

using namespace Eigen;

template<template <typename ...> class Container, typename Scalar>
void parse_stream(const std::string& str, Container<Scalar>& res, const char sep) {

  std::stringstream ss(str);

  bool non_space_sep = (sep != ' ');

  for (Scalar val; ss >> val;) {

    res.push_back(val);

    if (!non_space_sep) continue;
    ss.ignore();
  }

}

// read a csv file of initialization weights
// the file contains dimensions on the first line
// and values on the second
template <typename Scalar, Index Rank>
inline Tensor<Scalar, Rank> read_tensor_csv(const std::string& file_path, const char separator = ' ') {

  std::ifstream file;
  file.open(file_path, std::ios::in | std::ios::ate);

  if (!file) {
    std::cout << "Error opening file: " << file_path << std::endl;
    throw std::ifstream::failure("Error opening level file");
  }

  file.seekg(0, std::ios::beg);
  std::string str;
  std::getline(file, str);

  // dimensions
  std::vector<int> vdims;
  array<Index, Rank> dims;

  parse_stream(str, vdims, separator);

  int i = 0;
  for (const auto& d : vdims) {
    dims[i++] = d;
  }

  // reverse dimensions for future shuffling
  std::vector<Index> rev_dims(vdims.size());
  std::iota(rev_dims.begin(), rev_dims.end(), 0);
  std::reverse(rev_dims.begin(), rev_dims.end());
  
  // tensor
  std::vector<float> tensor_vec;
  std::getline(file, str);
  file.close();

  parse_stream(str, tensor_vec, separator);

  // convert tensor to col_major format
  TensorMap<Tensor<Scalar, Rank, RowMajor>> out_rows(tensor_vec.data(), dims);
  Tensor<Scalar, Rank> col_major = out_rows.swap_layout().shuffle(rev_dims);

  return col_major;
}
