#include <torch/extension.h>

#include <vector>

// CUDA forward declarations


std::vector<torch::Tensor> build_tensoRF_hier_map_cuda(
        torch::Tensor pnt_xyz,
        torch::Tensor gridSize,
        torch::Tensor xyz_min,
        torch::Tensor xyz_max,
        torch::Tensor units,
        torch::Tensor local_range,
        torch::Tensor local_dims,
        const int max_tensoRF);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> build_tensoRF_hier_map(
    torch::Tensor pnt_xyz,
    torch::Tensor gridSize,
    torch::Tensor xyz_min,
    torch::Tensor xyz_max,
    torch::Tensor units,
    torch::Tensor local_range,
    torch::Tensor local_dims, const int max_tensoRF) {
  CHECK_INPUT(pnt_xyz);
  CHECK_INPUT(gridSize);
  CHECK_INPUT(xyz_min);
  CHECK_INPUT(xyz_max);
  CHECK_INPUT(units);
  CHECK_INPUT(local_range);
  CHECK_INPUT(local_dims);
  assert(pnt_xyz.dim()==2);
  return build_tensoRF_hier_map_cuda(pnt_xyz, gridSize, xyz_min, xyz_max, units, local_range, local_dims, max_tensoRF);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build_tensoRF_hier_map", &build_tensoRF_hier_map, "build tensoRF indices map");
}


