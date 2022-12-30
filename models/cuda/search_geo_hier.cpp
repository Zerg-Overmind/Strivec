#include <torch/extension.h>

#include <vector>

// CUDA forward declarations


std::vector<torch::Tensor> build_tensoRF_map_hier_cuda(
        torch::Tensor pnt_xyz,
        torch::Tensor gridSize,
        torch::Tensor xyz_min,
        torch::Tensor xyz_max,
        torch::Tensor units,
        torch::Tensor local_range,
        torch::Tensor local_dims,
        const int max_tensoRF);


std::vector<torch::Tensor> sample_2_tensoRF_cvrg_hier_cuda(
        torch::Tensor xyz_sampled,
        torch::Tensor xyz_min,
        torch::Tensor xyz_max,
        torch::Tensor units,
        torch::Tensor lvl_units,
        torch::Tensor local_range,
        torch::Tensor local_dims,
        torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor tensoRF_count,
        torch::Tensor tensoRF_topindx,
        torch::Tensor geo_xyz,
        const int K,
        const bool KNN);

std::vector<torch::Tensor> sample_2_tensoRF_cvrg_hier_gs_cuda(
        torch::Tensor xyz_sampled,
        torch::Tensor xyz_min,
        torch::Tensor xyz_max,
        torch::Tensor units,
        torch::Tensor lvl_units,
        torch::Tensor local_range,
        torch::Tensor local_dims,
        torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor tensoRF_count,
        torch::Tensor tensoRF_topindx,
        torch::Tensor geo_xyz,
        const int K,
        const bool KNN);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



std::vector<torch::Tensor> sample_2_tensoRF_cvrg_hier_gs(torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max, torch::Tensor units, torch::Tensor lvl_units, torch::Tensor local_range, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds, torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_xyz, const int K, const bool KNN) {
  CHECK_INPUT(xyz_sampled);
  CHECK_INPUT(geo_xyz);
  assert(xyz_sampled.dim()==2);
  return sample_2_tensoRF_cvrg_hier_gs_cuda(xyz_sampled, xyz_min, xyz_max, units, lvl_units, local_range, local_dims, tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx, geo_xyz, K, KNN);
}


std::vector<torch::Tensor> sample_2_tensoRF_cvrg_hier(torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max, torch::Tensor units, torch::Tensor lvl_units, torch::Tensor local_range, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds, torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_xyz, const int K, const bool KNN) {
  CHECK_INPUT(xyz_sampled);
  CHECK_INPUT(geo_xyz);
  assert(xyz_sampled.dim()==2);
  return sample_2_tensoRF_cvrg_hier_cuda(xyz_sampled, xyz_min, xyz_max, units, lvl_units, local_range, local_dims, tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx, geo_xyz, K, KNN);
}

std::vector<torch::Tensor> build_tensoRF_map_hier(
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
  return build_tensoRF_map_hier_cuda(pnt_xyz, gridSize, xyz_min, xyz_max, units, local_range, local_dims, max_tensoRF);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build_tensoRF_map_hier", &build_tensoRF_map_hier, "build tensoRF indices map");
  m.def("sample_2_tensoRF_cvrg_hier", &sample_2_tensoRF_cvrg_hier, "Sampled points to get torsoRF");
  m.def("sample_2_tensoRF_cvrg_hier_gs", &sample_2_tensoRF_cvrg_hier_gs, "Sampled points to get torsoRF with gs sample");
}


