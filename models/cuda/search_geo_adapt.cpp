#include <torch/extension.h>

#include <vector>

// CUDA forward declarations


std::vector<torch::Tensor> build_cubic_tensoRF_map_hier_cuda(
        torch::Tensor pnt_xyz,
        torch::Tensor gridSize,
        torch::Tensor xyz_min,
        torch::Tensor xyz_max,
        torch::Tensor units,
        torch::Tensor radius,
        torch::Tensor local_range,
        torch::Tensor pnt_rmatrix,
        torch::Tensor local_dims,
        const int max_tensoRF);


std::vector<torch::Tensor> sample_2_rot_cubic_tensoRF_cvrg_cuda(
        torch::Tensor xyz_sampled,
        torch::Tensor xyz_min,
        torch::Tensor xyz_max,
        torch::Tensor units,
        torch::Tensor local_range,
        torch::Tensor local_dims,
        torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor tensoRF_count,
        torch::Tensor tensoRF_topindx,
        torch::Tensor geo_xyz,
        torch::Tensor geo_rot,
        torch::Tensor dim_cumsum_counter,
        const int K,
        const bool KNN);

torch::Tensor filter_tensoRF_cuda(
        torch::Tensor xyz_sampled,
        torch::Tensor xyz_inbbox,
        torch::Tensor xyz_min,
        torch::Tensor xyz_max,
        torch::Tensor units,
        torch::Tensor local_range,
        torch::Tensor local_dims,
        torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor tensoRF_count,
        torch::Tensor tensoRF_topindx,
        torch::Tensor geo_xyz,
        torch::Tensor geo_rot,
        const int K,
        const int ord_thresh);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


torch::Tensor filter_tensoRF(
    torch::Tensor xyz_sampled,
    torch::Tensor xyz_inbbox,
    torch::Tensor xyz_min,
    torch::Tensor xyz_max,
    torch::Tensor units,
    torch::Tensor local_range,
    torch::Tensor local_dims,
    torch::Tensor tensoRF_cvrg_inds,
    torch::Tensor tensoRF_count,
    torch::Tensor tensoRF_topindx,
    torch::Tensor geo_rot,
    torch::Tensor geo_xyz,
    const int K,
    const int ord_thresh) {
  CHECK_INPUT(xyz_sampled);
  CHECK_INPUT(geo_xyz);
  CHECK_INPUT(geo_rot);
  assert(xyz_sampled.dim()==3);
  return filter_tensoRF_cuda(xyz_sampled, xyz_inbbox, xyz_min, xyz_max, units, local_range, local_dims, tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx, geo_xyz, geo_rot, K, ord_thresh);
}


std::vector<torch::Tensor> sample_2_rot_cubic_tensoRF_cvrg(
    torch::Tensor xyz_sampled,
    torch::Tensor xyz_min,
    torch::Tensor xyz_max,
    torch::Tensor units,
    torch::Tensor local_range,
    torch::Tensor local_dims,
    torch::Tensor tensoRF_cvrg_inds,
    torch::Tensor tensoRF_count,
    torch::Tensor tensoRF_topindx,
    torch::Tensor geo_rot,
    torch::Tensor geo_xyz,
    torch::Tensor dim_cumsum_counter,
    const int K,
    const bool KNN) {
  CHECK_INPUT(xyz_sampled);
  CHECK_INPUT(geo_xyz);
  CHECK_INPUT(geo_rot);
  CHECK_INPUT(dim_cumsum_counter);
  assert(xyz_sampled.dim()==2);
  return sample_2_rot_cubic_tensoRF_cvrg_cuda(xyz_sampled, xyz_min, xyz_max, units, local_range, local_dims, tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx, geo_xyz, geo_rot, dim_cumsum_counter, K, KNN);
}


//xyz_sampled.contiguous(), self.aabb[0], self.aabb[1], self.units, self.local_range[l], self.local_dims[l], self.tensoRF_cvrg_inds[l], self.tensoRF_count[l], self.tensoRF_topindx[l], pnt_rmatrix[l], self.geo_xyz[l], self.dim_cumsum_counter, self.K_tensoRF[l], self.KNN

std::vector<torch::Tensor> build_cubic_tensoRF_map_hier(
        torch::Tensor pnt_xyz,
        torch::Tensor gridSize,
        torch::Tensor xyz_min,
        torch::Tensor xyz_max,
        torch::Tensor units,
        torch::Tensor radius,
        torch::Tensor local_range,
        torch::Tensor pnt_rmatrix,
        torch::Tensor local_dims,
        const int max_tensoRF){
  CHECK_INPUT(pnt_xyz);
  CHECK_INPUT(gridSize);
  CHECK_INPUT(xyz_min);
  CHECK_INPUT(xyz_max);
  CHECK_INPUT(units);
  CHECK_INPUT(radius);
  CHECK_INPUT(local_range);
  CHECK_INPUT(pnt_rmatrix);
  CHECK_INPUT(local_dims);
  assert(pnt_xyz.dim()==2);
  return build_cubic_tensoRF_map_hier_cuda(pnt_xyz, gridSize, xyz_min, xyz_max, units, radius, local_range, pnt_rmatrix, local_dims, max_tensoRF);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("sample_2_rot_cubic_tensoRF_cvrg", &sample_2_rot_cubic_tensoRF_cvrg, "Sampled points to get torsoRF");
  m.def("build_cubic_tensoRF_map_hier", &build_cubic_tensoRF_map_hier, "build cubic tensoRF indices map");
  m.def("filter_tensoRF", &filter_tensoRF, "filter tensoRF by threshold");
}


