#include <torch/extension.h>
#include <vector>



std::vector<torch::Tensor> grid_sample_from_tensoRF_backward_cuda(torch::Tensor local_gindx_s, torch::Tensor local_gindx_l, torch::Tensor local_gweight_s, torch::Tensor local_gweight_l, torch::Tensor final_tensoRF_id, torch::Tensor grad_planeout, torch::Tensor grad_lineout, int planesurf_num, int linesurf_num, int component_num, int res);


std::vector<torch::Tensor> grid_sample_from_tensoRF_cuda(
        torch::Tensor plane,
        torch::Tensor line1,
        torch::Tensor line2,
        torch::Tensor line3,
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


std::vector<torch::Tensor> cal_w_inds_cuda(
        torch::Tensor plane,
        torch::Tensor line1,
        torch::Tensor line2,
        torch::Tensor line3,
        torch::Tensor local_gindx_s,
        torch::Tensor local_gindx_l,
        torch::Tensor local_gweight_s,
        torch::Tensor local_gweight_l,
        torch::Tensor final_tensoRF_id);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)



std::vector<torch::Tensor> grid_sample_from_tensoRF(torch::Tensor plane, torch::Tensor line1, torch::Tensor line2, torch::Tensor line3, torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max, torch::Tensor units, torch::Tensor lvl_units, torch::Tensor local_range, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds, torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_xyz, const int K, const bool KNN) {
  CHECK_INPUT(plane);
  CHECK_INPUT(line1);
  CHECK_INPUT(line2);
  CHECK_INPUT(line3);
  CHECK_INPUT(geo_xyz);
  assert(xyz_sampled.dim()==2);
  return grid_sample_from_tensoRF_cuda(plane, line1, line2, line3, xyz_sampled, xyz_min, xyz_max, units, lvl_units, local_range, local_dims, tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx, geo_xyz, K, KNN);
}



std::vector<torch::Tensor> cal_w_inds(torch::Tensor plane, torch::Tensor line1, torch::Tensor line2, torch::Tensor line3, torch::Tensor local_gindx_s, torch::Tensor local_gindx_l, torch::Tensor local_gweight_s, torch::Tensor local_gweight_l, torch::Tensor final_tensoRF_id) {
  CHECK_INPUT(plane);
  CHECK_INPUT(line1);
  CHECK_INPUT(line2);
  CHECK_INPUT(line3);
  CHECK_INPUT(local_gindx_l);
  return cal_w_inds_cuda(plane, line1, line2, line3, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, final_tensoRF_id);
}

std::vector<torch::Tensor> grid_sample_from_tensoRF_backward(torch::Tensor local_gindx_s, torch::Tensor local_gindx_l, torch::Tensor local_gweight_s, torch::Tensor local_gweight_l, torch::Tensor final_tensoRF_id, torch::Tensor grad_planeout, torch::Tensor grad_lineout, int planesurf_num, int linesurf_num, int component_num, int res) {
  CHECK_INPUT(grad_planeout);
  CHECK_INPUT(grad_lineout);
  return grid_sample_from_tensoRF_backward_cuda(local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, final_tensoRF_id, grad_planeout, grad_lineout, planesurf_num, linesurf_num, component_num, res);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("cal_w_inds", &cal_w_inds, "Sampled points to get torsoRF with inds");
  m.def("grid_sample_from_tensoRF", &grid_sample_from_tensoRF, "Sampled points to get torsoRF with gs sample");
  m.def("grid_sample_from_tensoRF_backward", &grid_sample_from_tensoRF_backward, "Backward pass of the tensorf");
}


