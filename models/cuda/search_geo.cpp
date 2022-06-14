#include <torch/extension.h>

#include <vector>

// CUDA forward declarations


torch::Tensor filter_xyz_cvrg_cuda(
        torch::Tensor xyz_sampled,
        torch::Tensor xyz_min,
        torch::Tensor xyz_max,
        torch::Tensor units,
        torch::Tensor tensoRF_cvrg_mask);

torch::Tensor filter_xyz_rot_cvrg_cuda(
        torch::Tensor geo_xyz,
        torch::Tensor geo_rot,
        torch::Tensor xyz_sampled,
        torch::Tensor xyz_min,
        torch::Tensor xyz_max,
        torch::Tensor units,
        torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor tensoRF_count,
        torch::Tensor tensoRF_topindx,
        torch::Tensor local_range);

torch::Tensor filter_xyz_sphere_cvrg_cuda(
        torch::Tensor xyz_sampled,
        torch::Tensor xyz_min,
        torch::Tensor xyz_max,
        torch::Tensor units,
        const float radiusl, const float radiush,
        torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor tensoRF_count,
        torch::Tensor tensoRF_topindx,
        torch::Tensor pnt_xyz);

std::vector<torch::Tensor> build_tensoRF_map_cuda(
        torch::Tensor pnt_xyz,
        torch::Tensor gridSize,
        torch::Tensor xyz_min,
        torch::Tensor xyz_max,
        torch::Tensor units,
        torch::Tensor local_range,
        torch::Tensor local_dims,
        const int max_tensoRF);

std::vector<torch::Tensor> build_sphere_tensoRF_map_cuda(
        torch::Tensor pnt_xyz,
        torch::Tensor gridSize,
        torch::Tensor xyz_min,
        torch::Tensor xyz_max,
        torch::Tensor units,
        const float radiusl, const float radiush,
        torch::Tensor local_dims,
        const int max_tensoRF);

torch::Tensor filter_ray_by_cvrg_cuda(
        torch::Tensor xyz_sampled,
        torch::Tensor mask_inbox,
        torch::Tensor units,
        torch::Tensor xyz_min,
        torch::Tensor xyz_max,
        torch::Tensor tensoRF_cvrg_inds);

std::vector<torch::Tensor> sample_pts_on_rays_dist_cvrg_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor units, torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far, const float stepdist, torch::Tensor shift);

std::vector<torch::Tensor> sample_pts_on_rays_cvrg_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor units, torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far, const float stepdist);

std::vector<torch::Tensor> sample_pts_on_rays_rot_cvrg_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor pnt_xyz, torch::Tensor pnt_rmatrix, torch::Tensor tensoRF_cvrg_inds, torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor units, torch::Tensor local_range, torch::Tensor xyz_min, torch::Tensor xyz_max, const float near, const float far, const float stepdist);

std::vector<torch::Tensor> sample_pts_on_rays_ji_cvrg_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor units, torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far, const float stepdist, torch::Tensor shift);

std::vector<torch::Tensor> sample_pts_on_rays_sphere_cvrg_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor pnt_xyz, torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx,
        torch::Tensor units, const float radiusl, const float radiush, torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far, const float stepdist);

std::vector<torch::Tensor> sample_2_sphere_tensoRF_cvrg_cuda(
        torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max,
        torch::Tensor units, const float radiusl, const float radiush, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_xyz, const int K, const bool KNN);

std::vector<torch::Tensor> sample_2_tensoRF_cvrg_cuda(
        torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max,
        torch::Tensor units, torch::Tensor local_range, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_xyz_recenter, torch::Tensor geo_xyz, const int K, const bool KNN);

std::vector<torch::Tensor> sample_2_rot_tensoRF_cvrg_cuda(
        torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max,
        torch::Tensor units, torch::Tensor local_range, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_rot, torch::Tensor geo_xyz, const int K, const bool KNN);


std::vector<torch::Tensor> sample_2_rotdist_tensoRF_cvrg_cuda(
        torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max,
        torch::Tensor units, torch::Tensor local_range, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_rot, torch::Tensor geo_xyz, const int K, const bool KNN);

std::vector<torch::Tensor> inds_cvrg_cuda(
        torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max,
        torch::Tensor units, torch::Tensor local_range, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_xyz_recenter, torch::Tensor geo_xyz);

// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> build_tensoRF_map(
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
  return build_tensoRF_map_cuda(pnt_xyz, gridSize, xyz_min, xyz_max, units, local_range, local_dims, max_tensoRF);
}


std::vector<torch::Tensor> build_sphere_tensoRF_map(
    torch::Tensor pnt_xyz,
    torch::Tensor gridSize,
    torch::Tensor xyz_min,
    torch::Tensor xyz_max,
    torch::Tensor units,
    const float radiusl, const float radiush,
    torch::Tensor local_dims, const int max_tensoRF) {
  CHECK_INPUT(pnt_xyz);
  CHECK_INPUT(gridSize);
  CHECK_INPUT(xyz_min);
  CHECK_INPUT(xyz_max);
  CHECK_INPUT(local_dims);
  assert(pnt_xyz.dim()==2);
  return build_sphere_tensoRF_map_cuda(pnt_xyz, gridSize, xyz_min, xyz_max, units, radiusl, radiush, local_dims, max_tensoRF);
}


torch::Tensor filter_ray_by_cvrg(torch::Tensor xyz_sampled, torch::Tensor mask_inbox, torch::Tensor units, torch::Tensor xyz_min, torch::Tensor xyz_max, torch::Tensor tensoRF_cvrg_inds) {
  CHECK_INPUT(xyz_sampled);
  CHECK_INPUT(mask_inbox);
  CHECK_INPUT(units);
  CHECK_INPUT(xyz_min);
  CHECK_INPUT(xyz_max);
  CHECK_INPUT(tensoRF_cvrg_inds);
  assert(xyz_sampled.dim()==3);
  assert(mask_inbox.dim()==2);
  assert(tensoRF_cvrg_inds.dim()==3);
  return filter_ray_by_cvrg_cuda(xyz_sampled, mask_inbox, units, xyz_min, xyz_max, tensoRF_cvrg_inds);
}


std::vector<torch::Tensor> sample_pts_on_rays_dist_cvrg(
        torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor units, torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far, const float stepdist, torch::Tensor shift) {
  CHECK_INPUT(rays_o);
  CHECK_INPUT(rays_d);
  CHECK_INPUT(xyz_min);
  CHECK_INPUT(xyz_max);
  assert(rays_o.dim()==2);
  assert(rays_o.size(1)==3);
  return sample_pts_on_rays_dist_cvrg_cuda(rays_o, rays_d, tensoRF_cvrg_inds, units, xyz_min, xyz_max, near, far, stepdist, shift);
}


std::vector<torch::Tensor> sample_pts_on_rays_cvrg(
        torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor units, torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far, const float stepdist) {
  CHECK_INPUT(rays_o);
  CHECK_INPUT(rays_d);
  CHECK_INPUT(xyz_min);
  CHECK_INPUT(xyz_max);
  assert(rays_o.dim()==2);
  assert(rays_o.size(1)==3);
  return sample_pts_on_rays_cvrg_cuda(rays_o, rays_d, tensoRF_cvrg_inds, units, xyz_min, xyz_max, near, far, stepdist);
}


std::vector<torch::Tensor> sample_pts_on_rays_rot_cvrg(
        torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor pnt_xyz, torch::Tensor pnt_rmatrix, torch::Tensor tensoRF_cvrg_inds, torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor units, torch::Tensor local_range, torch::Tensor xyz_min, torch::Tensor xyz_max, const float near, const float far, const float stepdist) {
  CHECK_INPUT(rays_o);
  CHECK_INPUT(rays_d);
  CHECK_INPUT(xyz_min);
  CHECK_INPUT(xyz_max);
  assert(rays_o.dim()==2);
  assert(rays_o.size(1)==3);
  return sample_pts_on_rays_rot_cvrg_cuda(rays_o, rays_d, pnt_xyz, pnt_rmatrix, tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx, units, local_range, xyz_min, xyz_max, near, far, stepdist);
}


std::vector<torch::Tensor> sample_pts_on_rays_ji_cvrg(
        torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor units, torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far, const float stepdist, torch::Tensor shift) {
  CHECK_INPUT(rays_o);
  CHECK_INPUT(rays_d);
  CHECK_INPUT(xyz_min);
  CHECK_INPUT(xyz_max);
  CHECK_INPUT(shift);
  assert(rays_o.dim()==2);
  assert(rays_o.size(1)==3);
  return sample_pts_on_rays_ji_cvrg_cuda(rays_o, rays_d, tensoRF_cvrg_inds, units, xyz_min, xyz_max, near, far, stepdist, shift);
}

std::vector<torch::Tensor> sample_pts_on_rays_sphere_cvrg(
        torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor pnt_xyz, torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx,
        torch::Tensor units, const float radiusl, const float radiush, torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far, const float stepdist) {
  CHECK_INPUT(rays_o);
  CHECK_INPUT(rays_d);
  CHECK_INPUT(xyz_min);
  CHECK_INPUT(xyz_max);
  assert(rays_o.dim()==2);
  assert(rays_o.size(1)==3);
  return sample_pts_on_rays_sphere_cvrg_cuda(rays_o, rays_d, pnt_xyz, tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx, units, radiusl, radiush, xyz_min, xyz_max, near, far, stepdist);
}


std::vector<torch::Tensor> sample_2_tensoRF_cvrg(torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max, torch::Tensor units, torch::Tensor local_range, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds, torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_xyz_recenter, torch::Tensor geo_xyz, const int K, const bool KNN) {
  CHECK_INPUT(xyz_sampled);
  CHECK_INPUT(geo_xyz_recenter);
  CHECK_INPUT(geo_xyz);
  assert(xyz_sampled.dim()==2);
  return sample_2_tensoRF_cvrg_cuda(xyz_sampled, xyz_min, xyz_max, units, local_range, local_dims, tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx, geo_xyz_recenter, geo_xyz, K, KNN);
}


std::vector<torch::Tensor> sample_2_rot_tensoRF_cvrg(torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max, torch::Tensor units, torch::Tensor local_range, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds, torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_rot, torch::Tensor geo_xyz, const int K, const bool KNN) {
  CHECK_INPUT(xyz_sampled);
  CHECK_INPUT(geo_xyz);
  CHECK_INPUT(geo_rot);
  assert(xyz_sampled.dim()==2);
  return sample_2_rot_tensoRF_cvrg_cuda(xyz_sampled, xyz_min, xyz_max, units, local_range, local_dims, tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx, geo_rot, geo_xyz, K, KNN);
}


std::vector<torch::Tensor> sample_2_rotdist_tensoRF_cvrg(torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max, torch::Tensor units, torch::Tensor local_range, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds, torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_rot, torch::Tensor geo_xyz, const int K, const bool KNN) {
  CHECK_INPUT(xyz_sampled);
  CHECK_INPUT(geo_xyz);
  CHECK_INPUT(geo_rot);
  assert(xyz_sampled.dim()==2);
  return sample_2_rotdist_tensoRF_cvrg_cuda(xyz_sampled, xyz_min, xyz_max, units, local_range, local_dims, tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx, geo_rot, geo_xyz, K, KNN);
}


std::vector<torch::Tensor> sample_2_sphere_tensoRF_cvrg(torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max, torch::Tensor units, const float radiusl, const float radiush, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds, torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_xyz, const int K, const bool KNN) {
  CHECK_INPUT(xyz_sampled);
  CHECK_INPUT(geo_xyz);
  assert(xyz_sampled.dim()==2);
  assert(tensoRF_topindx.dim()==2);
  return sample_2_sphere_tensoRF_cvrg_cuda(xyz_sampled, xyz_min, xyz_max, units, radiusl, radiush, local_dims, tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx, geo_xyz, K, KNN);
}


std::vector<torch::Tensor> inds_cvrg(torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max, torch::Tensor units, torch::Tensor local_range, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds, torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_xyz_recenter, torch::Tensor geo_xyz) {
  CHECK_INPUT(xyz_sampled);
  CHECK_INPUT(geo_xyz_recenter);
  CHECK_INPUT(geo_xyz);
  assert(xyz_sampled.dim()==2);
  return inds_cvrg_cuda(xyz_sampled, xyz_min, xyz_max, units, local_range, local_dims, tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx, geo_xyz_recenter, geo_xyz);
}


torch::Tensor filter_xyz_cvrg(torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max, torch::Tensor units, torch::Tensor tensoRF_cvrg_mask) {
  CHECK_INPUT(xyz_sampled);
  CHECK_INPUT(xyz_min);
  CHECK_INPUT(xyz_max);
  CHECK_INPUT(units);
  CHECK_INPUT(tensoRF_cvrg_mask);
  assert(xyz_sampled.dim()==2);
  assert(tensoRF_cvrg_mask.dim()==3);
  return filter_xyz_cvrg_cuda(xyz_sampled, xyz_min, xyz_max, units, tensoRF_cvrg_mask);
}


torch::Tensor filter_xyz_rot_cvrg(torch::Tensor geo_xyz, torch::Tensor geo_rot, torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max, torch::Tensor units, torch::Tensor tensoRF_cvrg_inds, torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor local_range) {
  CHECK_INPUT(xyz_sampled);
  CHECK_INPUT(xyz_min);
  CHECK_INPUT(xyz_max);
  CHECK_INPUT(units);
  CHECK_INPUT(tensoRF_cvrg_inds);
  assert(xyz_sampled.dim()==2);
  return filter_xyz_rot_cvrg_cuda(geo_xyz, geo_rot, xyz_sampled, xyz_min, xyz_max, units, tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx, local_range);
}


torch::Tensor filter_xyz_sphere_cvrg(torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max, torch::Tensor units, const float radiusl, const float radiush, torch::Tensor tensoRF_cvrg_inds, torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor pnt_xyz) {
  CHECK_INPUT(xyz_sampled);
  CHECK_INPUT(xyz_min);
  CHECK_INPUT(xyz_max);
  CHECK_INPUT(units);
  CHECK_INPUT(tensoRF_cvrg_inds);
  CHECK_INPUT(pnt_xyz);
  assert(xyz_sampled.dim()==2);
  assert(tensoRF_cvrg_inds.dim()==3);
  assert(pnt_xyz.dim()==2);
  return filter_xyz_sphere_cvrg_cuda(xyz_sampled, xyz_min, xyz_max, units, radiusl, radiush, tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx, pnt_xyz);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("build_tensoRF_map", &build_tensoRF_map, "build tensoRF indices map");
  m.def("build_sphere_tensoRF_map", &build_sphere_tensoRF_map, "build spherical tensoRF indices map");
  m.def("filter_ray_by_cvrg", &filter_ray_by_cvrg, "filter ray if within coverage");
  m.def("sample_pts_on_rays_cvrg", &sample_pts_on_rays_cvrg, "Sample points on rays within coverage");
  m.def("sample_pts_on_rays_rot_cvrg", &sample_pts_on_rays_rot_cvrg, "Sample points on rays within coverage");
  m.def("sample_pts_on_rays_ji_cvrg", &sample_pts_on_rays_ji_cvrg, "Sample points on rays within coverage");
  m.def("sample_pts_on_rays_dist_cvrg", &sample_pts_on_rays_dist_cvrg, "Sample points on rays within coverage");
  m.def("sample_pts_on_rays_sphere_cvrg", &sample_pts_on_rays_sphere_cvrg, "Sample points on rays within spherical coverage");
  m.def("sample_2_rot_tensoRF_cvrg", &sample_2_rot_tensoRF_cvrg, "Sampled points to get torsoRF");
  m.def("sample_2_rotdist_tensoRF_cvrg", &sample_2_rotdist_tensoRF_cvrg, "Sampled points to get torsoRF");
  m.def("sample_2_tensoRF_cvrg", &sample_2_tensoRF_cvrg, "Sampled points to get torsoRF");
  m.def("sample_2_sphere_tensoRF_cvrg", &sample_2_sphere_tensoRF_cvrg, "Sampled points to get torsoRF");
  m.def("inds_cvrg", &inds_cvrg, "Sampled points to get torsoRF");
  m.def("filter_xyz_cvrg", &filter_xyz_cvrg, "Sampled points to get torsoRF");
  m.def("filter_xyz_rot_cvrg", &filter_xyz_rot_cvrg, "Sampled points to get torsoRF");
  m.def("filter_xyz_sphere_cvrg", &filter_xyz_sphere_cvrg, "Sampled points to get spherical torsoRF");
}


