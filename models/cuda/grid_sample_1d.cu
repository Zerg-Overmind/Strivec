#include <torch/extension.h>
#include <math.h>       /* atan2 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>


#include <vector>




template <typename scalar_t>
__global__ void find_tensoRF_and_repos_masked_cuda_kernel(
        scalar_t* __restrict__ xyz_sampled,
        scalar_t* __restrict__ geo_xyz,
        scalar_t* __restrict__ xyz_min,
        int16_t* __restrict__ final_tensoRF_id,
        int64_t* __restrict__ local_dims,
        int16_t* __restrict__ local_gindx_s,
        int16_t* __restrict__ local_gindx_l,
        scalar_t* __restrict__ local_gweight_s,
        scalar_t* __restrict__ local_gweight_l,
        scalar_t* __restrict__ local_kernel_dist,
        scalar_t* __restrict__ local_norm_xyz,
        scalar_t* __restrict__ units,
        scalar_t* __restrict__ lvl_units,
        scalar_t* __restrict__ local_range,
        int32_t* __restrict__ tensoRF_cvrg_inds,
        int8_t* __restrict__ tensoRF_count,
        int16_t* __restrict__ tensoRF_topindx,
        bool* __restrict__ tensoRF_mask,
        const int gridX,
        const int gridY,
        const int gridZ,
        const int gridYZ,
        const int n_sampleK,
        const int K,
        const int maxK
        ) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n_sampleK) {
    const int sample_id = idx / K;
    const int kid = idx % K;
    const int xyzshift = sample_id * 3;
    const float px = xyz_sampled[xyzshift];
    const float py = xyz_sampled[xyzshift + 1];
    const float pz = xyz_sampled[xyzshift + 2];

//    const int indx = min(max((int)((px - xyz_min[0]) / units[0]), 0), gridX-1);
//    const int indy = min(max((int)((py - xyz_min[1]) / units[1]), 0), gridY-1);
//    const int indz = min(max((int)((pz - xyz_min[2]) / units[2]), 0), gridZ-1);
    const int indx = (int)((px - xyz_min[0]) / units[0]);
    const int indy = (int)((py - xyz_min[1]) / units[1]);
    const int indz = (int)((pz - xyz_min[2]) / units[2]);

    const int inds = indx * gridYZ + indy * gridZ + indz;
    const int cvrg_id = tensoRF_cvrg_inds[inds];
    // printf("tensoRF_count[cvrg_id] %d %d %d %d;   ", kid, (int)tensoRF_count[cvrg_id], K, cvrg_id);
    if (kid < tensoRF_count[cvrg_id]){
        const int i_tid = tensoRF_topindx[cvrg_id * maxK + kid];
        const int offset_t = i_tid * 3;

        const float rel_x = px - geo_xyz[offset_t];
        const float rel_y = py - geo_xyz[offset_t+1];
        const float rel_z = pz - geo_xyz[offset_t+2];

        if (abs(rel_x) <= local_range[0] && abs(rel_y) <= local_range[1] && abs(rel_z) <= local_range[2]){
            const int offset_p = idx * 3;
            tensoRF_mask[idx] = true;
            final_tensoRF_id[idx] = i_tid;
            local_kernel_dist[idx] = sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z);

            const float softindx = min(max((rel_x + local_range[0]) / lvl_units[0], 0.0), (float)local_dims[0]);
            const float softindy = min(max((rel_y + local_range[1]) / lvl_units[1], 0.0), (float)local_dims[1]);
            const float softindz = min(max((rel_z + local_range[2]) / lvl_units[2], 0.0), (float)local_dims[2]);

            local_norm_xyz[offset_p  ] = rel_x / local_range[0];
            local_norm_xyz[offset_p+1] = rel_y / local_range[1];
            local_norm_xyz[offset_p+2] = rel_z / local_range[2];

            const int indlx = min((int)softindx, (int)local_dims[0]-1);
            const int indly = min((int)softindy, (int)local_dims[1]-1);
            const int indlz = min((int)softindz, (int)local_dims[2]-1);

            const float res_x = softindx - indlx;
            const float res_y = softindy - indly;
            const float res_z = softindz - indlz;

            local_gweight_s[offset_p  ] = 1 - res_x;
            local_gweight_s[offset_p+1] = 1 - res_y;
            local_gweight_s[offset_p+2] = 1 - res_z;
            local_gweight_l[offset_p  ] = res_x;
            local_gweight_l[offset_p+1] = res_y;
            local_gweight_l[offset_p+2] = res_z;

            local_gindx_s[offset_p  ] = indlx;
            local_gindx_s[offset_p+1] = indly;
            local_gindx_s[offset_p+2] = indlz;
            local_gindx_l[offset_p  ] = indlx + 1;
            local_gindx_l[offset_p+1] = indly + 1;
            local_gindx_l[offset_p+2] = indlz + 1;
        }
    }
  }
}




__global__ void __fill_agg_id(
        int32_t* __restrict__ cvrg_count,
        int32_t* __restrict__ cvrg_cumsum,
        int64_t* __restrict__ final_agg_id,
        const int n_sample) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<n_sample && cvrg_count[idx] > 0) {
//        if (cvrg_cumsum[n_sample] == 0) {
//            printf("cvrg_cumsum !!!!!!!", cvrg_cumsum[n_sample]);
//        }
        const int cur_agg_start = (idx!=0) ? cvrg_cumsum[idx-1] : 0;
        const int cur_agg_end = cvrg_cumsum[idx];
        // if (cur_agg_start==cur_agg_end) printf(" cur_agg_start=cur_agg_end %d ", cur_agg_end);
        for (int i = cur_agg_start; i < cur_agg_end; i++){
            final_agg_id[i] = idx;
        }
    }
}



__global__ void cal_intrp(
        const int res,
        const int thead_total,
        const int cvrg_len,
        const int num_component,
        float* __restrict__ plane,
        float* __restrict__ line1,
        float* __restrict__ line2,
        float* __restrict__ line3,
        int16_t* __restrict__ local_gindx_s,
        int16_t* __restrict__ local_gindx_l,
        int16_t* __restrict__ final_tensoRF_id,
        float* __restrict__ local_gweight_s,
        float* __restrict__ local_gweight_l,
        float* __restrict__ plane_out,
        float* __restrict__ line_out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<thead_total) {
        const int sample_id = idx / num_component;
        const int comp_id = idx % num_component;
        const int tensorf_id = final_tensoRF_id[sample_id];
        const int plane_shift = comp_id * res * res; // + 1 * num_component * res * res;
        const int line_shift = (tensorf_id * num_component + comp_id) * res;
        const int inds_shift = sample_id * 3;

        const int local_gindx_s_x = local_gindx_s[inds_shift];
        const int local_gindx_s_y = local_gindx_s[inds_shift+1];
        const int local_gindx_s_z = local_gindx_s[inds_shift+2];
        const int local_gindx_l_x = local_gindx_l[inds_shift];
        const int local_gindx_l_y = local_gindx_l[inds_shift+1];
        const int local_gindx_l_z = local_gindx_l[inds_shift+2];

        const float local_gweight_s_x = local_gweight_s[inds_shift];
        const float local_gweight_s_y = local_gweight_s[inds_shift+1];
        const float local_gweight_s_z = local_gweight_s[inds_shift+2];
        const float local_gweight_l_x = local_gweight_l[inds_shift];
        const float local_gweight_l_y = local_gweight_l[inds_shift+1];
        const float local_gweight_l_z = local_gweight_l[inds_shift+2];

        const int out_shift = (sample_id * num_component + comp_id) * 3;

        plane_out[out_shift] = plane[plane_shift + local_gindx_s_y * res + local_gindx_s_z] * local_gweight_s_y * local_gweight_s_z + plane[plane_shift + local_gindx_s_y * res + local_gindx_l_z] * local_gweight_s_y * local_gweight_l_z + plane[plane_shift + local_gindx_l_y * res + local_gindx_s_z] * local_gweight_l_y * local_gweight_s_z + plane[plane_shift + local_gindx_l_y * res + local_gindx_l_z] * local_gweight_l_y * local_gweight_l_z;
        line_out[out_shift] = line1[line_shift + local_gindx_s_x] * local_gweight_s_x + line1[line_shift + local_gindx_l_x] * local_gweight_l_x;

        plane_out[out_shift + 1] = plane[plane_shift + local_gindx_s_z * res + local_gindx_s_x] * local_gweight_s_x * local_gweight_s_z + plane[plane_shift + local_gindx_l_z * res + local_gindx_s_x] * local_gweight_s_x * local_gweight_l_z + plane[plane_shift + local_gindx_s_z * res + local_gindx_l_x] * local_gweight_l_x * local_gweight_s_z + plane[plane_shift + local_gindx_l_z * res + local_gindx_l_x] * local_gweight_l_x * local_gweight_l_z;
        line_out[out_shift + 1] = line2[line_shift + local_gindx_s_y] * local_gweight_s_y + line2[line_shift + local_gindx_l_y] * local_gweight_l_y;

        plane_out[out_shift + 2] = plane[plane_shift + local_gindx_s_x * res + local_gindx_s_y] * local_gweight_s_x * local_gweight_s_y + plane[plane_shift + local_gindx_s_x * res + local_gindx_l_y] * local_gweight_s_x * local_gweight_l_y + plane[plane_shift + local_gindx_l_x * res + local_gindx_s_y] * local_gweight_l_x * local_gweight_s_y + plane[plane_shift + local_gindx_l_x * res + local_gindx_l_y] * local_gweight_l_x * local_gweight_l_y;
        line_out[out_shift + 2] = line3[line_shift + local_gindx_s_z] * local_gweight_s_z + line3[line_shift + local_gindx_l_z] * local_gweight_l_z;
    }
}



std::vector<torch::Tensor> grid_sample_from_tensoRF_cuda(
        torch::Tensor plane, torch::Tensor line1, torch::Tensor line2, torch::Tensor line3,
        torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max, torch::Tensor units, torch::Tensor lvl_units, torch::Tensor local_range, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_xyz, const int K, const bool KNN) {

  const int threads = 256;
  const int n_pts = geo_xyz.size(0);
  const int n_sample = xyz_sampled.size(0);
  const int maxK = tensoRF_topindx.size(1);
  const int num_all_cvrg = tensoRF_count.size(0);
  const int gridX = tensoRF_cvrg_inds.size(0);
  const int gridY = tensoRF_cvrg_inds.size(1);
  const int gridZ = tensoRF_cvrg_inds.size(2);
  const int gridYZ = gridZ * gridY;
  const int num_component = line1.size(1);

  const int n_sampleK = n_sample * K;
  auto local_gindx_s = torch::empty({n_sampleK, 3}, torch::dtype(torch::kInt16).device(torch::kCUDA));
  auto local_gindx_l = torch::empty({n_sampleK, 3}, torch::dtype(torch::kInt16).device(torch::kCUDA));
  auto local_gweight_s = torch::empty({n_sampleK, 3}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
  auto local_gweight_l = torch::empty({n_sampleK, 3}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
  auto local_norm_xyz = torch::empty({n_sampleK, 3}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
  auto local_kernel_dist = torch::empty({n_sampleK}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
  auto final_tensoRF_id = torch::empty({n_sampleK}, torch::dtype(torch::kInt16).device(torch::kCUDA));
  auto tensoRF_mask = torch::zeros({n_sample, K}, torch::dtype(torch::kBool).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(xyz_sampled.type(), "find_tensoRF_and_repos_masked_cuda", ([&] {
      find_tensoRF_and_repos_masked_cuda_kernel<scalar_t><<<(n_sampleK+threads-1)/threads, threads>>>(
        xyz_sampled.data<scalar_t>(),
        geo_xyz.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        final_tensoRF_id.data<int16_t>(),
        local_dims.data<int64_t>(),
        local_gindx_s.data<int16_t>(),
        local_gindx_l.data<int16_t>(),
        local_gweight_s.data<scalar_t>(),
        local_gweight_l.data<scalar_t>(),
        local_kernel_dist.data<scalar_t>(),
        local_norm_xyz.data<scalar_t>(),
        units.data<scalar_t>(),
        lvl_units.data<scalar_t>(),
        local_range.data<scalar_t>(),
        tensoRF_cvrg_inds.data<int32_t>(),
        tensoRF_count.data<int8_t>(),
        tensoRF_topindx.data<int16_t>(),
        tensoRF_mask.data<bool>(),
        gridX,
        gridY,
        gridZ,
        gridYZ,
        n_sampleK,
        K,
        maxK);
  }));
//  torch::cuda::synchronize();

  auto cvrg_count = tensoRF_mask.sum(1, false, torch::kInt32);
  const int cvrg_len = cvrg_count.sum().item<int>();
  auto final_agg_id = torch::empty({cvrg_len}, torch::dtype(torch::kInt64).device(torch::kCUDA));
//  torch::cuda::synchronize();
  auto plane_out = torch::empty({cvrg_len, num_component, 3}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
  auto line_out = torch::empty({cvrg_len, num_component, 3}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
  if (cvrg_len > 0){
      auto cvrg_cumsum = cvrg_count.cumsum(0, torch::kInt32);
      __fill_agg_id<<<(n_sample+threads-1)/threads, threads>>>(cvrg_count.data<int32_t>(), cvrg_cumsum.data<int32_t>(), final_agg_id.data<int64_t>(), n_sample);
      tensoRF_mask = tensoRF_mask.reshape(-1);
      local_gindx_s = local_gindx_s.index({tensoRF_mask, "..."});
      local_gindx_l = local_gindx_l.index({tensoRF_mask, "..."});
      local_gweight_s = local_gweight_s.index({tensoRF_mask, "..."});
      local_gweight_l = local_gweight_l.index({tensoRF_mask, "..."});
      local_kernel_dist = local_kernel_dist.index({tensoRF_mask, "..."});
      final_tensoRF_id = final_tensoRF_id.index({tensoRF_mask, "..."});
      local_norm_xyz = local_norm_xyz.index({tensoRF_mask, "..."});
      const int thead_total = cvrg_len*num_component;
      const int res = line1.size(2);
      cal_intrp<<<(thead_total+threads-1)/threads, threads>>>(res, thead_total, cvrg_len, num_component, plane.data<float>(), line1.data<float>(), line2.data<float>(), line3.data<float>(), local_gindx_s.data<int16_t>(), local_gindx_l.data<int16_t>(), final_tensoRF_id.data<int16_t>(), local_gweight_s.data<float>(), local_gweight_l.data<float>(), plane_out.data<float>(), line_out.data<float>());
//      torch::cuda::synchronize();
  } else {
      local_gindx_s = torch::empty({0, 3}, torch::dtype(torch::kInt16).device(torch::kCUDA));
      local_gindx_l = torch::empty({0, 3}, torch::dtype(torch::kInt16).device(torch::kCUDA));
      local_gweight_s = torch::empty({0, 3}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
      local_gweight_l = torch::empty({0, 3}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
      local_norm_xyz = torch::empty({0, 3}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
      local_kernel_dist = torch::empty({0}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
      final_tensoRF_id = torch::empty({0}, torch::dtype(torch::kInt16).device(torch::kCUDA));
  }
  return {plane_out, line_out, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, final_tensoRF_id, final_agg_id, local_norm_xyz};
}



__global__ void cal_intrp_back(
        const int res,
        const int thead_total,
        const int cvrg_len,
        const int num_component,
        float* __restrict__ grad_planeout,
        float* __restrict__ grad_lineout,
        int16_t* __restrict__ local_gindx_s,
        int16_t* __restrict__ local_gindx_l,
        int16_t* __restrict__ final_tensoRF_id,
        float* __restrict__ local_gweight_s,
        float* __restrict__ local_gweight_l,
        float* __restrict__ grad_plane,
        float* __restrict__ grad_line_1,
        float* __restrict__ grad_line_2,
        float* __restrict__ grad_line_3
        ) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<thead_total) {
        const int sample_id = idx / num_component;
        const int comp_id = idx % num_component;
//        const int sample_id = idx % cvrg_len;
//        const int comp_id = idx / cvrg_len;
        const int tensorf_id = final_tensoRF_id[sample_id];
        const int plane_shift = comp_id * res * res; // + 1 * num_component * res * res ;
        const int line_shift = (tensorf_id * num_component + comp_id) * res;
        const int inds_shift = sample_id * 3;
        const int local_gindx_s_x = local_gindx_s[inds_shift];
        const int local_gindx_s_y = local_gindx_s[inds_shift+1];
        const int local_gindx_s_z = local_gindx_s[inds_shift+2];
        const int local_gindx_l_x = local_gindx_l[inds_shift];
        const int local_gindx_l_y = local_gindx_l[inds_shift+1];
        const int local_gindx_l_z = local_gindx_l[inds_shift+2];

        const float local_gweight_s_x = local_gweight_s[inds_shift];
        const float local_gweight_s_y = local_gweight_s[inds_shift+1];
        const float local_gweight_s_z = local_gweight_s[inds_shift+2];
        const float local_gweight_l_x = local_gweight_l[inds_shift];
        const float local_gweight_l_y = local_gweight_l[inds_shift+1];
        const float local_gweight_l_z = local_gweight_l[inds_shift+2];
//        const int realid = sample_id * num_component + comp_id;
        const int grad_shift = idx * 3;
        float* plane_shifted = grad_plane + plane_shift;

        // y z
        float plane_der_1 = grad_planeout[grad_shift];
        float* line_shifted_1 = grad_line_1 + line_shift;
//        printf("plane_shifted %f ;", &plane_shifted);
        atomicAdd(plane_shifted + local_gindx_s_y * res + local_gindx_s_z, plane_der_1 * local_gweight_s_y * local_gweight_s_z);

        atomicAdd(plane_shifted + local_gindx_s_y * res + local_gindx_l_z, plane_der_1 * local_gweight_s_y * local_gweight_l_z);

        atomicAdd(plane_shifted + local_gindx_l_y * res + local_gindx_s_z, plane_der_1 * local_gweight_l_y * local_gweight_s_z);

        atomicAdd(plane_shifted + local_gindx_l_y * res + local_gindx_l_z, plane_der_1 * local_gweight_l_y * local_gweight_l_z);

        float line_der_1 = grad_lineout[grad_shift];
        atomicAdd(line_shifted_1 + local_gindx_s_x, line_der_1 * local_gweight_s_x);
        atomicAdd(line_shifted_1 + local_gindx_l_x, line_der_1 * local_gweight_l_x);

        // x z
        float plane_der_2 = grad_planeout[grad_shift + 1];
        float* line_shifted_2 = grad_line_2 + line_shift;
        atomicAdd(plane_shifted + local_gindx_s_z * res + local_gindx_s_x, plane_der_2 * local_gweight_s_x * local_gweight_s_z);

        atomicAdd(plane_shifted + local_gindx_l_z * res + local_gindx_s_x, plane_der_2 * local_gweight_s_x * local_gweight_l_z);

        atomicAdd(plane_shifted + local_gindx_s_z * res + local_gindx_l_x, plane_der_2 * local_gweight_l_x * local_gweight_s_z);

        atomicAdd(plane_shifted + local_gindx_l_z * res + local_gindx_l_x, plane_der_2 * local_gweight_l_x * local_gweight_l_z);

        float line_der_2 = grad_lineout[grad_shift + 1];

        atomicAdd(line_shifted_2 + local_gindx_s_y, line_der_2 * local_gweight_s_y);
        atomicAdd(line_shifted_2 + local_gindx_l_y, line_der_2 * local_gweight_l_y);

        float plane_der_3 = grad_planeout[grad_shift + 2];
        float* line_shifted_3 = grad_line_3 + line_shift;
//        printf("plane_shifted %f ;", &plane_shifted);
        atomicAdd(plane_shifted + local_gindx_s_x * res + local_gindx_s_y, plane_der_3 * local_gweight_s_x * local_gweight_s_y);

        atomicAdd(plane_shifted + local_gindx_s_x * res + local_gindx_l_y, plane_der_3 * local_gweight_s_x * local_gweight_l_y);

        atomicAdd(plane_shifted + local_gindx_l_x * res + local_gindx_s_y, plane_der_3 * local_gweight_l_x * local_gweight_s_y);

        atomicAdd(plane_shifted + local_gindx_l_x * res + local_gindx_l_y, plane_der_3 * local_gweight_l_x * local_gweight_l_y);

        float line_der_3 = grad_lineout[grad_shift + 2];
        atomicAdd(line_shifted_3 + local_gindx_s_z, line_der_3 * local_gweight_s_z);
        atomicAdd(line_shifted_3 + local_gindx_l_z, line_der_3 * local_gweight_l_z);

    }
}


std::vector<torch::Tensor> grid_sample_from_tensoRF_backward_cuda(torch::Tensor local_gindx_s, torch::Tensor local_gindx_l, torch::Tensor local_gweight_s, torch::Tensor local_gweight_l, torch::Tensor final_tensoRF_id, torch::Tensor grad_planeout, torch::Tensor grad_lineout, int planesurf_num, int linesurf_num, int num_component, int res) {
    const int threads = 256;
    auto grad_plane = torch::zeros({planesurf_num, num_component, res, res}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto grad_line1 = torch::zeros({linesurf_num, num_component, res}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto grad_line2 = torch::zeros({linesurf_num, num_component, res}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    auto grad_line3 = torch::zeros({linesurf_num, num_component, res}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    const int cvrg_len = local_gindx_s.size(0);
    const int thead_total = cvrg_len * num_component;
//    torch::cuda::synchronize();
    cal_intrp_back<<<(thead_total+threads-1)/threads, threads>>>(res, thead_total, cvrg_len, num_component, grad_planeout.data<float>(), grad_lineout.data<float>(), local_gindx_s.data<int16_t>(), local_gindx_l.data<int16_t>(), final_tensoRF_id.data<int16_t>(), local_gweight_s.data<float>(), local_gweight_l.data<float>(), grad_plane.data<float>(), grad_line1.data<float>(), grad_line2.data<float>(), grad_line3.data<float>());
//    torch::cuda::synchronize();

  return {grad_plane, grad_line1, grad_line2, grad_line3};
}



std::vector<torch::Tensor> cal_w_inds_cuda(
        torch::Tensor plane, torch::Tensor line1, torch::Tensor line2, torch::Tensor line3, torch::Tensor local_gindx_s, torch::Tensor local_gindx_l, torch::Tensor local_gweight_s, torch::Tensor local_gweight_l, torch::Tensor final_tensoRF_id
        ) {
    const int cvrg_len = local_gindx_s.size(0);
    const int num_component = line1.size(1);
    const int threads = 256;
    const int thead_total = cvrg_len * num_component;
    const int res = line1.size(2);
    auto plane_out = torch::zeros({cvrg_len, num_component, 3}, torch::dtype(line1.dtype()).device(torch::kCUDA));
    auto line_out = torch::zeros({cvrg_len, num_component, 3}, torch::dtype(line1.dtype()).device(torch::kCUDA));
    cal_intrp<<<(thead_total+threads-1)/threads, threads>>>(res, thead_total, cvrg_len, num_component, plane.data<float>(), line1.data<float>(), line2.data<float>(), line3.data<float>(), local_gindx_s.data<int16_t>(), local_gindx_l.data<int16_t>(), final_tensoRF_id.data<int16_t>(), local_gweight_s.data<float>(), local_gweight_l.data<float>(), plane_out.data<float>(), line_out.data<float>());

    return {plane_out, line_out};
}