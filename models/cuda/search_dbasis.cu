#include <torch/extension.h>
#include <math.h>       /* atan2 */
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>


#include <vector>

/*
   Points sampling helper functions.
 */

template <typename scalar_t>
__global__ void mask_tensoRF(
        scalar_t* __restrict__ xyz_sampled,
        int8_t* __restrict__ xyz_inbbox,
        scalar_t* __restrict__ geo_xyz,
        scalar_t* __restrict__ geo_rot,
        scalar_t* __restrict__ xyz_min,
        int16_t* __restrict__ local_dims,
        scalar_t* __restrict__ units,
        scalar_t* __restrict__ local_range,
        int32_t* __restrict__ tensoRF_cvrg_inds,
        int8_t* __restrict__ tensoRF_count,
        int16_t* __restrict__ tensoRF_topindx,
        bool* __restrict__ tensoRF_mask,
        const int gridX,
        const int gridY,
        const int gridZ,
        const int n_rays,
        const int n_sample,
        const int maxK,
        const int K,
        const int ord_thresh
        ) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n_rays) {
    int sample_shift, sample_xyzshift, indx, indy, indz, inds, cvrg_id, tid, tf_count, offset_t, offset_r, rel_x, rel_y, rel_z, rx, ry, rz;
    float px, py, pz;
    int ord_count = 0;
    for (int i = 0; i < n_sample; i++){
        sample_shift = idx * n_sample + i;
        if(xyz_inbbox[sample_shift] < 1) continue;
        sample_xyzshift = sample_shift * 3;
        px = xyz_sampled[sample_xyzshift];
        py = xyz_sampled[sample_xyzshift + 1];
        pz = xyz_sampled[sample_xyzshift + 2];

        indx = min(gridX-1, (int)((px - xyz_min[0]) / units[0]));
        indy = min(gridY-1, (int)((py - xyz_min[1]) / units[1]));
        indz = min(gridZ-1, (int)((pz - xyz_min[2]) / units[2]));
        inds = indx * gridY * gridZ + indy * gridZ + indz;
        cvrg_id = tensoRF_cvrg_inds[inds];
        if (cvrg_id >= 0){
            tf_count = 0;
            for(int kid=0; kid < tensoRF_count[cvrg_id]; kid++) {
                tid = tensoRF_topindx[cvrg_id * maxK + kid];
                offset_t = tid * 3;
                offset_r = tid * 9;

                rel_x = px - geo_xyz[offset_t];
                rel_y = py - geo_xyz[offset_t+1];
                rel_z = pz - geo_xyz[offset_t+2];

                rx = rel_x * geo_rot[offset_r] + rel_y * geo_rot[offset_r+3]  + rel_z * geo_rot[offset_r+6];

                ry = rel_x * geo_rot[offset_r+1] + rel_y * geo_rot[offset_r+4]  + rel_z * geo_rot[offset_r+7];

                rz = rel_x * geo_rot[offset_r+2] + rel_y * geo_rot[offset_r+5]  + rel_z * geo_rot[offset_r+8];

                if (abs(rx) <= local_range[offset_t] && abs(ry) <= local_range[offset_t+1] && abs(rz) <= local_range[offset_t+2]){
                    tf_count++;
                    tensoRF_mask[tid] = true;
                }
            }
            if (tf_count > 0) ord_count++;
            if (ord_count >= ord_thresh) break;
        }
    }
  }
}


template <typename scalar_t>
__global__ void find_pca_tensoRF_and_repos_cuda_kernel(
        scalar_t* __restrict__ xyz_sampled,
        scalar_t* __restrict__ geo_xyz,
        scalar_t* __restrict__ geo_rot,
        scalar_t* __restrict__ xyz_min,
        int64_t* __restrict__ final_tensoRF_id,
        int16_t* __restrict__ local_dims,
        int64_t* __restrict__ local_gindx_s,
        int64_t* __restrict__ local_gindx_l,
        scalar_t* __restrict__ local_gweight_s,
        scalar_t* __restrict__ local_gweight_l,
        scalar_t* __restrict__ local_kernel_dist,
        scalar_t* __restrict__ units,
        scalar_t* __restrict__ local_range,
        int32_t* __restrict__ tensoRF_cvrg_inds,
        int8_t* __restrict__ tensoRF_count,
        int16_t* __restrict__ tensoRF_topindx,
        int32_t* __restrict__ dim_cumsum_counter,
        bool* __restrict__ tensoRF_mask,
        const int gridX,
        const int gridY,
        const int gridZ,
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
    const int indx = min(gridX-1, (int)((px - xyz_min[0]) / units[0]));
    const int indy = min(gridY-1, (int)((py - xyz_min[1]) / units[1]));
    const int indz = min(gridZ-1, (int)((pz - xyz_min[2]) / units[2]));
    const int inds = indx * gridY * gridZ + indy * gridZ + indz;
    const int cvrg_id = tensoRF_cvrg_inds[inds];
    // printf("tensoRF_count[cvrg_id] %d %d %d %d;   ", kid, (int)tensoRF_count[cvrg_id], K, cvrg_id);
    if (kid < tensoRF_count[cvrg_id]){
        const int i_tid = tensoRF_topindx[cvrg_id * maxK + kid];
        final_tensoRF_id[idx] = i_tid;
        const int offset_t = i_tid * 3;
        const int offset_r = i_tid * 9;

        const float rel_x = px - geo_xyz[offset_t];
        const float rel_y = py - geo_xyz[offset_t+1];
        const float rel_z = pz - geo_xyz[offset_t+2];

        const float rx = rel_x * geo_rot[offset_r] + rel_y * geo_rot[offset_r+3]  + rel_z * geo_rot[offset_r+6];

        const float ry = rel_x * geo_rot[offset_r+1] + rel_y * geo_rot[offset_r+4]  + rel_z * geo_rot[offset_r+7];

        const float rz = rel_x * geo_rot[offset_r+2] + rel_y * geo_rot[offset_r+5]  + rel_z * geo_rot[offset_r+8];

        if (abs(rx) <= local_range[offset_t] && abs(ry) <= local_range[offset_t+1] && abs(rz) <= local_range[offset_t+2]){
            const int offset_p = idx * 3;
            tensoRF_mask[idx] = true;
            local_kernel_dist[idx] = sqrt(rx * rx + ry * ry + rz * rz);

            const float softindx = (rx + local_range[offset_t]) / units[0];
            const float softindy = (ry + local_range[offset_t+1]) / units[1];
            const float softindz = (rz + local_range[offset_t+2]) / units[2];

            int indlx = min(max((int)softindx, 0), local_dims[offset_t]-1);
            int indly = min(max((int)softindy, 0), local_dims[offset_t+1]-1);
            int indlz = min(max((int)softindz, 0), local_dims[offset_t+2]-1);

            const float res_x = softindx - indlx;
            const float res_y = softindy - indly;
            const float res_z = softindz - indlz;

            indlx = indlx + dim_cumsum_counter[offset_t];
            indly = indly + dim_cumsum_counter[offset_t + 1];
            indlz = indlz + dim_cumsum_counter[offset_t + 2];

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

template <typename scalar_t>
__global__ void count_tensoRF_cvrg_cuda_kernel(
        scalar_t* __restrict__ xyz_sampled,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ units,
        int8_t* __restrict__ tensoRF_count,
        int32_t* __restrict__ tensoRF_cvrg_inds,
        int32_t* __restrict__ cvrg_inds,
        int32_t* __restrict__ cvrg_count,
        const int gridYZ,
        const int gridZ,
        const int n_sample) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n_sample) {
     const int xyzshift = idx * 3;
     const int indx = (xyz_sampled[xyzshift] - xyz_min[0]) / units[0];
     const int indy = (xyz_sampled[xyzshift + 1] - xyz_min[1]) / units[1];
     const int indz = (xyz_sampled[xyzshift + 2] - xyz_min[2]) / units[2];

     const int inds = indx * gridYZ + indy * gridZ + indz;
     const int cvrg_id = tensoRF_cvrg_inds[inds];
     if (cvrg_id >= 0){
        cvrg_inds[idx] = cvrg_id;
        cvrg_count[idx] = tensoRF_count[cvrg_id];
     }
  }
}




__global__ void __fill_agg_id(
        int64_t* __restrict__ cvrg_count,
        int64_t* __restrict__ cvrg_cumsum,
        int64_t* __restrict__ final_agg_id,
        const int n_sample) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<n_sample) {
        const int cur_agg_start = (idx!=0) ? cvrg_cumsum[idx-1] : 0;
        const int cur_agg_end = cvrg_cumsum[idx];
        // if (cur_agg_start==cur_agg_end) printf(" cur_agg_start=cur_agg_end %d ", cur_agg_end);
        for (int i = cur_agg_start; i < cur_agg_end; i++){
            final_agg_id[i] = idx;
        }
    }
}


template <typename scalar_t>
__global__ void get_geo_inds_cuda_kernel(
        scalar_t* __restrict__ local_range,
        int64_t* __restrict__ gridSize,
        scalar_t* __restrict__ units,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        scalar_t* __restrict__ pnt_xyz,
        int32_t* __restrict__ tensoRF_cvrg_inds,
        const int n_pts
        ) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<n_pts) {
     const int i_shift = idx * 3;
     const float px = pnt_xyz[i_shift];
     const float py = pnt_xyz[i_shift+1];
     const float pz = pnt_xyz[i_shift+2];
     
     // the xyz index of every tenserf center in unit
     const int xind = (px - xyz_min[0]) / units[0]; 
     const int yind = (py - xyz_min[1]) / units[1];
     const int zind = (pz - xyz_min[2]) / units[2];
     const int gx = gridSize[0];
     const int gy = gridSize[1];
     const int gz = gridSize[2];
     const int lx = ceil(local_range[0] / units[0]);
     const int ly = ceil(local_range[1] / units[0]);
     const int lz = ceil(local_range[2] / units[0]);
     
     // shift between the actual local range and the predefined local range in unit
     const int xmin = max(min(xind-lx, gx), 0);
     const int xmax = max(min(xind+lx+1, gx), 0);
     const int ymin = max(min(yind-ly, gy), 0);
     const int ymax = max(min(yind+ly+1, gy), 0);
     const int zmin = max(min(zind-lz, gz), 0);
     const int zmax = max(min(zind+lz+1, gz), 0);
     for (int i = xmin; i < xmax; i++){
        int shiftx = i * gy * gz; // shift for i_th unit
        for (int j = ymin; j < ymax; j++){
            int shifty = j * gz;
            for (int k = zmin; k < zmax; k++){
                if (min(abs(px - xyz_min[0] - i * units[0]), abs(px - xyz_min[0] - (i+1) * units[0])) < local_range[0] && min(abs(py - xyz_min[1] - j * units[1]), abs(py - xyz_min[1] - (j+1) * units[1])) < local_range[1] && min(abs(pz - xyz_min[2] - k * units[2]), abs(pz - xyz_min[2] - (k+1) * units[2])) < local_range[2]) {
                    tensoRF_cvrg_inds[shiftx + shifty + k] = 1;
                }
            }
        }
     }
  }
}


template <typename scalar_t>
__global__ void get_cubic_geo_inds_cuda_kernel(
        scalar_t* __restrict__ local_range,
        int64_t* __restrict__ gridSize,
        scalar_t* __restrict__ units,
        scalar_t* __restrict__ radius,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        scalar_t* __restrict__ pnt_xyz,
        scalar_t* __restrict__ geo_rot,
        int32_t* __restrict__ tensoRF_cvrg_inds,
        const int n_pts
        ) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<n_pts) {
     const int i_shift = idx * 3;
     const int i_shift_R = i_shift * 3;
     const float px = pnt_xyz[i_shift];
     const float py = pnt_xyz[i_shift+1];
     const float pz = pnt_xyz[i_shift+2];
     float local_range_every_0 = local_range[i_shift];
     float local_range_every_1 = local_range[i_shift+1];
     float local_range_every_2 = local_range[i_shift+2];
     float r_k = radius[idx];
     const int gx = gridSize[0];
     const int gy = gridSize[1];
     const int gz = gridSize[2];

     const int linds_x = (px - r_k - xyz_min[0]) / units[0];
     const int hinds_x = (px + r_k - xyz_min[0]) / units[0];
     const int linds_y = (py - r_k - xyz_min[1]) / units[0];
     const int hinds_y = (py + r_k - xyz_min[1]) / units[0];
     const int linds_z = (pz - r_k - xyz_min[2]) / units[0];
     const int hinds_z = (pz + r_k - xyz_min[2]) / units[0];

     const int xmin = max(min(linds_x, gx), 0);
     const int xmax = max(min(hinds_x+1, gx), 0);
     const int ymin = max(min(linds_y, gy), 0);
     const int ymax = max(min(hinds_y+1, gy), 0);
     const int zmin = max(min(linds_z, gz), 0);
     const int zmax = max(min(hinds_z+1, gz), 0);
     float x_n, y_n, z_n, r_xn, r_yn, r_zn;
     for (int i = xmin; i < xmax; i++){
        int shiftx = i * gy * gz; // shift for i_th unit
        for (int j = ymin; j < ymax; j++){
            int shifty = j * gz;
            for (int k = zmin; k < zmax; k++){ 
                x_n = px - xyz_min[0] - (i+0.5) * units[0];
                y_n = py - xyz_min[1] - (j+0.5) * units[1];
                z_n = pz - xyz_min[2] - (k+0.5) * units[2];
                // rotation
                r_xn = x_n * geo_rot[i_shift_R] + y_n * geo_rot[i_shift_R+3]  + z_n * geo_rot[i_shift_R+6];
                r_yn = x_n * geo_rot[i_shift_R+1] + y_n * geo_rot[i_shift_R+4]  + z_n * geo_rot[i_shift_R+7];
                r_zn = x_n * geo_rot[i_shift_R+2] + y_n * geo_rot[i_shift_R+5]  + z_n * geo_rot[i_shift_R+8];
                if (abs(r_xn) < local_range_every_0 && abs(r_yn) < local_range_every_1 && abs(r_zn) < local_range_every_2) {
                    tensoRF_cvrg_inds[shiftx + shifty + k] = 1;
                }
            }
        }
     }
  }
}


template <typename scalar_t>
__global__ void fill_cubic_geo_inds_cuda_kernel(
        scalar_t* __restrict__ local_range,
        int64_t* __restrict__ gridSize,
        int8_t* __restrict__ tensoRF_count,
        int16_t* __restrict__ tensoRF_topindx,
        scalar_t* __restrict__ units,
        scalar_t* __restrict__ radius,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        scalar_t* __restrict__ pnt_xyz,
        scalar_t* __restrict__ geo_rot,
        int32_t* __restrict__ tensoRF_cvrg_inds,
        const int max_tensoRF,
        const int n_pts,
        const int gridSizeAll
        ) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<gridSizeAll && tensoRF_cvrg_inds[idx] >= 0) {
     const int cvrg_id = tensoRF_cvrg_inds[idx];
     const int tshift = cvrg_id * max_tensoRF;
     const int indx = idx / (gridSize[1] * gridSize[2]);
     const int indy = (idx - (gridSize[1] * gridSize[2]) * indx) / gridSize[2];
     const int indz = idx % gridSize[2];
     const float cx = xyz_min[0] + (indx + 0.5) * units[0];
     const float cy = xyz_min[1] + (indy + 0.5) * units[1];
     const float cz = xyz_min[2] + (indz + 0.5) * units[2];
     float xyz2Buffer[8];
     int kid = 0, far_ind = 0; 
     float far2 = 0.0;

     for (int i = 0; i < n_pts; i++){
         const int i_shift = i * 3;
         const int i_shift_R = i_shift * 3;
         float xdiff = abs(pnt_xyz[i_shift] - cx);
         float ydiff = abs(pnt_xyz[i_shift+1] - cy);
         float zdiff = abs(pnt_xyz[i_shift+2] - cz);
         float local_range_every_0 = local_range[i_shift];
         float local_range_every_1 = local_range[i_shift+1];
         float local_range_every_2 = local_range[i_shift+2];

         float rx_diff = xdiff * geo_rot[i_shift_R] + ydiff * geo_rot[i_shift_R+3]  + zdiff * geo_rot[i_shift_R+6];
         float ry_diff = xdiff * geo_rot[i_shift_R+1] + ydiff * geo_rot[i_shift_R+4]  + zdiff * geo_rot[i_shift_R+7];
         float rz_diff = xdiff * geo_rot[i_shift_R+2] + ydiff * geo_rot[i_shift_R+5]  + zdiff * geo_rot[i_shift_R+8];

         if (rx_diff < local_range_every_0 && ry_diff < local_range_every_1 && rz_diff < local_range_every_2){

            float xyz2 = rx_diff * rx_diff + ry_diff * ry_diff + rz_diff * rz_diff;
            if (kid++ < max_tensoRF) {
                tensoRF_topindx[tshift + kid - 1] = i;
                xyz2Buffer[kid-1] = xyz2;
                if (xyz2 > far2){
                    far2 = xyz2;
                    far_ind = kid - 1;
                }
            } else {
                if (xyz2 < far2) {
                    tensoRF_topindx[tshift + far_ind] = i;
                    xyz2Buffer[far_ind] = xyz2;
                    far2 = xyz2;
                    for (int j = 0; j < max_tensoRF; j++) {
                        if (xyz2Buffer[j] > far2) {
                            far2 = xyz2Buffer[j];
                            far_ind = j;
                        }
                    }
                }
            }
         }
     }
     tensoRF_count[cvrg_id] = min(max_tensoRF, kid);
  }
}


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
        const int max_tensoRF) {
  const int threads = 256;
  const int n_pts = pnt_xyz.size(0);
  const int gridSizex = gridSize[0].item<int>();
  const int gridSizey = gridSize[1].item<int>();
  const int gridSizez = gridSize[2].item<int>();
  const int gridSizeAll = gridSizex * gridSizey * gridSizez;
  auto tensoRF_cvrg_inds = torch::zeros({gridSizex, gridSizey, gridSizez}, torch::dtype(torch::kInt32).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(pnt_xyz.type(), "get_cubic_geo_inds", ([&] {
    get_cubic_geo_inds_cuda_kernel<scalar_t><<<(n_pts+threads-1)/threads, threads>>>(
        local_range.data<scalar_t>(),
        gridSize.data<int64_t>(),
        units.data<scalar_t>(),
        radius.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        pnt_xyz.data<scalar_t>(),
        pnt_rmatrix.data<scalar_t>(),
        tensoRF_cvrg_inds.data<int32_t>(),
        n_pts);
  }));

  tensoRF_cvrg_inds = torch::cumsum(tensoRF_cvrg_inds.view(-1), 0, torch::kInt32) * tensoRF_cvrg_inds.view(-1);
  const int num_cvrg = tensoRF_cvrg_inds.max().item<int>();
  tensoRF_cvrg_inds = (tensoRF_cvrg_inds - 1).view({gridSizex, gridSizey, gridSizez});

  auto tensoRF_topindx = torch::full({num_cvrg, max_tensoRF}, -1, torch::dtype(torch::kInt16).device(torch::kCUDA));
  auto tensoRF_count = torch::zeros({num_cvrg}, torch::dtype(torch::kInt8).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(pnt_xyz.type(), "fill_cubic_geo_inds", ([&] {
    fill_cubic_geo_inds_cuda_kernel<scalar_t><<<(gridSizeAll+threads-1)/threads, threads>>>(
        local_range.data<scalar_t>(),
        gridSize.data<int64_t>(),
        tensoRF_count.data<int8_t>(),
        tensoRF_topindx.data<int16_t>(),
        units.data<scalar_t>(),
        radius.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        pnt_xyz.data<scalar_t>(),
        pnt_rmatrix.data<scalar_t>(),
        tensoRF_cvrg_inds.data<int32_t>(),
        max_tensoRF,
        n_pts,
        gridSizeAll);
  }));
  return {tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx};
}




std::vector<torch::Tensor> sample_2_rot_cubic_tensoRF_cvrg_cuda(torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max, torch::Tensor units, torch::Tensor local_range, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds, torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_xyz, torch::Tensor geo_rot, torch::Tensor dim_cumsum_counter, const int K, const bool KNN) {

  const int threads = 256;
  const int n_pts = geo_xyz.size(0);
  const int n_sample = xyz_sampled.size(0);
  const int maxK = tensoRF_topindx.size(1);
  const int gridX = tensoRF_cvrg_inds.size(0);
  const int gridY = tensoRF_cvrg_inds.size(1);
  const int gridZ = tensoRF_cvrg_inds.size(2);

  const int n_sampleK = n_sample * K;
  auto local_gindx_s = torch::empty({n_sampleK, 3}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto local_gindx_l = torch::empty({n_sampleK, 3}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto local_gweight_s = torch::empty({n_sampleK, 3}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
  auto local_gweight_l = torch::empty({n_sampleK, 3}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
  auto local_kernel_dist = torch::empty({n_sampleK}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
  auto final_tensoRF_id = torch::empty({n_sampleK}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto tensoRF_mask = torch::zeros({n_sample, K}, torch::dtype(torch::kBool).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(xyz_sampled.type(), "find_pca_tensoRF_and_repos_cuda_kernel", ([&] {
      find_pca_tensoRF_and_repos_cuda_kernel<scalar_t><<<(n_sampleK+threads-1)/threads, threads>>>(
        xyz_sampled.data<scalar_t>(),
        geo_xyz.data<scalar_t>(),
        geo_rot.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        final_tensoRF_id.data<int64_t>(),
        local_dims.data<int16_t>(),
        local_gindx_s.data<int64_t>(),
        local_gindx_l.data<int64_t>(),
        local_gweight_s.data<scalar_t>(),
        local_gweight_l.data<scalar_t>(),
        local_kernel_dist.data<scalar_t>(),
        units.data<scalar_t>(),
        local_range.data<scalar_t>(),
        tensoRF_cvrg_inds.data<int32_t>(),
        tensoRF_count.data<int8_t>(),
        tensoRF_topindx.data<int16_t>(),
        dim_cumsum_counter.data<int32_t>(),
        tensoRF_mask.data<bool>(),
        gridX,
        gridY,
        gridZ,
        n_sampleK,
        K,
        maxK);
  }));

  auto cvrg_count = tensoRF_mask.sum(1);
  auto cvrg_cumsum = cvrg_count.cumsum(0);
  const int cvrg_len = cvrg_count.sum().item<int>();

  auto final_agg_id = torch::empty({cvrg_len}, torch::dtype(torch::kInt64).device(torch::kCUDA));

  __fill_agg_id<<<(n_sample+threads-1)/threads, threads>>>(cvrg_count.data<int64_t>(), cvrg_cumsum.data<int64_t>(), final_agg_id.data<int64_t>(), n_sample);
  // torch::cuda::synchronize();

  return {tensoRF_mask.reshape(-1), local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, final_tensoRF_id, final_agg_id};
}





torch::Tensor filter_tensoRF_cuda(torch::Tensor xyz_sampled, torch::Tensor xyz_inbbox, torch::Tensor xyz_min, torch::Tensor xyz_max, torch::Tensor units, torch::Tensor local_range, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds, torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_xyz, torch::Tensor geo_rot, const int K, const int ord_thresh) {

  const int threads = 256;
  const int n_pts = geo_xyz.size(0);
  const int n_rays = xyz_sampled.size(0);
  const int n_sample = xyz_sampled.size(1);
  const int maxK = tensoRF_topindx.size(1);
  const int gridX = tensoRF_cvrg_inds.size(0);
  const int gridY = tensoRF_cvrg_inds.size(1);
  const int gridZ = tensoRF_cvrg_inds.size(2);

  auto tensoRF_mask = torch::zeros({n_pts}, torch::dtype(torch::kBool).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(xyz_sampled.type(), "mask_tensoRF", ([&] {
      mask_tensoRF<scalar_t><<<(n_rays+threads-1)/threads, threads>>>(
        xyz_sampled.data<scalar_t>(),
        xyz_inbbox.data<int8_t>(),
        geo_xyz.data<scalar_t>(),
        geo_rot.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        local_dims.data<int16_t>(),
        units.data<scalar_t>(),
        local_range.data<scalar_t>(),
        tensoRF_cvrg_inds.data<int32_t>(),
        tensoRF_count.data<int8_t>(),
        tensoRF_topindx.data<int16_t>(),
        tensoRF_mask.data<bool>(),
        gridX,
        gridY,
        gridZ,
        n_rays,
        n_sample,
        maxK,
        K,
        ord_thresh);
  }));
  return tensoRF_mask;
}
