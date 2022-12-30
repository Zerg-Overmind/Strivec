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
__global__ void find_tensoRF_and_repos_masked_cuda_kernel(
        scalar_t* __restrict__ xyz_sampled,
        scalar_t* __restrict__ geo_xyz,
        scalar_t* __restrict__ xyz_min,
        int64_t* __restrict__ final_tensoRF_id,
        int64_t* __restrict__ local_dims,
        int64_t* __restrict__ local_gindx_s,
        int64_t* __restrict__ local_gindx_l,
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


template <typename scalar_t>
__global__ void find_tensoRF_and_repos_normxyz_cuda_kernel(
        scalar_t* __restrict__ xyz_sampled,
        scalar_t* __restrict__ geo_xyz,
        int64_t* __restrict__ final_agg_id,
        int64_t* __restrict__ final_tensoRF_id,
        scalar_t* __restrict__ local_range,
        int64_t* __restrict__ local_dims,
        int64_t* __restrict__ local_gindx_s,
        int64_t* __restrict__ local_gindx_l,
        scalar_t* __restrict__ local_gweight_s,
        scalar_t* __restrict__ local_gweight_l,
        scalar_t* __restrict__ local_norm_xyz,
        scalar_t* __restrict__ local_kernel_dist,
        scalar_t* __restrict__ lvl_units,
        int16_t* __restrict__ tensoRF_topindx,
        int32_t* __restrict__ cvrg_inds,
        int32_t* __restrict__ cvrg_cumsum,
        int32_t* __restrict__ cvrg_count,
        const int cvrg_len,
        const int K,
        const int maxK
        ) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < cvrg_len) {
    const int i_agg = final_agg_id[idx];
    const int tensoRF_shift = idx - ((i_agg!=0) ? cvrg_cumsum[i_agg-1] : 0);
    const int cvrg_ind = cvrg_inds[i_agg];
    //if (tensoRF_shift != 0) {
    //    printf("tensoRF_shift %d", tensoRF_shift);
    //}

    const int i_tid = tensoRF_topindx[cvrg_ind * maxK + tensoRF_shift];
//    if (cvrg_ind * maxK + tensoRF_shift >=  507776) {
//        printf("tensoRF_topindx %d %d %d %d;  ", 507776 , cvrg_ind, maxK, tensoRF_shift);
//    }
    final_tensoRF_id[idx] = i_tid;
    const int offset_a = i_agg * 3;
    const int offset_t = i_tid * 3;
    const int offset_p = idx * 3;

    const float px = xyz_sampled[offset_a];
    const float py = xyz_sampled[offset_a + 1];
    const float pz = xyz_sampled[offset_a + 2];

    const float rel_x = px - geo_xyz[offset_t];
    const float rel_y = py - geo_xyz[offset_t+1];
    const float rel_z = pz - geo_xyz[offset_t+2];

    local_kernel_dist[idx] = sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z);

    const float softindx = (rel_x + local_range[0]) / lvl_units[0];
    const float softindy = (rel_y + local_range[1]) / lvl_units[1];
    const float softindz = (rel_z + local_range[2]) / lvl_units[2];

    local_norm_xyz[offset_p  ] = rel_x / local_range[0];
    local_norm_xyz[offset_p+1] = rel_y / local_range[1];
    local_norm_xyz[offset_p+2] = rel_z / local_range[2];
//    printf("local_range 0 %f, local_range 1 %f, local_range 2 %f;  ", local_range[0], local_range[1], local_range[2]);
//    printf("lvl_units 0 %f, lvl_units 1 %f, lvl_units 2 %f;  ", lvl_units[0], lvl_units[1], lvl_units[2]);

    const int indlx = min(max((int)softindx, 0), (int)local_dims[0]-1);
    const int indly = min(max((int)softindy, 0), (int)local_dims[1]-1);
    const int indlz = min(max((int)softindz, 0), (int)local_dims[2]-1);

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


template <typename scalar_t>
__global__ void find_tensoRF_and_repos_cuda_kernel(
        scalar_t* __restrict__ xyz_sampled,
        scalar_t* __restrict__ geo_xyz,
        int64_t* __restrict__ final_agg_id,
        int64_t* __restrict__ final_tensoRF_id,
        scalar_t* __restrict__ local_range,
        int64_t* __restrict__ local_dims,
        int64_t* __restrict__ local_gindx_s,
        int64_t* __restrict__ local_gindx_l,
        scalar_t* __restrict__ local_gweight_s,
        scalar_t* __restrict__ local_gweight_l,
        scalar_t* __restrict__ local_kernel_dist,
        scalar_t* __restrict__ lvl_units,
        int16_t* __restrict__ tensoRF_topindx,
        int32_t* __restrict__ cvrg_inds,
        int32_t* __restrict__ cvrg_cumsum,
        int32_t* __restrict__ cvrg_count,
        const int cvrg_len,
        const int K,
        const int maxK
        ) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < cvrg_len) {
    const int i_agg = final_agg_id[idx];
    const int tensoRF_shift = idx - ((i_agg!=0) ? cvrg_cumsum[i_agg-1] : 0);
    const int cvrg_ind = cvrg_inds[i_agg];
    //if (tensoRF_shift != 0) {
    //    printf("tensoRF_shift %d", tensoRF_shift);
    //}

    const int i_tid = tensoRF_topindx[cvrg_ind * maxK + tensoRF_shift];
//    if (cvrg_ind * maxK + tensoRF_shift >=  507776) {
//        printf("tensoRF_topindx %d %d %d %d;  ", 507776 , cvrg_ind, maxK, tensoRF_shift);
//    }
    final_tensoRF_id[idx] = i_tid;
    const int offset_a = i_agg * 3;
    const int offset_t = i_tid * 3;
    const int offset_p = idx * 3;

    const float px = xyz_sampled[offset_a];
    const float py = xyz_sampled[offset_a + 1];
    const float pz = xyz_sampled[offset_a + 2];

    const float rel_x = px - geo_xyz[offset_t];
    const float rel_y = py - geo_xyz[offset_t+1];
    const float rel_z = pz - geo_xyz[offset_t+2];

    local_kernel_dist[idx] = sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z);
    //if (local_kernel_dist[idx] > 0.4){
    //    printf("rel_x %f, rel_y %f, rel_z %f;  ", rel_x, rel_y, rel_z);
    //}

    const float softindx = (rel_x + local_range[0]) / lvl_units[0];
    const float softindy = (rel_y + local_range[1]) / lvl_units[1];
    const float softindz = (rel_z + local_range[2]) / lvl_units[2];
//    printf("local_range 0 %f, local_range 1 %f, local_range 2 %f;  ", local_range[0], local_range[1], local_range[2]);
//    printf("lvl_units 0 %f, lvl_units 1 %f, lvl_units 2 %f;  ", lvl_units[0], lvl_units[1], lvl_units[2]);

    const int indlx = min(max((int)softindx, 0), (int)local_dims[0]-1);
    const int indly = min(max((int)softindy, 0), (int)local_dims[1]-1);
    const int indlz = min(max((int)softindz, 0), (int)local_dims[2]-1);

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
        const int gridX,
        const int gridY,
        const int gridZ,
        const int n_sample) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n_sample) {
     const int xyzshift = idx * 3;
//     const int indx = (xyz_sampled[xyzshift] - xyz_min[0]) / units[0];
//     const int indy = (xyz_sampled[xyzshift + 1] - xyz_min[1]) / units[1];
//     const int indz = (xyz_sampled[xyzshift + 2] - xyz_min[2]) / units[2];
     const int indx = min(max((int)((xyz_sampled[xyzshift] - xyz_min[0]) / units[0]), 0), gridX-1);
     const int indy = min(max((int)((xyz_sampled[xyzshift + 1] - xyz_min[1]) / units[1]), 0), gridY-1);
     const int indz = min(max((int)((xyz_sampled[xyzshift + 2] - xyz_min[2]) / units[2]), 0), gridZ-1);

     const int inds = indx * gridYZ + indy * gridZ + indz;
     const int cvrg_id = tensoRF_cvrg_inds[inds];
     if (cvrg_id >= 0){
        cvrg_inds[idx] = cvrg_id;
        cvrg_count[idx] = tensoRF_count[cvrg_id];
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
                if (min(abs(px - xyz_min[0] - i * units[0]), abs(px - xyz_min[0] - (i+1) * units[0])) <= local_range[0] && min(abs(py - xyz_min[1] - j * units[1]), abs(py - xyz_min[1] - (j+1) * units[1])) <= local_range[1] && min(abs(pz - xyz_min[2] - k * units[2]), abs(pz - xyz_min[2] - (k+1) * units[2])) <= local_range[2]) {
                    tensoRF_cvrg_inds[shiftx + shifty + k] = 1;
                }
            }
        }
     }
  }
}



template <typename scalar_t>
__global__ void fill_geo_inds_cuda_kernel(
        scalar_t* __restrict__ local_range,
        int64_t* __restrict__ gridSize,
        int8_t* __restrict__ tensoRF_count,
        int16_t* __restrict__ tensoRF_topindx,
        scalar_t* __restrict__ units,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        scalar_t* __restrict__ pnt_xyz,
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
     float half_unitx = units[0] * 0.5;
     float half_unity = units[1] * 0.5;
     float half_unitz = units[2] * 0.5;
     for (int i = 0; i < n_pts; i++){
         const int i_shift = i * 3;
         float xdiff = abs(pnt_xyz[i_shift] - cx);
         float ydiff = abs(pnt_xyz[i_shift+1] - cy);
         float zdiff = abs(pnt_xyz[i_shift+2] - cz);

         if (xdiff < local_range[0] + half_unitx && ydiff < local_range[1] + half_unity && zdiff < local_range[2] + half_unitz){
            float xyz2 = xdiff * xdiff + ydiff * ydiff + zdiff * zdiff;
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



std::vector<torch::Tensor> build_tensoRF_map_hier_cuda(
        torch::Tensor pnt_xyz,
        torch::Tensor gridSize,
        torch::Tensor xyz_min,
        torch::Tensor xyz_max,
        torch::Tensor units,
        torch::Tensor local_range,
        torch::Tensor local_dims,
        const int max_tensoRF) {
  const int threads = 256;
  const int n_pts = pnt_xyz.size(0);
  const int gridSizex = gridSize[0].item<int>();
  const int gridSizey = gridSize[1].item<int>();
  const int gridSizez = gridSize[2].item<int>();
  const int gridSizeAll = gridSizex * gridSizey * gridSizez;
  auto tensoRF_cvrg_inds = torch::zeros({gridSizex, gridSizey, gridSizez}, torch::dtype(torch::kInt32).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(pnt_xyz.type(), "get_geo_inds", ([&] {
    get_geo_inds_cuda_kernel<scalar_t><<<(n_pts+threads-1)/threads, threads>>>(
        local_range.data<scalar_t>(),
        gridSize.data<int64_t>(),
        units.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        pnt_xyz.data<scalar_t>(),
        tensoRF_cvrg_inds.data<int32_t>(),
        n_pts);
  }));

  tensoRF_cvrg_inds = torch::cumsum(tensoRF_cvrg_inds.view(-1), 0, torch::kInt32) * tensoRF_cvrg_inds.view(-1);
  const int num_cvrg = tensoRF_cvrg_inds.max().item<int>();
  tensoRF_cvrg_inds = (tensoRF_cvrg_inds - 1).view({gridSizex, gridSizey, gridSizez});

  auto tensoRF_topindx = torch::full({num_cvrg, max_tensoRF}, -1, torch::dtype(torch::kInt16).device(torch::kCUDA));
  auto tensoRF_count = torch::zeros({num_cvrg}, torch::dtype(torch::kInt8).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(pnt_xyz.type(), "fill_geo_inds", ([&] {
    fill_geo_inds_cuda_kernel<scalar_t><<<(gridSizeAll+threads-1)/threads, threads>>>(
        local_range.data<scalar_t>(),
        gridSize.data<int64_t>(),
        tensoRF_count.data<int8_t>(),
        tensoRF_topindx.data<int16_t>(),
        units.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        pnt_xyz.data<scalar_t>(),
        tensoRF_cvrg_inds.data<int32_t>(),
        max_tensoRF,
        n_pts,
        gridSizeAll);
  }));
  return {tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx};
}




std::vector<torch::Tensor> sample_2_tensoRF_cvrg_hier_cuda(
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

  auto cvrg_inds = torch::empty({n_sample}, torch::dtype(torch::kInt32).device(torch::kCUDA));
  auto cvrg_count = torch::zeros({n_sample}, torch::dtype(torch::kInt32).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(xyz_sampled.type(), "count_tensoRF_cvrg_cuda", ([&] {
    count_tensoRF_cvrg_cuda_kernel<scalar_t><<<(n_sample+threads-1)/threads, threads>>>(
        xyz_sampled.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        units.data<scalar_t>(),
        tensoRF_count.data<int8_t>(),
        tensoRF_cvrg_inds.data<int32_t>(),
        cvrg_inds.data<int32_t>(),
        cvrg_count.data<int32_t>(),
        gridY * gridZ,
        gridX,
        gridY,
        gridZ,
        n_sample);
  }));
//  torch::cuda::synchronize();
  auto cvrg_cumsum = cvrg_count.cumsum(0, torch::kInt32);
  const int cvrg_len = cvrg_count.sum().item<int>();

  auto final_tensoRF_id = torch::empty({cvrg_len}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto final_agg_id = torch::empty({cvrg_len}, torch::dtype(torch::kInt64).device(torch::kCUDA));

  auto local_gindx_s = torch::empty({cvrg_len, 3}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto local_gindx_l = torch::empty({cvrg_len, 3}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto local_gweight_s = torch::empty({cvrg_len, 3}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
  auto local_gweight_l = torch::empty({cvrg_len, 3}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
  auto local_kernel_dist = torch::empty({cvrg_len}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
  if (cvrg_len > 0){
      __fill_agg_id<<<(n_sample+threads-1)/threads, threads>>>(cvrg_count.data<int32_t>(), cvrg_cumsum.data<int32_t>(), final_agg_id.data<int64_t>(), n_sample);

      //
      AT_DISPATCH_FLOATING_TYPES(xyz_sampled.type(), "find_tensoRF_and_repos_cuda", ([&] {
        find_tensoRF_and_repos_cuda_kernel<scalar_t><<<(cvrg_len+threads-1)/threads, threads>>>(
            xyz_sampled.data<scalar_t>(),
            geo_xyz.data<scalar_t>(),
            final_agg_id.data<int64_t>(),
            final_tensoRF_id.data<int64_t>(),
            local_range.data<scalar_t>(),
            local_dims.data<int64_t>(),
            local_gindx_s.data<int64_t>(),
            local_gindx_l.data<int64_t>(),
            local_gweight_s.data<scalar_t>(),
            local_gweight_l.data<scalar_t>(),
            local_kernel_dist.data<scalar_t>(),
            lvl_units.data<scalar_t>(),
            tensoRF_topindx.data<int16_t>(),
            cvrg_inds.data<int32_t>(),
            cvrg_cumsum.data<int32_t>(),
            cvrg_count.data<int32_t>(),
            cvrg_len,
            K,
            maxK);
       }));
  }
  // torch::cuda::synchronize();
  return {local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, final_tensoRF_id, final_agg_id};
}


std::vector<torch::Tensor> sample_2_tensoRF_cvrg_hier_gs_cuda(
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

  const int n_sampleK = n_sample * K;
  auto local_gindx_s = torch::empty({n_sampleK, 3}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto local_gindx_l = torch::empty({n_sampleK, 3}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto local_gweight_s = torch::empty({n_sampleK, 3}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
  auto local_gweight_l = torch::empty({n_sampleK, 3}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
  auto local_norm_xyz = torch::empty({n_sampleK, 3}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
  auto local_kernel_dist = torch::empty({n_sampleK}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
  auto final_tensoRF_id = torch::empty({n_sampleK}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto tensoRF_mask = torch::zeros({n_sample, K}, torch::dtype(torch::kBool).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(xyz_sampled.type(), "find_tensoRF_and_repos_masked_cuda", ([&] {
      find_tensoRF_and_repos_masked_cuda_kernel<scalar_t><<<(n_sampleK+threads-1)/threads, threads>>>(
        xyz_sampled.data<scalar_t>(),
        geo_xyz.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        final_tensoRF_id.data<int64_t>(),
        local_dims.data<int64_t>(),
        local_gindx_s.data<int64_t>(),
        local_gindx_l.data<int64_t>(),
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
//      torch::cuda::synchronize();
  } else {
      local_gindx_s = torch::empty({0, 3}, torch::dtype(torch::kInt64).device(torch::kCUDA));
      local_gindx_l = torch::empty({0, 3}, torch::dtype(torch::kInt64).device(torch::kCUDA));
      local_gweight_s = torch::empty({0, 3}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
      local_gweight_l = torch::empty({0, 3}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
      local_norm_xyz = torch::empty({0, 3}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
      local_kernel_dist = torch::empty({0}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
      final_tensoRF_id = torch::empty({0}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  }
  return {local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, final_tensoRF_id, final_agg_id, local_norm_xyz};
}



