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
__global__ void find_rotdist_tensoRF_and_repos_cuda_kernel(
        scalar_t* __restrict__ xyz_sampled,
        scalar_t* __restrict__ geo_xyz,
        scalar_t* __restrict__ geo_rot,
        scalar_t* __restrict__ xyz_min,
        int64_t* __restrict__ final_tensoRF_id,
        int64_t* __restrict__ local_dims,
        scalar_t* __restrict__ dist,
        scalar_t* __restrict__ local_kernel_dist,
        scalar_t* __restrict__ units,
        scalar_t* __restrict__ local_range,
        int64_t* __restrict__ tensoRF_cvrg_inds,
        int64_t* __restrict__ tensoRF_count,
        int64_t* __restrict__ tensoRF_topindx,
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


        if (abs(rx) < local_range[0] && abs(ry) < local_range[1] && abs(rz) < local_range[2]){
            const int offset_p = idx * 3;
            tensoRF_mask[idx] = true;
            local_kernel_dist[idx] = sqrt(rx * rx + ry * ry + rz * rz);
            dist[offset_p] = rel_x;
            dist[offset_p+1] = rel_y;
            dist[offset_p+2] = rel_z;
        }
    }
  }
}


template <typename scalar_t>
__global__ void find_rotdist_tensoRF_and_repos_rand_cuda_kernel(
        scalar_t* __restrict__ xyz_sampled,
        scalar_t* __restrict__ geo_xyz,
        scalar_t* __restrict__ geo_rot,
        scalar_t* __restrict__ xyz_min,
        int64_t* __restrict__ final_tensoRF_id,
        int64_t* __restrict__ local_dims,
        scalar_t* __restrict__ dist,
        scalar_t* __restrict__ local_kernel_dist,
        scalar_t* __restrict__ units,
        scalar_t* __restrict__ local_range,
        int64_t* __restrict__ tensoRF_cvrg_inds,
        int64_t* __restrict__ tensoRF_count,
        int64_t* __restrict__ tensoRF_topindx,
        bool* __restrict__ tensoRF_mask,
        const int gridX,
        const int gridY,
        const int gridZ,
        const int n_sample,
        const int K,
        const int maxK,
        const unsigned long seconds
        ) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n_sample) {
    const int xyzshift = idx * 3;
    const float px = xyz_sampled[xyzshift];
    const float py = xyz_sampled[xyzshift + 1];
    const float pz = xyz_sampled[xyzshift + 2];
    const int indx = min(gridX-1, (int)((px - xyz_min[0]) / units[0]));
    const int indy = min(gridY-1, (int)((py - xyz_min[1]) / units[1]));
    const int indz = min(gridZ-1, (int)((pz - xyz_min[2]) / units[2]));

    const int inds = indx * gridY * gridZ + indy * gridZ + indz;
    const int cvrg_id = tensoRF_cvrg_inds[inds];
    // printf("tensoRF_count[cvrg_id] %d %d %d %d;   ", kid, (int)tensoRF_count[cvrg_id], K, cvrg_id);
    int kid, i_tid, offset_t, offset_r, offset_s, offset_p, count = 0, placement[8] = {0};
    float rel_x, rel_y, rel_z, rx, ry, rz;
    curandState state;
    const int idCount = tensoRF_count[cvrg_id];
    const int num_select = min(K, idCount);
    curand_init(seconds+idx, 0, 0, &state);
    for (int i = 0; i < num_select; i++){
        kid = min(idCount-1, (int)(curand_uniform(&state) * idCount));
        // printf("placement[kid] %d,%d ", kid, placement[kid]);
        if (placement[kid] == 0){
            placement[kid] = 1;
            i_tid = tensoRF_topindx[cvrg_id * maxK + kid];
            offset_t = i_tid * 3;
            offset_r = i_tid * 9;

            rel_x = px - geo_xyz[offset_t];
            rel_y = py - geo_xyz[offset_t+1];
            rel_z = pz - geo_xyz[offset_t+2];

            rx = rel_x * geo_rot[offset_r] + rel_y * geo_rot[offset_r+3]  + rel_z * geo_rot[offset_r+6];

            ry = rel_x * geo_rot[offset_r+1] + rel_y * geo_rot[offset_r+4]  + rel_z * geo_rot[offset_r+7];

            rz = rel_x * geo_rot[offset_r+2] + rel_y * geo_rot[offset_r+5]  + rel_z * geo_rot[offset_r+8];

            if (abs(rx) < local_range[0] && abs(ry) < local_range[1] && abs(rz) < local_range[2]){
                offset_s = idx * K + i;
                offset_p = offset_s * 3;
                tensoRF_mask[offset_s] = true;
                final_tensoRF_id[offset_s] = i_tid;
                local_kernel_dist[offset_s] = sqrt(rx * rx + ry * ry + rz * rz);
                dist[offset_p] = rel_x;
                dist[offset_p+1] = rel_y;
                dist[offset_p+2] = rel_z;
                count++;
            }
        }
    }

    const int startid = min(idCount-1, (int)(curand_uniform(&state) * idCount));
    if (count == 0){
        for (int i = startid; i < idCount + startid; i++){
            kid = i % idCount;
            if (placement[kid] == 0){
                i_tid = tensoRF_topindx[cvrg_id * maxK + kid];
                offset_t = i_tid * 3;
                offset_r = i_tid * 9;
                rel_x = px - geo_xyz[offset_t];
                rel_y = py - geo_xyz[offset_t+1];
                rel_z = pz - geo_xyz[offset_t+2];
                rx = rel_x * geo_rot[offset_r] + rel_y * geo_rot[offset_r+3]  + rel_z * geo_rot[offset_r+6];

                ry = rel_x * geo_rot[offset_r+1] + rel_y * geo_rot[offset_r+4]  + rel_z * geo_rot[offset_r+7];

                rz = rel_x * geo_rot[offset_r+2] + rel_y * geo_rot[offset_r+5]  + rel_z * geo_rot[offset_r+8];

                if (abs(rx) < local_range[0] && abs(ry) < local_range[1] && abs(rz) < local_range[2]){
                    offset_s = idx * K;
                    offset_p = offset_s * 3;
                    tensoRF_mask[offset_s] = true;
                    final_tensoRF_id[offset_s] = i_tid;
                    local_kernel_dist[offset_s] = sqrt(rx * rx + ry * ry + rz * rz);
                    dist[offset_p] = rel_x;
                    dist[offset_p+1] = rel_y;
                    dist[offset_p+2] = rel_z;
                    count++;
                    break;
                }
            }
            // if (i > startid) printf("  count %d / %d ", kid, idCount);
        }
    }
  }
}


template <typename scalar_t>
__global__ void find_rot_tensoRF_and_repos_cuda_kernel(
        scalar_t* __restrict__ xyz_sampled,
        scalar_t* __restrict__ geo_xyz,
        scalar_t* __restrict__ geo_rot,
        scalar_t* __restrict__ xyz_min,
        int64_t* __restrict__ final_tensoRF_id,
        int64_t* __restrict__ local_dims,
        int64_t* __restrict__ local_gindx_s,
        int64_t* __restrict__ local_gindx_l,
        scalar_t* __restrict__ local_gweight_s,
        scalar_t* __restrict__ local_gweight_l,
        scalar_t* __restrict__ local_kernel_dist,
        scalar_t* __restrict__ units,
        scalar_t* __restrict__ local_range,
        int64_t* __restrict__ tensoRF_cvrg_inds,
        int64_t* __restrict__ tensoRF_count,
        int64_t* __restrict__ tensoRF_topindx,
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


        if (abs(rx) < local_range[0] && abs(ry) < local_range[1] && abs(rz) < local_range[2]){
            const int offset_p = idx * 3;
            tensoRF_mask[idx] = true;
            local_kernel_dist[idx] = sqrt(rx * rx + ry * ry + rz * rz);

            const float softindx = (rx + local_range[0]) / units[0];
            const float softindy = (ry + local_range[1]) / units[1];
            const float softindz = (rz + local_range[2]) / units[2];

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
  }
}


template <typename scalar_t>
__global__ void find_rot_tensoRF_and_repos_rand_cuda_kernel(
        scalar_t* __restrict__ xyz_sampled,
        scalar_t* __restrict__ geo_xyz,
        scalar_t* __restrict__ geo_rot,
        scalar_t* __restrict__ xyz_min,
        int64_t* __restrict__ final_tensoRF_id,
        int64_t* __restrict__ local_dims,
        int64_t* __restrict__ local_gindx_s,
        int64_t* __restrict__ local_gindx_l,
        scalar_t* __restrict__ local_gweight_s,
        scalar_t* __restrict__ local_gweight_l,
        scalar_t* __restrict__ local_kernel_dist,
        scalar_t* __restrict__ units,
        scalar_t* __restrict__ local_range,
        int64_t* __restrict__ tensoRF_cvrg_inds,
        int64_t* __restrict__ tensoRF_count,
        int64_t* __restrict__ tensoRF_topindx,
        bool* __restrict__ tensoRF_mask,
        const int gridX,
        const int gridY,
        const int gridZ,
        const int n_sample,
        const int K,
        const int maxK,
        const unsigned long seconds
        ) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n_sample) {
    const int xyzshift = idx * 3;
    const float px = xyz_sampled[xyzshift];
    const float py = xyz_sampled[xyzshift + 1];
    const float pz = xyz_sampled[xyzshift + 2];
    const int indx = min(gridX-1, (int)((px - xyz_min[0]) / units[0]));
    const int indy = min(gridY-1, (int)((py - xyz_min[1]) / units[1]));
    const int indz = min(gridZ-1, (int)((pz - xyz_min[2]) / units[2]));

    const int inds = indx * gridY * gridZ + indy * gridZ + indz;
    const int cvrg_id = tensoRF_cvrg_inds[inds];
    // printf("tensoRF_count[cvrg_id] %d %d %d %d;   ", kid, (int)tensoRF_count[cvrg_id], K, cvrg_id);
    int kid, i_tid, offset_t, offset_r, offset_s, offset_p, indlx, indly, indlz, count = 0, placement[8] = {0};
    float rel_x, rel_y, rel_z, rx, ry, rz, softindx, softindy, softindz, res_x, res_y, res_z;
    curandState state;
    const int idCount = tensoRF_count[cvrg_id];
    const int num_select = min(K, idCount);
    curand_init(seconds+idx, 0, 0, &state);
    for (int i = 0; i < num_select; i++){
        kid = min(idCount-1, (int)(curand_uniform(&state) * idCount));
        // printf("placement[kid] %d,%d ", kid, placement[kid]);
        if (placement[kid] == 0){
            placement[kid] = 1;
            i_tid = tensoRF_topindx[cvrg_id * maxK + kid];
            offset_t = i_tid * 3;
            offset_r = i_tid * 9;

            rel_x = px - geo_xyz[offset_t];
            rel_y = py - geo_xyz[offset_t+1];
            rel_z = pz - geo_xyz[offset_t+2];

            rx = rel_x * geo_rot[offset_r] + rel_y * geo_rot[offset_r+3]  + rel_z * geo_rot[offset_r+6];

            ry = rel_x * geo_rot[offset_r+1] + rel_y * geo_rot[offset_r+4]  + rel_z * geo_rot[offset_r+7];

            rz = rel_x * geo_rot[offset_r+2] + rel_y * geo_rot[offset_r+5]  + rel_z * geo_rot[offset_r+8];

            if (abs(rx) < local_range[0] && abs(ry) < local_range[1] && abs(rz) < local_range[2]){
                offset_s = idx * K + i;
                offset_p = offset_s * 3;
                tensoRF_mask[offset_s] = true;
                final_tensoRF_id[offset_s] = i_tid;
                local_kernel_dist[offset_s] = sqrt(rx * rx + ry * ry + rz * rz);

                softindx = (rx + local_range[0]) / units[0];
                softindy = (ry + local_range[1]) / units[1];
                softindz = (rz + local_range[2]) / units[2];

                indlx = min(max((int)softindx, 0), (int)local_dims[0]-1);
                indly = min(max((int)softindy, 0), (int)local_dims[1]-1);
                indlz = min(max((int)softindz, 0), (int)local_dims[2]-1);

                res_x = softindx - indlx;
                res_y = softindy - indly;
                res_z = softindz - indlz;

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
                count++;
            }
        }
    }

    const int startid = min(idCount-1, (int)(curand_uniform(&state) * idCount));
    if (count == 0){
        for (int i = startid; i < idCount + startid; i++){
            kid = i % idCount;
            if (placement[kid] == 0){
                i_tid = tensoRF_topindx[cvrg_id * maxK + kid];
                offset_t = i_tid * 3;
                offset_r = i_tid * 9;
                rel_x = px - geo_xyz[offset_t];
                rel_y = py - geo_xyz[offset_t+1];
                rel_z = pz - geo_xyz[offset_t+2];
                rx = rel_x * geo_rot[offset_r] + rel_y * geo_rot[offset_r+3]  + rel_z * geo_rot[offset_r+6];

                ry = rel_x * geo_rot[offset_r+1] + rel_y * geo_rot[offset_r+4]  + rel_z * geo_rot[offset_r+7];

                rz = rel_x * geo_rot[offset_r+2] + rel_y * geo_rot[offset_r+5]  + rel_z * geo_rot[offset_r+8];

                if (abs(rx) < local_range[0] && abs(ry) < local_range[1] && abs(rz) < local_range[2]){
                    offset_s = idx * K;
                    offset_p = offset_s * 3;
                    tensoRF_mask[offset_s] = true;
                    final_tensoRF_id[offset_s] = i_tid;
                    local_kernel_dist[offset_s] = sqrt(rx * rx + ry * ry + rz * rz);
                    softindx = (rx + local_range[0]) / units[0];
                    softindy = (ry + local_range[1]) / units[1];
                    softindz = (rz + local_range[2]) / units[2];

                    indlx = min(max((int)softindx, 0), (int)local_dims[0]-1);
                    indly = min(max((int)softindy, 0), (int)local_dims[1]-1);
                    indlz = min(max((int)softindz, 0), (int)local_dims[2]-1);

                    res_x = softindx - indlx;
                    res_y = softindy - indly;
                    res_z = softindz - indlz;

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
                    count++;
                    break;
                }
            }
            // if (i > startid) printf("  count %d / %d ", kid, idCount);
        }
    }

  }
}


template <typename scalar_t>
__global__ void find_sphere_tensoRF_and_repos_cuda_kernel(
        scalar_t* __restrict__ xyz_sampled,
        scalar_t* __restrict__ geo_xyz,
        scalar_t* __restrict__ xyz_min,
        int64_t* __restrict__ final_tensoRF_id,
        const float radiusl,
        const float radiush,
        int64_t* __restrict__ local_dims,
        int64_t* __restrict__ local_gindx_s,
        int64_t* __restrict__ local_gindx_l,
        scalar_t* __restrict__ local_gweight_s,
        scalar_t* __restrict__ local_gweight_l,
        scalar_t* __restrict__ local_kernel_dist,
        scalar_t* __restrict__ units,
        int64_t* __restrict__ tensoRF_cvrg_inds,
        int64_t* __restrict__ tensoRF_count,
        int64_t* __restrict__ tensoRF_topindx,
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
    const int indy = min(gridY-1, (int)((py - xyz_min[1]) / units[0]));
    const int indz = min(gridZ-1, (int)((pz - xyz_min[2]) / units[0]));

    const int inds = indx * gridY * gridZ + indy * gridZ + indz;
    const int cvrg_id = tensoRF_cvrg_inds[inds];
    // printf("tensoRF_count[cvrg_id] %d %d %d %d;   ", kid, (int)tensoRF_count[cvrg_id], K, cvrg_id);
    if (kid < tensoRF_count[cvrg_id]){
        const int i_tid = tensoRF_topindx[cvrg_id * maxK + kid];
        final_tensoRF_id[idx] = i_tid;
        const int offset_t = i_tid * 3;

        const float rel_x = px - geo_xyz[offset_t];
        const float rel_y = py - geo_xyz[offset_t+1];
        const float rel_z = pz - geo_xyz[offset_t+2];

        const float r = sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z);
        // printf("tensoRF_count[cvrg_id] %d %d %f %f;   ", kid, (int)tensoRF_count[cvrg_id], r, radiusl);
        if (r <= radiush && r>= radiusl){
            const int offset_p = idx * 3;
            tensoRF_mask[idx] = true;
            // printf(" true %d; ", idx);

            const float theta = atan2(rel_y, rel_x);
            const float phi = acos(rel_z / r);
            // printf("theta %f, phi %f    ;", theta / M_PI, phi / M_PI);
            local_kernel_dist[idx] = r;

            const float softindx = (r - radiusl) / units[0];
            const float softindy = (theta + M_PI) / units[1];
            const float softindz = phi / units[2];


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
  }
}


template <typename scalar_t>
__global__ void find_tensoRF_and_repos_cuda_kernel(
        scalar_t* __restrict__ xyz_sampled,
        scalar_t* __restrict__ geo_xyz,
        scalar_t* __restrict__ geo_xyz_recenter,
        int64_t* __restrict__ final_agg_id,
        int64_t* __restrict__ final_tensoRF_id,
        scalar_t* __restrict__ local_range,
        int64_t* __restrict__ local_dims,
        int64_t* __restrict__ local_gindx_s,
        int64_t* __restrict__ local_gindx_l,
        scalar_t* __restrict__ local_gweight_s,
        scalar_t* __restrict__ local_gweight_l,
        scalar_t* __restrict__ local_kernel_dist,
        scalar_t* __restrict__ units,
        int64_t* __restrict__ tensoRF_topindx,
        int64_t* __restrict__ cvrg_inds,
        int64_t* __restrict__ cvrg_cumsum,
        int64_t* __restrict__ cvrg_count,
        const int cvrg_len,
        const int K,
        const int maxK
        ) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < cvrg_len) {
    const int i_agg = final_agg_id[idx];
    const int tensoRF_shift = idx - ((i_agg!=0) ? cvrg_cumsum[i_agg-1] : 0);
    const int cvrg_ind = cvrg_inds[i_agg];
    const int i_tid = tensoRF_topindx[cvrg_ind * maxK + tensoRF_shift];
    final_tensoRF_id[idx] = i_tid;
    const int offset_a = i_agg * 3;
    const int offset_t = i_tid * 3;
    const int offset_p = idx * 3;

    const float px = xyz_sampled[offset_a];
    const float py = xyz_sampled[offset_a + 1];
    const float pz = xyz_sampled[offset_a + 2];

    const float rel_c_x = px - geo_xyz_recenter[offset_t];
    const float rel_c_y = py - geo_xyz_recenter[offset_t+1];
    const float rel_c_z = pz - geo_xyz_recenter[offset_t+2];

    const float rel_x = px - geo_xyz[offset_t];
    const float rel_y = py - geo_xyz[offset_t+1];
    const float rel_z = pz - geo_xyz[offset_t+2];

    local_kernel_dist[idx] = sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z);
    //if (local_kernel_dist[idx] > 0.4){
    //    printf("rel_x %f, rel_y %f, rel_z %f;  ", rel_x, rel_y, rel_z);
    //}

    const float softindx = (rel_c_x + local_range[0]) / units[0];
    const float softindy = (rel_c_y + local_range[1]) / units[1];
    const float softindz = (rel_c_z + local_range[2]) / units[2];

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
__global__ void find_tensoRF_and_repos_cuda_kernel_bk(
    scalar_t* __restrict__ xyz_sampled,
    scalar_t* __restrict__ geo_xyz,
    scalar_t* __restrict__ geo_xyz_recenter,
    scalar_t* __restrict__ xyz_min,
    int64_t* __restrict__ final_tensoRF_id,
    int64_t* __restrict__ local_dims,
    int64_t* __restrict__ local_gindx_s,
    int64_t* __restrict__ local_gindx_l,
    scalar_t* __restrict__ local_gweight_s,
    scalar_t* __restrict__ local_gweight_l,
    scalar_t* __restrict__ local_kernel_dist,
    scalar_t* __restrict__ units,
    scalar_t* __restrict__ local_range,
    int64_t* __restrict__ tensoRF_cvrg_inds,
    int64_t* __restrict__ tensoRF_count,
    int64_t* __restrict__ tensoRF_topindx,
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

        const float rel_c_x = px - geo_xyz_recenter[offset_t];
        const float rel_c_y = py - geo_xyz_recenter[offset_t+1];
        const float rel_c_z = pz - geo_xyz_recenter[offset_t+2];

        const float rel_x = px - geo_xyz[offset_t];
        const float rel_y = py - geo_xyz[offset_t+1];
        const float rel_z = pz - geo_xyz[offset_t+2];

        const int offset_p = idx * 3;
        tensoRF_mask[idx] = true;

        local_kernel_dist[idx] = sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z);

        const float softindx = (rel_c_x + local_range[0]) / units[0];
        const float softindy = (rel_c_y + local_range[1]) / units[1];
        const float softindz = (rel_c_z + local_range[2]) / units[2];

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
__global__ void filter_xyz_by_cvrg_cuda_kernel(
        scalar_t* __restrict__ xyz_sampled,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        scalar_t* __restrict__ units,
        const int gridX, const int gridY, const int gridZ,
        bool* __restrict__ tensoRF_cvrg_mask,
        bool* __restrict__ mask,
        const int n_sample) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n_sample) {
     const int xyzshift = idx * 3;
     const float xval = xyz_sampled[xyzshift],  yval = xyz_sampled[xyzshift + 1],  zval = xyz_sampled[xyzshift + 2];
     if (xval >= xyz_min[0] && xval <= xyz_max[0] && yval >= xyz_min[1] && yval <= xyz_max[1] && zval >= xyz_min[2] && zval <= xyz_max[2]){
         const int indx = min(gridX-1, (int)((xval - xyz_min[0]) / units[0]));
         const int indy = min(gridY-1, (int)((yval - xyz_min[1]) / units[1]));
         const int indz = min(gridZ-1, (int)((zval - xyz_min[2]) / units[2]));
         const int inds = indx * gridY * gridZ + indy * gridZ + indz;
         // if (indx < 0 || indx >= gridX) printf(" indx %d xval %f gridX %d; ", indx, xval, gridX);
         // if (indy < 0 || indy >= gridY) printf(" indy %d yval %f gridY %d; ", indy, yval, gridY);
         // if (indz < 0 || indz >= gridZ) printf(" indz %d zval %f gridZ %d; ", indz, zval, gridZ);
         mask[idx] = tensoRF_cvrg_mask[inds];
     }
  }
}


template <typename scalar_t>
__global__ void filter_xyz_by_rot_cvrg_cuda_kernel(
        scalar_t* __restrict__ geo_xyz,
        scalar_t* __restrict__ geo_rot,
        scalar_t* __restrict__ xyz_sampled,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        scalar_t* __restrict__ units,
        const int gridX, const int gridY, const int gridZ, const int K,
        int64_t* __restrict__ tensoRF_cvrg_inds,
        int64_t* __restrict__ tensoRF_count,
        int64_t* __restrict__ tensoRF_topindx,
        scalar_t* __restrict__ local_range,
        bool* __restrict__ mask,
        const int n_sample) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n_sample) {
     const int xyzshift = idx * 3;
     const float xval = xyz_sampled[xyzshift],  yval = xyz_sampled[xyzshift + 1],  zval = xyz_sampled[xyzshift + 2];
     if (xval >= xyz_min[0] && xval <= xyz_max[0] && yval >= xyz_min[1] && yval <= xyz_max[1] && zval >= xyz_min[2] && zval <= xyz_max[2]){

        const int indx = min(gridX-1, (int)((xval - xyz_min[0]) / units[0]));
        const int indy = min(gridY-1, (int)((yval - xyz_min[1]) / units[1]));
        const int indz = min(gridZ-1, (int)((zval - xyz_min[2]) / units[2]));
        const int cvrg_id = tensoRF_cvrg_inds[indx * gridY * gridZ + indy * gridZ + indz];
        if (cvrg_id >= 0){
           float rel_x, rel_y, rel_z, rx, ry, rz;
           const int tshift = cvrg_id * K;
           const int amount = tensoRF_count[cvrg_id];
           int offset_t = -1, offset_r = -1;
           for (int i = 0; i < amount; i++){
               offset_t = tensoRF_topindx[tshift + i] * 3;
               offset_r = offset_t * 3;
               rel_x = xval - geo_xyz[offset_t];
               rel_y = yval - geo_xyz[offset_t+1];
               rel_z = zval - geo_xyz[offset_t+2];
               rx = rel_x * geo_rot[offset_r] + rel_y * geo_rot[offset_r+3]  + rel_z * geo_rot[offset_r+6];
               ry = rel_x * geo_rot[offset_r+1] + rel_y * geo_rot[offset_r+4]  + rel_z * geo_rot[offset_r+7];
               rz = rel_x * geo_rot[offset_r+2] + rel_y * geo_rot[offset_r+5]  + rel_z * geo_rot[offset_r+8];

               if (abs(rx) < local_range[0] && abs(ry) < local_range[1] && abs(rz) < local_range[2]){
                   mask[idx] = true;
                   break;
               }
           }
        }
     }
  }
}


template <typename scalar_t>
__global__ void filter_xyz_by_sphere_cvrg_cuda_kernel(
        scalar_t* __restrict__ xyz_sampled,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        scalar_t* __restrict__ pnt_xyz,
        scalar_t* __restrict__ units,
        const float radiusl,
        const float radiush,
        const int gridX, const int gridY, const int gridZ,
        int64_t* __restrict__ tensoRF_cvrg_inds,
        int64_t* __restrict__ tensoRF_count,
        int64_t* __restrict__ tensoRF_topindx,
        bool* __restrict__ mask,
        const int K,
        const int n_sample) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n_sample) {
     const int xyzshift = idx * 3;
     const float xval = xyz_sampled[xyzshift],  yval = xyz_sampled[xyzshift + 1],  zval = xyz_sampled[xyzshift + 2];
     if (xval >= xyz_min[0] && xval <= xyz_max[0] && yval >= xyz_min[1] && yval <= xyz_max[1] && zval >= xyz_min[2] && zval <= xyz_max[2]){
         const int indx = min(gridX-1, (int)((xval - xyz_min[0]) / units[0]));
         const int indy = min(gridY-1, (int)((yval - xyz_min[1]) / units[0]));
         const int indz = min(gridZ-1, (int)((zval - xyz_min[2]) / units[0]));
         const int inds = indx * gridY * gridZ + indy * gridZ + indz;
         // if (indx < 0 || indx >= gridX) printf(" indx %d xval %f gridX %d; ", indx, xval, gridX);
         // if (indy < 0 || indy >= gridY) printf(" indy %d yval %f gridY %d; ", indy, yval, gridY);
         // if (indz < 0 || indz >= gridZ) printf(" indz %d zval %f gridZ %d; ", indz, zval, gridZ);
         const int cvrg_id = tensoRF_cvrg_inds[inds];
         const float radiusl_sqr = radiusl * radiusl;
         const float radiush_sqr = radiush * radiush;
         if (cvrg_id >= 0){
            const int tshift = cvrg_id * K;
            const int amount = tensoRF_count[cvrg_id];
            int geoshift = -1;
            float x = 0, y = 0, z = 0, rsqr = 0;
            for (int i = 0; i < amount; i++){
                geoshift = tensoRF_topindx[tshift + i] * 3;
                x = pnt_xyz[geoshift] - xval;
                y = pnt_xyz[geoshift+1] - yval;
                z = pnt_xyz[geoshift+2] - zval;
                rsqr = x*x + y*y + z*z;
                if (rsqr <= radiush_sqr && rsqr >= radiusl_sqr) {
                    mask[idx] = true;
                    break;
                }
            }
         }
     }
  }
}


template <typename scalar_t>
__global__ void fill_grids_tensoRF_cuda_kernel(
        scalar_t* __restrict__ xyz_sampled,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ units,
        int64_t* __restrict__ tensoRF_count,
        int64_t* __restrict__ tensoRF_topindx,
        int64_t* __restrict__ cvrg_inds_map,
        int64_t* __restrict__ sample_grids,
        int64_t* __restrict__ sample_grids_tensoRF,
        int64_t* __restrict__ sample_grids_tensoRF_count,
        const int gridYZ,
        const int gridZ,
        const int K,
        const int num_grids) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < num_grids) {
     const int gridshift = idx * 3;
     const int grid_indx = sample_grids[gridshift];
     const int grid_indy = sample_grids[gridshift + 1];
     const int grid_indz = sample_grids[gridshift + 2];
     const int tensoRF_shift = idx * 8 * K;
     int tensoRF_count = 0;
  }
}


template <typename scalar_t>
__global__ void fill_sample_ind_mask_cuda_kernel(
        scalar_t* __restrict__ xyz_sampled,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ units,
        int64_t* __restrict__ tensoRF_cvrg_inds,
        int64_t* __restrict__ cvrg_inds_map,
        bool* __restrict__ grid_inds_mask,
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
     cvrg_inds_map[inds] = tensoRF_cvrg_inds[inds];
     grid_inds_mask[inds] = true;
     grid_inds_mask[inds + gridYZ] = true;
     grid_inds_mask[inds + gridZ] = true;
     grid_inds_mask[inds + 1] = true;
     grid_inds_mask[inds + gridZ + 1] = true;
     grid_inds_mask[inds + gridYZ + gridZ] = true;
     grid_inds_mask[inds + gridYZ + 1] = true;
     grid_inds_mask[inds + gridYZ + gridZ + 1] = true;
  }
}


template <typename scalar_t>
__global__ void count_sphere_tensoRF_cvrg_cuda_kernel(
        scalar_t* __restrict__ xyz_sampled,
        scalar_t* __restrict__ geo_xyz,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ units,
        int64_t* __restrict__ tensoRF_count,
        int64_t* __restrict__ tensoRF_cvrg_inds,
        int64_t* __restrict__ cvrg_inds,
        int64_t* __restrict__ cvrg_count,
        const int gridYZ,
        const int gridZ,
        const int n_sample) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n_sample) {
     const int xyzshift = idx * 3;
     const int indx = (xyz_sampled[xyzshift] - xyz_min[0]) / units[0];
     const int indy = (xyz_sampled[xyzshift + 1] - xyz_min[1]) / units[0];
     const int indz = (xyz_sampled[xyzshift + 2] - xyz_min[2]) / units[0];

     const int inds = indx * gridYZ + indy * gridZ + indz;
     const int cvrg_id = tensoRF_cvrg_inds[inds];
     cvrg_inds[idx] = cvrg_id;
     cvrg_count[idx] = tensoRF_count[cvrg_id];
     // const int cur_count = tensoRF_count[cvrg_id];
  }
}

template <typename scalar_t>
__global__ void count_tensoRF_cvrg_cuda_kernel(
        scalar_t* __restrict__ xyz_sampled,
        scalar_t* __restrict__ geo_xyz_recenter,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ units,
        int64_t* __restrict__ tensoRF_count,
        int64_t* __restrict__ tensoRF_cvrg_inds,
        int64_t* __restrict__ cvrg_inds,
        int64_t* __restrict__ cvrg_count,
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

template <typename scalar_t>
__global__ void infer_n_samples_cuda_kernel(
        scalar_t* __restrict__ t_min,
        scalar_t* __restrict__ t_max,
        const float stepdist,
        const int n_rays,
        int64_t* __restrict__ n_samples) {
  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
    // at least 1 point for easier implementation in the later sample_pts_on_rays_cuda
    n_samples[i_ray] = max(ceil((t_max[i_ray]-t_min[i_ray]) / stepdist), 1.);
  }
}


template <typename scalar_t>
__global__ void infer_n_samples_shift_cuda_kernel(
        scalar_t* __restrict__ t_min,
        scalar_t* __restrict__ t_max,
        scalar_t* __restrict__ shift,
        const int n_rays,
        int64_t* __restrict__ n_samples) {
  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
    // at least 1 point for easier implementation in the later sample_pts_on_rays_cuda
    n_samples[i_ray] = max(ceil((t_max[i_ray]-t_min[i_ray]) / shift[i_ray]), 1.);
  }
}

template <typename scalar_t>
__global__ void sample_pts_on_rays_cvrg_cuda_kernel(
        scalar_t* __restrict__ rays_start,
        scalar_t* __restrict__ rays_dir,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        scalar_t* __restrict__ units,
        const int gridX,
        const int gridY,
        const int gridZ,
        int64_t* __restrict__ ray_id,
        int64_t* __restrict__ step_id,
        const float stepdist, const int total_len,
        scalar_t* __restrict__ rays_pts,
        bool* __restrict__ tensoRF_cvrg_mask,
        bool* __restrict__ mask_valid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<total_len) {
    const int i_ray = ray_id[idx];
    const int i_step = step_id[idx];

    const int offset_p = idx * 3;
    const int offset_r = i_ray * 3;
    const float dist = stepdist * i_step;
    const float px = rays_start[offset_r  ] + rays_dir[offset_r  ] * dist;
    const float py = rays_start[offset_r+1] + rays_dir[offset_r+1] * dist;
    const float pz = rays_start[offset_r+2] + rays_dir[offset_r+2] * dist;
    rays_pts[offset_p  ] = px;
    rays_pts[offset_p+1] = py;
    rays_pts[offset_p+2] = pz;
    bool out_bound = (xyz_min[0]>px) | (xyz_min[1]>py) | (xyz_min[2]>pz) | \
                        (xyz_max[0]<px) | (xyz_max[1]<py) | (xyz_max[2]<pz);
    if (!out_bound) {
        const int indx = (px - xyz_min[0]) / units[0];
        const int indy = (py - xyz_min[1]) / units[1];
        const int indz = (pz - xyz_min[2]) / units[2];
        mask_valid[idx] = tensoRF_cvrg_mask[max(min(gridX-1,indx),0) * gridY * gridZ + max(min(gridY-1,indy),0) * gridZ + max(min(gridZ-1,indz),0)];
    }
  }
}


template <typename scalar_t>
__global__ void sample_pts_on_rays_rot_cvrg_cuda_kernel(
        scalar_t* __restrict__ rays_start,
        scalar_t* __restrict__ rays_dir,
        scalar_t* __restrict__ geo_xyz,
        scalar_t* __restrict__ geo_rot,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        scalar_t* __restrict__ units,
        scalar_t* __restrict__ local_range,
        const int gridX,
        const int gridY,
        const int gridZ,
        const int K,
        int64_t* __restrict__ ray_id,
        int64_t* __restrict__ step_id,
        const float stepdist,
        const int total_len,
        scalar_t* __restrict__ rays_pts,
        int64_t* __restrict__ tensoRF_cvrg_inds,
        int64_t* __restrict__ tensoRF_count,
        int64_t* __restrict__ tensoRF_topindx,
        bool* __restrict__ mask_valid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<total_len) {
    const int i_ray = ray_id[idx];
    const int i_step = step_id[idx];

    const int offset_p = idx * 3;
    const int offset_r = i_ray * 3;
    const float dist = stepdist * i_step;
    const float px = rays_start[offset_r  ] + rays_dir[offset_r  ] * dist;
    const float py = rays_start[offset_r+1] + rays_dir[offset_r+1] * dist;
    const float pz = rays_start[offset_r+2] + rays_dir[offset_r+2] * dist;
    rays_pts[offset_p  ] = px;
    rays_pts[offset_p+1] = py;
    rays_pts[offset_p+2] = pz;
    bool out_bound = (xyz_min[0]>px) | (xyz_min[1]>py) | (xyz_min[2]>pz) | \
                        (xyz_max[0]<px) | (xyz_max[1]<py) | (xyz_max[2]<pz);
    if (!out_bound) {
        const int indx = max(min(gridX-1,(int)((px - xyz_min[0]) / units[0])),0);
        const int indy = max(min(gridY-1,(int)((py - xyz_min[1]) / units[0])),0);
        const int indz = max(min(gridZ-1,(int)((pz - xyz_min[2]) / units[0])),0);
        const int cvrg_id = tensoRF_cvrg_inds[indx * gridY * gridZ + indy * gridZ + indz];
        if (cvrg_id >= 0){
           float rel_x, rel_y, rel_z, rx, ry, rz;
           const int tshift = cvrg_id * K;
           const int amount = tensoRF_count[cvrg_id];
           int offset_t = -1, offset_r = -1;
           for (int i = 0; i < amount; i++){
               offset_t = tensoRF_topindx[tshift + i] * 3;
               offset_r = offset_t * 3;
               rel_x = px - geo_xyz[offset_t];
               rel_y = py - geo_xyz[offset_t+1];
               rel_z = pz - geo_xyz[offset_t+2];
               rx = rel_x * geo_rot[offset_r] + rel_y * geo_rot[offset_r+3]  + rel_z * geo_rot[offset_r+6];
               ry = rel_x * geo_rot[offset_r+1] + rel_y * geo_rot[offset_r+4]  + rel_z * geo_rot[offset_r+7];
               rz = rel_x * geo_rot[offset_r+2] + rel_y * geo_rot[offset_r+5]  + rel_z * geo_rot[offset_r+8];

               if (abs(rx) < local_range[0] && abs(ry) < local_range[1] && abs(rz) < local_range[2]){
                   mask_valid[idx] = true;
                   break;
               }
           }
        }
    }
  }
}


template <typename scalar_t>
__global__ void sample_pts_on_rays_dist_cuda_kernel(
        scalar_t* __restrict__ rays_start,
        scalar_t* __restrict__ rays_dir,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        int64_t* __restrict__ ray_id,
        int64_t* __restrict__ step_id,
        const float stepdist,
        scalar_t* __restrict__ shift,
        const int total_len,
        scalar_t* __restrict__ rays_pts,
        bool* __restrict__ mask_outbbox) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<total_len) {
    const int i_ray = ray_id[idx];
    const int i_step = step_id[idx];

    const int offset_p = idx * 3;
    const int offset_r = i_ray * 3;
    const float dist = stepdist * i_step + shift[i_ray];
    const float px = rays_start[offset_r  ] + rays_dir[offset_r  ] * dist;
    const float py = rays_start[offset_r+1] + rays_dir[offset_r+1] * dist;
    const float pz = rays_start[offset_r+2] + rays_dir[offset_r+2] * dist;
    rays_pts[offset_p  ] = px;
    rays_pts[offset_p+1] = py;
    rays_pts[offset_p+2] = pz;
    mask_outbbox[idx] = (xyz_min[0]>px) | (xyz_min[1]>py) | (xyz_min[2]>pz) | \
                        (xyz_max[0]<px) | (xyz_max[1]<py) | (xyz_max[2]<pz);
  }
}


template <typename scalar_t>
__global__ void sample_pts_on_rays_dist_cvrg_cuda_kernel(
        scalar_t* __restrict__ rays_start,
        scalar_t* __restrict__ rays_dir,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        scalar_t* __restrict__ units,
        const int gridX,
        const int gridY,
        const int gridZ,
        int64_t* __restrict__ ray_id,
        int64_t* __restrict__ step_id,
        const float stepdist,
        scalar_t* __restrict__ shift,
        const int total_len,
        scalar_t* __restrict__ rays_pts,
        bool* __restrict__ tensoRF_cvrg_mask,
        bool* __restrict__ mask_valid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<total_len) {
    const int i_ray = ray_id[idx];
    const int i_step = step_id[idx];

    const int offset_p = idx * 3;
    const int offset_r = i_ray * 3;
    const float dist = shift[i_ray] * i_step;
    const float px = rays_start[offset_r  ] + rays_dir[offset_r  ] * dist;
    const float py = rays_start[offset_r+1] + rays_dir[offset_r+1] * dist;
    const float pz = rays_start[offset_r+2] + rays_dir[offset_r+2] * dist;
    rays_pts[offset_p  ] = px;
    rays_pts[offset_p+1] = py;
    rays_pts[offset_p+2] = pz;
    bool out_bound = (xyz_min[0]>px) | (xyz_min[1]>py) | (xyz_min[2]>pz) | \
                        (xyz_max[0]<px) | (xyz_max[1]<py) | (xyz_max[2]<pz);
    if (!out_bound) {
        const int indx = (px - xyz_min[0]) / units[0];
        const int indy = (py - xyz_min[1]) / units[1];
        const int indz = (pz - xyz_min[2]) / units[2];
        mask_valid[idx] = tensoRF_cvrg_mask[max(min(gridX-1,indx),0) * gridY * gridZ + max(min(gridY-1,indy),0) * gridZ + max(min(gridZ-1,indz),0)];
    }
  }
}



template <typename scalar_t>
__global__ void sample_pts_on_rays_sphere_cvrg_cuda_kernel(
        scalar_t* __restrict__ rays_start,
        scalar_t* __restrict__ rays_dir,
        scalar_t* __restrict__ pnt_xyz,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        scalar_t* __restrict__ units,
        const float radiusl,
        const float radiush,
        const int gridX,
        const int gridY,
        const int gridZ,
        const int K,
        int64_t* __restrict__ ray_id,
        int64_t* __restrict__ step_id,
        const float stepdist, const int total_len,
        scalar_t* __restrict__ rays_pts,
        int64_t* __restrict__ tensoRF_cvrg_inds,
        int64_t* __restrict__ tensoRF_count,
        int64_t* __restrict__ tensoRF_topindx,
        bool* __restrict__ mask_valid) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<total_len) {
    const int i_ray = ray_id[idx];
    const int i_step = step_id[idx];

    const int offset_p = idx * 3;
    const int offset_r = i_ray * 3;
    const float dist = stepdist * i_step;
    const float px = rays_start[offset_r  ] + rays_dir[offset_r  ] * dist;
    const float py = rays_start[offset_r+1] + rays_dir[offset_r+1] * dist;
    const float pz = rays_start[offset_r+2] + rays_dir[offset_r+2] * dist;
    rays_pts[offset_p  ] = px;
    rays_pts[offset_p+1] = py;
    rays_pts[offset_p+2] = pz;
    bool out_bound = (xyz_min[0]>px) | (xyz_min[1]>py) | (xyz_min[2]>pz) | \
                        (xyz_max[0]<px) | (xyz_max[1]<py) | (xyz_max[2]<pz);
    if (!out_bound) {
        const int indx = max(min(gridX-1,(int)((px - xyz_min[0]) / units[0])),0);
        const int indy = max(min(gridY-1,(int)((py - xyz_min[1]) / units[0])),0);
        const int indz = max(min(gridZ-1,(int)((pz - xyz_min[2]) / units[0])),0);

        const int inds = indx * gridY * gridZ + indy * gridZ + indz;
         // if (indx < 0 || indx >= gridX) printf(" indx %d xval %f gridX %d; ", indx, xval, gridX);
         // if (indy < 0 || indy >= gridY) printf(" indy %d yval %f gridY %d; ", indy, yval, gridY);
         // if (indz < 0 || indz >= gridZ) printf(" indz %d zval %f gridZ %d; ", indz, zval, gridZ);
        const int cvrg_id = tensoRF_cvrg_inds[inds];
        const float radiusl_sqr = radiusl * radiusl;
        const float radiush_sqr = radiush * radiush;
        if (cvrg_id >= 0){
           const int tshift = cvrg_id * K;
           const int amount = tensoRF_count[cvrg_id];
           int tid = -1;
           float x = 0, y = 0, z = 0, rsqr = 0.0;
           for (int i = 0; i < amount; i++){
               tid = tensoRF_topindx[tshift + i] * 3;
               x = px - pnt_xyz[tid];
               y = py - pnt_xyz[tid+1];
               z = pz - pnt_xyz[tid+2];
               // printf("xyzsqr %f %f;   ", x*x + y*y + z*z, radius_sqr);
               rsqr = x*x + y*y + z*z;
               if (rsqr <= radiush_sqr && rsqr >= radiusl_sqr) {
                   mask_valid[idx] = true;
                   break;
               }
           }
        }
    }
  }
}

template <typename scalar_t>
__global__ void infer_t_minmax_shift_cuda_kernel(
        scalar_t* __restrict__ rays_o,
        scalar_t* __restrict__ rays_d,
        scalar_t* __restrict__ shift,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        const float near, const float far, const int n_rays,
        scalar_t* __restrict__ t_min,
        scalar_t* __restrict__ t_max) {
  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
    const int offset = i_ray * 3;
    float vx = ((rays_d[offset  ]==0) ? 1e-6 : rays_d[offset  ]);
    float vy = ((rays_d[offset+1]==0) ? 1e-6 : rays_d[offset+1]);
    float vz = ((rays_d[offset+2]==0) ? 1e-6 : rays_d[offset+2]);
    float ax = (xyz_max[0] - rays_o[offset  ]) / vx;
    float ay = (xyz_max[1] - rays_o[offset+1]) / vy;
    float az = (xyz_max[2] - rays_o[offset+2]) / vz;
    float bx = (xyz_min[0] - rays_o[offset  ]) / vx;
    float by = (xyz_min[1] - rays_o[offset+1]) / vy;
    float bz = (xyz_min[2] - rays_o[offset+2]) / vz;
    t_min[i_ray] = max(min(max(max(min(ax, bx), min(ay, by)), min(az, bz)), far), near) - shift[i_ray];
    t_max[i_ray] = max(min(min(min(max(ax, bx), max(ay, by)), max(az, bz)), far), near);
  }
}


template <typename scalar_t>
__global__ void infer_t_minmax_cuda_kernel(
        scalar_t* __restrict__ rays_o,
        scalar_t* __restrict__ rays_d,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        const float near, const float far, const int n_rays,
        scalar_t* __restrict__ t_min,
        scalar_t* __restrict__ t_max) {
  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
    const int offset = i_ray * 3;
    float vx = ((rays_d[offset  ]==0) ? 1e-6 : rays_d[offset  ]);
    float vy = ((rays_d[offset+1]==0) ? 1e-6 : rays_d[offset+1]);
    float vz = ((rays_d[offset+2]==0) ? 1e-6 : rays_d[offset+2]);
    float ax = (xyz_max[0] - rays_o[offset  ]) / vx;
    float ay = (xyz_max[1] - rays_o[offset+1]) / vy;
    float az = (xyz_max[2] - rays_o[offset+2]) / vz;
    float bx = (xyz_min[0] - rays_o[offset  ]) / vx;
    float by = (xyz_min[1] - rays_o[offset+1]) / vy;
    float bz = (xyz_min[2] - rays_o[offset+2]) / vz;
    t_min[i_ray] = max(min(max(max(min(ax, bx), min(ay, by)), min(az, bz)), far), near);
    t_max[i_ray] = max(min(min(min(max(ax, bx), max(ay, by)), max(az, bz)), far), near);
  }
}

__global__ void __set_1_at_ray_seg_start(
        int64_t* __restrict__ ray_id,
        int64_t* __restrict__ N_steps_cumsum,
        const int n_rays) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(0<idx && idx<n_rays) {
    ray_id[N_steps_cumsum[idx-1]] = 1;
  }
}


__global__ void __set_step_id(
        int64_t* __restrict__ step_id,
        int64_t* __restrict__ ray_id,
        int64_t* __restrict__ N_steps_cumsum,
        const int total_len) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<total_len) {
      const int rid = ray_id[idx];
      step_id[idx] = idx - ((rid!=0) ? N_steps_cumsum[rid-1] : 0);
    }
}

__global__ void __set_n_at_ray_seg_start(
        int64_t* __restrict__ ray_id,
        int64_t* __restrict__ N_steps_cumsum,
        int64_t* __restrict__ N_steps,
        const int n_rays) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(0<idx && idx<n_rays && N_steps[idx] > 0) {
    int count = 1;
    for (int j = idx-1; j > 0; j--){
        if (N_steps[j] > 0) break;
        count++;
    }
    ray_id[N_steps_cumsum[idx-1]] = count;
  }
}


template <typename scalar_t>
__global__ void infer_ray_start_dir_cuda_kernel(
        scalar_t* __restrict__ rays_o,
        scalar_t* __restrict__ rays_d,
        scalar_t* __restrict__ t_min,
        const int n_rays,
        scalar_t* __restrict__ rays_start,
        scalar_t* __restrict__ rays_dir) {
  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
    const int offset = i_ray * 3;
    const float rnorm = sqrt(
            rays_d[offset  ]*rays_d[offset  ] +\
            rays_d[offset+1]*rays_d[offset+1] +\
            rays_d[offset+2]*rays_d[offset+2]);
    rays_start[offset  ] = rays_o[offset  ] + rays_d[offset  ] * t_min[i_ray];
    rays_start[offset+1] = rays_o[offset+1] + rays_d[offset+1] * t_min[i_ray];
    rays_start[offset+2] = rays_o[offset+2] + rays_d[offset+2] * t_min[i_ray];
    rays_dir  [offset  ] = rays_d[offset  ] / rnorm;
    rays_dir  [offset+1] = rays_d[offset+1] / rnorm;
    rays_dir  [offset+2] = rays_d[offset+2] / rnorm;
  }
}


template <typename scalar_t>
__global__ void filter_ray_by_cvrg_cuda_kernel(
    scalar_t* __restrict__ xyz_sampled,
    bool* __restrict__ mask_inbox,
    scalar_t* __restrict__ units,
    scalar_t* __restrict__ xyz_min,
    scalar_t* __restrict__ xyz_max,
    bool* __restrict__ tensoRF_cvrg_mask,
    const int n_rays,
    const int n_samp,
    const int gridX,
    const int gridY,
    const int gridZ,
    bool* __restrict__ sample_mask
   ) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  const int i_ray = idx / n_samp;
  if(idx<n_rays*n_samp && i_ray < n_rays && mask_inbox[idx]) {
     const int xyzshift = idx * 3;
     const int indx = min(gridX-1, (int)((xyz_sampled[xyzshift] - xyz_min[0]) / units[0]));
     const int indy = min(gridY-1, (int)((xyz_sampled[xyzshift + 1] - xyz_min[1]) / units[1]));
     const int indz = min(gridZ-1, (int)((xyz_sampled[xyzshift + 2] - xyz_min[2]) / units[2]));
     const int inds = indx * gridY * gridZ + indy * gridZ + indz;
     sample_mask[idx] = tensoRF_cvrg_mask[inds];
  }
}


template <typename scalar_t>
__global__ void get_geo_inds_cuda_kernel_bk(
        scalar_t* __restrict__ local_range,
        int64_t* __restrict__ local_dims,
        int64_t* __restrict__ gridSize,
        scalar_t* __restrict__ units,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        scalar_t* __restrict__ pnt_xyz,
        scalar_t* __restrict__ pnt_xyz_recenter,
        int64_t* __restrict__ tensoRF_cvrg_inds,
        const int n_pts
        ) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<n_pts) {
     const int i_shift = idx * 3;
     const float px = pnt_xyz[i_shift];
     const float py = pnt_xyz[i_shift+1];
     const float pz = pnt_xyz[i_shift+2];

     const int xind = (px - xyz_min[0]) / units[0];
     const int yind = (py - xyz_min[1]) / units[1];
     const int zind = (pz - xyz_min[2]) / units[2];
     const int gx = gridSize[0];
     const int gy = gridSize[1];
     const int gz = gridSize[2];
     const int lx = (local_dims[0] - 1) / 2;
     const int ly = (local_dims[1] - 1) / 2;
     const int lz = (local_dims[2] - 1) / 2;
     pnt_xyz_recenter[i_shift] = xyz_min[0] + (xind + 0.5) * units[0];
     pnt_xyz_recenter[i_shift+1] = xyz_min[1] + (yind + 0.5) * units[1];
     pnt_xyz_recenter[i_shift+2] = xyz_min[2] + (zind + 0.5) * units[2];
     const int xmin = max(min(xind-lx, gx), 0);
     const int xmax = max(min(xind+lx+1, gx), 0);
     const int ymin = max(min(yind-ly, gy), 0);
     const int ymax = max(min(yind+ly+1, gy), 0);
     const int zmin = max(min(zind-lz, gz), 0);
     const int zmax = max(min(zind+lz+1, gz), 0);
     for (int i = xmin; i < xmax; i++){
        int shiftx = i * gy * gz;
        for (int j = ymin; j < ymax; j++){
            int shifty = j * gz;
            for (int k = zmin; k < zmax; k++){
                tensoRF_cvrg_inds[shiftx + shifty + k] = 1;
            }
        }
     }
  }
}

template <typename scalar_t>
__global__ void get_geo_inds_cuda_kernel(
        scalar_t* __restrict__ local_range,
        int64_t* __restrict__ local_dims,
        int64_t* __restrict__ gridSize,
        scalar_t* __restrict__ units,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        scalar_t* __restrict__ pnt_xyz,
        scalar_t* __restrict__ pnt_xyz_recenter,
        int64_t* __restrict__ tensoRF_cvrg_inds,
        const int n_pts
        ) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<n_pts) {
     const int i_shift = idx * 3;
     const float px = pnt_xyz[i_shift];
     const float py = pnt_xyz[i_shift+1];
     const float pz = pnt_xyz[i_shift+2];

     const int xind = (px - xyz_min[0]) / units[0];
     const int yind = (py - xyz_min[1]) / units[1];
     const int zind = (pz - xyz_min[2]) / units[2];
     const int gx = gridSize[0];
     const int gy = gridSize[1];
     const int gz = gridSize[2];
     const int lx = floor(local_dims[0] / 2);
     const int ly = floor(local_dims[1] / 2);
     const int lz = floor(local_dims[2] / 2);
     pnt_xyz_recenter[i_shift] = xyz_min[0] + (xind + 0.5) * units[0];
     pnt_xyz_recenter[i_shift+1] = xyz_min[1] + (yind + 0.5) * units[1];
     pnt_xyz_recenter[i_shift+2] = xyz_min[2] + (zind + 0.5) * units[2];
     const int xmin = max(min(xind-lx, gx), 0);
     const int xmax = max(min(xind+lx+1, gx), 0);
     const int ymin = max(min(yind-ly, gy), 0);
     const int ymax = max(min(yind+ly+1, gy), 0);
     const int zmin = max(min(zind-lz, gz), 0);
     const int zmax = max(min(zind+lz+1, gz), 0);
     for (int i = xmin; i < xmax; i++){
        int shiftx = i * gy * gz;
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
__global__ void get_geo_sphere_inds_cuda_kernel(
        const float radiusl,
        const float radiush,
        int64_t* __restrict__ gridSize,
        scalar_t* __restrict__ units,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        scalar_t* __restrict__ pnt_xyz,
        int64_t* __restrict__ tensoRF_cvrg_inds,
        const int n_pts
        ) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<n_pts) {
     const int i_shift = idx * 3;
     const float px = pnt_xyz[i_shift];
     const float py = pnt_xyz[i_shift+1];
     const float pz = pnt_xyz[i_shift+2];
     const int gx = gridSize[0];
     const int gy = gridSize[1];
     const int gz = gridSize[2];

     const int linds_x = (px - radiush - xyz_min[0]) / units[0];
     const int hinds_x = (px + radiush - xyz_min[0]) / units[0];
     const int linds_y = (py - radiush - xyz_min[1]) / units[0];
     const int hinds_y = (py + radiush - xyz_min[1]) / units[0];
     const int linds_z = (pz - radiush - xyz_min[2]) / units[0];
     const int hinds_z = (pz + radiush - xyz_min[2]) / units[0];

     const int xmin = max(min(linds_x, gx), 0);
     const int xmax = max(min(hinds_x+1, gx), 0);
     const int ymin = max(min(linds_y, gy), 0);
     const int ymax = max(min(hinds_y+1, gy), 0);
     const int zmin = max(min(linds_z, gz), 0);
     const int zmax = max(min(hinds_z+1, gz), 0);
     float cx = 0, cy = 0, cz = 0;
     float radiusl_sqr = radiusl * radiusl;
     float radiush_sqr = radiush * radiush;
     float sqr1,sqr2,sqr3,sqr4,sqr5,sqr6,sqr7,sqr8;
     for (int i = xmin; i < xmax; i++){
        int shiftx = i * gy * gz;
        for (int j = ymin; j < ymax; j++){
            int shifty = j * gz;
            for (int k = zmin; k < zmax; k++){
                cx = xyz_min[0] + i * units[0];
                cy = xyz_min[1] + j * units[0];
                cz = xyz_min[2] + k * units[0];
                sqr1 = (cx - px)*(cx - px) + (cy - py)*(cy - py) + (cz - pz)*(cz - pz);
                sqr2 = (cx + units[0] - px)*(cx + units[0] - px) + (cy - py)*(cy - py) + (cz - pz)*(cz - pz);
                sqr3 = (cx - px)*(cx - px) + (cy + units[0] - py)*(cy + units[0] - py) + (cz - pz)*(cz - pz);
                sqr4 = (cx - px)*(cx - px) + (cy - py)*(cy - py) + (cz + units[0] - pz)*(cz + units[0] - pz);
                sqr5 = (cx + units[0] - px)*(cx + units[0] - px) + (cy + units[0] - py)*(cy + units[0] - py) + (cz - pz)*(cz - pz);
                sqr6 = (cx + units[0] - px)*(cx + units[0] - px) + (cy - py)*(cy - py) + (cz + units[0] - pz)*(cz + units[0] - pz);
                sqr7 = (cx - px)*(cx - px) + (cy + units[0] - py)*(cy + units[0] - py) + (cz + units[0] - pz)*(cz + units[0] - pz);
                sqr8 = (cx + units[0] - px)*(cx + units[0] - px) + (cy + units[0] - py)*(cy + units[0] - py) + (cz + units[0] - pz)*(cz + units[0] - pz);
                if ((sqr1 <= radiush_sqr && sqr1 >= radiusl_sqr) || (sqr2 <= radiush_sqr && sqr2 >= radiusl_sqr) || (sqr3 <= radiush_sqr && sqr3 >= radiusl_sqr) || (sqr4 <= radiush_sqr && sqr4 >= radiusl_sqr) || (sqr5 <= radiush_sqr && sqr5 >= radiusl_sqr) || (sqr6 <= radiush_sqr && sqr6 >= radiusl_sqr) || (sqr7 <= radiush_sqr && sqr7 >= radiusl_sqr) || (sqr8 <= radiush_sqr && sqr8 >= radiusl_sqr)){
                    tensoRF_cvrg_inds[shiftx + shifty + k] = 1;
                }
            }
        }
     }
  }
}


template <typename scalar_t>
__global__ void fill_geo_inds_cuda_kernel(
        int64_t* __restrict__ local_dims,
        int64_t* __restrict__ gridSize,
        int64_t* __restrict__ tensoRF_count,
        int64_t* __restrict__ tensoRF_topindx,
        scalar_t* __restrict__ units,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        scalar_t* __restrict__ pnt_xyz,
        scalar_t* __restrict__ pnt_xyz_recenter,
        int64_t* __restrict__ tensoRF_cvrg_inds,
        const int max_tensoRF,
        const int n_pts,
        const int gridSizeAll
        ) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<gridSizeAll && tensoRF_cvrg_inds[idx] >= 0) {
     const int cvrg_id = tensoRF_cvrg_inds[idx];
     const int tshift = cvrg_id * max_tensoRF;
     const float range_dx = local_dims[0] * 0.5 * units[0];
     const float range_dy = local_dims[1] * 0.5 * units[1];
     const float range_dz = local_dims[2] * 0.5 * units[2];
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
         const float pcx = pnt_xyz_recenter[i_shift];
         const float pcy = pnt_xyz_recenter[i_shift+1];
         const float pcz = pnt_xyz_recenter[i_shift+2];
         if (abs(pcx - cx) < range_dx && abs(pcy - cy) < range_dy && abs(pcz - cz) < range_dz){
            float xdiff = pnt_xyz[i_shift] - cx;
            float ydiff = pnt_xyz[i_shift+1] - cy;
            float zdiff = pnt_xyz[i_shift+2] - cz;
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


template <typename scalar_t>
__global__ void fill_geo_sphere_inds_cuda_kernel(
        const float radiusl,
        const float radiush,
        int64_t* __restrict__ gridSize,
        int64_t* __restrict__ tensoRF_count,
        int64_t* __restrict__ tensoRF_topindx,
        scalar_t* __restrict__ units,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        scalar_t* __restrict__ pnt_xyz,
        int64_t* __restrict__ tensoRF_cvrg_inds,
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
     const float cy = xyz_min[1] + (indy + 0.5) * units[0];
     const float cz = xyz_min[2] + (indz + 0.5) * units[0];
     float xyz2Buffer[8];
     int kid = 0, far_ind = 0;
     float far2 = 0.0, xyz2 = 0, radiusl_sqr = radiusl * radiusl, radiush_sqr = radiush * radiush;
     float ushift = 0.5 * units[0];
     float sqr1,sqr2,sqr3,sqr4,sqr5,sqr6,sqr7,sqr8;
     for (int i = 0; i < n_pts; i++){
         const int i_shift = i * 3;
         const float px = pnt_xyz[i_shift];
         const float py = pnt_xyz[i_shift+1];
         const float pz = pnt_xyz[i_shift+2];
         sqr1 = (px - cx - ushift) * (px - cx - ushift) + (py - cy - ushift) * (py - cy - ushift) + (pz - cz - ushift) * (pz - cz - ushift);
         sqr2 = (px - cx + ushift) * (px - cx + ushift) + (py - cy - ushift) * (py - cy - ushift) + (pz - cz - ushift) * (pz - cz - ushift);
         sqr3 = (px - cx - ushift) * (px - cx - ushift) + (py - cy + ushift) * (py - cy + ushift) + (pz - cz - ushift) * (pz - cz - ushift);
         sqr4 = (px - cx - ushift) * (px - cx - ushift) + (py - cy - ushift) * (py - cy - ushift) + (pz - cz + ushift) * (pz - cz + ushift);
         sqr5 = (px - cx + ushift) * (px - cx + ushift) + (py - cy - ushift) * (py - cy - ushift) + (pz - cz + ushift) * (pz - cz + ushift);
         sqr6 = (px - cx - ushift) * (px - cx - ushift) + (py - cy + ushift) * (py - cy + ushift) + (pz - cz + ushift) * (pz - cz + ushift);
         sqr7 = (px - cx + ushift) * (px - cx + ushift) + (py - cy + ushift) * (py - cy + ushift) + (pz - cz - ushift) * (pz - cz - ushift);
         sqr8 = (px - cx + ushift) * (px - cx + ushift) + (py - cy + ushift) * (py - cy + ushift) + (pz - cz + ushift) * (pz - cz + ushift);
         if ((sqr1 <= radiush_sqr && sqr1 >= radiusl_sqr) ||
            (sqr2 <= radiush_sqr && sqr2 >= radiusl_sqr) ||
            (sqr3 <= radiush_sqr && sqr3 >= radiusl_sqr) ||
            (sqr4 <= radiush_sqr && sqr4 >= radiusl_sqr) ||
            (sqr5 <= radiush_sqr && sqr5 >= radiusl_sqr) ||
            (sqr6 <= radiush_sqr && sqr6 >= radiusl_sqr) ||
            (sqr7 <= radiush_sqr && sqr7 >= radiusl_sqr) ||
            (sqr8 <= radiush_sqr && sqr8 >= radiusl_sqr)
         ){
            xyz2 = (px - cx) * (px - cx) + (py - cy) * (py - cy) + (pz - cz) * (pz - cz);
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


std::vector<torch::Tensor> build_tensoRF_map_cuda(
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
  auto pnt_xyz_recenter = torch::zeros({n_pts, 3}, pnt_xyz.options());
  auto tensoRF_cvrg_inds = torch::zeros({gridSizex, gridSizey, gridSizez}, torch::dtype(torch::kInt64).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(pnt_xyz.type(), "get_geo_inds", ([&] {
    get_geo_inds_cuda_kernel<scalar_t><<<(n_pts+threads-1)/threads, threads>>>(
        local_range.data<scalar_t>(),
        local_dims.data<int64_t>(),
        gridSize.data<int64_t>(),
        units.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        pnt_xyz.data<scalar_t>(),
        pnt_xyz_recenter.data<scalar_t>(),
        tensoRF_cvrg_inds.data<int64_t>(),
        n_pts);
  }));

  tensoRF_cvrg_inds = tensoRF_cvrg_inds.view(-1).cumsum(0) * tensoRF_cvrg_inds.view(-1);
  const int num_cvrg = tensoRF_cvrg_inds.max().item<int>();
  tensoRF_cvrg_inds = (tensoRF_cvrg_inds - 1).view({gridSizex, gridSizey, gridSizez});

  auto tensoRF_topindx = torch::full({num_cvrg, max_tensoRF}, -1, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto tensoRF_count = torch::zeros({num_cvrg}, torch::dtype(torch::kInt64).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(pnt_xyz.type(), "fill_geo_inds", ([&] {
    fill_geo_inds_cuda_kernel<scalar_t><<<(gridSizeAll+threads-1)/threads, threads>>>(
        local_dims.data<int64_t>(),
        gridSize.data<int64_t>(),
        tensoRF_count.data<int64_t>(),
        tensoRF_topindx.data<int64_t>(),
        units.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        pnt_xyz.data<scalar_t>(),
        pnt_xyz_recenter.data<scalar_t>(),
        tensoRF_cvrg_inds.data<int64_t>(),
        max_tensoRF,
        n_pts,
        gridSizeAll);
  }));
  return {tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx, pnt_xyz_recenter};
}



std::vector<torch::Tensor> build_sphere_tensoRF_map_cuda(
        torch::Tensor pnt_xyz,
        torch::Tensor gridSize,
        torch::Tensor xyz_min,
        torch::Tensor xyz_max,
        torch::Tensor units,
        const float radiusl,
        const float radiush,
        torch::Tensor local_dims,
        const int max_tensoRF) {
  const int threads = 256;
  const int n_pts = pnt_xyz.size(0);
  const int gridSizex = gridSize[0].item<int>();
  const int gridSizey = gridSize[1].item<int>();
  const int gridSizez = gridSize[2].item<int>();
  const int gridSizeAll = gridSizex * gridSizey * gridSizez;
  auto tensoRF_cvrg_inds = torch::zeros({gridSizex, gridSizey, gridSizez}, torch::dtype(torch::kInt64).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(pnt_xyz.type(), "get_geo_sphere_inds", ([&] {
    get_geo_sphere_inds_cuda_kernel<scalar_t><<<(n_pts+threads-1)/threads, threads>>>(
        radiusl,
        radiush,
        gridSize.data<int64_t>(),
        units.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        pnt_xyz.data<scalar_t>(),
        tensoRF_cvrg_inds.data<int64_t>(),
        n_pts);
  }));

  tensoRF_cvrg_inds = tensoRF_cvrg_inds.view(-1).cumsum(0) * tensoRF_cvrg_inds.view(-1);
  const int num_cvrg = tensoRF_cvrg_inds.max().item<int>();
  tensoRF_cvrg_inds = (tensoRF_cvrg_inds - 1).view({gridSizex, gridSizey, gridSizez});
  // printf("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!num_cvrg %d   ", num_cvrg);
  auto tensoRF_topindx = torch::full({num_cvrg, max_tensoRF}, -1, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto tensoRF_count = torch::zeros({num_cvrg}, torch::dtype(torch::kInt64).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(pnt_xyz.type(), "fill_geo_sphere_inds", ([&] {
    fill_geo_sphere_inds_cuda_kernel<scalar_t><<<(gridSizeAll+threads-1)/threads, threads>>>(
        radiusl,
        radiush,
        gridSize.data<int64_t>(),
        tensoRF_count.data<int64_t>(),
        tensoRF_topindx.data<int64_t>(),
        units.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        pnt_xyz.data<scalar_t>(),
        tensoRF_cvrg_inds.data<int64_t>(),
        max_tensoRF,
        n_pts,
        gridSizeAll);
  }));
  return {tensoRF_cvrg_inds, tensoRF_count, tensoRF_topindx};
}


torch::Tensor filter_ray_by_cvrg_cuda(
        torch::Tensor xyz_sampled,
        torch::Tensor mask_inbox,
        torch::Tensor units,
        torch::Tensor xyz_min,
        torch::Tensor xyz_max,
        torch::Tensor tensoRF_cvrg_mask){
  const int n_rays = xyz_sampled.size(0);
  const int n_samp = xyz_sampled.size(1);
  const int gridX = tensoRF_cvrg_mask.size(0);
  const int gridY = tensoRF_cvrg_mask.size(1);
  const int gridZ = tensoRF_cvrg_mask.size(2);
  const int threads = 256; // 256

  auto sample_mask = torch::zeros({n_rays, n_samp}, torch::dtype(torch::kBool).device(torch::kCUDA));
  if(n_rays==0 || n_samp==0) {
    return sample_mask;
  }
  const int blocks = (n_rays * n_samp + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(xyz_sampled.type(), "filter_ray_by_cvrg_cuda", ([&] {
    filter_ray_by_cvrg_cuda_kernel<scalar_t><<<blocks, threads>>>(
        xyz_sampled.data<scalar_t>(),
        mask_inbox.data<bool>(),
        units.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        tensoRF_cvrg_mask.data<bool>(),
        n_rays,
        n_samp,
        gridX,
        gridY,
        gridZ,
        sample_mask.data<bool>());
  }));
  return sample_mask;
}


std::vector<torch::Tensor> infer_t_minmax_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far) {
  const int n_rays = rays_o.size(0);
  auto t_min = torch::empty({n_rays}, rays_o.options());
  auto t_max = torch::empty({n_rays}, rays_o.options());

  const int threads = 256;
  const int blocks = (n_rays + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "infer_t_minmax_cuda", ([&] {
    infer_t_minmax_cuda_kernel<scalar_t><<<blocks, threads>>>(
        rays_o.data<scalar_t>(),
        rays_d.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        near, far, n_rays,
        t_min.data<scalar_t>(),
        t_max.data<scalar_t>());
  }));

  return {t_min, t_max};
}

std::vector<torch::Tensor> infer_t_minmax_shift_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor shift, torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far) {
  const int n_rays = rays_o.size(0);
  auto t_min = torch::empty({n_rays}, rays_o.options());
  auto t_max = torch::empty({n_rays}, rays_o.options());

  const int threads = 256;
  const int blocks = (n_rays + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "infer_t_minmax_shift_cuda", ([&] {
    infer_t_minmax_shift_cuda_kernel<scalar_t><<<blocks, threads>>>(
        rays_o.data<scalar_t>(),
        rays_d.data<scalar_t>(),
        shift.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        near, far, n_rays,
        t_min.data<scalar_t>(),
        t_max.data<scalar_t>());
  }));

  return {t_min, t_max};
}


std::vector<torch::Tensor> infer_ray_start_dir_cuda(torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor t_min) {
  const int n_rays = rays_o.size(0);
  const int threads = 256;
  const int blocks = (n_rays + threads - 1) / threads;
  auto rays_start = torch::empty_like(rays_o);
  auto rays_dir = torch::empty_like(rays_o);
  AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "infer_ray_start_dir_cuda", ([&] {
    infer_ray_start_dir_cuda_kernel<scalar_t><<<blocks, threads>>>(
        rays_o.data<scalar_t>(),
        rays_d.data<scalar_t>(),
        t_min.data<scalar_t>(),
        n_rays,
        rays_start.data<scalar_t>(),
        rays_dir.data<scalar_t>());
  }));
  return {rays_start, rays_dir};
}


torch::Tensor infer_n_samples_shift_cuda(torch::Tensor t_min, torch::Tensor t_max, torch::Tensor shift) {
  const int n_rays = t_min.size(0);
  auto n_samples = torch::empty({n_rays}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  const int threads = 256;
  const int blocks = (n_rays + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(t_min.type(), "infer_n_samples_shift_cuda", ([&] {
    infer_n_samples_shift_cuda_kernel<scalar_t><<<blocks, threads>>>(
        t_min.data<scalar_t>(),
        t_max.data<scalar_t>(),
        shift.data<scalar_t>(),
        n_rays,
        n_samples.data<int64_t>());
  }));
  return n_samples;
}

torch::Tensor infer_n_samples_cuda(torch::Tensor t_min, torch::Tensor t_max, const float stepdist) {
  const int n_rays = t_min.size(0);
  auto n_samples = torch::empty({n_rays}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  const int threads = 256;
  const int blocks = (n_rays + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(t_min.type(), "infer_n_samples_cuda", ([&] {
    infer_n_samples_cuda_kernel<scalar_t><<<blocks, threads>>>(
        t_min.data<scalar_t>(),
        t_max.data<scalar_t>(),
        stepdist,
        n_rays,
        n_samples.data<int64_t>());
  }));
  return n_samples;
}

std::vector<torch::Tensor> sample_pts_on_rays_cvrg_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d,
        torch::Tensor tensoRF_cvrg_mask, torch::Tensor units,
        torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far, const float stepdist) {
  const int threads = 256;
  const int n_rays = rays_o.size(0);

  const int gridX = tensoRF_cvrg_mask.size(0);
  const int gridY = tensoRF_cvrg_mask.size(1);
  const int gridZ = tensoRF_cvrg_mask.size(2);

  // Compute ray-bbox intersection
  auto t_minmax = infer_t_minmax_cuda(rays_o, rays_d, xyz_min, xyz_max, near, far);
  auto t_min = t_minmax[0];
  auto t_max = t_minmax[1];

  // Compute the number of points required.
  // Assign ray index and step index to each.
  auto N_steps = infer_n_samples_cuda(t_min, t_max, stepdist);
  auto N_steps_cumsum = N_steps.cumsum(0);
  const int total_len = N_steps.sum().item<int>();
  auto ray_id = torch::zeros({total_len}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  __set_1_at_ray_seg_start<<<(n_rays+threads-1)/threads, threads>>>(ray_id.data<int64_t>(), N_steps_cumsum.data<int64_t>(), n_rays);
  ray_id.cumsum_(0);
  auto step_id = torch::empty({total_len}, ray_id.options());
  __set_step_id<<<(total_len+threads-1)/threads, threads>>>(
        step_id.data<int64_t>(), ray_id.data<int64_t>(), N_steps_cumsum.data<int64_t>(), total_len);

  // Compute the global xyz of each point
  auto rays_start_dir = infer_ray_start_dir_cuda(rays_o, rays_d, t_min);
  auto rays_start = rays_start_dir[0];
  auto rays_dir = rays_start_dir[1];

  auto rays_pts = torch::empty({total_len, 3}, torch::dtype(rays_o.dtype()).device(torch::kCUDA));
  auto mask_valid = torch::zeros({total_len}, torch::dtype(torch::kBool).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "sample_pts_on_rays_cuda", ([&] {
    sample_pts_on_rays_cvrg_cuda_kernel<scalar_t><<<(total_len+threads-1)/threads, threads>>>(
        rays_start.data<scalar_t>(),
        rays_dir.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        units.data<scalar_t>(),
        gridX, gridY, gridZ,
        ray_id.data<int64_t>(),
        step_id.data<int64_t>(),
        stepdist, total_len,
        rays_pts.data<scalar_t>(),
        tensoRF_cvrg_mask.data<bool>(),
        mask_valid.data<bool>());
  }));
  return {rays_pts, mask_valid, ray_id, step_id, N_steps, t_min, t_max};
}


std::vector<torch::Tensor> sample_pts_on_rays_rot_cvrg_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d,
        torch::Tensor pnt_xyz, torch::Tensor pnt_rot,
        torch::Tensor tensoRF_cvrg_inds, torch::Tensor tensoRF_count,
        torch::Tensor tensoRF_topindx, torch::Tensor units, torch::Tensor local_range,
        torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far, const float stepdist) {
  const int threads = 256;
  const int n_rays = rays_o.size(0);

  const int gridX = tensoRF_cvrg_inds.size(0);
  const int gridY = tensoRF_cvrg_inds.size(1);
  const int gridZ = tensoRF_cvrg_inds.size(2);
  const int K = tensoRF_topindx.size(1);

  // Compute ray-bbox intersection
  auto t_minmax = infer_t_minmax_cuda(rays_o, rays_d, xyz_min, xyz_max, near, far);
  auto t_min = t_minmax[0];
  auto t_max = t_minmax[1];

  // Compute the number of points required.
  // Assign ray index and step index to each.
  auto N_steps = infer_n_samples_cuda(t_min, t_max, stepdist);
  auto N_steps_cumsum = N_steps.cumsum(0);
  const int total_len = N_steps.sum().item<int>();
  auto ray_id = torch::zeros({total_len}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  __set_1_at_ray_seg_start<<<(n_rays+threads-1)/threads, threads>>>(ray_id.data<int64_t>(), N_steps_cumsum.data<int64_t>(), n_rays);
  ray_id.cumsum_(0);
  auto step_id = torch::empty({total_len}, ray_id.options());
  __set_step_id<<<(total_len+threads-1)/threads, threads>>>(step_id.data<int64_t>(), ray_id.data<int64_t>(), N_steps_cumsum.data<int64_t>(), total_len);

  // Compute the global xyz of each point
  auto rays_start_dir = infer_ray_start_dir_cuda(rays_o, rays_d, t_min);
  auto rays_start = rays_start_dir[0];
  auto rays_dir = rays_start_dir[1];

  auto rays_pts = torch::empty({total_len, 3}, torch::dtype(rays_o.dtype()).device(torch::kCUDA));
  auto mask_valid = torch::zeros({total_len}, torch::dtype(torch::kBool).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "sample_pts_on_rays_rot_cvrg_cuda", ([&] {
    sample_pts_on_rays_rot_cvrg_cuda_kernel<scalar_t><<<(total_len+threads-1)/threads, threads>>>(
        rays_start.data<scalar_t>(),
        rays_dir.data<scalar_t>(),
        pnt_xyz.data<scalar_t>(),
        pnt_rot.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        units.data<scalar_t>(),
        local_range.data<scalar_t>(),
        gridX, gridY, gridZ, K,
        ray_id.data<int64_t>(),
        step_id.data<int64_t>(),
        stepdist, total_len,
        rays_pts.data<scalar_t>(),
        tensoRF_cvrg_inds.data<int64_t>(),
        tensoRF_count.data<int64_t>(),
        tensoRF_topindx.data<int64_t>(),
        mask_valid.data<bool>());
  }));
  return {rays_pts, mask_valid, ray_id, step_id, N_steps, t_min, t_max};
}


std::vector<torch::Tensor> sample_pts_on_rays_dist_cvrg_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d,
        torch::Tensor tensoRF_cvrg_mask, torch::Tensor units,
        torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far, const float stepdist, torch::Tensor shift) {
  const int threads = 256;
  const int n_rays = rays_o.size(0);

  const int gridX = tensoRF_cvrg_mask.size(0);
  const int gridY = tensoRF_cvrg_mask.size(1);
  const int gridZ = tensoRF_cvrg_mask.size(2);

  // Compute ray-bbox intersection
  auto t_minmax = infer_t_minmax_cuda(rays_o, rays_d, xyz_min, xyz_max, near, far);
  auto t_min = t_minmax[0];
  auto t_max = t_minmax[1];

  // Compute the number of points required.
  // Assign ray index and step index to each.
  auto N_steps = infer_n_samples_shift_cuda(t_min, t_max, shift);
  auto N_steps_cumsum = N_steps.cumsum(0);
  const int total_len = N_steps.sum().item<int>();
  auto ray_id = torch::zeros({total_len}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  __set_1_at_ray_seg_start<<<(n_rays+threads-1)/threads, threads>>>(ray_id.data<int64_t>(), N_steps_cumsum.data<int64_t>(), n_rays);
  ray_id.cumsum_(0);
  auto step_id = torch::empty({total_len}, ray_id.options());
  __set_step_id<<<(total_len+threads-1)/threads, threads>>>(
        step_id.data<int64_t>(), ray_id.data<int64_t>(), N_steps_cumsum.data<int64_t>(), total_len);

  // Compute the global xyz of each point
  auto rays_start_dir = infer_ray_start_dir_cuda(rays_o, rays_d, t_min);
  auto rays_start = rays_start_dir[0];
  auto rays_dir = rays_start_dir[1];

  auto rays_pts = torch::empty({total_len, 3}, torch::dtype(rays_o.dtype()).device(torch::kCUDA));
  auto mask_valid = torch::zeros({total_len}, torch::dtype(torch::kBool).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "sample_pts_on_rays_cuda", ([&] {
    sample_pts_on_rays_dist_cvrg_cuda_kernel<scalar_t><<<(total_len+threads-1)/threads, threads>>>(
        rays_start.data<scalar_t>(),
        rays_dir.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        units.data<scalar_t>(),
        gridX, gridY, gridZ,
        ray_id.data<int64_t>(),
        step_id.data<int64_t>(),
        stepdist,
        shift.data<scalar_t>(),
        total_len,
        rays_pts.data<scalar_t>(),
        tensoRF_cvrg_mask.data<bool>(),
        mask_valid.data<bool>());
  }));
  return {rays_pts, mask_valid, ray_id, step_id, N_steps, t_min, t_max};
}

std::vector<torch::Tensor> sample_pts_on_rays_ji_cvrg_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d,
        torch::Tensor tensoRF_cvrg_mask, torch::Tensor units,
        torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far, const float stepdist, torch::Tensor shift) {
  const int threads = 256;
  const int n_rays = rays_o.size(0);

  const int gridX = tensoRF_cvrg_mask.size(0);
  const int gridY = tensoRF_cvrg_mask.size(1);
  const int gridZ = tensoRF_cvrg_mask.size(2);

  // Compute ray-bbox intersection
  auto t_minmax = infer_t_minmax_shift_cuda(rays_o, rays_d, shift, xyz_min, xyz_max, near, far);
  auto t_min = t_minmax[0];
  auto t_max = t_minmax[1];

  // Compute the number of points required.
  // Assign ray index and step index to each.
  auto N_steps = infer_n_samples_cuda(t_min, t_max, stepdist);
  auto N_steps_cumsum = N_steps.cumsum(0);
  const int total_len = N_steps.sum().item<int>();
  auto ray_id = torch::zeros({total_len}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  __set_1_at_ray_seg_start<<<(n_rays+threads-1)/threads, threads>>>(ray_id.data<int64_t>(), N_steps_cumsum.data<int64_t>(), n_rays);
  ray_id.cumsum_(0);
  auto step_id = torch::empty({total_len}, ray_id.options());
  __set_step_id<<<(total_len+threads-1)/threads, threads>>>(
        step_id.data<int64_t>(), ray_id.data<int64_t>(), N_steps_cumsum.data<int64_t>(), total_len);

  // Compute the global xyz of each point
  auto rays_start_dir = infer_ray_start_dir_cuda(rays_o, rays_d, t_min);
  auto rays_start = rays_start_dir[0];
  auto rays_dir = rays_start_dir[1];

  auto rays_pts = torch::empty({total_len, 3}, torch::dtype(rays_o.dtype()).device(torch::kCUDA));
  auto mask_valid = torch::zeros({total_len}, torch::dtype(torch::kBool).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "sample_pts_on_rays_cuda", ([&] {
    sample_pts_on_rays_cvrg_cuda_kernel<scalar_t><<<(total_len+threads-1)/threads, threads>>>(
        rays_start.data<scalar_t>(),
        rays_dir.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        units.data<scalar_t>(),
        gridX, gridY, gridZ,
        ray_id.data<int64_t>(),
        step_id.data<int64_t>(),
        stepdist, total_len,
        rays_pts.data<scalar_t>(),
        tensoRF_cvrg_mask.data<bool>(),
        mask_valid.data<bool>());
  }));
  return {rays_pts, mask_valid, ray_id, step_id, N_steps, t_min, t_max};
}


std::vector<torch::Tensor> sample_pts_on_rays_sphere_cvrg_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d,
        torch::Tensor pnt_xyz,
        torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx,
        torch::Tensor units,
        const float radiusl,
        const float radiush,
        torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far, const float stepdist) {
  const int threads = 256;
  const int n_rays = rays_o.size(0);

  const int gridX = tensoRF_cvrg_inds.size(0);
  const int gridY = tensoRF_cvrg_inds.size(1);
  const int gridZ = tensoRF_cvrg_inds.size(2);
  const int K = tensoRF_topindx.size(1);
  // Compute ray-bbox intersection
  auto t_minmax = infer_t_minmax_cuda(rays_o, rays_d, xyz_min, xyz_max, near, far);
  auto t_min = t_minmax[0];
  auto t_max = t_minmax[1];

  // Compute the number of points required.
  // Assign ray index and step index to each.
  auto N_steps = infer_n_samples_cuda(t_min, t_max, stepdist);
  auto N_steps_cumsum = N_steps.cumsum(0);
  const int total_len = N_steps.sum().item<int>();
  auto ray_id = torch::zeros({total_len}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  __set_1_at_ray_seg_start<<<(n_rays+threads-1)/threads, threads>>>(ray_id.data<int64_t>(), N_steps_cumsum.data<int64_t>(), n_rays);
  ray_id.cumsum_(0);
  auto step_id = torch::empty({total_len}, ray_id.options());
  __set_step_id<<<(total_len+threads-1)/threads, threads>>>(
        step_id.data<int64_t>(), ray_id.data<int64_t>(), N_steps_cumsum.data<int64_t>(), total_len);

  // Compute the global xyz of each point
  auto rays_start_dir = infer_ray_start_dir_cuda(rays_o, rays_d, t_min);
  auto rays_start = rays_start_dir[0];
  auto rays_dir = rays_start_dir[1];

  auto rays_pts = torch::empty({total_len, 3}, torch::dtype(rays_o.dtype()).device(torch::kCUDA));
  auto mask_valid = torch::zeros({total_len}, torch::dtype(torch::kBool).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "sample_pts_on_rays_sphere_cuda", ([&] {
    sample_pts_on_rays_sphere_cvrg_cuda_kernel<scalar_t><<<(total_len+threads-1)/threads, threads>>>(
        rays_start.data<scalar_t>(),
        rays_dir.data<scalar_t>(),
        pnt_xyz.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        units.data<scalar_t>(),
        radiusl,
        radiush,
        gridX, gridY, gridZ, K,
        ray_id.data<int64_t>(),
        step_id.data<int64_t>(),
        stepdist, total_len,
        rays_pts.data<scalar_t>(),
        tensoRF_cvrg_inds.data<int64_t>(),
        tensoRF_count.data<int64_t>(),
        tensoRF_topindx.data<int64_t>(),
        mask_valid.data<bool>());
  }));
  return {rays_pts, mask_valid, ray_id, step_id, N_steps, t_min, t_max};
}


std::vector<torch::Tensor> sample_2_tensoRF_cvrg_cuda(
        torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max, torch::Tensor units, torch::Tensor local_range, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_xyz_recenter, torch::Tensor geo_xyz, const int K, const bool KNN) {

  const int threads = 256;
  const int n_pts = geo_xyz.size(0);
  const int n_sample = xyz_sampled.size(0);
  const int maxK = tensoRF_topindx.size(1);
  const int num_all_cvrg = tensoRF_count.size(0);
  const int gridX = tensoRF_cvrg_inds.size(0);
  const int gridY = tensoRF_cvrg_inds.size(1);
  const int gridZ = tensoRF_cvrg_inds.size(2);

  auto cvrg_inds = torch::empty({n_sample}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto cvrg_count = torch::zeros({n_sample}, torch::dtype(torch::kInt64).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(xyz_sampled.type(), "count_tensoRF_cvrg_cuda", ([&] {
    count_tensoRF_cvrg_cuda_kernel<scalar_t><<<(n_sample+threads-1)/threads, threads>>>(
        xyz_sampled.data<scalar_t>(),
        geo_xyz_recenter.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        units.data<scalar_t>(),
        tensoRF_count.data<int64_t>(),
        tensoRF_cvrg_inds.data<int64_t>(),
        cvrg_inds.data<int64_t>(),
        cvrg_count.data<int64_t>(),
        gridY * gridZ,
        gridZ,
        n_sample);
  }));

  auto cvrg_cumsum = cvrg_count.cumsum(0);
  const int cvrg_len = cvrg_count.sum().item<int>();
  // const int64_t cvrg_len = cvrg_count.sum().item<int64_t>();

  auto final_tensoRF_id = torch::empty({cvrg_len}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto final_agg_id = torch::empty({cvrg_len}, torch::dtype(torch::kInt64).device(torch::kCUDA));

  auto local_gindx_s = torch::empty({cvrg_len, 3}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto local_gindx_l = torch::empty({cvrg_len, 3}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto local_gweight_s = torch::empty({cvrg_len, 3}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
  auto local_gweight_l = torch::empty({cvrg_len, 3}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
  auto local_kernel_dist = torch::empty({cvrg_len}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
  if (cvrg_len > 0){
      __fill_agg_id<<<(n_sample+threads-1)/threads, threads>>>(cvrg_count.data<int64_t>(), cvrg_cumsum.data<int64_t>(), final_agg_id.data<int64_t>(), n_sample);
      // torch::cuda::synchronize();
      AT_DISPATCH_FLOATING_TYPES(xyz_sampled.type(), "find_tensoRF_and_repos_cuda", ([&] {
        find_tensoRF_and_repos_cuda_kernel<scalar_t><<<(cvrg_len+threads-1)/threads, threads>>>(
            xyz_sampled.data<scalar_t>(),
            geo_xyz.data<scalar_t>(),
            geo_xyz_recenter.data<scalar_t>(),
            final_agg_id.data<int64_t>(),
            final_tensoRF_id.data<int64_t>(),
            local_range.data<scalar_t>(),
            local_dims.data<int64_t>(),
            local_gindx_s.data<int64_t>(),
            local_gindx_l.data<int64_t>(),
            local_gweight_s.data<scalar_t>(),
            local_gweight_l.data<scalar_t>(),
            local_kernel_dist.data<scalar_t>(),
            units.data<scalar_t>(),
            tensoRF_topindx.data<int64_t>(),
            cvrg_inds.data<int64_t>(),
            cvrg_cumsum.data<int64_t>(),
            cvrg_count.data<int64_t>(),
            cvrg_len,
            K,
            maxK);
       }));
  }
  // torch::cuda::synchronize();
  return {local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, final_tensoRF_id, final_agg_id};
}



std::vector<torch::Tensor> sample_2_tensoRF_cvrg_cuda_bk(
        torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max, torch::Tensor units,
        torch::Tensor local_range, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_xyz_recenter, torch::Tensor geo_xyz, const int K, const bool KNN) {

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

  AT_DISPATCH_FLOATING_TYPES(xyz_sampled.type(), "find_tensoRF_and_repos_cuda_bk", ([&] {
      find_tensoRF_and_repos_cuda_kernel_bk<scalar_t><<<(n_sampleK+threads-1)/threads, threads>>>(
        xyz_sampled.data<scalar_t>(),
        geo_xyz.data<scalar_t>(),
        geo_xyz_recenter.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        final_tensoRF_id.data<int64_t>(),
        local_dims.data<int64_t>(),
        local_gindx_s.data<int64_t>(),
        local_gindx_l.data<int64_t>(),
        local_gweight_s.data<scalar_t>(),
        local_gweight_l.data<scalar_t>(),
        local_kernel_dist.data<scalar_t>(),
        units.data<scalar_t>(),
        local_range.data<scalar_t>(),
        tensoRF_cvrg_inds.data<int64_t>(),
        tensoRF_count.data<int64_t>(),
        tensoRF_topindx.data<int64_t>(),
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

  return {tensoRF_mask.reshape(-1), local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, final_tensoRF_id, final_agg_id};
}

std::vector<torch::Tensor> sample_2_rot_tensoRF_cvrg_cuda(
        torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max, torch::Tensor units,
        torch::Tensor local_range, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_rot, torch::Tensor geo_xyz, const int K, const bool KNN) {

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
  if (KNN) {
  AT_DISPATCH_FLOATING_TYPES(xyz_sampled.type(), "find_rot_tensoRF_and_repos_cuda", ([&] {
      find_rot_tensoRF_and_repos_cuda_kernel<scalar_t><<<(n_sampleK+threads-1)/threads, threads>>>(
        xyz_sampled.data<scalar_t>(),
        geo_xyz.data<scalar_t>(),
        geo_rot.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        final_tensoRF_id.data<int64_t>(),
        local_dims.data<int64_t>(),
        local_gindx_s.data<int64_t>(),
        local_gindx_l.data<int64_t>(),
        local_gweight_s.data<scalar_t>(),
        local_gweight_l.data<scalar_t>(),
        local_kernel_dist.data<scalar_t>(),
        units.data<scalar_t>(),
        local_range.data<scalar_t>(),
        tensoRF_cvrg_inds.data<int64_t>(),
        tensoRF_count.data<int64_t>(),
        tensoRF_topindx.data<int64_t>(),
        tensoRF_mask.data<bool>(),
        gridX,
        gridY,
        gridZ,
        n_sampleK,
        K,
        maxK);
  }));
  } else {
      timeval curTime;
      gettimeofday(&curTime, NULL);
      unsigned long seconds = curTime.tv_usec;

      AT_DISPATCH_FLOATING_TYPES(xyz_sampled.type(), "find_rot_tensoRF_and_repos_rand_cuda", ([&] {
        find_rot_tensoRF_and_repos_rand_cuda_kernel<scalar_t><<<(n_sample+threads-1)/threads, threads>>>(
            xyz_sampled.data<scalar_t>(),
            geo_xyz.data<scalar_t>(),
            geo_rot.data<scalar_t>(),
            xyz_min.data<scalar_t>(),
            final_tensoRF_id.data<int64_t>(),
            local_dims.data<int64_t>(),
            local_gindx_s.data<int64_t>(),
            local_gindx_l.data<int64_t>(),
            local_gweight_s.data<scalar_t>(),
            local_gweight_l.data<scalar_t>(),
            local_kernel_dist.data<scalar_t>(),
            units.data<scalar_t>(),
            local_range.data<scalar_t>(),
            tensoRF_cvrg_inds.data<int64_t>(),
            tensoRF_count.data<int64_t>(),
            tensoRF_topindx.data<int64_t>(),
            tensoRF_mask.data<bool>(),
            gridX,
            gridY,
            gridZ,
            n_sample,
            K,
            maxK,
            seconds);
      }));
  }
  // torch::cuda::synchronize();

  auto cvrg_count = tensoRF_mask.sum(1);
  auto cvrg_cumsum = cvrg_count.cumsum(0);
  const int cvrg_len = cvrg_count.sum().item<int>();

  auto final_agg_id = torch::empty({cvrg_len}, torch::dtype(torch::kInt64).device(torch::kCUDA));

  __fill_agg_id<<<(n_sample+threads-1)/threads, threads>>>(cvrg_count.data<int64_t>(), cvrg_cumsum.data<int64_t>(), final_agg_id.data<int64_t>(), n_sample);
  // torch::cuda::synchronize();

  return {tensoRF_mask.reshape(-1), local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, final_tensoRF_id, final_agg_id};
}


std::vector<torch::Tensor> sample_2_rotdist_tensoRF_cvrg_cuda(
        torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max, torch::Tensor units,
        torch::Tensor local_range, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_rot, torch::Tensor geo_xyz, const int K, const bool KNN) {

  const int threads = 256;
  const int n_pts = geo_xyz.size(0);
  const int n_sample = xyz_sampled.size(0);
  const int maxK = tensoRF_topindx.size(1);
  const int gridX = tensoRF_cvrg_inds.size(0);
  const int gridY = tensoRF_cvrg_inds.size(1);
  const int gridZ = tensoRF_cvrg_inds.size(2);

  const int n_sampleK = n_sample * K;

  auto dist = torch::empty({n_sampleK, 3}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
  auto local_kernel_dist = torch::empty({n_sampleK}, torch::dtype(xyz_sampled.dtype()).device(torch::kCUDA));
  auto final_tensoRF_id = torch::empty({n_sampleK}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto tensoRF_mask = torch::zeros({n_sample, K}, torch::dtype(torch::kBool).device(torch::kCUDA));

  if (KNN) {
      AT_DISPATCH_FLOATING_TYPES(xyz_sampled.type(), "find_rotdist_tensoRF_and_repos_cuda", ([&] {
        find_rotdist_tensoRF_and_repos_cuda_kernel<scalar_t><<<(n_sampleK+threads-1)/threads, threads>>>(
            xyz_sampled.data<scalar_t>(),
            geo_xyz.data<scalar_t>(),
            geo_rot.data<scalar_t>(),
            xyz_min.data<scalar_t>(),
            final_tensoRF_id.data<int64_t>(),
            local_dims.data<int64_t>(),
            dist.data<scalar_t>(),
            local_kernel_dist.data<scalar_t>(),
            units.data<scalar_t>(),
            local_range.data<scalar_t>(),
            tensoRF_cvrg_inds.data<int64_t>(),
            tensoRF_count.data<int64_t>(),
            tensoRF_topindx.data<int64_t>(),
            tensoRF_mask.data<bool>(),
            gridX,
            gridY,
            gridZ,
            n_sampleK,
            K,
            maxK);
      }));
  } else {
      timeval curTime;
      gettimeofday(&curTime, NULL);
      unsigned long seconds = curTime.tv_usec;

      AT_DISPATCH_FLOATING_TYPES(xyz_sampled.type(), "find_rotdist_tensoRF_and_repos_rand_cuda", ([&] {
        find_rotdist_tensoRF_and_repos_rand_cuda_kernel<scalar_t><<<(n_sample+threads-1)/threads, threads>>>(
            xyz_sampled.data<scalar_t>(),
            geo_xyz.data<scalar_t>(),
            geo_rot.data<scalar_t>(),
            xyz_min.data<scalar_t>(),
            final_tensoRF_id.data<int64_t>(),
            local_dims.data<int64_t>(),
            dist.data<scalar_t>(),
            local_kernel_dist.data<scalar_t>(),
            units.data<scalar_t>(),
            local_range.data<scalar_t>(),
            tensoRF_cvrg_inds.data<int64_t>(),
            tensoRF_count.data<int64_t>(),
            tensoRF_topindx.data<int64_t>(),
            tensoRF_mask.data<bool>(),
            gridX,
            gridY,
            gridZ,
            n_sample,
            K,
            maxK,
            seconds);
      }));
  }
//  torch::cuda::synchronize();

  auto cvrg_count = tensoRF_mask.sum(1);
  auto cvrg_cumsum = cvrg_count.cumsum(0);
  const int cvrg_len = cvrg_count.sum().item<int>();

  auto final_agg_id = torch::empty({cvrg_len}, torch::dtype(torch::kInt64).device(torch::kCUDA));

  __fill_agg_id<<<(n_sample+threads-1)/threads, threads>>>(cvrg_count.data<int64_t>(), cvrg_cumsum.data<int64_t>(), final_agg_id.data<int64_t>(), n_sample);
  // torch::cuda::synchronize();

  return {tensoRF_mask.reshape(-1), dist, local_kernel_dist, final_tensoRF_id, final_agg_id};
}


std::vector<torch::Tensor> sample_2_sphere_tensoRF_cvrg_cuda(
        torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max, torch::Tensor units, const float radiusl, const float radiush, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds, torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_xyz, const int K, const bool KNN) {

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

  AT_DISPATCH_FLOATING_TYPES(xyz_sampled.type(), "find_sphere_tensoRF_and_repos_cuda", ([&] {
    find_sphere_tensoRF_and_repos_cuda_kernel<scalar_t><<<(n_sampleK+threads-1)/threads, threads>>>(
        xyz_sampled.data<scalar_t>(),
        geo_xyz.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        final_tensoRF_id.data<int64_t>(),
        radiusl,
        radiush,
        local_dims.data<int64_t>(),
        local_gindx_s.data<int64_t>(),
        local_gindx_l.data<int64_t>(),
        local_gweight_s.data<scalar_t>(),
        local_gweight_l.data<scalar_t>(),
        local_kernel_dist.data<scalar_t>(),
        units.data<scalar_t>(),
        tensoRF_cvrg_inds.data<int64_t>(),
        tensoRF_count.data<int64_t>(),
        tensoRF_topindx.data<int64_t>(),
        tensoRF_mask.data<bool>(),
        gridX,
        gridY,
        gridZ,
        n_sampleK,
        K,
        maxK);
  }));
  // torch::cuda::synchronize();

  auto cvrg_count = tensoRF_mask.sum(1);
  auto cvrg_cumsum = cvrg_count.cumsum(0);
  const int cvrg_len = cvrg_count.sum().item<int>();

  auto final_agg_id = torch::empty({cvrg_len}, torch::dtype(torch::kInt64).device(torch::kCUDA));

  __fill_agg_id<<<(n_sample+threads-1)/threads, threads>>>(cvrg_count.data<int64_t>(), cvrg_cumsum.data<int64_t>(), final_agg_id.data<int64_t>(), n_sample);
  // torch::cuda::synchronize();

  return {tensoRF_mask.reshape(-1), local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, final_tensoRF_id, final_agg_id};
}



std::vector<torch::Tensor> inds_cvrg_cuda(
        torch::Tensor xyz_sampled, torch::Tensor xyz_min, torch::Tensor xyz_max, torch::Tensor units,
        torch::Tensor local_range, torch::Tensor local_dims, torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor tensoRF_count, torch::Tensor tensoRF_topindx, torch::Tensor geo_xyz_recenter, torch::Tensor geo_xyz) {

  const int threads = 256;
  const int n_pts = geo_xyz.size(0);
  const int n_sample = xyz_sampled.size(0);
  const int K = tensoRF_topindx.size(1);
  const int num_all_cvrg = tensoRF_count.size(0);
  const int gridX = tensoRF_cvrg_inds.size(0);
  const int gridY = tensoRF_cvrg_inds.size(1);
  const int gridZ = tensoRF_cvrg_inds.size(2);

  auto cvrg_inds_map = torch::full({gridX, gridY, gridZ}, -1, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto grid_inds_mask = torch::zeros({gridX+1, gridY+1, gridZ+1}, torch::dtype(torch::kBool).device(torch::kCUDA));
  AT_DISPATCH_FLOATING_TYPES(xyz_sampled.type(), "fill_sample_ind_mask_cuda", ([&] {
    fill_sample_ind_mask_cuda_kernel<scalar_t><<<(n_sample+threads-1)/threads, threads>>>(
        xyz_sampled.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        units.data<scalar_t>(),
        tensoRF_cvrg_inds.data<int64_t>(),
        cvrg_inds_map.data<int64_t>(),
        grid_inds_mask.data<bool>(),
        gridY * gridZ,
        gridZ,
        n_sample);
  }));

  auto sample_grids = torch::nonzero(grid_inds_mask);
  const int num_grids = sample_grids.size(0);
  auto sample_grids_tensoRF = torch::full({num_grids, 8*K}, -1, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto sample_grids_tensoRF_count = torch::zeros({num_grids}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  AT_DISPATCH_FLOATING_TYPES(xyz_sampled.type(), "fill_grids_tensoRF_cuda", ([&] {
    fill_grids_tensoRF_cuda_kernel<scalar_t><<<(8 * num_grids+threads-1)/threads, threads>>>(
        xyz_sampled.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        units.data<scalar_t>(),
        tensoRF_count.data<int64_t>(),
        tensoRF_topindx.data<int64_t>(),
        cvrg_inds_map.data<int64_t>(),
        sample_grids.data<int64_t>(),
        sample_grids_tensoRF.data<int64_t>(),
        sample_grids_tensoRF_count.data<int64_t>(),
        gridY * gridZ,
        gridZ,
        K,
        num_grids);
  }));
  //return {global_gindx, local_gindx, tensoRF_id};
  return {};
}



torch::Tensor filter_xyz_cvrg_cuda(
        torch::Tensor xyz_sampled,
        torch::Tensor xyz_min,
        torch::Tensor xyz_max,
        torch::Tensor units,
        torch::Tensor tensoRF_cvrg_mask){

  const int threads = 256;
  const int n_sample = xyz_sampled.size(0);
  const int gridX = tensoRF_cvrg_mask.size(0);
  const int gridY = tensoRF_cvrg_mask.size(1);
  const int gridZ = tensoRF_cvrg_mask.size(2);
  auto mask = torch::zeros({n_sample}, torch::dtype(torch::kBool).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(xyz_sampled.type(), "filter_xyz_by_cvrg_cuda", ([&] {
    filter_xyz_by_cvrg_cuda_kernel<scalar_t><<<(n_sample+threads-1)/threads, threads>>>(
        xyz_sampled.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        units.data<scalar_t>(),
        gridX, gridY, gridZ,
        tensoRF_cvrg_mask.data<bool>(),
        mask.data<bool>(),
        n_sample);
  }));
  return mask;
}



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
        torch::Tensor local_range){

  const int threads = 256;
  const int n_sample = xyz_sampled.size(0);
  const int gridX = tensoRF_cvrg_inds.size(0);
  const int gridY = tensoRF_cvrg_inds.size(1);
  const int gridZ = tensoRF_cvrg_inds.size(2);
  const int K = tensoRF_topindx.size(1);
  auto mask = torch::zeros({n_sample}, torch::dtype(torch::kBool).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(xyz_sampled.type(), "filter_xyz_by_rot_cvrg_cuda", ([&] {
    filter_xyz_by_rot_cvrg_cuda_kernel<scalar_t><<<(n_sample+threads-1)/threads, threads>>>(
        geo_xyz.data<scalar_t>(),
        geo_rot.data<scalar_t>(),
        xyz_sampled.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        units.data<scalar_t>(),
        gridX, gridY, gridZ, K,
        tensoRF_cvrg_inds.data<int64_t>(),
        tensoRF_count.data<int64_t>(),
        tensoRF_topindx.data<int64_t>(),
        local_range.data<scalar_t>(),
        mask.data<bool>(),
        n_sample);
  }));
  return mask;
}


torch::Tensor filter_xyz_sphere_cvrg_cuda(
        torch::Tensor xyz_sampled,
        torch::Tensor xyz_min,
        torch::Tensor xyz_max,
        torch::Tensor units,
        const float radiusl,
        const float radiush,
        torch::Tensor tensoRF_cvrg_inds,
        torch::Tensor tensoRF_count,
        torch::Tensor tensoRF_topindx,
        torch::Tensor pnt_xyz){

  const int threads = 256;
  const int n_sample = xyz_sampled.size(0);
  const int gridX = tensoRF_cvrg_inds.size(0);
  const int gridY = tensoRF_cvrg_inds.size(1);
  const int gridZ = tensoRF_cvrg_inds.size(2);
  const int K = tensoRF_topindx.size(1);
  auto mask = torch::zeros({n_sample}, torch::dtype(torch::kBool).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(xyz_sampled.type(), "filter_xyz_by_cvrg_cuda", ([&] {
    filter_xyz_by_sphere_cvrg_cuda_kernel<scalar_t><<<(n_sample+threads-1)/threads, threads>>>(
        xyz_sampled.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        pnt_xyz.data<scalar_t>(),
        units.data<scalar_t>(),
        radiusl,
        radiush,
        gridX, gridY, gridZ,
        tensoRF_cvrg_inds.data<int64_t>(),
        tensoRF_count.data<int64_t>(),
        tensoRF_topindx.data<int64_t>(),
        mask.data<bool>(),
        K,
        n_sample);
  }));
  return mask;
}