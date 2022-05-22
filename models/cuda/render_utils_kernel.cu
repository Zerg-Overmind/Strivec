#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>

/*
   Points sampling helper functions.
 */
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


template <typename scalar_t>
__global__ void infer_t_minmax_geo_cuda_kernel(
        scalar_t* __restrict__ rays_o,
        scalar_t* __restrict__ rays_d,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        const float near, const float far, const int n_rays,
        scalar_t* __restrict__ ray_tminmax,
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
    t_min[i_ray] = (ray_tminmax[2*i_ray] < 0) ? -1.0 : max(min(max(max(max(min(ax, bx), min(ay, by)), min(az, bz)), ray_tminmax[2*i_ray]), far), near);
    t_max[i_ray] = (ray_tminmax[2*i_ray] < 0) ? -1.0 : max(min(min(min(min(max(ax, bx), max(ay, by)), max(az, bz)), ray_tminmax[2*i_ray+1]), far), near);
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
__global__ void infer_n_samples_geo_cuda_kernel(
        scalar_t* __restrict__ t_min,
        scalar_t* __restrict__ t_max,
        const float stepdist,
        const int n_rays,
        int64_t* __restrict__ n_samples) {
  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
    n_samples[i_ray] = max(ceil((t_max[i_ray]-t_min[i_ray]) / stepdist), 0.);
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


std::vector<torch::Tensor> infer_t_minmax_geo_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far, torch::Tensor ray_tminmax) {
  const int n_rays = rays_o.size(0);
  auto t_min = torch::empty({n_rays}, rays_o.options());
  auto t_max = torch::empty({n_rays}, rays_o.options());

  const int threads = 256;
  const int blocks = (n_rays + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "infer_t_minmax_geo_cuda", ([&] {
    infer_t_minmax_geo_cuda_kernel<scalar_t><<<blocks, threads>>>(
        rays_o.data<scalar_t>(),
        rays_d.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        near, far, n_rays,
        ray_tminmax.data<scalar_t>(),
        t_min.data<scalar_t>(),
        t_max.data<scalar_t>());
  }));

  return {t_min, t_max};
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

torch::Tensor infer_n_samples_geo_cuda(torch::Tensor t_min, torch::Tensor t_max, const float stepdist) {
  const int n_rays = t_min.size(0);
  auto n_samples = torch::empty({n_rays}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  const int threads = 256;
  const int blocks = (n_rays + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(t_min.type(), "infer_n_samples_cuda", ([&] {
    infer_n_samples_geo_cuda_kernel<scalar_t><<<blocks, threads>>>(
        t_min.data<scalar_t>(),
        t_max.data<scalar_t>(),
        stepdist,
        n_rays,
        n_samples.data<int64_t>());
  }));
  return n_samples;
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


/*
   Sampling query points on rays.
 */
__global__ void __set_1_at_ray_seg_start(
        int64_t* __restrict__ ray_id,
        int64_t* __restrict__ N_steps_cumsum,
        const int n_rays) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(0<idx && idx<n_rays) {
    ray_id[N_steps_cumsum[idx-1]] = 1;
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


__global__ void __set_check_step_id(
        int64_t* __restrict__ step_id,
        int64_t* __restrict__ ray_id,
        int64_t* __restrict__ N_steps_cumsum,
        int64_t* __restrict__ N_steps,
        const int total_len) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<total_len) {
      const int rid = ray_id[idx];
      step_id[idx] = idx - ((rid!=0) ? N_steps_cumsum[rid-1] : 0);
      int curN_steps = N_steps[rid];
      int curid = step_id[idx];
      if (curN_steps == 0 or curid >= curN_steps) printf("curN_steps %d; curid %d      ;", curN_steps, curid);
    }
}


__global__ void __fill_rayid_stepid_tensoRFid(
        int64_t* __restrict__ ray_id,
        int64_t* __restrict__ step_id,
        int64_t* __restrict__ tensoRF_sample_counts,
        int64_t* __restrict__ tensoRF_sample_cumsum,
        int64_t* __restrict__ tensoRF_id_at_sample,
        int64_t* __restrict__ num_tensoRF_ray_cumsum,
        int64_t* __restrict__ tensoRF_per_ray,
        int64_t* __restrict__ final_ray_id,
        int64_t* __restrict__ final_step_id,
        int64_t* __restrict__ final_tensoRF_id,
        int64_t* __restrict__ final_agg_id,
        int64_t* __restrict__ num_final_sample_cumsum,
        const int total_len) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<total_len) {
        const int rid = ray_id[idx];
        const int sid = step_id[idx];
        const int cur_tensoRF_num = tensoRF_sample_counts[idx];
        const int tensoRF_shift = ((rid!=0) ? num_tensoRF_ray_cumsum[rid-1] : 0) + sid * tensoRF_per_ray[rid];
        const int final_tensoRF_shift = (idx!=0) ? tensoRF_sample_cumsum[idx-1] : 0;
        const int final_sample_idx = num_final_sample_cumsum[idx] - 1;
        if (cur_tensoRF_num > 0) {
            final_ray_id[final_sample_idx] = rid;
            final_step_id[final_sample_idx] = sid;
            for (int i=0; i < cur_tensoRF_num; i++){
                final_tensoRF_id[final_tensoRF_shift+i] = tensoRF_id_at_sample[tensoRF_shift+i];
                // printf("final_tensoRF_shift: %d %" PRId64 " ;", final_tensoRF_shift, final_tensoRF_id[final_tensoRF_shift+i]);
                final_agg_id[final_tensoRF_shift+i] = final_sample_idx;
            }
        }
    }
}

__global__ void __set_num_tensoRF_ray(
        int64_t* __restrict__ num_tensoRF_ray,
        int64_t* __restrict__ N_steps,
        int64_t* __restrict__ tensoRF_per_ray,
        const int n_rays) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<n_rays) {
      num_tensoRF_ray[idx] = N_steps[idx] * tensoRF_per_ray[idx];
      // const int cur_tensoRF_per_ray = tensoRF_per_ray[idx];
      // const int nstep = N_steps[idx];
      // if (cur_tensoRF_per_ray == 0 && nstep != 0) printf("cur_tensoRF_per_ray %d N_steps %d    ", cur_tensoRF_per_ray, nstep);
    }
}

template <typename scalar_t>
__global__ void sample_pts_on_rays_cuda_kernel(
        scalar_t* __restrict__ rays_start,
        scalar_t* __restrict__ rays_dir,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        int64_t* __restrict__ ray_id,
        int64_t* __restrict__ step_id,
        const float stepdist, const int total_len,
        scalar_t* __restrict__ rays_pts,
        bool* __restrict__ mask_outbbox) {
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
    mask_outbbox[idx] = (xyz_min[0]>px) | (xyz_min[1]>py) | (xyz_min[2]>pz) | \
                        (xyz_max[0]<px) | (xyz_max[1]<py) | (xyz_max[2]<pz);
  }
}


template <typename scalar_t>
__global__ void relpos_sample_to_tensoRF_cuda_kernel(
        scalar_t* __restrict__ rays_dir,
        scalar_t* __restrict__ pnts,
        int64_t* __restrict__ final_ray_id,
        int64_t* __restrict__ final_step_id,
        int64_t* __restrict__ final_agg_id,
        int64_t* __restrict__ final_tensoRF_id,
        const float stepdist, const int tensoRF_sample_len,
        scalar_t* __restrict__ local_range,
        int64_t* __restrict__ local_dims,
        scalar_t* __restrict__ local_stepsize,
        int64_t* __restrict__ local_gindx_s,
        int64_t* __restrict__ local_gindx_l,
        scalar_t* __restrict__ local_gweight_s,
        scalar_t* __restrict__ local_gweight_l,
        scalar_t* __restrict__ local_kernel_dist,
        scalar_t* __restrict__ xyz_sampled
        ) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<tensoRF_sample_len) {
    const int i_tid = final_tensoRF_id[idx];
    const int i_agg = final_agg_id[idx];
    const int i_ray = final_ray_id[i_agg];
    const int i_step = final_step_id[i_agg];

    const int offset_p = idx * 3;
    const int offset_r = i_ray * 3;
    const int offset_t = i_tid * 3;
    const int offset_a = i_agg * 3;
    const float dist = stepdist * i_step;

    const float dx = rays_dir[offset_r];
    const float dy = rays_dir[offset_r+1];
    const float dz = rays_dir[offset_r+2];

    const float px = xyz_sampled[offset_a];
    const float py = xyz_sampled[offset_a + 1];
    const float pz = xyz_sampled[offset_a + 2];

    const float rel_x = px - pnts[offset_t];
    const float rel_y = py - pnts[offset_t+1];
    const float rel_z = pz - pnts[offset_t+2];

    local_kernel_dist[idx] = sqrt(rel_x * rel_x + rel_y * rel_y + rel_z * rel_z);

    const float softindx = (rel_x + local_range[0]) / local_stepsize[0];
    const float softindy = (rel_y + local_range[1]) / local_stepsize[1];
    const float softindz = (rel_z + local_range[2]) / local_stepsize[2];

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
__global__ void sample_xyz_dir_cuda_kernel(
    scalar_t* __restrict__ rays_start,
    scalar_t* __restrict__ rays_dir,
    int64_t* __restrict__ final_ray_id,
    int64_t* __restrict__ final_step_id,
    const float stepdist, const int total_final_sample,
    scalar_t* __restrict__ xyz_sampled,
    scalar_t* __restrict__ sample_dir
    ) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx<total_final_sample) {
        const int i_ray = final_ray_id[idx];
        const int i_step = final_step_id[idx];
        const int offset_p = idx * 3;
        const int offset_r = i_ray * 3;
        const float dist = stepdist * i_step;
        const float dx = rays_dir[offset_r];
        const float dy = rays_dir[offset_r+1];
        const float dz = rays_dir[offset_r+2];
        sample_dir[offset_p  ] = dx;
        sample_dir[offset_p+1] = dy;
        sample_dir[offset_p+2] = dz;
        xyz_sampled[offset_p  ] = rays_start[offset_r  ] + dx * dist;
        xyz_sampled[offset_p+1] = rays_start[offset_r+1] + dy * dist;
        xyz_sampled[offset_p+2] = rays_start[offset_r+2] + dz * dist;
    }
}


template <typename scalar_t>
__global__ void count_tensor_at_sample_cuda_kernel(
        scalar_t* __restrict__ t_min,
        scalar_t* __restrict__ rays_d,
        scalar_t* __restrict__ xyz_min,
        scalar_t* __restrict__ xyz_max,
        scalar_t* __restrict__ tensoRF_tminmax,
        int64_t* __restrict__ ray_id,
        int64_t* __restrict__ step_id,
        const float stepdist, const int total_len,
        int64_t* __restrict__ tensoRF_cumsum,
        int64_t* __restrict__ tensoRF_id_at_ray,
        int64_t* __restrict__ N_steps,
        int64_t* __restrict__ tensoRF_per_ray,
        int64_t* __restrict__ num_tensoRF_ray,
        int64_t* __restrict__ num_tensoRF_ray_cumsum,
        int64_t* __restrict__ tensoRF_sample_counts,
        int64_t* __restrict__ tensoRF_sample_mask,
        int64_t* __restrict__ tensoRF_id_at_sample
  ) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<total_len) {
    const int i_ray = ray_id[idx];
    const int i_step = step_id[idx];
    const float t_min_val = t_min[i_ray];
    const float dist = stepdist * i_step + t_min_val;
    const int shiftray = (i_ray!=0) ? tensoRF_cumsum[i_ray-1] : 0;
    const int cur_tensoRF_per_ray = tensoRF_per_ray[i_ray];
    const int nstep = N_steps[i_ray];
    // if (cur_tensoRF_per_ray == 0 || nstep == 0) printf("cur_tensoRF_per_ray %d N_steps %d    ", cur_tensoRF_per_ray, nstep);
    const int sample_shift = ((i_ray!=0) ? num_tensoRF_ray_cumsum[i_ray-1] : 0) + i_step * tensoRF_per_ray[i_ray];
    int count = 0;
    for(int i = 0; i < cur_tensoRF_per_ray; ++i) {
        int cur_shiftray = shiftray + i;
        float tmin = tensoRF_tminmax[cur_shiftray * 2];
        float tmax = tensoRF_tminmax[cur_shiftray * 2 + 1];
        // if (tmax - tmin > 0.2) printf("tmin %f , %f ;", tmin, tmax);
        if (dist >= tmin && dist <= tmax){
            tensoRF_id_at_sample[sample_shift + count] = tensoRF_id_at_ray[cur_shiftray];
            count++;
        }
    }
    tensoRF_sample_counts[idx] = count;
    tensoRF_sample_mask[idx] = (count > 0) ? 1 : 0;
    // if (count < 1) printf(" %d ", count);
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
__global__ void find_tensor_by_projection_cuda_kernel(
    scalar_t* __restrict__ rays_o,
    scalar_t* __restrict__ rays_d,
    scalar_t* __restrict__ pnts,
    scalar_t* __restrict__ half_range_sqr,
    scalar_t* __restrict__ tensoRF_tminmax,
    scalar_t* __restrict__ ray_tminmax,
    int64_t* __restrict__ tensoRF_cumsum,
    int64_t* __restrict__ tensoRF_id_at_ray,
    int64_t* __restrict__ tensoRF_per_ray,
    const int n_pts,
    const int n_rays
   ) {

  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray < n_rays) {
     const int shift = (i_ray!=0) ? tensoRF_cumsum[i_ray-1] : 0;
     int count = 0;
     float tmin = 100.0;
     float tmax = 0.0;
     if (tensoRF_per_ray[i_ray] > 0){
         const int i_ray3 = 3 * i_ray;
         float ox = rays_o[i_ray3];
         float oy = rays_o[i_ray3 + 1];
         float oz = rays_o[i_ray3 + 2];
         float dx = rays_d[i_ray3];
         float dy = rays_d[i_ray3 + 1];
         float dz = rays_d[i_ray3 + 2];
         for(int j=0; j<n_pts; ++j) {
            float px = pnts[j*3] - ox;
            float py = pnts[j*3+1] - oy;
            float pz = pnts[j*3+2] - oz;
            float dpx = px * dx + py * dy + pz * dz;
            float half_cut_sqr = half_range_sqr[0] - (px*px + py*py + pz*pz - dpx * dpx);
            if (half_cut_sqr >= 0){
               int cur_idx = shift + count;
               tensoRF_id_at_ray[cur_idx] = j;
               float half_cut = sqrt(half_cut_sqr);
               tensoRF_tminmax[cur_idx * 2] = dpx - half_cut;
               tensoRF_tminmax[cur_idx * 2 + 1] = dpx + half_cut;
               tmin = min(tmin, dpx - half_cut);
               tmax = max(tmax, dpx + half_cut);
               count++;
            }
         }
     }
     ray_tminmax[i_ray * 2] = (count == 0) ? -1.0 : tmin;
     ray_tminmax[i_ray * 2 + 1] = (count == 0) ? -1.0 : tmax;
  }
}

std::vector<torch::Tensor> sample_pts_on_rays_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d,
        torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far, const float stepdist) {
  const int threads = 256;
  const int n_rays = rays_o.size(0);

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
  auto mask_outbbox = torch::empty({total_len}, torch::dtype(torch::kBool).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "sample_pts_on_rays_cuda", ([&] {
    sample_pts_on_rays_cuda_kernel<scalar_t><<<(total_len+threads-1)/threads, threads>>>(
        rays_start.data<scalar_t>(),
        rays_dir.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        ray_id.data<int64_t>(),
        step_id.data<int64_t>(),
        stepdist, total_len,
        rays_pts.data<scalar_t>(),
        mask_outbbox.data<bool>());
  }));
  return {rays_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max};
}


std::vector<torch::Tensor> sample_pts_on_rays_geo_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor pnts, torch::Tensor tensoRF_per_ray,
        torch::Tensor half_range_sqr, torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far, const float stepdist, torch::Tensor local_range, torch::Tensor local_dims, torch::Tensor local_stepsize) {
  const int threads = 256;
  const int n_rays = rays_o.size(0);
  const int n_pts = pnts.size(0);

  // fill crossed tensoRF for each ray
  auto tensoRF_cumsum = tensoRF_per_ray.cumsum(0);
  const int tensoRF_len = tensoRF_per_ray.sum().item<int>();
  auto tensoRF_id_at_ray = torch::empty({tensoRF_len}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto tensoRF_tminmax = torch::empty({tensoRF_len*2}, rays_o.options());
  auto ray_tminmax = torch::empty({n_rays*2}, rays_o.options());

  AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "find_tensor_by_projection_cuda_kernel", ([&] {
    find_tensor_by_projection_cuda_kernel<scalar_t><<<(n_rays+threads-1)/threads, threads>>>(
        rays_o.data<scalar_t>(),
        rays_d.data<scalar_t>(),
        pnts.data<scalar_t>(),
        half_range_sqr.data<scalar_t>(),
        tensoRF_tminmax.data<scalar_t>(),
        ray_tminmax.data<scalar_t>(),
        tensoRF_cumsum.data<int64_t>(),
        tensoRF_id_at_ray.data<int64_t>(),
        tensoRF_per_ray.data<int64_t>(),
        n_pts,
        n_rays);
  }));

  // Compute ray-bbox intersection
  auto t_minmax = infer_t_minmax_geo_cuda(rays_o, rays_d, xyz_min, xyz_max, near, far, ray_tminmax);
  auto t_min = t_minmax[0];
  auto t_max = t_minmax[1];

  // Compute the number of points required.
  // Assign ray index and step index to each.
  auto N_steps = infer_n_samples_geo_cuda(t_min, t_max, stepdist);
  auto N_steps_cumsum = N_steps.cumsum(0);
  int total_len = N_steps.sum().item<int>();
  auto ray_id = torch::zeros({total_len}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  __set_n_at_ray_seg_start<<<(n_rays+threads-1)/threads, threads>>>(ray_id.data<int64_t>(), N_steps_cumsum.data<int64_t>(), N_steps.data<int64_t>(), n_rays);
  // __set_1_at_ray_seg_start<<<(n_rays+threads-1)/threads, threads>>>(ray_id.data<int64_t>(), N_steps_cumsum.data<int64_t>(), n_rays);
  ray_id.cumsum_(0);
  auto step_id = torch::empty({total_len}, ray_id.options());
  if (total_len > 0) {
      __set_step_id<<<(total_len+threads-1)/threads, threads>>>(
            step_id.data<int64_t>(), ray_id.data<int64_t>(), N_steps_cumsum.data<int64_t>(), total_len); // __set_step_id
  }
  auto num_tensoRF_ray = torch::zeros({n_rays}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  __set_num_tensoRF_ray<<<(n_rays+threads-1)/threads, threads>>>(
        num_tensoRF_ray.data<int64_t>(), N_steps.data<int64_t>(), tensoRF_per_ray.data<int64_t>(), n_rays);
  auto num_tensoRF_ray_cumsum = num_tensoRF_ray.cumsum(0);
  const int total_len_tensoRF = num_tensoRF_ray.sum().item<int>();

  auto tensoRF_sample_counts = torch::zeros({total_len}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto tensoRF_sample_mask = torch::zeros({total_len}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto tensoRF_id_at_sample = torch::full({total_len_tensoRF}, -1, torch::dtype(torch::kInt64).device(torch::kCUDA));
  if (total_len > 0) {
      AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "count_tensor_at_sample_cuda", ([&] {
        count_tensor_at_sample_cuda_kernel<scalar_t><<<(total_len+threads-1)/threads, threads>>>(
            t_min.data<scalar_t>(),
            rays_d.data<scalar_t>(),
            xyz_min.data<scalar_t>(),
            xyz_max.data<scalar_t>(),
            tensoRF_tminmax.data<scalar_t>(),
            ray_id.data<int64_t>(),
            step_id.data<int64_t>(),
            stepdist, total_len,
            tensoRF_cumsum.data<int64_t>(),
            tensoRF_id_at_ray.data<int64_t>(),
            N_steps.data<int64_t>(),
            tensoRF_per_ray.data<int64_t>(),
            num_tensoRF_ray.data<int64_t>(),
            num_tensoRF_ray_cumsum.data<int64_t>(),
            tensoRF_sample_counts.data<int64_t>(),
            tensoRF_sample_mask.data<int64_t>(),
            tensoRF_id_at_sample.data<int64_t>()
        );
      }));
  }
  // torch::cuda::synchronize();
  auto tensoRF_sample_cumsum = tensoRF_sample_counts.cumsum(0);
  const int tensoRF_sample_len = tensoRF_sample_counts.sum().item<int>();

  auto num_final_sample_cumsum = tensoRF_sample_mask.cumsum(0);
  int total_final_sample = tensoRF_sample_mask.sum().item<int>();

  auto final_ray_id = torch::empty({total_final_sample}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto final_step_id = torch::empty({total_final_sample}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto final_tensoRF_id = torch::empty({tensoRF_sample_len}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto final_agg_id = torch::zeros({tensoRF_sample_len}, torch::dtype(torch::kInt64).device(torch::kCUDA));

  if (total_len > 0) {
      __fill_rayid_stepid_tensoRFid<<<(total_len+threads-1)/threads, threads>>>(ray_id.data<int64_t>(), step_id.data<int64_t>(), tensoRF_sample_counts.data<int64_t>(), tensoRF_sample_cumsum.data<int64_t>(), tensoRF_id_at_sample.data<int64_t>(), num_tensoRF_ray_cumsum.data<int64_t>(), tensoRF_per_ray.data<int64_t>(), final_ray_id.data<int64_t>(), final_step_id.data<int64_t>(), final_tensoRF_id.data<int64_t>(), final_agg_id.data<int64_t>(), num_final_sample_cumsum.data<int64_t>(), total_len);
  }

  // Compute the global xyz of each point
  auto rays_start_dir = infer_ray_start_dir_cuda(rays_o, rays_d, t_min);
  auto rays_start = rays_start_dir[0];
  auto rays_dir = rays_start_dir[1];

  auto local_gindx_s = torch::empty({tensoRF_sample_len, 3}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto local_gindx_l = torch::empty({tensoRF_sample_len, 3}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto local_gweight_s = torch::empty({tensoRF_sample_len, 3}, torch::dtype(rays_o.dtype()).device(torch::kCUDA));
  auto local_gweight_l = torch::empty({tensoRF_sample_len, 3}, torch::dtype(rays_o.dtype()).device(torch::kCUDA));
  auto local_kernel_dist = torch::empty({tensoRF_sample_len}, torch::dtype(rays_o.dtype()).device(torch::kCUDA));
  auto xyz_sampled = torch::empty({total_final_sample, 3}, torch::dtype(rays_o.dtype()).device(torch::kCUDA));
  auto sample_dir = torch::empty({total_final_sample, 3}, torch::dtype(rays_o.dtype()).device(torch::kCUDA));
  if (total_final_sample > 0) {
      AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "relpos_sample_to_tensoRF_cuda", ([&] {
       sample_xyz_dir_cuda_kernel<scalar_t><<<(total_final_sample+threads-1)/threads, threads>>>(
            rays_start.data<scalar_t>(),
            rays_dir.data<scalar_t>(),
            final_ray_id.data<int64_t>(),
            final_step_id.data<int64_t>(),
            stepdist, total_final_sample,
            xyz_sampled.data<scalar_t>(),
            sample_dir.data<scalar_t>()
            );
       }));

      AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "relpos_sample_to_tensoRF_cuda", ([&] {
        relpos_sample_to_tensoRF_cuda_kernel<scalar_t><<<(tensoRF_sample_len+threads-1)/threads, threads>>>(
            rays_dir.data<scalar_t>(),
            pnts.data<scalar_t>(),
            final_ray_id.data<int64_t>(),
            final_step_id.data<int64_t>(),
            final_agg_id.data<int64_t>(),
            final_tensoRF_id.data<int64_t>(),
            stepdist, tensoRF_sample_len,
            local_range.data<scalar_t>(),
            local_dims.data<int64_t>(),
            local_stepsize.data<scalar_t>(),
            local_gindx_s.data<int64_t>(),
            local_gindx_l.data<int64_t>(),
            local_gweight_s.data<scalar_t>(),
            local_gweight_l.data<scalar_t>(),
            local_kernel_dist.data<scalar_t>(),
            xyz_sampled.data<scalar_t>()
            );
       }));
  }
  return {xyz_sampled, sample_dir, local_gindx_s, local_gindx_l, local_gweight_s, local_gweight_l, local_kernel_dist, final_ray_id, final_step_id, final_tensoRF_id, final_agg_id, t_min, t_max};
}




std::vector<torch::Tensor> sample_pts_on_rays_dist_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d,
        torch::Tensor xyz_min, torch::Tensor xyz_max,
        const float near, const float far, const float stepdist, torch::Tensor shift) {
  const int threads = 256;
  const int n_rays = rays_o.size(0);

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
  __set_1_at_ray_seg_start<<<(n_rays+threads-1)/threads, threads>>>(
        ray_id.data<int64_t>(), N_steps_cumsum.data<int64_t>(), n_rays);
  ray_id.cumsum_(0);
  auto step_id = torch::empty({total_len}, ray_id.options());
  __set_step_id<<<(total_len+threads-1)/threads, threads>>>(
        step_id.data<int64_t>(), ray_id.data<int64_t>(), N_steps_cumsum.data<int64_t>(), total_len);

  // Compute the global xyz of each point
  auto rays_start_dir = infer_ray_start_dir_cuda(rays_o, rays_d, t_min);
  auto rays_start = rays_start_dir[0];
  auto rays_dir = rays_start_dir[1];

  auto rays_pts = torch::empty({total_len, 3}, torch::dtype(rays_o.dtype()).device(torch::kCUDA));
  auto interval = torch::empty({total_len}, torch::dtype(rays_o.dtype()).device(torch::kCUDA));
  auto mask_outbbox = torch::empty({total_len}, torch::dtype(torch::kBool).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "sample_pts_on_rays_dist_cuda", ([&] {
    sample_pts_on_rays_dist_cuda_kernel<scalar_t><<<(total_len+threads-1)/threads, threads>>>(
        rays_start.data<scalar_t>(),
        rays_dir.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        ray_id.data<int64_t>(),
        step_id.data<int64_t>(),
        stepdist,
        shift.data<scalar_t>(),
        total_len,
        rays_pts.data<scalar_t>(),
        mask_outbbox.data<bool>());
  }));
  return {rays_pts, mask_outbbox, ray_id, step_id, N_steps, t_min, t_max};
}

template <typename scalar_t>
__global__ void sample_ndc_pts_on_rays_cuda_kernel(
        const scalar_t* __restrict__ rays_o,
        const scalar_t* __restrict__ rays_d,
        const scalar_t* __restrict__ xyz_min,
        const scalar_t* __restrict__ xyz_max,
        const int N_samples, const int n_rays,
        scalar_t* __restrict__ rays_pts,
        bool* __restrict__ mask_outbbox) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx<N_samples*n_rays) {
    const int i_ray = idx / N_samples;
    const int i_step = idx % N_samples;

    const int offset_p = idx * 3;
    const int offset_r = i_ray * 3;
    const float dist = ((float)i_step) / (N_samples-1);
    const float px = rays_o[offset_r  ] + rays_d[offset_r  ] * dist;
    const float py = rays_o[offset_r+1] + rays_d[offset_r+1] * dist;
    const float pz = rays_o[offset_r+2] + rays_d[offset_r+2] * dist;
    rays_pts[offset_p  ] = px;
    rays_pts[offset_p+1] = py;
    rays_pts[offset_p+2] = pz;
    mask_outbbox[idx] = (xyz_min[0]>px) | (xyz_min[1]>py) | (xyz_min[2]>pz) | \
                        (xyz_max[0]<px) | (xyz_max[1]<py) | (xyz_max[2]<pz);
  }
}

std::vector<torch::Tensor> sample_ndc_pts_on_rays_cuda(
        torch::Tensor rays_o, torch::Tensor rays_d,
        torch::Tensor xyz_min, torch::Tensor xyz_max,
        const int N_samples) {
  const int threads = 256;
  const int n_rays = rays_o.size(0);

  auto rays_pts = torch::empty({n_rays, N_samples, 3}, torch::dtype(rays_o.dtype()).device(torch::kCUDA));
  auto mask_outbbox = torch::empty({n_rays, N_samples}, torch::dtype(torch::kBool).device(torch::kCUDA));

  AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "sample_ndc_pts_on_rays_cuda", ([&] {
    sample_ndc_pts_on_rays_cuda_kernel<scalar_t><<<(n_rays*N_samples+threads-1)/threads, threads>>>(
        rays_o.data<scalar_t>(),
        rays_d.data<scalar_t>(),
        xyz_min.data<scalar_t>(),
        xyz_max.data<scalar_t>(),
        N_samples, n_rays,
        rays_pts.data<scalar_t>(),
        mask_outbbox.data<bool>());
  }));
  return {rays_pts, mask_outbbox};
}


/*
   MaskCache lookup to skip known freespace.
 */

static __forceinline__ __device__
bool check_xyz(int i, int j, int k, int sz_i, int sz_j, int sz_k) {
  return (0 <= i) && (i < sz_i) && (0 <= j) && (j < sz_j) && (0 <= k) && (k < sz_k);
}


template <typename scalar_t>
__global__ void maskcache_lookup_cuda_kernel(
    bool* __restrict__ world,
    scalar_t* __restrict__ xyz,
    bool* __restrict__ out,
    scalar_t* __restrict__ xyz2ijk_scale,
    scalar_t* __restrict__ xyz2ijk_shift,
    const int sz_i, const int sz_j, const int sz_k, const int n_pts) {

  const int i_pt = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_pt<n_pts) {
    const int offset = i_pt * 3;
    const int i = round(xyz[offset  ] * xyz2ijk_scale[0] + xyz2ijk_shift[0]);
    const int j = round(xyz[offset+1] * xyz2ijk_scale[1] + xyz2ijk_shift[1]);
    const int k = round(xyz[offset+2] * xyz2ijk_scale[2] + xyz2ijk_shift[2]);
    if(check_xyz(i, j, k, sz_i, sz_j, sz_k)) {
      out[i_pt] = world[i*sz_j*sz_k + j*sz_k + k];
    }
  }
}

torch::Tensor maskcache_lookup_cuda(
        torch::Tensor world,
        torch::Tensor xyz,
        torch::Tensor xyz2ijk_scale,
        torch::Tensor xyz2ijk_shift) {

  const int sz_i = world.size(0);
  const int sz_j = world.size(1);
  const int sz_k = world.size(2);
  const int n_pts = xyz.size(0);

  auto out = torch::zeros({n_pts}, torch::dtype(torch::kBool).device(torch::kCUDA));
  if(n_pts==0) {
    return out;
  }

  const int threads = 256;
  const int blocks = (n_pts + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(xyz.type(), "maskcache_lookup_cuda", ([&] {
    maskcache_lookup_cuda_kernel<scalar_t><<<blocks, threads>>>(
        world.data<bool>(),
        xyz.data<scalar_t>(),
        out.data<bool>(),
        xyz2ijk_scale.data<scalar_t>(),
        xyz2ijk_shift.data<scalar_t>(),
        sz_i, sz_j, sz_k, n_pts);
  }));

  return out;
}


/*
    Ray marching helper function.
 */
template <typename scalar_t>
__global__ void raw2alpha_cuda_kernel(
    scalar_t* __restrict__ density,
    const float shift, const float interval, const int n_pts,
    scalar_t* __restrict__ exp_d,
    scalar_t* __restrict__ alpha) {

  const int i_pt = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_pt<n_pts) {
    const scalar_t e = exp(density[i_pt] + shift); // can be inf
    exp_d[i_pt] = e;
    alpha[i_pt] = 1 - pow(1 + e, -interval);
  }
}

std::vector<torch::Tensor> raw2alpha_cuda(torch::Tensor density, const float shift, const float interval) {

  const int n_pts = density.size(0);
  auto exp_d = torch::empty_like(density);
  auto alpha = torch::empty_like(density);
  if(n_pts==0) {
    return {exp_d, alpha};
  }

  const int threads = 256;
  const int blocks = (n_pts + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(density.type(), "raw2alpha_cuda", ([&] {
    raw2alpha_cuda_kernel<scalar_t><<<blocks, threads>>>(
        density.data<scalar_t>(),
        shift, interval, n_pts,
        exp_d.data<scalar_t>(),
        alpha.data<scalar_t>());
  }));

  return {exp_d, alpha};
}

template <typename scalar_t>
__global__ void raw2alpha_backward_cuda_kernel(
    scalar_t* __restrict__ exp_d,
    scalar_t* __restrict__ grad_back,
    const float interval, const int n_pts,
    scalar_t* __restrict__ grad) {

  const int i_pt = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_pt<n_pts) {
    grad[i_pt] = min(exp_d[i_pt], 1e10) * pow(1+exp_d[i_pt], -interval-1) * interval * grad_back[i_pt];
  }
}

torch::Tensor raw2alpha_backward_cuda(torch::Tensor exp_d, torch::Tensor grad_back, const float interval) {

  const int n_pts = exp_d.size(0);
  auto grad = torch::empty_like(exp_d);
  if(n_pts==0) {
    return grad;
  }

  const int threads = 256;
  const int blocks = (n_pts + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(exp_d.type(), "raw2alpha_backward_cuda", ([&] {
    raw2alpha_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
        exp_d.data<scalar_t>(),
        grad_back.data<scalar_t>(),
        interval, n_pts,
        grad.data<scalar_t>());
  }));

  return grad;
}


/*
    Ray marching helper function.
 */
template <typename scalar_t>
__global__ void raw2alpha_randstep_cuda_kernel(
    scalar_t* __restrict__ density,
    const float shift,
    scalar_t* __restrict__ interval,
    const int n_pts,
    scalar_t* __restrict__ exp_d,
    scalar_t* __restrict__ alpha) {

  const int i_pt = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_pt<n_pts) {
    const scalar_t e = exp(density[i_pt] + shift); // can be inf
    exp_d[i_pt] = e;
    alpha[i_pt] = 1 - pow(1 + e, -interval[i_pt]);
  }
}

std::vector<torch::Tensor> raw2alpha_randstep_cuda(torch::Tensor density, const float shift, torch::Tensor interval) {

  const int n_pts = density.size(0);
  auto exp_d = torch::empty_like(density);
  auto alpha = torch::empty_like(density);
  if(n_pts==0) {
    return {exp_d, alpha};
  }

  const int threads = 256;
  const int blocks = (n_pts + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(density.type(), "raw2alpha_randstep_cuda", ([&] {
    raw2alpha_randstep_cuda_kernel<scalar_t><<<blocks, threads>>>(
        density.data<scalar_t>(),
        shift,
        interval.data<scalar_t>(),
        n_pts,
        exp_d.data<scalar_t>(),
        alpha.data<scalar_t>());
  }));

  return {exp_d, alpha};
}

template <typename scalar_t>
__global__ void raw2alpha_randstep_backward_cuda_kernel(
    scalar_t* __restrict__ exp_d,
    scalar_t* __restrict__ grad_back,
    scalar_t* __restrict__ interval,
    const int n_pts,
    scalar_t* __restrict__ grad) {

  const int i_pt = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_pt<n_pts) {
    grad[i_pt] = min(exp_d[i_pt], 1e10) * pow(1+exp_d[i_pt], -interval[i_pt]-1) * interval[i_pt] * grad_back[i_pt];
  }
}

torch::Tensor raw2alpha_randstep_backward_cuda(torch::Tensor exp_d, torch::Tensor grad_back, torch::Tensor interval) {

  const int n_pts = exp_d.size(0);
  auto grad = torch::empty_like(exp_d);
  if(n_pts==0) {
    return grad;
  }

  const int threads = 256;
  const int blocks = (n_pts + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(exp_d.type(), "raw2alpha_randstep_backward_cuda", ([&] {
    raw2alpha_randstep_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
        exp_d.data<scalar_t>(),
        grad_back.data<scalar_t>(),
        interval.data<scalar_t>(), n_pts,
        grad.data<scalar_t>());
  }));

  return grad;
}


template <typename scalar_t>
__global__ void alpha2weight_cuda_kernel(
    scalar_t* __restrict__ alpha,
    const int n_rays,
    scalar_t* __restrict__ weight,
    scalar_t* __restrict__ T,
    scalar_t* __restrict__ alphainv_last,
    int64_t* __restrict__ i_start,
    int64_t* __restrict__ i_end) {

  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
    const int i_s = i_start[i_ray];
    const int i_e_max = i_end[i_ray];

    float T_cum = 1.;
    int i;
    for(i=i_s; i<i_e_max; ++i) {
      T[i] = T_cum;
      weight[i] = T_cum * alpha[i];
      T_cum *= (1. - alpha[i]);
      if(T_cum<1e-3) {
        i+=1;
        break;
      }
    }
    i_end[i_ray] = i;
    alphainv_last[i_ray] = T_cum;
  }
}


template <typename scalar_t>
__global__ void filter_ray_by_points_cuda_kernel(
    scalar_t* __restrict__ xyz,
    scalar_t* __restrict__ pnts,
    scalar_t* __restrict__ half_range,
    const int n_pts,
    const int n_rays,
    const int n_samples,
    int64_t* __restrict__ ray_mask
   ) {

  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_pts) {
    for(int j=0; j<n_pts; ++j) {
       float px = pnts[j*3];
       float py = pnts[j*3+1];
       float pz = pnts[j*3+2];
       for(int i=0; i<n_samples; ++i) {
          int shift = (i_ray * n_samples + i) * 3;
          if(abs(xyz[shift]-px) <= half_range[0] && abs(xyz[shift+1]-py) <= half_range[1] && abs(xyz[shift+2]-pz) <= half_range[2]){
            ray_mask[i_ray] = 1;
            return;
          }
       }
    }
  }
}


template <typename scalar_t>
__global__ void filter_ray_by_projection_cuda_kernel(
    scalar_t* __restrict__ rays_o,
    scalar_t* __restrict__ rays_d,
    scalar_t* __restrict__ pnts,
    scalar_t* __restrict__ half_range_sqr,
    const int n_pts,
    const int n_rays,
    int64_t* __restrict__ ray_mask
   ) {

  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
     const int i_ray3 = 3 * i_ray;
     float ox = rays_o[i_ray3];
     float oy = rays_o[i_ray3 + 1];
     float oz = rays_o[i_ray3 + 2];
     float dx = rays_d[i_ray3];
     float dy = rays_d[i_ray3 + 1];
     float dz = rays_d[i_ray3 + 2];
     for(int j=0; j<n_pts; ++j) {
        float px = pnts[j*3] - ox;
        float py = pnts[j*3+1] - oy;
        float pz = pnts[j*3+2] - oz;
        float dpx = px * dx + py * dy + pz * dz;
        if (px*px + py*py + pz*pz - dpx * dpx <= half_range_sqr[0]){
           ray_mask[i_ray] = ray_mask[i_ray] + 1;
        }
     }
  }
}



__global__ void __set_i_for_segment_start_end(
        int64_t* __restrict__ ray_id,
        const int n_pts,
        int64_t* __restrict__ i_start,
        int64_t* __restrict__ i_end) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if(0<index && index<n_pts && ray_id[index]!=ray_id[index-1]) {
    i_start[ray_id[index]] = index;
    i_end[ray_id[index-1]] = index;
  }
}

std::vector<torch::Tensor> alpha2weight_cuda(torch::Tensor alpha, torch::Tensor ray_id, const int n_rays) {

  const int n_pts = alpha.size(0);
  const int threads = 256;

  auto weight = torch::zeros_like(alpha);
  auto T = torch::ones_like(alpha);
  auto alphainv_last = torch::ones({n_rays}, alpha.options());
  auto i_start = torch::zeros({n_rays}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  auto i_end = torch::zeros({n_rays}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  if(n_pts==0) {
    return {weight, T, alphainv_last, i_start, i_end};
  }

  __set_i_for_segment_start_end<<<(n_pts+threads-1)/threads, threads>>>(
          ray_id.data<int64_t>(), n_pts, i_start.data<int64_t>(), i_end.data<int64_t>());
  i_end[ray_id[n_pts-1]] = n_pts;

  const int blocks = (n_rays + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(alpha.type(), "alpha2weight_cuda", ([&] {
    alpha2weight_cuda_kernel<scalar_t><<<blocks, threads>>>(
        alpha.data<scalar_t>(),
        n_rays,
        weight.data<scalar_t>(),
        T.data<scalar_t>(),
        alphainv_last.data<scalar_t>(),
        i_start.data<int64_t>(),
        i_end.data<int64_t>());
  }));

  return {weight, T, alphainv_last, i_start, i_end};
}


torch::Tensor filter_ray_by_points_cuda(torch::Tensor xyz, torch::Tensor pnts, torch::Tensor half_range) {
  const int n_rays = xyz.size(0);
  const int n_samples = xyz.size(1);
  const int n_pts = pnts.size(0);
  const int threads = 256; //256

  auto ray_mask = torch::zeros({n_rays}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  if(n_rays==0 || n_pts==0) {
    return ray_mask;
  }
  const int blocks = (n_rays + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(xyz.type(), "filter_ray_by_points_cuda", ([&] {
    filter_ray_by_points_cuda_kernel<scalar_t><<<blocks, threads>>>(
        xyz.data<scalar_t>(),
        pnts.data<scalar_t>(),
        half_range.data<scalar_t>(),
        n_pts,
        n_rays,
        n_samples,
        ray_mask.data<int64_t>());
  }));
  return ray_mask;
}


torch::Tensor filter_ray_by_projection_cuda(torch::Tensor rays_o, torch::Tensor rays_d, torch::Tensor pnts, torch::Tensor half_range_sqr) {
  const int n_rays = rays_o.size(0);
  const int n_pts = pnts.size(0);
  const int threads = 256; // 256

  auto ray_mask = torch::zeros({n_rays}, torch::dtype(torch::kInt64).device(torch::kCUDA));
  if(n_rays==0 || n_pts==0) {
    return ray_mask;
  }
  const int blocks = (n_rays + threads - 1) / threads;
  AT_DISPATCH_FLOATING_TYPES(rays_o.type(), "filter_ray_by_projection_cuda", ([&] {
    filter_ray_by_projection_cuda_kernel<scalar_t><<<blocks, threads>>>(
        rays_o.data<scalar_t>(),
        rays_d.data<scalar_t>(),
        pnts.data<scalar_t>(),
        half_range_sqr.data<scalar_t>(),
        n_pts,
        n_rays,
        ray_mask.data<int64_t>());
  }));
  return ray_mask;
}


template <typename scalar_t>
__global__ void alpha2weight_backward_cuda_kernel(
    scalar_t* __restrict__ alpha,
    scalar_t* __restrict__ weight,
    scalar_t* __restrict__ T,
    scalar_t* __restrict__ alphainv_last,
    int64_t* __restrict__ i_start,
    int64_t* __restrict__ i_end,
    const int n_rays,
    scalar_t* __restrict__ grad_weights,
    scalar_t* __restrict__ grad_last,
    scalar_t* __restrict__ grad) {

  const int i_ray = blockIdx.x * blockDim.x + threadIdx.x;
  if(i_ray<n_rays) {
    const int i_s = i_start[i_ray];
    const int i_e = i_end[i_ray];

    float back_cum = grad_last[i_ray] * alphainv_last[i_ray];
    for(int i=i_e-1; i>=i_s; --i) {
      grad[i] = grad_weights[i] * T[i] - back_cum / (1-alpha[i] + 1e-10);
      back_cum += grad_weights[i] * weight[i];
    }
  }
}

torch::Tensor alpha2weight_backward_cuda(
        torch::Tensor alpha, torch::Tensor weight, torch::Tensor T, torch::Tensor alphainv_last,
        torch::Tensor i_start, torch::Tensor i_end, const int n_rays,
        torch::Tensor grad_weights, torch::Tensor grad_last) {

  auto grad = torch::zeros_like(alpha);
  if(n_rays==0) {
    return grad;
  }

  const int threads = 256;
  const int blocks = (n_rays + threads - 1) / threads;

  AT_DISPATCH_FLOATING_TYPES(alpha.type(), "alpha2weight_backward_cuda", ([&] {
    alpha2weight_backward_cuda_kernel<scalar_t><<<blocks, threads>>>(
        alpha.data<scalar_t>(),
        weight.data<scalar_t>(),
        T.data<scalar_t>(),
        alphainv_last.data<scalar_t>(),
        i_start.data<int64_t>(),
        i_end.data<int64_t>(),
        n_rays,
        grad_weights.data<scalar_t>(),
        grad_last.data<scalar_t>(),
        grad.data<scalar_t>());
  }));

  return grad;
}

