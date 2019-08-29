#include <ATen/ATen.h>
#include <THC/THCAtomics.cuh>

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)

#define THREADS_PER_BLOCK 1024

inline int GET_BLOCKS(const int N) {
  int optimal_block_num = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
  int max_block_num = 65000;
  return min(optimal_block_num, max_block_num);
}

template <typename scalar_t>
__global__ void PSROIPoolForward(const int nthreads, const scalar_t* bottom_data,
    const scalar_t spatial_scale, const int height, const int width,
    const int channels, const int pooled_height, const int pooled_width,
    const int group_size, const int output_dim,
    const scalar_t* bottom_rois, scalar_t* top_data, int* mapping_channel_data)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
        // (n, c, ph, pw) is an element in the pooled output
        int pw = index % pooled_width;
      	int ph = (index / pooled_width) % pooled_height;
      	int ctop = (index / pooled_width / pooled_height) % output_dim;
      	int n = index / pooled_width / pooled_height / output_dim;

        bottom_rois += n * 5;
        int roi_batch_ind = bottom_rois[0];
	    scalar_t roi_start_w =
        	static_cast<scalar_t>(bottom_rois[1]) * spatial_scale;
      	scalar_t roi_start_h =
        	static_cast<scalar_t>(bottom_rois[2]) * spatial_scale;
      	scalar_t roi_end_w =
        	static_cast<scalar_t>(bottom_rois[3] + 1.) * spatial_scale;
      	scalar_t roi_end_h =
        	static_cast<scalar_t>(bottom_rois[4] + 1.) * spatial_scale;

        // Force malformed ROIs to be 1x1
        scalar_t roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
      	scalar_t roi_height = max(roi_end_h - roi_start_h, 0.1);

        scalar_t bin_size_h = (scalar_t)(roi_height) / (scalar_t)(pooled_height);
        scalar_t bin_size_w = (scalar_t)(roi_width) / (scalar_t)(pooled_width);

        int hstart = static_cast<scalar_t>(ph) * bin_size_h
                          + roi_start_h;
      	int wstart = static_cast<scalar_t>(pw) * bin_size_w
                          + roi_start_w;
      	int hend = static_cast<scalar_t>(ph + 1) * bin_size_h
                        + roi_start_h;
      	int wend = static_cast<scalar_t>(pw + 1) * bin_size_w
                        + roi_start_w;

        // Add roi offsets and clip to input boundaries
        hstart = min(max(hstart, 0), height);
      	hend = min(max(hend, 0), height);
      	wstart = min(max(wstart, 0), width);
      	wend = min(max(wend, 0), width);
        bool is_empty = (hend <= hstart) || (wend <= wstart);

        int gw = pw;
      	int gh = ph;
      	int c = (ctop*group_size + gh)*group_size + gw;

        bottom_data += (roi_batch_ind * channels + c) * height * width;
        scalar_t out_sum = 0;
      	for (int y = hstart; y < hend; ++y) {
      	  for (int x = wstart; x < wend; ++x) {
      	    int y_low = (int)y;
            int x_low = (int)x;
            int y_high;
            int x_high;

            if (y_low >= height - 1) {
              y_high = y_low = height - 1;
              y = (scalar_t)y_low;
            } else {
              y_high = y_low + 1;
            }

            if (x_low >= width - 1) {
              x_high = x_low = width - 1;
              x = (scalar_t)x_low;
            } else {
              x_high = x_low + 1;
            }
            scalar_t ly = y - y_low;
            scalar_t lx = x - x_low;
            scalar_t hy = 1. - ly;
            scalar_t hx = 1. - lx;
            // do bilinear interpolation
            scalar_t lt = bottom_data[y_low * width + x_low];
            scalar_t rt = bottom_data[y_low * width + x_high];
            scalar_t lb = bottom_data[y_high * width + x_low];
            scalar_t rb = bottom_data[y_high * width + x_high];
            scalar_t w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

            scalar_t val = (w1 * lt + w2 * rt + w3 * lb + w4 * rb);
      	    out_sum += val;
      	  }
      	}
      	scalar_t bin_area = (hend - hstart)*(wend - wstart);
      	top_data[index] = is_empty? scalar_t(0) : out_sum/bin_area;
      	mapping_channel_data[index] = c;
    }
}


int PSROIPoolForwardLauncher(
    const at::Tensor features, const at::Tensor rois, at::Tensor mapping_channel, const float spatial_scale, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height,
    const int pooled_width, const int group_size, const int output_dim,
    at::Tensor output)
{
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    features.scalar_type(),"PSROIPoolLaucherForward",([&] {
    const scalar_t *bottom_data = features.data<scalar_t>();
    const scalar_t *bottom_rois = rois.data<scalar_t>();
    scalar_t *top_data = output.data<scalar_t>();
    int *mapping_channel_data = mapping_channel.data<int>();

    const int output_size = output_dim * pooled_height * pooled_width * num_rois;
    PSROIPoolForward<scalar_t>
    <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>(
      output_size, bottom_data, spatial_scale, height, width, channels, pooled_height,
      pooled_width, group_size, output_dim, bottom_rois, top_data, mapping_channel_data);
    }));
    THCudaCheck(cudaGetLastError());

    return 1;
}

template <typename scalar_t>
__global__ void PSROIPoolBackward(const int nthreads, const scalar_t* top_diff,
    const int* mapping_channel, const int num_rois, const scalar_t spatial_scale,
    const int height, const int width, const int channels,
    const int pooled_height, const int pooled_width, const int output_dim, scalar_t* bottom_diff,
    const scalar_t* bottom_rois) {
    CUDA_1D_KERNEL_LOOP(index, nthreads)
    {
      int pw = index % pooled_width;
      int ph = (index / pooled_width) % pooled_height;
      int n = index / pooled_width / pooled_height / output_dim;

      // [start, end) interval for spatial sampling
      bottom_rois += n * 5;
      int roi_batch_ind = bottom_rois[0];
      scalar_t roi_start_w = static_cast<scalar_t>(bottom_rois[1]) * spatial_scale;
      scalar_t roi_start_h = static_cast<scalar_t>(bottom_rois[2]) * spatial_scale;
      scalar_t roi_end_w = static_cast<scalar_t>(bottom_rois[3] + 1.) * spatial_scale;
      scalar_t roi_end_h = static_cast<scalar_t>(bottom_rois[4] + 1.) * spatial_scale;

      // Force too small ROIs to be 1x1
      scalar_t roi_width = max(roi_end_w - roi_start_w, 0.1);  // avoid 0
      scalar_t roi_height = max(roi_end_h - roi_start_h, 0.1);

      // Compute w and h at bottom
      scalar_t bin_size_h = roi_height / static_cast<scalar_t>(pooled_height);
      scalar_t bin_size_w = roi_width / static_cast<scalar_t>(pooled_width);

      int hstart = static_cast<scalar_t>(ph)* bin_size_h + roi_start_h;
      int wstart = static_cast<scalar_t>(pw)* bin_size_w + roi_start_w;
      int hend = static_cast<scalar_t>(ph + 1) * bin_size_h + roi_start_h;
      int wend = static_cast<scalar_t>(pw + 1) * bin_size_w + roi_start_w;
      // Add roi offsets and clip to input boundaries
      hstart = min(max(hstart, 0), height);
      hend = min(max(hend, 0), height);
      wstart = min(max(wstart, 0), width);
      wend = min(max(wend, 0), width);
      bool is_empty = (hend <= hstart) || (wend <= wstart);

      // Compute c at bottom
      int c = mapping_channel[index];
      scalar_t* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;
      scalar_t bin_area = (hend - hstart)*(wend - wstart);
      scalar_t diff_val = is_empty ? scalar_t(0) : top_diff[index] / bin_area;
      for (int y = hstart; y < hend; ++y) {
        for (int x = wstart; x < wend; ++x) {
          int y_low = (int)y;
          int x_low = (int)x;
          int y_high;
          int x_high;

          if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (scalar_t)y_low;
          } else {
            y_high = y_low + 1;
          }

          if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (scalar_t)x_low;
          } else {
            x_high = x_low + 1;
          }
          scalar_t ly = y - y_low;
          scalar_t lx = x - x_low;
          scalar_t hy = 1. - ly;
          scalar_t hx = 1. - lx;
          // do bilinear interpolation
          scalar_t w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

          atomicAdd(offset_bottom_diff + y_low  * width + x_low,  diff_val*w1);
          atomicAdd(offset_bottom_diff + y_low  * width + x_high, diff_val*w2);
          atomicAdd(offset_bottom_diff + y_high * width + x_low,  diff_val*w3);
          atomicAdd(offset_bottom_diff + y_high * width + x_high, diff_val*w4);
        }
      }
  }
}


int PSROIPoolBackwardLauncher(const at::Tensor top_grad, const at::Tensor rois, const at::Tensor mapping_channel, const int batch_size, const int num_rois, const float spatial_scale, const int channels,
    const int height, const int width, const int pooled_width,
    const int pooled_height, const int output_dim,
    at::Tensor bottom_grad)
{
    const int output_size = output_dim * pooled_height * pooled_width * num_rois;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    top_grad.scalar_type(), "PSROIPoolLaucherBackward", ([&] {
        const scalar_t *top_diff = top_grad.data<scalar_t>();
        const scalar_t *rois_data = rois.data<scalar_t>();
        const int *mapping_channel_data = mapping_channel.data<int>();
        scalar_t *bottom_diff = bottom_grad.data<scalar_t>();

    PSROIPoolBackward<scalar_t>
    <<<GET_BLOCKS(output_size), THREADS_PER_BLOCK>>>
    (output_size, top_diff, mapping_channel_data, num_rois, spatial_scale, height, width, channels, pooled_height,
      pooled_width, output_dim, bottom_diff, rois_data);
    }));
    THCudaCheck(cudaGetLastError());

    return 1;
}




