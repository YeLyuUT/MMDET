#include <torch/extension.h>
#include <cmath>
#include <vector>
#include "psroi_pooling_kernel.h"

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int psroi_pooling_forward_cuda(int pooled_height, int pooled_width, float spatial_scale, int sample_num, int group_size, int output_dim,
                                at::Tensor features, at::Tensor rois, at::Tensor output, at::Tensor mapping_channel)
{
	CHECK_INPUT(features);
    CHECK_INPUT(rois);
    CHECK_INPUT(output);
    CHECK_INPUT(mapping_channel);
	//Get # of Rois
	int num_rois = rois.size(0);
	int size_rois = rois.size(1);
	if (size_rois!=5)
	{
		printf("wrong roi size\n");
        return 0;
	}
	int channels = features.size(1);
    int height = features.size(2);
    int width = features.size(3);

	PSROIPoolForwardLauncher(
    features, rois, mapping_channel, spatial_scale, sample_num,
    num_rois, height, width, channels,
    pooled_height, pooled_width, group_size, output_dim, output);
	return 1;
}


int psroi_pooling_backward_cuda(int pooled_height, int pooled_width, float spatial_scale, int sample_num, int output_dim,
at::Tensor top_grad, at::Tensor rois, at::Tensor bottom_grad, at::Tensor mapping_channel)
{
    CHECK_INPUT(top_grad);
    CHECK_INPUT(rois);
    CHECK_INPUT(bottom_grad);
    CHECK_INPUT(mapping_channel);

    // Number of ROIs
    int num_rois = rois.size(0);
    int size_rois = rois.size(1);
    if (size_rois != 5)
    {
        printf("wrong roi size\n");
        return 0;
    }
    int batch_size = bottom_grad.size(0);
    int channels = bottom_grad.size(1);
    int height = bottom_grad.size(2);
    int width = bottom_grad.size(3);

    PSROIPoolBackwardLauncher(
    top_grad, rois, mapping_channel, batch_size,
    num_rois, spatial_scale, sample_num, channels, height, width,
    pooled_width, pooled_height, output_dim, bottom_grad);
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &psroi_pooling_forward_cuda, "PSRoi_Pooling forward (CUDA)");
  m.def("backward", &psroi_pooling_backward_cuda, "PSRoi_Pooling backward (CUDA)");
}