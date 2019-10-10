#ifndef PS_ROI_POOLING_KERNEL
#define PS_ROI_POOLING_KERNEL

int PSROIPoolForwardLauncher(
    const at::Tensor features, const at::Tensor rois, at::Tensor mapping_channel, const float spatial_scale,
    const int sample_num, const int num_rois, const int height,
    const int width, const int channels, const int pooled_height, const int pooled_width,
    const int group_size, const int output_dim, at::Tensor output);

int PSROIPoolBackwardLauncher(const at::Tensor top_grad, const at::Tensor rois, const at::Tensor mapping_channel, const int batch_size,
    const int num_rois, const float spatial_scale, const int sample_num, const int channels, const int height, const int width,
    const int pooled_width, const int pooled_height, const int output_dim, at::Tensor bottom_grad);


#endif