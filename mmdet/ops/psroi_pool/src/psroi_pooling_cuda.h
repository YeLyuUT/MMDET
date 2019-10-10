#ifndef PS_ROI_POOLING_OP
#define PS_ROI_POOLING_OP

int psroi_pooling_forward_cuda( int pooled_height, int pooled_width, float spatial_scale, int sample_num, int group_size, int output_dim,
                        THCudaTensor * features, THCudaTensor * rois, THCudaTensor * output, THCudaIntTensor * mappingchannel);

int psroi_pooling_backward_cuda(int pooled_height, int pooled_width, float spatial_scale, int sample_num, int output_dim,
                        THCudaTensor * top_grad, THCudaTensor * rois, THCudaTensor * bottom_grad, THCudaIntTensor * mappingchannel);

#endif