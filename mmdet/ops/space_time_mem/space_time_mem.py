import torch
import torch.nn as nn
import numpy as np

class PointwiseGraphNN(nn.Module):
  # space_time_mem is used to augment the features2
  # relation percent 1.0 for training, 0.25 for testing.
  def __init__(self, C_in, C_qk=None, relation_percent=1.0):
    super(PointwiseGraphNN, self).__init__()
    if C_qk is None:
      C_qk = int(C_in / 8)

    self.query = nn.Conv2d(C_in, C_qk, kernel_size=1)
    self.key = nn.Conv2d(C_in, C_qk, kernel_size=1)
    self.value = nn.Conv2d(C_in, C_in, kernel_size=1)
    self.C_qk = C_qk
    self.relation_percent = relation_percent
    self.softmax = nn.Softmax(dim=1)
    self.init_weights()
    assert relation_percent>0 and relation_percent<=1.0

  def init_weights(self):
    nn.init.normal_(self.query.weight, 0, 0.01)
    nn.init.constant_(self.query.bias, 0)
    nn.init.normal_(self.key.weight, 0, 0.01)
    nn.init.constant_(self.key.bias, 0)
    nn.init.normal_(self.value.weight, 0, 0.01)
    nn.init.constant_(self.value.bias, 0)

  def forward(self, space_time_mem, features2):
    '''
    For each pixel, calculate the relations between each other.
    Augment feature2 by feature1.

    :param features1: [T,N,C,H1,W1]
    :param features2: [N,C,H2,W2]
    '''
    T, N, C, H1, W1 = space_time_mem.shape
    space_time_mem_4d = space_time_mem.view(T * N, C, H1, W1)
    query_feat1 = self.query(space_time_mem_4d)
    key_feat2 = self.key(features2)
    value_feat1 = self.value(space_time_mem_4d)
    query_feat1_transpose = query_feat1.view(T, N, self.C_qk, H1, W1).permute(1, 2, 0, 3, 4)
    value_feat1_transpose = value_feat1.view(T, N, C, H1, W1).permute(1, 2, 0, 3, 4)
    relation_matrix = self.pointwise_relation_matrix(query_feat1_transpose, key_feat2)
    aug_feat2 = self.augment_feature2(relation_matrix, value_feat1_transpose, features2)
    return aug_feat2

  def pointwise_relation_matrix(self, space_time_mem_5d, features2):
    N1, C1, T, H1, W1 = space_time_mem_5d.shape
    N2, C2, H2, W2 = features2.shape

    m1 = space_time_mem_5d.reshape(N1, C1, T*H1*W1).permute(0, 2, 1)
    m2 = features2.view(N2, C2, H2*W2)
    # relation_matrix of size (N, T*H1*W1, H2*W2)
    relation_matrix = torch.bmm(m1, m2)
    relation_matrix = relation_matrix/np.sqrt(C1)
    relation_matrix = self.softmax(relation_matrix)
    return relation_matrix

  def augment_feature2(self, relation_matrix, value_feat1_5d, features2):
    N1, C1, T1, H1, W1 = value_feat1_5d.shape
    value_feat1_5d = value_feat1_5d.reshape((N1, C1, T1 * H1 * W1))
    # TODO sparsify the graph.
    rel_sum = torch.sum(relation_matrix,dim=(0,2))
    indices = torch.argsort(rel_sum, descending=True)

    if self.relation_percent<1.0:
      NR, R1, R2 = relation_matrix.shape
      kept_num = int(R1*self.relation_percent)
      indices = indices[:kept_num]

    relation_matrix = torch.index_select(relation_matrix, 1, indices)
    value_feat1_5d = torch.index_select(value_feat1_5d, 2, indices)

    augmented_features2 = torch.bmm(value_feat1_5d, relation_matrix).view(features2.shape)
    #augmented_features2 = augmented_features2 + features2
    return augmented_features2





