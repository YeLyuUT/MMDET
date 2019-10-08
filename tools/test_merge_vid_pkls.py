import mmcv
import os
import os.path as osp
import glob
import argparse

def parse_args():
  parser = argparse.ArgumentParser(description='Merge vids detect results.')
  parser.add_argument(
    '--vid_jsons_dir',
    help='directory contains json anns for all vids',
    type=str,
    default='./data/imagenet/ImageSets/VID_val_jsons')
  parser.add_argument(
    '--output_pkls_dir',
    help='directory contains pkl outputs for all vids',
    type=str,
    default='./work_dirs/VID_val_pkls_output')
  parser.add_argument(
    '--out_file',
    help='merged filename',
    type=str,
    default='./work_dirs/vid_val_output.pkl')
  parser.add_argument(
    '--threshold',
    help='filter out det with score lower than threshold.',
    type=float,
    default=0.)
  args = parser.parse_args()
  return args

def main():
  args = parse_args()
  vid_jsons_dir = args.vid_jsons_dir
  output_pkls_dir = args.output_pkls_dir
  out_file = args.out_file
  threshold = args.threshold
  
  assert osp.isdir(output_pkls_dir)
  fList_vid = glob.glob1(vid_jsons_dir, "*.json")
  fCounter_vid = len(fList_vid)
  fList = glob.glob1(output_pkls_dir, "*.pkl")
  fCounter = len(fList)
  assert fCounter_vid==fCounter,'number of videos are not equal. %d!=%d'%(fCounter_vid, fCounter)
  all_dets = []
  last_count = 0
  for i in range(fCounter):
    json_vid_bbox = osp.join(vid_jsons_dir, '%06d.json' % (i))
    pkl_out_bbox = osp.join(output_pkls_dir, '%06d.pkl' % (i))
    anns = mmcv.load(json_vid_bbox)
    dets = mmcv.load(pkl_out_bbox)
    for det in dets:
      det = list(det)
      for ind, clsdet in enumerate(det):
        if threshold>0 and len(clsdet)>0:
          det[ind] = clsdet[clsdet[:,-1]>threshold]
      all_dets.append(det)
    last_count=last_count+len(anns)
   
  print(len(all_dets))
  print(last_count)
  mmcv.dump(all_dets, out_file)

if __name__=='__main__':
  main()
