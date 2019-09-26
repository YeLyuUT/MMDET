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
    '--output_jsons_dir',
    help='directory contains json outputs for all vids',
    type=str,
    default='./work_dirs/VID_val_jsons_output')
  parser.add_argument(
    '--out_file',
    help='merged filename',
    type=str,
    default='./work_dirs/vid_val_output.json')
  args = parser.parse_args()
  return args

def main():
  args = parse_args()
  vid_jsons_dir = args.vid_jsons_dir
  output_jsons_dir = args.output_jsons_dir
  out_file = args.out_file

  assert osp.isdir(output_jsons_dir)
  fList_vid = glob.glob1(vid_jsons_dir, "*.json")
  fCounter_vid = len(fList_vid)
  fList = glob.glob1(output_jsons_dir, "*.json")
  fCounter = len(fList)
  assert fCounter_vid==fCounter,'number of videos are not equal. %d!=%d'%(fCounter_vid, fCounter)
  last_count = 0
  all_dets = []
  for i in range(fCounter):
    json_vid_bbox = osp.join(vid_jsons_dir, '%06d.json' % (i))
    json_out_bbox = osp.join(output_jsons_dir, '%06d.bbox.json' % (i))
    anns = mmcv.load(json_vid_bbox)
    dets = mmcv.load(json_out_bbox)
    for det in dets:
      det['image_id'] = det['image_id']+last_count
      all_dets.append(det)
    last_count = last_count+len(anns)
  print(last_count)
  mmcv.dump(all_dets, out_file)

if __name__=='__main__':
  main()