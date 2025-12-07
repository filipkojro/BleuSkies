import cv2
import numpy as np
import matplotlib.pyplot as plt

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
class Segmentator:
  def __init__(self) -> None:
    self.cfg_panoptic = get_cfg()
    self.cfg_panoptic.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
    self.cfg_panoptic.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

    self.cfg_panoptic.MODEL.DEVICE = 'cpu'

    self.panoptic_predictor = DefaultPredictor(self.cfg_panoptic)

    self.metadata = MetadataCatalog.get(self.cfg_panoptic.DATASETS.TRAIN[0])

    self.thing_classes = self.metadata.thing_classes
    self.stuff_classes = self.metadata.stuff_classes

    self.global_classes = ['']

    for i in range(len(self.thing_classes)):
      self.global_classes.append(self.thing_classes[i])
    for i in range(len(self.stuff_classes)):
      self.global_classes.append(self.stuff_classes[i])
    self.global_classes = np.array(sorted(self.global_classes), dtype=str)
    

  def get_segments(self, image):

    panoptic_seg, segments_info = self.panoptic_predictor(image)["panoptic_seg"]
    panoptic_seg_np = np.array(panoptic_seg.to("cpu"))
    panoptic_seg_g_labels = np.empty_like(panoptic_seg_np, dtype=int)

    for segment in segments_info:
      segment_desc = {}

      segment_desc['id'] = segment['id']

      if segment['isthing']:
        segment_desc['label'] = self.thing_classes[segment['category_id']]
      else:
        segment_desc['label'] = self.stuff_classes[segment['category_id']]

      panoptic_seg_g_labels[panoptic_seg_np == segment['id']] = np.argwhere(self.global_classes == segment_desc['label'])[0,0]

    panoptic_seg_g_labels[panoptic_seg_g_labels == None] = 0

    return panoptic_seg_g_labels
  

def cut_chunks(image, n_rows, n_cols):
  chunks = []
  for y in range(n_rows):
    row = []
    for x in range(n_cols):
      row.append(image[int(y * (image.shape[0]/n_cols)) : int((y+1) * (image.shape[0]/n_cols)),
                       int(x * (image.shape[1]/n_rows)) : int((x+1) * (image.shape[1]/n_rows))])
    chunks.append(row)
  return chunks


def glue_chunks(chunks):
  return np.block(chunks)


def segment_image(segmentator, path_to_file):
    
    im = cv2.imread(path_to_file)

    segments = segmentator.get_segments(im)

    

    return segments

    # plt.imshow(np.array(segments), cmap='turbo')
    # plt.colorbar()
    # # for label in labels:
    # #     print(label)
    # plt.savefig("output.png", dpi=300, bbox_inches='tight', pad_inches=0)



if __name__ == "__main__":
# example chunks

    in_image = np.arange(6*6).reshape((6,6))
    print(in_image)

    in_image = cut_chunks(in_image, 4, 4)
    print(in_image)

    in_image = glue_chunks(in_image)
    print(in_image)

    segmentator = Segmentator()

    im = cv2.imread("./proj_000-3.png")

    segments, labels = segmentator.get_segments(im, merged_labels=False)

    plt.imshow(np.array(segments), cmap='turbo')
    plt.colorbar()
    for label in labels:
        print(label)
        plt.show()
