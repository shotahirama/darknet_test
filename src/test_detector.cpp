#include <opencv2/opencv.hpp>
#include <vector>

extern "C" {
#ifdef __cplusplus
#include "box.h"
#include "cost_layer.h"
#include "demo.h"
#include "network.h"
#include "option_list.h"
#include "parser.h"
#include "region_layer.h"
#include "utils.h"
#endif
}

int main(int argc, char *argv[]) {
  list *options = read_data_cfg("cfg/coco.data");
  char *name_list = option_find_str(options, "names", "data/names.list");
  char **names = get_labels(name_list);

  std::cout << "hogehoge" << std::endl;
  image **alphabet = load_alphabet();
  network net = parse_network_cfg(argv[1]);
  if (argv[2]) {
    load_weights(&net, argv[2]);
  }
  set_batch_network(&net, 1);
  srand(2222222);
  char buff[256];
  char *input = buff;
  float nms = 0.4;
  strncpy(input, argv[3], 256);
  image im = load_image_color(input, 0, 0);
  image sized = resize_image(im, net.w, net.h);
  layer l = net.layers[net.n - 1];
  printf("size : %d, %d, %d", l.w, l.h, l.n);

  std::vector<box> boxes(l.w * l.h * l.n);
  std::vector<float *> probs(l.w * l.h * l.n);
  float *probs_in = static_cast<float *>(calloc(probs.size() * l.classes, sizeof(float)));
  for (int i = 0; i < l.w * l.h * l.n; i++) {
    probs[i] = probs_in;
    probs_in += l.classes;
  }
  float *X = sized.data;
  network_predict(net, X);
  get_region_boxes(l, 1, 1, 0.24, probs.data(), boxes.data(), 0, 0);
  if (nms) {
    do_nms_sort(boxes.data(), probs.data(), l.w * l.h * l.n, l.classes, nms);
  }
  draw_detections(im, l.w * l.h * l.n, 0.24, boxes.data(), probs.data(), names, alphabet, l.classes);
  show_image(im, "predictions");

  free_image(im);
  free_image(sized);
  free(probs.data());
  cvWaitKey();
  cvDestroyAllWindows();

  return 0;
}
