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

image mat_to_image(cv::Mat src) {
  int h = src.rows;
  int w = src.cols;
  int c = src.channels();
  image out = make_image(w, h, c);
  int countdata = 0;
  for (int i = 0; i < c; i++) {
    for (int j = 0; j < h; j++) {
      for (int k = 0; k < w; k++) {
        out.data[countdata++] = src.data[j * src.step + k * c + i] / 255.0;
      }
    }
  }
  rgbgr_image(out);
  return out;
}

int main(int argc, char *argv[]) {
  list *options = read_data_cfg("cfg/coco.data");
  char *name_list = option_find_str(options, "names", "data/names.list");
  char **names = get_labels(name_list);

  image **alphabet = load_alphabet();
  network net = parse_network_cfg(argv[1]);
  if (argv[2]) {
    load_weights(&net, argv[2]);
  }
  set_batch_network(&net, 1);
  srand(2222222);
  float nms = 0.4;
  cv::Mat cv_image = cv::imread(argv[3]);
  cv::Mat cv_resize_image;
  cv::resize(cv_image, cv_resize_image, cv::Size(416, 416));
  image im = mat_to_image(cv_image);
  image sized = resize_image(im, net.w, net.h);
  layer l = net.layers[net.n - 1];
  printf("size : %d, %d, %d\n", l.w, l.h, l.n);

  std::vector<box> boxes(l.w * l.h * l.n);
  std::vector<float *> probs(l.w * l.h * l.n);
  //    float *probs_in = static_cast<float *>(calloc(probs.size() * l.classes, sizeof(float)));
  for (int i = 0; i < netsize; i++) {
    probs[i] = static_cast<float *>(calloc(l.classes, sizeof(float)));
  }
  float *X = sized.data;
  network_predict(net, X);
  get_region_boxes(l, 1, 1, 0.24, probs.data(), boxes.data(), 0, 0);
  if (nms) {
    do_nms_sort(boxes.data(), probs.data(), l.w * l.h * l.n, l.classes, nms);
  }
  for (int i = 0; i < l.w * l.h * l.n; i++) {
    int class_id = max_index(probs[i], l.classes);
    float prob = probs[i][class_id];
    if (prob > 0.24) {
      printf("%d, %s: %.0f%%\n", class_id, names[class_id], prob * 100);
      box b = boxes[i];
      int left = (b.x - b.w / 2.) * cv_image.cols;
      int right = (b.x + b.w / 2.) * cv_image.cols;
      int top = (b.y - b.h / 2.) * cv_image.rows;
      int bot = (b.y + b.h / 2.) * cv_image.rows;

      if (left < 0) left = 0;
      if (right > im.w - 1) right = im.w - 1;
      if (top < 0) top = 0;
      if (bot > im.h - 1) bot = im.h - 1;

      cv::Scalar rngclr = cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
      cv::rectangle(cv_image, cv::Point(left, top), cv::Point(right, bot), rngclr, 4);
      int baseline = 0;
      cv::Size textsize = cv::getTextSize(std::string(names[class_id]), cv::FONT_HERSHEY_SIMPLEX, 0.8, 1, &baseline);
      cv::rectangle(cv_image, cv::Point(left - 2, top), cv::Point(left + textsize.width, top - textsize.height - 15), rngclr, -1);
      cv::putText(cv_image, std::string(names[class_id]), cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 1, CV_AA);
    }
  }
  cv::imshow("test", cv_image);

  free_image(im);
  free_image(sized);
  for (int i = 0; i < probs.size(); i++) {
    free(probs[i]);
  }
  cvWaitKey();
  cvDestroyAllWindows();

  return 0;
}
