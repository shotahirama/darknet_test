#include <sys/time.h>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

extern "C" {
#undef __cplusplus
#include "box.h"
#include "cost_layer.h"
#include "demo.h"
#include "network.h"
#include "option_list.h"
#include "parser.h"
#include "region_layer.h"
#include "utils.h"
#define __cplusplus
}

class Darknet {
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

  network net_;
  float nms_;
  float thresh_;
  float hier_;
  std::vector<box> boxes_;
  std::vector<float *> probs_;
  char **names;
  std::vector<cv::Scalar> cv_color_;

 public:
  Darknet() : nms_(0.4), thresh_(0.24), hier_(0.5) {}

  ~Darknet() {}

  void load(std::string cfgfile, std::string weightfile = "") {
    list *options = read_data_cfg("cfg/coco.data");
    char *name_list = option_find_str(options, "names", "data/names.list");
    names = get_labels(name_list);

    net_ = parse_network_cfg(const_cast<char *>(cfgfile.c_str()));
    if (weightfile != "") {
      std::cout << "weight file >> " << weightfile << std::endl;
      load_weights(&net_, const_cast<char *>(weightfile.c_str()));
    }
    set_batch_network(&net_, 1);
    srand(2222222);
    for (int i = 0; i < 80; i++) {
      cv_color_.push_back(cv::Scalar(rand() % 255, rand() % 255, rand() % 255));
    }
  }

  void set_nms(float nms) { this->nms_ = nms; }

  void set_thresh(float thresh) { this->thresh_ = thresh; }

  void set_hier(float hier) { this->hier_ = hier; }

  cv::Mat detect(cv::Mat input_src) {
    cv::Mat src = input_src.clone();
    image im = mat_to_image(src);
    image sized = resize_image(im, net_.w, net_.h);
    layer l = net_.layers[net_.n - 1];
    //    printf("size : %d, %d, %d\n", l.w, l.h, l.n);
    int netsize = l.w * l.h * l.n;
    boxes_.resize(netsize);
    probs_.resize(netsize);
    for (int i = 0; i < netsize; i++) {
      probs_[i] = static_cast<float *>(calloc(l.classes, sizeof(float)));
    }
    //    float *X = sized.data;
    network_predict_gpu(net_, sized.data);
    get_region_boxes(l, 1, 1, thresh_, probs_.data(), boxes_.data(), 0, 0, hier_);
    if (nms_) {
      do_nms(boxes_.data(), probs_.data(), netsize, l.classes, nms_);
    }
    for (int i = 0; i < netsize; i++) {
      int class_id = max_index(probs_[i], l.classes);
      float prob = probs_[i][class_id];
      if (prob > thresh_) {
        printf("%d, %s: %.0f%%\n", class_id, names[class_id], prob * 100);
        box b = boxes_[i];
        int left = (b.x - b.w / 2.) * im.w;
        int right = (b.x + b.w / 2.) * im.w;
        int top = (b.y - b.h / 2.) * im.h;
        int bot = (b.y + b.h / 2.) * im.h;

        if (left < 0) left = 0;
        if (right > im.w - 1) right = im.w - 1;
        if (top < 0) top = 0;
        if (bot > im.h - 1) bot = im.h - 1;

        cv::Scalar rngclr = cv_color_[class_id];
        //            cv::Scalar(rand() % 255, rand() % 255, rand() % 255);
        cv::rectangle(src, cv::Point(left, top), cv::Point(right, bot), rngclr, 4);
        int baseline = 0;
        cv::Size textsize = cv::getTextSize(std::string(names[class_id]), cv::FONT_HERSHEY_SIMPLEX, 0.8, 1, &baseline);
        cv::rectangle(src, cv::Point(left - 2, top), cv::Point(left + textsize.width, top - textsize.height - 15), rngclr, -1);
        cv::putText(src, std::string(names[class_id]), cv::Point(left, top - 10), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 1, CV_AA);
      }
    }
    free_image(im);
    free_image(sized);
    for (size_t i = 0; i < probs_.size(); i++) {
      free(probs_[i]);
    }
    return src;
  }
};

double get_wall_time() {
  struct timeval time;
  if (gettimeofday(&time, NULL)) {
    return 0;
  }
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

int main(int argc, char *argv[]) {
  Darknet darknet;
  darknet.load(argv[1], argv[2]);
  if (argv[3]) {
    darknet.set_thresh(atof(argv[3]));
  }
  cv::VideoCapture cap(1);
  while (true) {
    double before = get_wall_time();
    cv::Mat img;
    cap >> img;
    cv::Mat dst = darknet.detect(img);
    double after = get_wall_time();
    double fps = 1.0 / (after - before);
    std::cout << "FPS = " << fps << std::endl;
    std::cout << "********************************" << std::endl;
    cv::imshow("test", dst);
    if (cvWaitKey(1) > 0) {
      break;
    }
  }

  return 0;
}
