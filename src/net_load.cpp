#include <iostream>

extern "C" {
#ifdef __cplusplus
#include "box.h"
#include "cost_layer.h"
#include "detection_layer.h"
#include "image.h"
#include "network.h"
#include "parser.h"
#include "region_layer.h"
#include "utils.h"
#endif
}

int main(int argc, char *argv[]) {
  network net = parse_network_cfg(argv[1]);
  if (argv[2] != NULL) {
    load_weights(&net, argv[2]);
  }
  set_batch_network(&net, 1);
  layer l = net.layers[net.n - 1];
  printf("size : %d, %d, %d\n", l.w, l.h, l.n);
  return 0;
}
