
#ifndef MYCLASS_H
#define MYCLASS_H

#include <map>
namespace horovod {
namespace common {
struct BcastState{
      int counter_bcast=0;
      int counter_allreduce=0;

      std::map <int,int> map_allreduce;
      std::map <int,int> map_bcast;

      int time_allreduce;
};

}
}

#endif

