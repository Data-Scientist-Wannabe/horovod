// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2019 Uber Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include "mpi_cuda_operations.h"

namespace horovod {
namespace common {

  unsigned int table_msg_size_lookup(unsigned int n ){
    n = n * sizeof(float );
    int msg_size[28] = {8,
                    16,
                    32,
                    64,
                    128,
                    256,
                    512,
                    1024,
                    2048,
                    4096,
                    8192,
                    16384,
                    32768,
                    65536,
                    131072,
                    262144,
                    524288,
                    1048576,
                    2097152,
                    4194304,
                    8388608,
                    16777216,
                    25165824,
                    33554432,
                    41943040,
                    50331648,
                    58720256,
                    67108864};
  for(int i = 0 ; i< 28; ++i){
    if (n < msg_size[i]){
      return msg_size[i] / sizeof(float);
    }
  }

  }

  unsigned int nextPowerOf2_cuda(unsigned int n)  
{  
    unsigned count = 0;  
      
    // First n in the below condition  
    // is for the case where n is 0  
    if (n && !(n & (n - 1)))  
        return n;  
      
    while( n != 0)  
    {  
        n >>= 1;  
        count += 1;  
    }  
      
    return 1 << count;  
} 

MPI_CUDAAllreduce::MPI_CUDAAllreduce(MPIContext* mpi_context,
                                     CUDAContext* cuda_context,
                                     HorovodGlobalState* global_state)
    : CUDAAllreduce(cuda_context, global_state),
      mpi_context_(mpi_context) {}

Status MPI_CUDAAllreduce::Execute(std::vector<TensorTableEntry>& entries, const Response& response) {
  auto& first_entry = entries[0];

  InitCUDA(entries);

  void* buffer_data;
  size_t buffer_len;
  int64_t num_elements = NumElements(entries);

  // Copy memory into the fusion buffer.
  auto& timeline = global_state_->timeline;
  if (entries.size() > 1 || 1) {
    timeline.ActivityStartAll(entries, MEMCPY_IN_FUSION_BUFFER);
    const void* fused_input_data;
    MemcpyInFusionBuffer(entries, fused_input_data, buffer_data, buffer_len);

    auto cuda_result = cudaStreamSynchronize(cuda_context_->streams[entries[0].device]);
    cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);


    num_elements = table_msg_size_lookup((int)num_elements);
    timeline.ActivityEndAll(entries);
  } else {
    buffer_data = (void*) first_entry.output->data();
    buffer_len = (size_t) first_entry.output->size();
  }

  // Do allreduce.
  timeline.ActivityStartAll(entries, MPI_ALLREDUCE);
  const void* sendbuf = entries.size() > 1 || first_entry.tensor->data() == first_entry.output->data()
                        ? MPI_IN_PLACE : first_entry.tensor->data();

  global_state_->counter_allreduce = global_state_->counter_allreduce + 1;
  std::map<int,int>::iterator it;

  int size_mpi,size_msg;


  
  auto start = std::chrono::high_resolution_clock::now();
  int op = MPI_Allreduce(sendbuf, buffer_data,
                         (int) num_elements,
                         mpi_context_->GetMPIDataType(first_entry.tensor),
                         mpi_context_->GetMPISumOp(first_entry.tensor->dtype()),
                         mpi_context_->GetMPICommunicator(Communicator::GLOBAL));

  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
  global_state_->time_allreduce = global_state_->time_allreduce + duration.count();


  MPI_Type_size(mpi_context_->GetMPIDataType(first_entry.tensor), &size_mpi);
  size_msg = size_mpi * (int)num_elements;
  it = global_state_->map_allreduce.find(size_msg);
  if (it == global_state_->map_allreduce.end()){
        global_state_->map_allreduce[size_msg]=1;
        global_state_->time_map_allreduce[size_msg] = duration.count();
    }
  else{
    global_state_->map_allreduce[size_msg]=global_state_->map_allreduce[size_msg]+1;
    global_state_->time_map_allreduce[size_msg] = global_state_->time_map_allreduce[size_msg] + duration.count();
  }


  if (op != MPI_SUCCESS) {
    throw std::logic_error("MPI_Allreduce failed, see MPI output for details.");
  }
  timeline.ActivityEndAll(entries);

  // Copy memory out of the fusion buffer.
  if (entries.size() > 1) {
    timeline.ActivityStartAll(entries, MEMCPY_OUT_FUSION_BUFFER);
    MemcpyOutFusionBuffer(buffer_data, entries);

    auto cuda_result = cudaStreamSynchronize(cuda_context_->streams[entries[0].device]);
    cuda_context_->ErrorCheck("cudaStreamSynchronize", cuda_result);

    timeline.ActivityEndAll(entries);
  }

  return Status::OK();
}

} // namespace common
} // namespace horovod
