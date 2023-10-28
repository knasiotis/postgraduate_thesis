/*written by knasiotis*/
#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H
#define _REDUCE_KERNEL_H_
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <stdio.h>

namespace cg = cooperative_groups;

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template <class T>
struct SharedMemory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <>
struct SharedMemory<double> {
  __device__ inline operator double *() {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }

  __device__ inline operator const double *() const {
    extern __shared__ double __smem_d[];
    return (double *)__smem_d;
  }
};

template <class T>
__device__ __forceinline__ T warpReduceSum(unsigned int mask, T mySum) {
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    mySum += __shfl_down_sync(mask, mySum, offset);
  }
  return mySum;
}

#if __CUDA_ARCH__ >= 800
// Specialize warpReduceFunc for int inputs to use __reduce_add_sync intrinsic
// when on SM 8.0 or higher
template <>
__device__ __forceinline__ int warpReduceSum<int>(unsigned int mask,
                                                  int mySum) {
  mySum = __reduce_add_sync(mask, mySum);
  return mySum;
}
#endif


template <class T>
__global__ void reduce2(T *g_idata, T *g_odata, unsigned int n) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  T *sdata = SharedMemory<T>();

  // load shared mem
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

  sdata[tid] = (i < n) ? g_idata[i] : 0;

  cg::sync(cta);

  // do reduction in shared mem
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }

    cg::sync(cta);
  }

  // write result for this block to global mem
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

//for generalization calculations
inline __device__ T boxVolume(interval_gpu<float> *s, long i, int dims){
    T volume=1;
    for (long j=i; j<i+dims;j++){
        volume*= width(s[j]);
    }
    return volume;
}

inline __device__ interval_gpu<VARTYPE> TorusFun(interval_gpu<VARTYPE> *s,const int i){
    return (s[i]*s[i]) + (s[i+1]*s[i+1]);
}

//NN Missing as it is not my intellectual property
inline __device__ interval_gpu<VARTYPE> nnfunc(interval_gpu<VARTYPE> *vX, const int idx,
    float ** vW1, float  ** vW2, float * vb1, 
    float * vb2, const int targetDim) {//to fill//}

inline __device__ interval_gpu<VARTYPE> GriewankFun(interval_gpu<VARTYPE> *x,const int i){

    interval_gpu<VARTYPE> sum=interval_gpu<VARTYPE>(0,0);
    interval_gpu<VARTYPE> prod=interval_gpu<VARTYPE>(1,1);
    int index = 0;
    for(int j=i; j<i+2;j++){
        #if CODE==0
        sum = sum + div_non_zero((x[j]*x[j]), interval_gpu<VARTYPE>(4000,4000));
        prod = prod * interval_cos(div_non_zero(x[j], interval_gpu<VARTYPE>(sqrtf(index+1),sqrtf(index+1))));
        #elif
        sum = sum + half_div_non_zero((x[j]*x[j]), interval_gpu<VARTYPE>(4000,4000));
        prod = prod * interval_cos(half_div_non_zero(x[j], interval_gpu<VARTYPE>(sqrtf(index+1),sqrtf(index+1))));
        #endif
        index++;
    }

    return sum-prod+interval_gpu<VARTYPE>(1,1);
  }

__global__ void sivia(interval_gpu<VARTYPE> *s, char* labels, const interval_gpu<VARTYPE> * Ybox,const long length, const int dims, int funcID){
    interval_gpu<VARTYPE> fxy;
    interval_gpu<VARTYPE> yb = Ybox[0];
    //add stride loop
    for (int i = (blockIdx.x * blockDim.x + threadIdx.x); i < length; i += (blockDim.x * gridDim.x)){
        //Evaluation of a box via Inclusion function
        switch(funcID){
            case 0:
            fxy = TorusFun(s,i*dims);
            break;
            case 1:
            fxy = GriewankFun(s,i*dims);
            break;
            default:
            fxy = TorusFun_opt(s,i*dims);
        }
        //labels= 0:in, 1:epsilon, 2:out
        labels[i]= (!fxy.is_subset(yb) + !fxy.intersects(yb));
    }
}


//for generalization calculations
__global__ void sivia_nn(interval_gpu<VARTYPE> *s, float* sums, 
    const interval_gpu<VARTYPE> * Ybox,
    const long length, const int dims, int funcID, float ** d_W1, float ** d_W2, 
    float * d_b1, float * d_b2){

    float sum=0;
    sums[(blockIdx.x * blockDim.x + threadIdx.x)] = 0;
    interval_gpu<VARTYPE> fxy;
    interval_gpu<VARTYPE> yb = Ybox[0];
    //add stride loop
    for (long i =(blockIdx.x * blockDim.x + threadIdx.x)*dims; i < length; i += (blockDim.x * gridDim.x)){
        //Evaluation of a box via Inclusion function
        fxy = nnfunc(s,i, d_W1, d_W2, d_b1, d_b2, 0);
        //printf("%f", boxVolume(s,i*dims, dims));
        #if CODE==0
        sum+=fxy.is_subset(yb)*boxVolume(s,i, dims);
        #elif 
        sum += half_is_subset(fxy, yb)*boxVolumeHalf(s,i, dims);
        #endif
    __syncthreads();
    //load sum in sums to perform reduction afterwards
    sums[(blockIdx.x * blockDim.x + threadIdx.x)] += sum;
}


//for generalization calculations
inline __device__ T boxVolumeHalf(interval_gpu<VARTYPE> *s, long i, const int dims){
    float volume=1;
    for (long j=i; j<i+dims;j++){
        volume*= __half2float(width(s[j]));
    }
    return volume;
}
inline long calculateBoxes(interval_gpu<float> *initBox, const int dims, const float eps){
  long boxes = 1;
  for (int i=0; i < dims; i++){
      boxes *= pow(2,ceil(log2(initBox[i].h_width()/eps)));
  }
  return boxes;
}



//bisection engineered to run on host device
inline __host__ void h_bisection(ivector_gpu *boxes, const int numBoxes,const float eps){
    //containers for bisection result
    interval_gpu<double> first = interval_gpu<double>::empty();
    interval_gpu<double> second = interval_gpu<double>::empty();
    
    //int numBoxes = calculateBoxes(initBox, eps); NOT THIS FUNCTIONS JOB DOESNT EXPLICITLY REMIND DELETION
    //ivector_gpu* boxes = new ivector_gpu[numBoxes]; HAVE TO BE MOVED TO MAIN
    int dims = boxes[0].size();
    int currentBox = 0; //to change with stride in device version
    int offset = 0; //offset keeps track of last inserted element
    while(currentBox<numBoxes){ //numBoxes has to be replaced with stride limit  in device version
        if (boxes[currentBox].h_max_diam()<=eps){
            currentBox++;
            continue;
        }
        for(int i=0; i<dims; i++){
            if(boxes[currentBox][i].h_width()>eps){
                h_bisect(boxes[currentBox],first,second,i);
                offset++; //offset goes until the end of the stride interval in the device
                boxes[offset]=boxes[currentBox];
                boxes[offset][i]=second;
                boxes[currentBox][i]=first;
                break;
            }
        }
    }
    //return boxes;
}

//bisection engineered to run on host device
template <class T>
inline __host__ void h_bisection_opt(interval_gpu<T> *boxes, long numBoxes, int dims, double eps){
    //containers for bisection result
    interval_gpu<T> first = interval_gpu<T>::empty();
    interval_gpu<T> second = interval_gpu<T>::empty();
    
    long currentBox = 0; //to change with stride in device version
    int offset = 0; //offset keeps track of last inserted element
    while(currentBox<numBoxes){ //numBoxes has to be replaced with stride limit  in device version
        //std::cout<< "currentBox >" << currentBox << " location >"  << currentBox*dims << " max diam >" << h_max_diam_opt(boxes, currentBox, dims) << std::endl; 
        if (h_max_diam_opt(boxes, currentBox, dims)<=eps){
            //std::cout<< "next box" << std::endl; 
            currentBox++;
            continue;
        }
        for(int i=0; i<dims; i++){
            //std::cout<< "major index >" << currentBox*dims << " minor index >" << currentBox*dims+i << " width >" <<  boxes[currentBox*dims+i].h_width() << std::endl; 
            if(boxes[currentBox*dims+i].h_width()>eps){
                h_bisect_opt(boxes[currentBox*dims+i],first,second);
                
                offset++; //offset defines starting point of next box
                //std::cout << "bisected >" << currentBox*dims+i << " target >" << offset*dims+i << std::endl; 
                //copy boxes to an offset area
                for(int j=0; j<dims; j++){
                    boxes[offset*dims+j]=boxes[currentBox*dims+j];
                }
                //std::cout << "new boxes placed at " << offset*dims << std::endl; 
                //replace values with the bisected ones
                boxes[offset*dims+i]=second;
                boxes[currentBox*dims+i]=first;
                break;
            }
        }
    }
}

__global__ void partialBisect(interval_gpu<VARTYPE> *boxes, const long length, const float eps, const int dims){

    interval_gpu<VARTYPE> first = interval_gpu<float>::empty();
    interval_gpu<VARTYPE> second = interval_gpu<float>::empty();

    for (long i = (blockIdx.x * blockDim.x + threadIdx.x); i < length; i += (blockDim.x * gridDim.x)){
        //<- tha borouse na bei ena akomh for loop edw opou rollarei oso leei mia parametros numBisections per cycle.
        //mporei kai oxi. proxwrame
        #if CODE == 0
        for (int j=0; j<dims; j++){
        #elif
        (__hgt(width(boxes[i*dims+j]),eps))
        #endif
            if (width(boxes[i*dims+j])>eps){
                bisect(boxes[i*dims+j],first,second);
                
                for(int k=0; k<dims; k++){
                    boxes[(i+length)*dims+k]=boxes[i*dims+k];
                }

                //replace values with the bisected ones
                boxes[(i+length)*dims+j]=second;
                boxes[i*dims+j]=first;
                break;
            }
        }
    }
}

inline void cpu_partialBisect(interval_gpu<float> *boxes, const long length, const float eps, const int dims){

    interval_gpu<float> first = interval_gpu<float>::empty();
    interval_gpu<float> second = interval_gpu<float>::empty();

    for (int i = 0; i < length; i++){
        for (int j=0; j<dims; j++){
            if (boxes[i*dims+j].h_width()>eps){
                h_bisect_opt(boxes[i*dims+j],first,second);
                
                for(int k=0; k<dims; k++){
                    boxes[(i+length)*dims+k]=boxes[i*dims+k];
                }

                //replace values with the bisected ones
                boxes[(i+length)*dims+j]=second;
                boxes[i*dims+j]=first;

                break;
            }
        }
    }
}   


/*
void bisectBenchmark(){
    double eps = 0.001;
    int dims = 2;
    ivector_gpu initBox = ivector_gpu(2);
	initBox[0]=interval_gpu<double>(-1.5,1.5);
	initBox[1]=interval_gpu<double>(-1.5,1.5); 
    long numBoxes = calculateBoxes(initBox, eps);
    interval_gpu<float> * boxes = new interval_gpu<float>[numBoxes*dims];
    interval_gpu<float> * d_boxes;
    boxes[0] = interval_gpu<float>(-1.5,1.5);
    boxes[1] = interval_gpu<float>(-1.5,1.5); 
    std::cout << "Bisection Benchmark" << std::endl;
    printf("Initial Box ([%f,%f],[%f,%f])\n",boxes[0].lower(),boxes[0].upper(),boxes[1].lower(),boxes[1].upper());
    std::cout << "Epsilon: " << eps << " Dimensions: " << dims << std::endl;
    std::cout << "Problem size: " << numBoxes << " boxes" << std::endl;
    CHECKED_CALL(cudaMalloc((void**)&d_boxes,numBoxes*dims*sizeof(*d_boxes)));
    CHECKED_CALL(cudaMemcpy(d_boxes, boxes , numBoxes*dims*sizeof(*d_boxes), cudaMemcpyHostToDevice));

    long length = 1;
    for (int i=1; i<log2(numBoxes) ; i++){
        auto gpuTimeStart = std::chrono::high_resolution_clock::now();
        partialBisect<<<38, 1024>>>(d_boxes, length, eps, dims); 
        CHECKED_CALL(cudaGetLastError());             
        CHECKED_CALL(cudaDeviceSynchronize());
        auto gpuTimeEnd = std::chrono::high_resolution_clock::now();
        auto gpuDuration = std::chrono::duration_cast<std::chrono::microseconds>(gpuTimeEnd - gpuTimeStart);

        auto cpuTimeStart = std::chrono::high_resolution_clock::now();
        cpu_partialBisect(boxes, length, eps, dims); 
        auto cpuTimeEnd = std::chrono::high_resolution_clock::now();
        auto cpuDuration = std::chrono::duration_cast<std::chrono::microseconds>(cpuTimeEnd - cpuTimeStart);
        std::cout << "Boxes: " << length << " | " << "CPU Duration: " << cpuDuration.count() << "us " << "GPU Duration: " << gpuDuration.count() << "us" << std::endl;
        length*=2;
    }

    delete[] boxes;
    cudaFree(d_boxes);
}*/


