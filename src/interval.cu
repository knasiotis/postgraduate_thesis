#include <chrono>
#include <iostream>
#include <stdio.h>
#include "helper_cuda.h"
#include "interval.h"
#include "cuda_interval.h"
#include "cpu_interval.h"
#include "functions.h"
#include "vibes.h"
#include "argshand.h" 
#include <thread>
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <algorithm>

// includes, project
//#include "reduction.h"
//#include "reduction_kernel.cu"

//NN Initialization (loads weights in GPU from a file)
//Missing, not my intellectual property
class NNConfig {};



extern "C" bool isPow2(unsigned int x) { return ((x & (x - 1)) == 0); }

void threadRoutine(interval_gpu<float> ** initialBoxes, long initialBoxesLength, int devID, int tid);
void parametersError();
void argParser(int argc, char *argv[]);
void exec();
void gpuSIVIA_nn(interval_gpu<VARTYPE> * boxes, interval_gpu<VARTYPE> * d_boxes, interval_gpu<VARTYPE> * d_Ybox,
            float * sums, float * d_sums, float * o_sums, double &totalSum, const long numBoxes, const int blocks, const int threads,
            std::chrono::milliseconds &initMemTransferDuration, std::chrono::milliseconds &bisectionDuration, 
            std::chrono::milliseconds &siviaDuration, std::chrono::milliseconds &memDuration, std::chrono::milliseconds &reductionDuration, NNConfig &nn);
void gpuSIVIA(interval_gpu<VARTYPE> * boxes, interval_gpu<VARTYPE> * d_boxes, interval_gpu<VARTYPE> * d_Ybox,
            char * labels, char * d_labels, const long long numBoxes, const int blocks, const int threads,
            std::chrono::milliseconds &initMemTransferDuration, std::chrono::milliseconds &bisectionDuration, 
            std::chrono::milliseconds &siviaDuration, std::chrono::milliseconds &memDuration);

//funcID 0 : Torus, 1: griewank
int funcID;
int devID_min;
int devID_max;
int dims;
float eps;
bool vc2=false;
interval_gpu<float> * startingBox;
interval_gpu<VARTYPE> * Ybox;
 
int main(int argc, char *argv[]) {
    //NNConfig nn("ESperfVColumn_2Cnet30tanlogBinary_Full.dat");
    argParser(argc, argv);
    std::cout << "Epsilon                                              : " << eps << std::endl;
    std::cout << "Dimensions                                           : " << dims << std::endl;
    //print initial box and epsilon along the function
    //execution
    exec();

    //cleaning
    delete[] startingBox;
    CHECKED_CALL(cudaFreeHost(Ybox));
    return 0;
} 
       
void exec(){
    interval_gpu<float> ** buffer;
    if(devID_min==devID_max){

        //single GPU execution
        //creates 1 thread on a target GPU
        buffer = new interval_gpu<float>*[1];
        buffer[0] = new interval_gpu<float>[dims];
        for(long i=0; i<dims; i++){
            buffer[0][i]=startingBox[i];
        }
        threadRoutine(buffer, 1, devID_min, 0);
      
    }else{
        long bisections=0;
        int gpuRange=0;
        std::cout << "GPU RANGE                                            : " << devID_min << "-" << devID_max << std::endl; 
        gpuRange = (devID_max-devID_min)+1;
        
        std::cout << "Number of GPUs                                       : " << gpuRange << std::endl; 
        
        bisections = ceil(log2(gpuRange));
        std::cout << "Number of Bisections                                 : " << bisections << std::endl; 
        
        long queueBoxes = pow(2,bisections);
        std::cout << "Boxes in initial queue                               : " << queueBoxes << std::endl; 
        interval_gpu<float> * queue = new interval_gpu<float>[queueBoxes*dims];

        //take first box from queue as a baseline
        for(long i=0; i<dims; i++){
            queue[i]=startingBox[i];
            //std::cout << "startingBox->queue : [" << queue[i].lower() << "," << queue[i].upper() << "]" << std::endl;
        }
        
        //bisect it log2() times
        for (long i=1; i<queueBoxes; i*=2){
            cpu_partialBisect(queue, i, eps, dims);
        }
        
        /*//debug queueBoxes
        for (int i=0; i<queueBoxes*dims; i+=dims){
            std::cout << "partialBisect->queue : [" << queue[i].lower() << "," << queue[i].upper() << "],";
            std::cout << "[" << queue[i+1].lower() << "," << queue[i+1].upper() << "]" << std::endl;
        }*/

        long * distribution = new long[gpuRange];
        for (long i=0; i<gpuRange; i++){
            distribution[i]=0;
        }
       
        int tempID=0;
        for (long i=0; i < queueBoxes; i++){
            distribution[tempID++]++;
            if (tempID==gpuRange){
                tempID=0;
            }
        }

        //debug distribution
        /*for (int i=0; i<gpuRange; i++){
            std::cout << "Distribution between threads :" << distribution[i] << ":boxes" << std::endl;
        }*/

        //create initial buffers
        //multi GPU execution
        //distributes boxes among threads
        buffer = new interval_gpu<float>*[gpuRange];
        for (long i=0; i<gpuRange; i++){
            buffer[i] = new interval_gpu<float>[dims*distribution[i]];
        }

        
        long count=0;
        for (long i=0; i<gpuRange; i++){
            for (long j=0; j < dims*distribution[i]; j++){
                buffer[i][j] = queue[count+j];
            }
            count+=dims*distribution[i];
        }

        //equal to num of GPUs
        //and assigns devID accordingly
        //spread threads
        std::thread *t = new std::thread[gpuRange];
        tempID = devID_min;
        for (long i=0; i<gpuRange; i++){
            //dont forget to replace the '0' after deployment
            t[i] = std::thread(threadRoutine, buffer, distribution[i], i, i);
            //t[i].join();
            //threadRoutine(buffer, distribution[i], 0, i);
            tempID++;
        }

        //wait to collect
        for (int i=0; i<gpuRange; i++){
            t[i].join();
        }

        //collect threads
        delete[] t;
        delete[] queue;
    }
    delete[] buffer;
}


   
void threadRoutine(interval_gpu<float> ** initialBoxes, long initialBoxesLength, int devID, int tid){
    auto threadStart = std::chrono::high_resolution_clock::now();
    int devCount;
    CHECKED_CALL(cudaGetDeviceCount(&devCount));
    std::cout<<"thread: " << tid << " device count: " << devCount << std::endl;
    cudaDeviceProp deviceProp;
    CHECKED_CALL(cudaGetDeviceProperties(&deviceProp, devID)); 
    CHECKED_CALL(cudaSetDevice(devID));
    CHECKED_CALL(cudaGetLastError());         
    CHECKED_CALL(cudaDeviceSynchronize());
    CHECKED_CALL(cudaGetLastError());
    NNConfig nn("ESperfVColumn_2Cnet30tanlogBinary_Full.dat");
    int blocks = 2*(deviceProp.multiProcessorCount * deviceProp.maxThreadsPerMultiProcessor)/(deviceProp.maxThreadsPerBlock/2);
    //int blocks = 32;
    int threads = deviceProp.maxThreadsPerBlock/2;
    blocks = floor(log2(blocks*threads));
    blocks = floor((pow(2,blocks))/(threads));
    std::cout << blocks << " Blocks " << threads << " Threads per Block " << std::endl;
    std::cout << "Total Threads running " << threads*blocks << std::endl;
    
    interval_gpu<float> * initBox = new interval_gpu<float>[dims];
    //take first box from queue as a baseline
    for(long i=0; i<dims; i++){
        initBox[i]=initialBoxes[tid][i];
    }
 
    std::chrono::milliseconds initMemTransferDuration = std::chrono::duration_cast<std::chrono::milliseconds>(threadStart-threadStart);
    std::chrono::milliseconds bisectionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(threadStart-threadStart);
    std::chrono::milliseconds siviaDuration = std::chrono::duration_cast<std::chrono::milliseconds>(threadStart-threadStart);
    std::chrono::milliseconds memDuration = std::chrono::duration_cast<std::chrono::milliseconds>(threadStart-threadStart);
    std::chrono::milliseconds reductionDuration = std::chrono::duration_cast<std::chrono::milliseconds>(threadStart-threadStart);
    interval_gpu<float> * queueBoxes;
    interval_gpu<VARTYPE> * boxes;
    interval_gpu<VARTYPE> * d_boxes;
    interval_gpu<VARTYPE> * d_Ybox;

    double totalSum=0;
 
    //replace with sums
    float * sums;
    float * d_sums;
    float * o_sums;

 
    //cuda mallocs
    CHECKED_CALL(cudaMalloc((void**)&d_Ybox,sizeof(*d_Ybox)));
    CHECKED_CALL(cudaMemcpy(d_Ybox, Ybox, sizeof(*d_Ybox),cudaMemcpyHostToDevice));
    //
 
    long boxesPerRun;
    long queueBoxesLength;
    size_t boxSize = sizeof(interval_gpu<VARTYPE>)*dims;
    size_t capacity = ((deviceProp.totalGlobalMem)/boxSize);
    //long capacity = 2048;
    long numBoxes = calculateBoxes(initBox, dims, eps);
    //long initialBisections = ceil(numBoxes/capacity); //partitions box if size>VRAM
    long initialBisections = ceil(log2((numBoxes/capacity)+1)); //partitions box if size>VRAM
  
    /*///debug  variables section
    std::cout << "GPU capacity (boxes)            : " << capacity << std::endl; 
    std::cout << "boxSize (bytes)                 : " << boxSize << std::endl; 
    std::cout << "numBoxes                        : " << numBoxes << std::endl; 
    std::cout << "initialBisections               : " << initialBisections << std::endl;
    ///end of debug  variables section*/
  
    //ALLOCATION SPACE
    if(initialBisections<1){ //whole box will fit in gpu
        boxesPerRun=numBoxes;
        queueBoxesLength = 1;
 
        //thread allocations
        queueBoxes = new interval_gpu<float>[queueBoxesLength*dims];
        boxes = (interval_gpu<VARTYPE> *)malloc(boxesPerRun*dims*sizeof(*boxes));
        //labels = (char*)malloc(boxesPerRun*sizeof(*labels));
        sums = (float*)malloc(blocks*threads*sizeof(*sums));
        //labels = new char[boxesPerRun];
        //CHECKED_CALL(cudaMallocHost((void**)&boxes, boxesPerRun*dims*sizeof(*boxes)));
        //CHECKED_CALL(cudaMallocHost((void**)&labels, boxesPerRun*sizeof(*labels)));
 
    }else{ //partitioning required
        boxesPerRun = numBoxes/pow(2,initialBisections); //total box expansion
        queueBoxesLength = pow(2,initialBisections);
        
        //thread allocations
        queueBoxes = new interval_gpu<float>[queueBoxesLength*dims];
        //boxes = new interval_gpu<VARTYPE>[boxesPerRun*dims];
        boxes = (interval_gpu<VARTYPE> *)malloc(boxesPerRun*dims*sizeof(*boxes));
        //labels = (char*)malloc(boxesPerRun*sizeof(*labels));
        sums = (float*)malloc(blocks*threads*sizeof(*sums));
        //labels = new char[boxesPerRun];
        //CHECKED_CALL(cudaMallocHost((void**)&boxes, boxesPerRun*dims*sizeof(*boxes)));
        //CHECKED_CALL(cudaMallocHost((void**)&labels, boxesPerRun*sizeof(*labels)));
        
    } 
    //gpu allocations
    CHECKED_CALL(cudaMalloc((void**)&d_boxes,boxesPerRun*dims*sizeof(*d_boxes)));
    CHECKED_CALL(cudaMalloc((void**)&d_sums,blocks*threads*sizeof(*d_sums)));
    CHECKED_CALL(cudaMalloc((void**)&o_sums,blocks*threads*sizeof(*o_sums)));

    /*///debug  variables section
    std::cout << "Box expansion per run           : " << boxesPerRun << std::endl; 
    std::cout << "Queue Size (or runs on the GPU) : " << queueBoxesLength << std::endl;
    ///end of debug  variables section*/
    //EXECUTION SPACE
    //take a box from initialBoxes -> add to queue

    for (long box=0; box<initialBoxesLength*dims; box+=dims){
        //move each box from initialBoxes to queueboxes
        //this means:
        //take a box from initialBoxes
        for(long i=0; i<dims; i++){
            queueBoxes[i]=initialBoxes[tid][box+i];
        }
        
        //bisect it log2(queueBoxesLength) times
        for (long i=1; i<queueBoxesLength; i*=2){
            cpu_partialBisect(queueBoxes, i, eps, dims);
        }

        //take a box from queue/partition -> add to buffer
        for (long qbox=0; qbox < queueBoxesLength*dims; qbox+=dims){
            
            for(long i=0;i<dims;i++){
                boxes[i]=queueBoxes[qbox+i];
            }
            //move each box from queueBoxes to boxes
            //this means take a box from queueBoxes and copy it in boxes
            //std::cout << "run " << qbox/6 << " of " << queueBoxesLength << std::endl;
            if (vc2){
                gpuSIVIA_nn(boxes, d_boxes, d_Ybox, sums, d_sums, o_sums, totalSum, boxesPerRun, blocks, threads, 
                        initMemTransferDuration, bisectionDuration, siviaDuration, memDuration, reductionDuration, nn);
            }
            else{
                gpuSIVIA(boxes, d_boxes, d_Ybox, labels, d_labels, boxesPerRun, blocks, threads, 
                        initMemTransferDuration, bisectionDuration, siviaDuration, memDuration);
            }
            //RESULTS HANDLE BLOCK
            // verification space
            #ifdef VIBES       
            vibes::beginDrawing();
            //vibes::newFigure("sivia");      
            std::string s;       
            for (long i=0;i<boxesPerRun;i++){    
                switch((short) labels[i]){      
                    case 0: 
                    s="k[r]"; 
                    break;

                    case 1:
                    s="k[y]";
                    break;

                    case 2:
                    s="k[b]"; 
                    break;
                } 
                vibes::drawBox(boxes[i*dims].lower(), boxes[i*dims].upper(), boxes[i*dims+1].lower(), boxes[i*dims+1].upper(), s);
            }    
            vibes::endDrawing(); 
            #endif   
            //we save the result smh here, maybe push
            //to another thread for file writing
            //END OF RESULTS HANDLE BLOCK
        }
    }  
    std::cout << "total sum = " << totalSum << std::endl;
    auto threadStop = std::chrono::high_resolution_clock::now();
    auto threadDuration = std::chrono::duration_cast<std::chrono::milliseconds>(threadStop - threadStart);
    std::cout << "Thread/GPU " << devID << " InitialBoxesLength                      : " << initialBoxesLength << std::endl;
    std::cout << "Thread/GPU " << devID << " Capacity (Boxes)                        : " << capacity << std::endl;
    std::cout << "Thread/GPU " << devID << " Capacity                                : " << deviceProp.totalGlobalMem << " :Bytes" << std::endl;
    std::cout << "Thread/GPU " << devID << " Total Boxes Processed                   : " << numBoxes*initialBoxesLength << std::endl;
    std::cout << "Thread/GPU " << devID << " Total Boxes Processed/GPU run           : " << boxesPerRun << std::endl;
    std::cout << "Thread/GPU " << devID << " Total Boxes Processed         Data Size : " << numBoxes*initialBoxesLength*boxSize << " :Bytes" << std::endl;
    std::cout << "Thread/GPU " << devID << " Total Boxes Processed/GPU run Data Size : " << boxesPerRun*boxSize << " :Bytes" << std::endl;
    std::cout << "Thread/GPU " << devID << " Number of GPU runs                      : " << (queueBoxesLength*initialBoxesLength) << std::endl;
    std::cout << "Thread/GPU " << devID << " Average Bisection Duration              : " << (float)bisectionDuration.count()/(queueBoxesLength*initialBoxesLength) << " :ms" << std::endl;
    std::cout << "Thread/GPU " << devID << " Average Eval & Classification Duration  : " << (float)siviaDuration.count()/(queueBoxesLength*initialBoxesLength) << " :ms" << std::endl;
    std::cout << "Thread/GPU " << devID << " Average SIVIA Duration                  : " << (float)(siviaDuration.count()+bisectionDuration.count())/(queueBoxesLength*initialBoxesLength) << " :ms" << std::endl;
    std::cout << "Thread/GPU " << devID << " Average Reduction Duration              : " << (float)(reductionDuration.count())/(queueBoxesLength*initialBoxesLength) << " :ms" << std::endl;
    std::cout << "Thread/GPU " << devID << " Average Push Memory transfer  Duration  : " << (float)initMemTransferDuration.count()/(queueBoxesLength*initialBoxesLength) << " :ms" << std::endl;
    std::cout << "Thread/GPU " << devID << " Average Pull Memory transfer  Duration  : " << (float)memDuration.count()/(queueBoxesLength*initialBoxesLength) << " :ms" << std::endl;
    std::cout << "Thread/GPU " << devID << " Average Whole Memory transfer Duration  : " << (float)(memDuration.count()+initMemTransferDuration.count())/(queueBoxesLength*initialBoxesLength) << " :ms" << std::endl;
    std::cout << "Thread/GPU " << devID << " Total Bisection Duration                : " << bisectionDuration.count() << " :ms" << std::endl;
    std::cout << "Thread/GPU " << devID << " Total Eval & Classification Duration    : " << siviaDuration.count() << " :ms" << std::endl;
    std::cout << "Thread/GPU " << devID << " Total SIVIA Duration                    : " << (siviaDuration.count()+bisectionDuration.count()) << " :ms" << std::endl;
    std::cout << "Thread/GPU " << devID << " Total Reduction Duration                : " << (reductionDuration.count()) << " :ms" << std::endl;
    std::cout << "Thread/GPU " << devID << " Total Push Memory transfer Duration     : " << initMemTransferDuration.count() << " :ms" << std::endl;
    std::cout << "Thread/GPU " << devID << " Total Pull Memory transfer Duration     : " << memDuration.count() << " :ms" << std::endl;
    std::cout << "Thread/GPU " << devID << " Total Memory transfers Duration         : " << memDuration.count()+initMemTransferDuration.count() << " :ms" << std::endl;
    std::cout << "Thread/GPU " << devID << " Total Thread Duration                   : " << threadDuration.count() << " :ms" << std::endl;
   
    delete[] queueBoxes;
    delete[] initBox;
    free(boxes);
    free(sums);
    //CHECKED_CALL(cudaFreeHost(labels));
    //CHECKED_CALL(cudaFreeHost(boxes));
    CHECKED_CALL(cudaFree(d_boxes));
    CHECKED_CALL(cudaFree(d_sums));
    CHECKED_CALL(cudaFree(o_sums));
    CHECKED_CALL(cudaFree(d_Ybox));
} 

   


//labels and Ybox will be allocated at the start of the thread, no need to repeat the allocation
//all the mallocs as well
//eps, dims global

//the main execution logic
//input: one or more boxes and an Initial length
//output: labels, boxes
int reducecount=0;
int siviacount=0;
void gpuSIVIA_nn(interval_gpu<VARTYPE> * boxes, interval_gpu<VARTYPE> * d_boxes, interval_gpu<VARTYPE> * d_Ybox,
            float * sums, float * d_sums, float * o_sums, double &totalSum, const long numBoxes, const int blocks, const int threads, 
            std::chrono::milliseconds &initMemTransferDuration, std::chrono::milliseconds &bisectionDuration, 
            std::chrono::milliseconds &siviaDuration, std::chrono::milliseconds &memDuration, std::chrono::milliseconds &reductionDuration, NNConfig &nn){
    
    auto initMemTransferStart = std::chrono::high_resolution_clock::now();
    
    //transfer of boxes
    CHECKED_CALL(cudaMemcpy(d_boxes, boxes, dims*sizeof(*boxes), cudaMemcpyHostToDevice));
    //CHECKED_CALL(cudaMemcpy(d_sums, sums, threads*blocks*sizeof(*d_sums), cudaMemcpyHostToDevice));
    //CHECKED_CALL(cudaMemcpy(d_labels, labels, sizeof(*d_labels), cudaMemcpyHostToDevice));
    
    auto initMemTransferStop = std::chrono::high_resolution_clock::now();
     
    //Bisection
    for (long i=1; i<numBoxes; i*=2){
        //blocks and threads will be drawn from context
        partialBisect<<<blocks, threads>>>(d_boxes, i, eps, dims); 
        CHECKED_CALL(cudaGetLastError());             
        CHECKED_CALL(cudaDeviceSynchronize());  
    }
    //std::cout << "perasa" << std::endl;
    auto bisectionStop = std::chrono::high_resolution_clock::now();
    /*int vn = nn.noinp();
    int vh = nn.nohid();
    int vm = nn.noout();
    float ** d_vW1 = nn.d_wmat1();
    float ** d_vW2 = nn.d_wmat2();
    float * d_vb1 = nn.d_bvec1();
    float * d_vb2 = nn.d_bvec2();*/
    //SIVIA
    //sivia_opt<<<blocks, threads>>>(d_boxes, d_labels, d_Ybox, numBoxes, dims, funcID);   
    //std::cout << blocks*threads << " " << std::endl;        
    sivia_gen<<<blocks, threads>>>(d_boxes, d_sums, d_Ybox, numBoxes, dims, funcID
    ,nn.d_wmat1(),nn.d_wmat2(),nn.d_bvec1(),nn.d_bvec2());
    CHECKED_CALL(cudaGetLastError());             
    CHECKED_CALL(cudaDeviceSynchronize());   
    //std::cout << "sivia_gen passed " << ++siviacount <<std::endl; 
    auto siviaStop = std::chrono::high_resolution_clock::now();
    int smemSize =
      (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);
    //REDUCTION
    //sivia_opt<<<blocks, threads>>>(d_boxes, d_labels, d_Ybox, numBoxes, dims, funcID);        the reduce kernel    
    //reduce(blocks*threads, threads, blocks, 0, d_sums, o_sums);
    reduce2<<<blocks, threads, smemSize>>>(d_sums, o_sums, blocks*threads);
    CHECKED_CALL(cudaGetLastError());             
    CHECKED_CALL(cudaDeviceSynchronize());  
    //std::cout << "reduce passed " << ++reducecount <<std::endl; 
    auto reductionStop = std::chrono::high_resolution_clock::now();
    
    //return the sum
    CHECKED_CALL(cudaMemcpy(sums, o_sums, blocks*sizeof(*o_sums), cudaMemcpyDeviceToHost));
    //CHECKED_CALL(cudaMemcpy(boxes, d_boxes, numBoxes*dims*sizeof(*d_boxes), cudaMemcpyDeviceToHost));


    auto memTransferStop = std::chrono::high_resolution_clock::now();
    for(int i=0;i<blocks;i++){
    totalSum+=sums[i];
    }
    reductionDuration += std::chrono::duration_cast<std::chrono::milliseconds>(reductionStop - siviaStop);
    initMemTransferDuration += std::chrono::duration_cast<std::chrono::milliseconds>(initMemTransferStop - initMemTransferStart);
    bisectionDuration += std::chrono::duration_cast<std::chrono::milliseconds>(bisectionStop - initMemTransferStop);
    siviaDuration += std::chrono::duration_cast<std::chrono::milliseconds>(siviaStop - bisectionStop);
    memDuration += std::chrono::duration_cast<std::chrono::milliseconds>(memTransferStop - siviaStop);
}

void gpuSIVIA(interval_gpu<VARTYPE> * boxes, interval_gpu<VARTYPE> * d_boxes, interval_gpu<VARTYPE> * d_Ybox,
            char * labels, char * d_labels, const long long numBoxes, const int blocks, const int threads, 
            std::chrono::milliseconds &initMemTransferDuration, std::chrono::milliseconds &bisectionDuration, 
            std::chrono::milliseconds &siviaDuration, std::chrono::milliseconds &memDuration){
    
    auto initMemTransferStart = std::chrono::high_resolution_clock::now();
    
    //transfer of boxes
    CHECKED_CALL(cudaMemcpy(d_boxes, boxes, dims*sizeof(*boxes), cudaMemcpyHostToDevice));
    CHECKED_CALL(cudaMemcpy(d_labels, labels, sizeof(*d_labels), cudaMemcpyHostToDevice));
    
    auto initMemTransferStop = std::chrono::high_resolution_clock::now();
    
    //Bisection
    for (long long i=1; i<numBoxes; i*=2){
        //blocks and threads will be drawn from context
        partialBisect<<<blocks, threads>>>(d_boxes, i, eps, dims); 
        CHECKED_CALL(cudaGetLastError());             
        CHECKED_CALL(cudaDeviceSynchronize());  
    }

    auto bisectionStop = std::chrono::high_resolution_clock::now();
     
    //SIVIA
    sivia_opt<<<blocks, threads>>>(d_boxes, d_labels, d_Ybox, numBoxes, dims, funcID);              
    CHECKED_CALL(cudaGetLastError());             
    CHECKED_CALL(cudaDeviceSynchronize());   

    auto siviaStop = std::chrono::high_resolution_clock::now();
    
    //return the results
    CHECKED_CALL(cudaMemcpy(labels, d_labels, numBoxes*sizeof(*d_labels), cudaMemcpyDeviceToHost));
    CHECKED_CALL(cudaMemcpy(boxes, d_boxes, numBoxes*dims*sizeof(*d_boxes), cudaMemcpyDeviceToHost));
    
    auto memTransferStop = std::chrono::high_resolution_clock::now();

    initMemTransferDuration += std::chrono::duration_cast<std::chrono::milliseconds>(initMemTransferStop - initMemTransferStart);
    bisectionDuration += std::chrono::duration_cast<std::chrono::milliseconds>(bisectionStop - initMemTransferStop);
    siviaDuration += std::chrono::duration_cast<std::chrono::milliseconds>(siviaStop - bisectionStop);
    memDuration += std::chrono::duration_cast<std::chrono::milliseconds>(memTransferStop - siviaStop);


}


void parametersError(){
    std::cout << "USAGE: ./main -f <function> -ep <epsilon> -id <GPU_ID>\nExamples:" << std::endl;
    //cout << "-f function: griewank | torus | xor. Default : Torus" << endl;
    std::cout << "-eps epsilon: The minimum size of each box. Default: 1e-2" << std::endl;
    std::cout << "-minID, -maxID: Set range of GPUs for cluster mode" << std::endl;
    std::cout << "-minID GPU_ID: As reported by nvidia-smi. Default: 0" << std::endl;
    std::cout << "-maxID GPU_ID: As reported by nvidia-smi. Default: 0" << std::endl;
    std::cout << "If minID == max ID: Single GPU mode" << std::endl;
}


void argParser(int argc, char *argv[]){
    if (ExistArg("-help",argc,argv) || ExistArg("-h",argc,argv)){
      parametersError();
      exit(-1);
    }

    if (ExistArg("-ep",argc,argv)){
        eps=std::stod(GetArg("-ep",argc,argv));
    }else {
        eps = 1e-2;
    }
    if (ExistArg("-minID",argc,argv)){
        devID_min=std::stoi(GetArg("-minID",argc,argv));
        if (ExistArg("-maxID",argc,argv)){
            devID_max = std::stoi(GetArg("-maxID",argc,argv));
        }else{
            devID_max=devID_min;
        }
    }else {
        devID_min = 0;
        devID_max = 0;
    }
 
    /*if (ExistArg("-t",argc,argv)){
        targetDim = stod(GetArg("-t",argc,argv));
    }else {
        targetDim = 0;
    }*/

    if (ExistArg("-f",argc,argv)){
        //griewank
        if (!strcmp(GetArg("-f",argc,argv), "griewank")){
            funcID=1;
            dims=2;
            startingBox = new interval_gpu<float>[2];
            for (int i=0;i<dims;i++){
                startingBox[i] = interval_gpu<float>(-10,10);
                std::cout << "["  << startingBox[i].lower() << "," << startingBox[i].upper() << "] ";
            }
            std::cout << "\n" << std::endl;
            CHECKED_CALL(cudaMallocHost((void**)&Ybox, sizeof(interval_gpu<VARTYPE>)));
            for (int i=0;i<1;i++){
                Ybox[i] = interval_gpu<VARTYPE>(1.5,3);
            }
        }
        //torus
        else if  (!strcmp(GetArg("-f",argc,argv), "torus")){
            //torus init
            std::cout << "\nTorus Function\n" << std::endl;
            funcID=0;
            dims=2;
            startingBox = new interval_gpu<float>[2];
            std::cout << "Initial Box : "; 
            for (int i=0;i<dims;i++){
                startingBox[i] = interval_gpu<float>(-1.5,1.5);
                std::cout << "["  << startingBox[i].lower() << "," << startingBox[i].upper() << "] ";
            }
            std::cout << "\n" << std::endl;
            CHECKED_CALL(cudaMallocHost((void**)&Ybox, sizeof(interval_gpu<VARTYPE>)));
            //Ybox = new interval_gpu<VARTYPE>[1];
            
            for (int i=0;i<1;i++){
                Ybox[i] = interval_gpu<VARTYPE>(1,2);
            }
        }
        else if  (!strcmp(GetArg("-f",argc,argv), "vc2")){
                        std::cout << "\nNN Function\n" << std::endl;
            funcID=2;
            dims=6;
            vc2=true;
            startingBox = new interval_gpu<float>[6];
            std::cout << "Initial Box : "; 
            for (int i=0;i<dims;i++){
                startingBox[i] = interval_gpu<float>(-1,1);
                std::cout << "["  << startingBox[i].lower() << "," << startingBox[i].upper() << "] ";
            }
            std::cout << "\n" << std::endl;
            CHECKED_CALL(cudaMallocHost((void**)&Ybox, sizeof(interval_gpu<VARTYPE>)));
            //Ybox = new interval_gpu<VARTYPE>[1];
            
            for (int i=0;i<1;i++){
                Ybox[i] = interval_gpu<VARTYPE>(0.8,1);
            }
        }
    } else {
        //torus init
        std::cout << "\nTorus Function\n" << std::endl;
        dims=2;
        funcID=0;
        startingBox = new interval_gpu<float>[2];
        std::cout << "Initial Box : "; 
        for (int i=0;i<dims;i++){
            startingBox[i] = interval_gpu<float>(-1.5,1.5);
            std::cout << "["  << startingBox[i].lower() << "," << startingBox[i].upper() << "]" << std::endl;
        } 
        std::cout << "\n" << std::endl;
        //Ybox = new interval_gpu<VARTYPE>[1];
        CHECKED_CALL(cudaMallocHost((void**)&Ybox, sizeof(interval_gpu<VARTYPE>)));
        for (int i=0;i<dims-1;i++){
            Ybox[i] = interval_gpu<VARTYPE>(1,2);
        }
    }
}

