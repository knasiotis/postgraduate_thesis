## Accelerating SIVIA (Set Inversion via Interval Analysis). An Interval Set Membership Technique to explain Neural Classifier Decisions.

The purpose of this thesis is to propose an alternative, breadth-first, Branch and Bound parallel algorithm to estimate the validation space of Neural Networks, which is a problem that requires exhaustive search of that space.


To reproduce the code, one should train and import a Neural Network of their choice. Unfortunately, due to the fact that popular frameworks for neural computations do not support interval arithmetic, this has to be done manually.
The above was the hardest step, the only thing left is to copy-paste and replace the files from Nvidia's 
[https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/interval](CUDA Interval Samples) repository.
 
