## Accelerating SIVIA (Set Inversion via Interval Analysis). An Interval Set Membership Technique to Evaluate the Generalization of Neural Classifiers.

The purpose of this thesis is to propose an alternative, breadth-first, Branch and Bound parallel algorithm to estimate the validation space of Neural Networks, which is a problem that requires exhaustive search of that space.


To reproduce the code, one should train and import a Neural Network of their choice as the MLP used in this thesis was not my creation. Unfortunately, due to the fact that popular frameworks for neural computations do not support interval arithmetic, this has to be done manually.
The above was the hardest step, the only thing left is to copy-paste and replace the files from Nvidia's 
[CUDA Interval Samples](https://github.com/NVIDIA/cuda-samples/tree/master/Samples/2_Concepts_and_Techniques/interval) repository.

Using ```-DCODE 0``` as a compilation argument in the Makefile runs the code with FP32(float) variables, otherwise it uses FP16(half) variables to store the boxes. There is no accuracy loss by using half variables, unless a very very small epsilon value is given as an input.

After the successful compilation, run ```./interval -help``` for guidance about the rest of the arguments.

Lastly, in order to run the sequential algorithm one must install the [Ibex Interval Library](https://github.com/ibex-team/ibex-lib) and (re)place the files from the sequential folder into the Ibex/examples one.


P.S. Don't forget to set the targeted SM architectures in the Makefile.
