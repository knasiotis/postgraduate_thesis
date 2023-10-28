//author:knasiotis
#ifndef FUNCTIONS_H
#define FUNCTIONS_H
#endif

using namespace std;

//nnconfig missing due to not being my intellectual property
class NNConfig {};


NNConfig nn("filename");


//2D Torus interval function
ibex::Interval TorusFun(ibex::IntervalVector x, int targetDim){
    if (x.size()>2)
    {
        cout << "this function requires 2 dimensions\nExiting..." << endl;
        exit(-1);
    }

    return ibex::pow(x[0],2) + ibex::pow(x[1],2);
  }

///////////////////////////////////////////////////////////////////////////////

//2D Griewank interval  function
ibex::Interval GriewankFun(ibex::IntervalVector x, int targetDim){
    if (x.size()>2)
    {
        cout << "this function requires 2 dimensions\nExiting..." << endl;
        exit(-1);
    }

    ibex::Interval sum=ibex::Interval(0,0);
    ibex::Interval prod=ibex::Interval(1,1);
    ibex::Interval result;
    for(int i=0; i<2;i++){
        sum += ibex::pow(x[i],2)/4000;
        prod *= cos(x[i]/sqrt(i+1));
    }

    return sum-prod+1;
  }

////////////////////////////////////////////////////////////////////////////////////

//XOR MLP :1 Hidden layer, 2 nodes, sigmoid activation
ibex::Interval mlpXOR(ibex::IntervalVector x, int targetDim){
  
  auto sigmoid = [](ibex::Interval h){
    return 1/(1+exp(-h));
  };
  //Weights borrowed from Victor Lavrenko - https://youtu.be/kNPGXgzxoHw
  ibex::Interval h1 = sigmoid(20*x[0] + 20*x[1] - 10);
  ibex::Interval h2 = sigmoid(-20*x[1] - 20*x[0] + 30);
  ibex::Interval h3 = sigmoid(20*h1+20*h2-30);
  return h3;
}


//missing due to not being my intellectual property
//its a 6-30-2 network
ibex::Interval nnfunc(ibex::IntervalVector vX, int targetDim) {
	return yout[targetDim];
}
