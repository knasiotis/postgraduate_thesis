//author:knasiotis

//libs
#include <cmath>
#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include "ibex.h"
#include "vibes.cpp"
//should it draw the boxes? uncomment if yes
//#define VIBES

/* My project files section*/
#include "functions.h"
#include "argshand.h"
/* end of section */

using namespace ibex;
using namespace std;



//argument parsing
void parametersError();
void argParser(int argc, char *argv[]);

//bisection strategy declaration zone
//queue<IntervalVector> bisectionST(IntervalVector initBox, double eps);
IntervalVector* bisectionARRAY(IntervalVector initBox, double eps);
vector<IntervalVector> bisectionMT(IntervalVector initBox, double eps);
vector<IntervalVector> bisectionCUDA(IntervalVector initBox, double eps);

//sivia declaration zone
double sivia(stack<IntervalVector> s, IntervalVector Ybox, double eps, int targetDim, auto func);
//void siviaST(queue<IntervalVector> s, IntervalVector Ybox, double eps, int targetDim, auto func); //pre-bisected input
void siviaARRAY(IntervalVector* s, IntervalVector Ybox, int* results, double eps, int targetDim, auto func); //pre-bisected input
void siviaMT(IntervalVector* s, IntervalVector Ybox, double eps, int targetDim, auto func); //pre-bisected input

//problem declaration zone
void torusInit(IntervalVector &initBox, IntervalVector &Ybox, double &eps, int &targetDim, auto &func);
void griewankInit(IntervalVector &initBox, IntervalVector &Ybox, double &eps, int &targetDim, auto &func);
void xorInit(IntervalVector &initBox, IntervalVector &Ybox, double &eps, int &targetDim, auto &func);
void vc2Init(IntervalVector &initBox, IntervalVector &Ybox, double &eps, int &targetDim, auto &func);

//utility
int calculateBoxes(IntervalVector initBox);

//Initialized variables    
IntervalVector initBox; //starting box
IntervalVector Ybox; //targetbox
double eps; //cutoff
int targetDim; //targetbox dimension

Interval (*func)(IntervalVector x, int targetDim);



//MAIN
int main(int argc, char *argv[]){
    argParser(argc, argv);
    //cout << "Initial box size: " << sizeof(initBox[0])*initBox.size() + sizeof(initBox) << " bytes"<< endl;
    //initial box grid
    double vnet=0;
    double vinput=1;
    stack<IntervalVector> s;
    auto start = chrono::high_resolution_clock::now();
    s.push(initBox);
    
    //ibex::Interval nnfunc
    //(double ** vW1, double ** vW2, double * vb1, double * vb2, 
    //ibex::IntervalVector vX, int vn, int vh, int vm, int iclass)
    //sivia section

    vnet+=sivia(s, Ybox, eps, 0, func);
    //s.push(initBox);
    //vnet+=sivia(s, Ybox, eps, 1, func);
    auto stop = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::milliseconds>(stop - start);
    std::cout << "Vanilla single thread sivia time: " << duration.count() << "ms" << std::endl;
    std::cout << vnet << std::endl;
    for(int i=0;i<initBox.size();i++){
        vinput*=initBox[i].diam();
    }
    std::cout<< "Gnet : " << (vnet/vinput)-(37/310.0) << endl;
    //batch bisection - grid preparation
    /*
    int * results = new int[calculateBoxes(initBox)];
    start = chrono::high_resolution_clock::now();
    IntervalVector * arr = bisectionARRAY(initBox, eps);
    
    stop = chrono::high_resolution_clock::now();

    siviaARRAY(arr, Ybox, results, eps, targetDim, func);
    auto stopSiviaARRAY = chrono::high_resolution_clock::now();
    auto durationBisectARRAY = chrono::duration_cast<chrono::milliseconds>(stop - start);
    auto durationSiviaARRAY = chrono::duration_cast<chrono::milliseconds>(stopSiviaARRAY - stop);
    auto totalduration = chrono::duration_cast<chrono::milliseconds>(stopSiviaARRAY - start);

    delete [] arr;

    cout << "Single thread bisection time: " << durationBisectARRAY.count() << "ms" << endl;
    cout << "Single thread sivia time: " << durationSiviaARRAY.count() << "ms" << endl;
    cout << "Single thread sivia boxes evaluated: " << calculateBoxes(initBox) << endl;
    cout << "Total time: " << totalduration.count() << "ms" << endl;*/
    return 0;
}

double sivia(stack<IntervalVector> s, IntervalVector Ybox, double eps, int targetDim, auto func){
    int boxcount = 0;
    double sum = 0;
    double prod = 1;
    std::vector<IntervalVector> in;
    #ifdef VIBES
    vibes::beginDrawing();
    vibes::newFigure("sivia");
    #endif

    while (!s.empty()) {
        //Extract from pending
		IntervalVector box=s.top();
        //std::cout << s.top() << std::endl;
		s.pop();
        boxcount++;
        //Evaluation of a box via Inclusion function
        ibex::Interval fxy = func(box, targetDim);
        //cout << fxy << endl;
        //Approval
		if (fxy.is_subset(Ybox[0])) {
            //in.push_back(box);
            prod = 1;
            for(int j=0;j<box.size();j++){
                prod*=box[j].diam();
            }
            sum+=prod;
            #ifdef VIBES
            vibes::drawBox(box[0].lb(), box[0].ub(), box[1].lb(), box[1].ub(), "k[r]");
            #endif
            continue;
		}
        //Rejection
		else if (!fxy.intersects(Ybox[0])) {
			#ifdef VIBES
            vibes::drawBox(box[0].lb(), box[0].ub(), box[1].lb(), box[1].ub(), "k[b]");
            #endif
            continue;
		} 
        //Width threshold satisfied
		else if (box.max_diam()<=eps){
			//  XIBt.AddNode(pNode);
            #ifdef VIBES
            vibes::drawBox(box[0].lb(), box[0].ub(), box[1].lb(), box[1].ub(), "k[y]");
            #endif
            continue;
		} 
 
        //Otherwise we bisect. Dimension with the greatest width is selected
        int i=box.extr_diam_index(false);
        pair<IntervalVector,IntervalVector> p=box.bisect(i);
        s.push(p.first);
        s.push(p.second);
	}
    #ifdef VIBES
    vibes::endDrawing();
    #endif
    cout << "Boxes created: " << boxcount << endl;
    cout << "partial Vnet[" << targetDim << "] : " << sum << endl;
    return sum;
    //cout << "Initbox size: " << (sizeof(initBox[0])*initBox.size()+sizeof(initBox))*boxcount+sizeof(s) << " bytes"<< endl;
}

/*
void siviaST(queue<IntervalVector> s, IntervalVector Ybox, double eps, int targetDim, auto func){
    #ifdef VIBES
    vibes::beginDrawing();
    vibes::newFigure("sivia");
    #endif
    while (!s.empty()) {

        //Extract from pending
		IntervalVector box=s.front();
		s.pop();
        //Evaluation of a box via Inclusion function
        ibex::Interval fxy = func(box);
        //cout << fxy << endl;
        //Approval
		if (fxy.is_subset(Ybox[targetDim])) {
			#ifdef VIBES
            vibes::drawBox(box[0].lb(), box[0].ub(), box[1].lb(), box[1].ub(), "k[r]");
            #endif
            continue;
		}
        //Rejection
		else if (!fxy.intersects(Ybox[targetDim])) {
			#ifdef VIBES
            vibes::drawBox(box[0].lb(), box[0].ub(), box[1].lb(), box[1].ub(), "k[b]");
            #endif
            continue;
		}
        //Width threshold satisfied
		else {
			#ifdef VIBES
            vibes::drawBox(box[0].lb(), box[0].ub(), box[1].lb(), box[1].ub(), "k[y]");
            #endif
            continue;
		}
	}
    #ifdef VIBES
    vibes::endDrawing();
    #endif
    
}*/

void siviaARRAY(IntervalVector* s, IntervalVector Ybox, int * results, double eps, int targetDim, auto func){
    #ifdef VIBES
    vibes::beginDrawing();
    vibes::newFigure("sivia");
    #endif
    int i = 0;
    int size = calculateBoxes(initBox);
    while (i<size) {

        //Extract from pending
		IntervalVector box=s[i];
        i++;
        //Evaluation of a box via Inclusion function
        ibex::Interval fxy = func(box);
        //cout << fxy << endl;
        //Approval
		if (fxy.is_subset(Ybox[targetDim])) {
			#ifdef VIBES
            vibes::drawBox(box[0].lb(), box[0].ub(), box[1].lb(), box[1].ub(), "k[r]");
            #endif
            results[i]=0;
            continue;
		}
        //Rejection
		else if (!fxy.intersects(Ybox[targetDim])) {
			#ifdef VIBES
            vibes::drawBox(box[0].lb(), box[0].ub(), box[1].lb(), box[1].ub(), "k[b]");
            #endif
            results[i]=2;
            continue;
		}
        //Width threshold satisfied
		else {
			#ifdef VIBES
            vibes::drawBox(box[0].lb(), box[0].ub(), box[1].lb(), box[1].ub(), "k[y]");
            #endif
            results[i]=1;
            continue;
		}
	}
    #ifdef VIBES
    vibes::endDrawing();
    #endif
    
}

void torusInit(IntervalVector &initBox, IntervalVector &Ybox, double &eps, int &targetDim, auto &func){
    //initial box definition
	initBox = IntervalVector::empty(2);

	initBox[0]=Interval(-1.5,1.5);
	initBox[1]=Interval(-1.5,1.5);

    Ybox = IntervalVector::empty(1);
    Ybox[targetDim]=Interval(1, 2);

    //function definition
    func = TorusFun;
}

void griewankInit(IntervalVector &initBox, IntervalVector &Ybox, double &eps, int &targetDim, auto &func){
    //initial box definition
	initBox = IntervalVector::empty(2);

	initBox[0]=Interval(-10,10);
	initBox[1]=Interval(-10,10);

    Ybox = IntervalVector::empty(1);
    Ybox[targetDim]=Interval(1.5, 3);

    //function definition
    func = GriewankFun;
}

void xorInit(IntervalVector &initBox, IntervalVector &Ybox, double &eps, int &targetDim, auto &func){
    //initial box definition
	initBox = IntervalVector::empty(2);

	initBox[0]=Interval(0,1);
	initBox[1]=Interval(0,1);

    Ybox = IntervalVector::empty(1);
    Ybox[targetDim]=Interval(0.8, 1);

    //function definition
    func = mlpXOR;
}

void vc2Init(IntervalVector &initBox, IntervalVector &Ybox, double &eps, int &targetDim, auto &func){
    initBox = IntervalVector::empty(6);
    for(int i=0;i<6;i++){
        initBox[i]=Interval(-1,1);
    }

    Ybox = IntervalVector::empty(1);
    Ybox[targetDim]=Interval(0.8, 1);

    func = nnfunc;
}

int calculateBoxes(IntervalVector initBox){
    int boxes = 1;
    //the amount of boxes may vary, it would be a waste to perform multiple allocations
    for (int i=0; i < initBox.size(); i++){
        boxes *= pow(2,ceil(log2(initBox[i].diam()/eps)));
    }
    
    return boxes;
}

/*
queue<IntervalVector> bisectionST(IntervalVector initBox, double eps){
    queue<IntervalVector> s;
    s.push(initBox);
    while(s.front().max_diam()>eps){
        IntervalVector temp = s.front();
        s.pop();
        pair<IntervalVector,IntervalVector> p=temp.bisect(temp.extr_diam_index(false));
        s.push(p.first);
        s.push(p.second);
    }

    cout << "Boxes created: " << s.size() << endl;
    cout << "Initbox size: " << (sizeof(initBox[0])*initBox.size()+sizeof(initBox))*s.size()+sizeof(s) << " bytes"<< endl;
    return s;
}*/

//works with static arrays compatible with cuda
IntervalVector* bisectionARRAY(IntervalVector initBox, double eps){
    int numBoxes = calculateBoxes(initBox);
    IntervalVector* boxes = new IntervalVector[numBoxes];
    int dims = initBox.size();
    int currentBox = 0;
    boxes[currentBox]=initBox;
    int offset = 0; //offset keeps track of last inserted element
    while(currentBox<numBoxes){
        if (boxes[currentBox].max_diam()<=eps){
            currentBox++;
            continue;
        }
        for(int i=0; i<dims; i++){
            if(boxes[currentBox][i].diam()>eps){
                pair<IntervalVector,IntervalVector> p=boxes[currentBox].bisect(i);
                offset++;
                boxes[currentBox]=p.first;
                boxes[offset]=p.second;
                break;
            }
        }

    }
    return boxes;
}

void parametersError(){
    cout << "USAGE: ./main -f <function> -ep <epsilon> -t <target>\nExamples:" << endl;
    cout << "-f function: griewank | torus | xor. Default : Torus" << endl;
    cout << "-eps epsilon: The minimum size of each box. Default: 1e-2" << endl;
    cout << "-t target: Default: 0" << endl;
}

void argParser(int argc, char *argv[]){
    if (ExistArg("-help",argc,argv) || ExistArg("-h",argc,argv)){
      parametersError();
      exit(-1);
    }

    if (ExistArg("-ep",argc,argv)){
        eps=stod(GetArg("-ep",argc,argv));
    }else {
        eps = 1e-2;
    }

    if (ExistArg("-t",argc,argv)){
        targetDim = stod(GetArg("-t",argc,argv));
    }else {
        targetDim = 0;
    }

    if (ExistArg("-f",argc,argv)){
        if (!strcmp(GetArg("-f",argc,argv), "griewank")){
            griewankInit(initBox, Ybox, eps, targetDim, func);
        }
        else if  (!strcmp(GetArg("-f",argc,argv), "torus")){
            torusInit(initBox, Ybox, eps, targetDim, func);
        }
        else if  (!strcmp(GetArg("-f",argc,argv), "xor")){
            xorInit(initBox, Ybox, eps, targetDim, func);
        }
        //for vertebral column
        else if  (!strcmp(GetArg("-f",argc,argv), "vc2")){
            vc2Init(initBox, Ybox, eps, targetDim, func);
        }
    } else {
        torusInit(initBox, Ybox, eps, targetDim, func);
    }
}



/*
ibex::Interval normalize(ibex::Interval x, double min, double max){
    double lb = ((x.lb()-min)/(max-min)); 
    double ub = ((x.ub()-min)/(max-min));

    ibex::Interval y(lb,ub);
    return y;
}*/