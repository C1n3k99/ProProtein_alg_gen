#include <chrono>
#include <ctime>
#include <iostream>
#include <fstream>
#include <vector>
#include <eigen3/Eigen/Geometry>
#include <bits/stdc++.h>
#include <chrono>
#include <mutex>
#include <thread>
#include <omp.h>
#include <stdlib.h>
using namespace std;

//additional parameters
double sphereRadius = 8;
int SPHERES;
int ATOMS;
int FRAMES;
int FRAMEONE;
int FRAMETWO;
int omp_thread_id;
#pragma omp threadprivate(FRAMEONE, FRAMETWO, omp_thread_id)
//Atoms[<frame>][<atom>][<coordinate>]
vector<vector<vector<double>>> A;
//Maps sphere to CA; CAAtomNumber[<sphere>]
vector<int> sphereCA;
//Numer of atoms in [<sphere>]
//vector<int> sphereSize;
//List of atoms in [<sphere>]
vector<vector<int>>* sphereAtoms;
// Number of individuals in each generation
//--------------------------------------------------

//assistant function to set frame number
int whichFrame(int i)
{
    if (i == 0) return FRAMEONE;
    return FRAMETWO;
}

//reading data from input pdb file.
void readFile(string filename)
{
    string line;
    ifstream myfile(filename);
    if (myfile.is_open())
    {
        int frame = 0;
        int atom;
        SPHERES = 0;
        FRAMES = 0;
        ATOMS = 0;
        A = {};
        sphereCA = {};
        //sphereSize = {};
        while (getline(myfile, line))
        {
            if (line[0] == 'M')
            {
                frame = stoi(line.substr(9, 5));
                frame--;
                A.push_back({});
                FRAMES++;
            }
            else if (line[0] == 'A')
            {
                atom = stoi(line.substr(6, 5));
                atom--;
                A[frame].push_back({});
                A[frame][atom].push_back(stod(line.substr(30, 8)));
                A[frame][atom].push_back(stod(line.substr(38, 8)));
                A[frame][atom].push_back(stod(line.substr(46, 8)));
                if (frame == 0)
                {
                    ATOMS++;
                    if (line[14] == 'A' and line[13] == 'C')
                    {
                        sphereCA.push_back(atom);
                        //sphereSize.push_back(0);
                        SPHERES++;
                    }
                }
            }
        }
        myfile.close();
    }
    else
    {
        cout << "Nie odnaleziono pliku!" << endl;
    }
}

//calculating distance between 2 atoms, used when allocating atoms to spheres.
double atomsDistanceCalc(int atom1, int atom2)
{
    double result = (A[FRAMEONE][atom1][0] - A[FRAMEONE][atom2][0]) * (A[FRAMEONE][atom1][0] - A[FRAMEONE][atom2][0]) +
        (A[FRAMEONE][atom1][1] - A[FRAMEONE][atom2][1]) * (A[FRAMEONE][atom1][1] - A[FRAMEONE][atom2][1]) +
        (A[FRAMEONE][atom1][2] - A[FRAMEONE][atom2][2]) * (A[FRAMEONE][atom1][2] - A[FRAMEONE][atom2][2]);
    return sqrt(result);
}

//allocating atoms into spheres, based on sphereRadius
void atomsAllocation()
{
    //sphereAtoms[omp_thread_id] = {};
    sphereAtoms[omp_thread_id].assign(SPHERES, {});
    // for (int i = 0; i < SPHERES; i++)
    // {
    //     sphereAtoms[omp_thread_id].push_back({});
    // }
    for (int i = 0; i < ATOMS; i++)
    {
        for (int j = 0; j < SPHERES; j++)
        {
            if (atomsDistanceCalc(i, sphereCA[j]) <= sphereRadius)
            {
                sphereAtoms[omp_thread_id][j].push_back(i);
                //sphereSize[j]++;
            }
        }
    }
}

// Find3DAffineTransform is from oleg-alexandrov repository on github, available here https://github.com/oleg-alexandrov/projects/blob/master/eigen/Kabsch.cpp [as of 27.01.2022]
// Given two sets of 3D points, find the rotation + translation + scale
// which best maps the first set to the second.
// Source: http://en.wikipedia.org/wiki/Kabsch_algorithm

// The input 3D points are stored as columns.
Eigen::Affine3d Find3DAffineTransform(Eigen::Matrix3Xd in, Eigen::Matrix3Xd out) {

    // Default output
    Eigen::Affine3d A;
    A.linear() = Eigen::Matrix3d::Identity(3, 3);
    A.translation() = Eigen::Vector3d::Zero();

    if (in.cols() != out.cols())
        throw "Find3DAffineTransform(): input data mis-match";

    // First find the scale, by finding the ratio of sums of some distances,
    // then bring the datasets to the same scale.
    double dist_in = 0, dist_out = 0;
    for (int col = 0; col < in.cols() - 1; col++) {
        dist_in += (in.col(col + 1) - in.col(col)).norm();
        dist_out += (out.col(col + 1) - out.col(col)).norm();
    }
    if (dist_in <= 0 || dist_out <= 0)
        return A;
    double scale = dist_out / dist_in;
    out /= scale;

    // Find the centroids then shift to the origin
    Eigen::Vector3d in_ctr = Eigen::Vector3d::Zero();
    Eigen::Vector3d out_ctr = Eigen::Vector3d::Zero();
    for (int col = 0; col < in.cols(); col++) {
        in_ctr += in.col(col);
        out_ctr += out.col(col);
    }
    in_ctr /= in.cols();
    out_ctr /= out.cols();
    for (int col = 0; col < in.cols(); col++) {
        in.col(col) -= in_ctr;
        out.col(col) -= out_ctr;
    }

    // SVD
    Eigen::MatrixXd Cov = in * out.transpose();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(Cov, Eigen::ComputeThinU | Eigen::ComputeThinV);

    // Find the rotation
    double d = (svd.matrixV() * svd.matrixU().transpose()).determinant();
    if (d > 0)
        d = 1.0;
    else
        d = -1.0;
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity(3, 3);
    I(2, 2) = d;
    Eigen::Matrix3d R = svd.matrixV() * I * svd.matrixU().transpose();

    // The final transform
    A.linear() = scale * R;
    A.translation() = scale * (out_ctr - R * in_ctr);

    return A;
}

//superpose changes vector of atoms frame 2 to map atoms from frame 1 in the way to minimise RMSD between both frames
void superpose(const vector<vector<double>>& frame1, vector<vector<double>>& frame2)
{
    int atomsInSphere = frame1.size();
    Eigen::Matrix3Xd S1(3, atomsInSphere);
    Eigen::Matrix3Xd S2(3, atomsInSphere);
    for (int j = 0; j < atomsInSphere; j++)
    {
        for (int k = 0; k < 3; k++)
        {
            S1(k, j) = frame1[j][k];
            S2(k, j) = frame2[j][k];
        }
    }
    Eigen::Affine3d RT = Find3DAffineTransform(S2, S1);
    S2 = RT.linear() * S2;
    for (int j = 0; j < atomsInSphere; j++)
    {
        S2.block<3, 1>(0, j) += RT.translation();
    }
    for (int j = 0; j < atomsInSphere; j++)
    {
        for (int k = 0; k < 3; k++)
        {
            frame2[j][k] = S2(k, j);
        }
    }
}

//calculating RMSD on spheres, on choosen frames
double calculateRMSDSuperpose()
{
    double result = 0;
    //double tempResult;
    vector<vector<vector<double>>> sphereMatrix;
    //vector<vector<double>> tempMatrix;
    //double tempRMSD;
    for (int s = 0; s < SPHERES; s++)
    {
        int atomsInSphere = sphereAtoms[omp_thread_id][s].size();
        //sphereMatrix = {};
        sphereMatrix.assign(2, {});
        sphereMatrix[0].assign(atomsInSphere, {});
        sphereMatrix[1].assign(atomsInSphere, {});
        //for (int i = 0; i < 2; i++)
        //{
            //sphereMatrix.push_back({});
        for (int j = 0; j < atomsInSphere; j++)
        {
            //sphereMatrix[i].push_back(A[whichFrame(i)][sphereAtoms[omp_thread_id][s][j]]);
            sphereMatrix[0][j] = A[FRAMEONE][sphereAtoms[omp_thread_id][s][j]];
            sphereMatrix[1][j] = A[FRAMETWO][sphereAtoms[omp_thread_id][s][j]];
        }
            //if (i > 0)
            // {
            //     tempResult = 0;
            //     tempMatrix = sphereMatrix[i];
            //     superpose(sphereMatrix[i - 1], tempMatrix);
            //     for (int j = 0; j < atomsInSphere; j++)
            //     {
            //         for (int k = 0; k < 3; k++)
            //         {
            //             // tempRMSD = pow(tempMatrix[j][k] - sphereMatrix[i - 1][j][k], 2);
            //             // tempResult += tempRMSD;
            //             tempRMSD = tempMatrix[j][k] - sphereMatrix[i - 1][j][k];
            //             tempResult += tempRMSD * tempRMSD;
            //         }
            //     }
            //     //tempResult /= ((double)atomsInSphere * (double)3);
            //     tempResult /= atomsInSphere * 3.0;
            //     tempResult = sqrt(tempResult);
            //     result += tempResult;
            // }
        //}
        double tempResult = 0;
        superpose(sphereMatrix[0], sphereMatrix[1]);
        for (int j = 0; j < atomsInSphere; j++) {
            for (int k = 0; k < 3; k++ ) {
                double tempRMSD = sphereMatrix[1][j][k] - sphereMatrix[0][j][k];
                tempResult += tempRMSD * tempRMSD;
            }
        }
        tempResult /= atomsInSphere * 3.0;
        tempResult = sqrt(tempResult);
        result += tempResult;
    }
    return result;
}

//main function to calculate RMSD between 2 frames
double calculateRMSD(int firstFrame, int secondFrame)
{
    FRAMEONE = firstFrame;
    FRAMETWO = secondFrame;
    atomsAllocation();
    return calculateRMSDSuperpose();
}


// Function to generate random numbers in given range
int random_num(int start, int end)
{
    int range = (end-start); //(end-start)+1;
    int random_int = start+(rand()%range);
    return random_int;
}

double cal_fitness(vector<int> T)
{
    return calculateRMSD(T[0], T[1]);
};

vector<int> mutation(vector<int> par1, vector<vector<double>> pop) {
    double p2 = (double) rand() / (RAND_MAX);
    if (p2<0.5){
        int r = random_num(0, FRAMES);
        while (r == par1[0] || r == par1[1]) {
            r = random_num(0, FRAMES);
        }
        vector<int> T = {par1[0], r};
        while (any_of(pop.begin(), pop.end(), [&](vector<double>& p) { 
            if (((int)p[0]==T[0] && (int)p[1]==T[1]) || ((int)p[1]==T[0] && (int)p[0]==T[1])) {
                return true;
            }
            return false;
        })) 
        {
            r = random_num(0, FRAMES);
            while (r == par1[0] || r == par1[1]) {
                r = random_num(0, FRAMES);
            }
            T[1] = r;
        }
        return T;
    }
    int r = random_num(0, FRAMES);
    while (r == par1[0] || r == par1[1]) {
        r = random_num(0, FRAMES);
    }
    vector<int> T = {r, par1[1]};
    while (any_of(pop.begin(), pop.end(), [&](vector<double>& p) { 
        if (((int)p[0]==T[0] && (int)p[1]==T[1]) || ((int)p[1]==T[0] && (int)p[0]==T[1])) {
            return true;
        }
        return false;
    })) 
    {
        r = random_num(0, FRAMES);
        while (r == par1[0] || r == par1[1]) {
            r = random_num(0, FRAMES);
        }
        T[0] = r;
    }
    return T;
}

vector<int> crossover(vector<int> par1, vector<int> par2, vector<vector<double>> pop)
{
    double p2 = (double) rand() / (RAND_MAX);
    if (p2 < 0.5){
        if (par1[1] != par2[0]) {
            vector<int> T = {par2[0], par1[1]};
            if (any_of(pop.begin(), pop.end(), [&](vector<double>& p) { 
                if (((int)p[0]==T[0] && (int)p[1]==T[1]) || ((int)p[1]==T[0] && (int)p[0]==T[1])) {
                    return true;
                }
                return false;
            })) 
            {
                return mutation(par1, pop);
            }
            return T;
        }
    }
    vector<int> T = {par1[0], par2[1]};
    if (any_of(pop.begin(), pop.end(), [&](vector<double>& p) { 
        if (((int)p[0]==T[0] && (int)p[1]==T[1]) || ((int)p[1]==T[0] && (int)p[0]==T[1])) {
            return true;
        }
        return false;
    })) 
    {
        return mutation(par1, pop);
    }
    return T;
}

int main(int argc, char* argv[])
{   
    string inputFile = "trajectory_test.pdb";
    double timelimit;
    int threads_num;
    int repetitions;
    int population_size;
    double cross_prob;
    int maxpopwithoutbetterfit;
    double elite;
    if (argc > 1)
    {
        inputFile = string(argv[1]);
        readFile(inputFile);
    }
    else
    {
        readFile(inputFile);
    }
    if (argc > 2)
    {
        timelimit = atof(argv[2]);
    } else {
        timelimit = 600.0;
    }
    if (argc > 3)
    {
        threads_num = atoi(argv[3]);
    } else {
        threads_num = 8;
    }
    if (argc > 4)
    {
        repetitions = atoi(argv[4]);
    } else {
        repetitions = 5;
    }
    if (argc > 5)
    {
        population_size = atoi(argv[5]);
    } else {
        population_size = 1000;
    }
    if (argc > 6)
    {
        cross_prob = atof(argv[6]);
    } else {
        cross_prob = 0.7;
    }
    if (argc > 7)
    {
        maxpopwithoutbetterfit = atoi(argv[7]);
    } else {
        maxpopwithoutbetterfit = 50;
    }
    if (argc > 8)
    {
        elite = atof(argv[8]);
    } else {
        elite = 0.1;
    }
    int repetition = 0;
    while (repetition<repetitions) {
        auto start = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = chrono::high_resolution_clock::now() - start;
        srand((unsigned)(time(0)));

        int generation = 0;

        vector<vector<double>> population;

        double all_time_best = 0.0;
        int best_trj1 = 0;
        int best_trj2 = 0;
        float threshold = 0.05;
        int currentPopWithoutBetterFit = 0;
        mutex barrier_mutex;
        vector<thread> threads;
        omp_set_num_threads(threads_num);
        omp_lock_t writelock;
        omp_init_lock(&writelock);

        #pragma omp parallel 
        {
            omp_thread_id = omp_get_thread_num();
            if (omp_thread_id==0) {
                sphereAtoms = new vector<vector<int>>[omp_get_num_threads()];
                // sphereAtoms = new vector<vector<int>>[1];
            }
            #pragma omp barrier
            while(currentPopWithoutBetterFit < maxpopwithoutbetterfit && elapsed.count() < timelimit) //dodac warunek stopu czasowy 5-10min
            {
                #pragma omp for 
                for(int i = population.size();i<population_size;i++)
                {   
                    int p = random_num(0, FRAMES);
                    int p2 = random_num(0, FRAMES);
                    while (p2==p) {
                        p2 = random_num(0, FRAMES);
                    }
                    while (any_of(population.begin(), population.end(), [&](vector<double>& x) { 
                        if (((int)x[0]==p && (int)x[1]==p2) || ((int)x[1]==p && (int)x[0]==p2)) {
                            return true;
                        }
                        return false;
                    })) 
                    {
                        p = random_num(0, FRAMES);
                        p2 = random_num(0, FRAMES);
                        while (p2==p) {
                            p2 = random_num(0, FRAMES);
                        }
                    }
                    
                    double fit = cal_fitness({p, p2});
                    omp_set_lock(&writelock);
                    population.push_back({(double)p, (double)p2, fit});
                    omp_unset_lock(&writelock);
                }
                #pragma omp barrier
                #pragma omp for 
                for(int i = 0;i<population_size;i++)
                {
                    double p = (double) rand() / (RAND_MAX);
                    if (p < cross_prob) {
                        int r = random_num(0, (int)population.size());
                        vector<int> parent1 = {(int)population[r][0], (int)population[r][1]};
                        int r2 = random_num(0, (int)population.size());
                        while (r2==r) {
                            r2 = random_num(0, (int)population.size());
                        }
                        vector<int> parent2 = {(int)population[r2][0], (int)population[r2][1]};
                        
                        vector<int> offspring = crossover(parent1, parent2, population);
                        
                        double offFit = cal_fitness(offspring);
                        omp_set_lock(&writelock);
                        if (population[r][2] > population[r2][2] && offFit > population[r2][2]) {
                            population[r2]={(double)offspring[0], (double)offspring[1], offFit};
                        } else if (population[r2][2] > population[r][2] && offFit > population[r][2]) {
                            population[r]={(double)offspring[0], (double)offspring[1], offFit};
                        }
                        omp_unset_lock(&writelock);
                    } else {
                        int r = random_num(0, (int)population.size());
                        vector<int> parent1 = {(int)population[r][0], (int)population[r][1]};
                        vector<int> offspring = mutation(parent1, population);
                        
                        double offFit = cal_fitness(offspring);
                        omp_set_lock(&writelock);
                        population[r]={(double)offspring[0], (double)offspring[1], offFit};
                        omp_unset_lock(&writelock);
                    }
                }
                if (omp_thread_id==0) {
                    sort(population.begin(), population.end(),[](const vector<double>& a, const vector<double>& b)
                    { 
                        return a[2] > b[2]; 
                    });
                    population.resize(elite*population_size);
                    if (all_time_best+threshold<=population[0][2]) {
                        currentPopWithoutBetterFit = 0;
                    }
                    else {
                        currentPopWithoutBetterFit++;
                    }
                    if (population[0][2] > all_time_best) {
                        all_time_best = population[0][2];
                        best_trj1 = population[0][0];
                        best_trj2 = population[0][1];
                    }
                    generation++;
                    elapsed = chrono::high_resolution_clock::now() - start;
                    // cout<< "Generation: " << generation << "\t";
                    // cout<< "Time: " << elapsed.count() << "\t";
                    // cout<< "first Trajectory: "<< population[0][0] <<"\t";
                    // cout<< "second Trajectory: "<< population[0][1] <<"\t";
                    // cout<< "Fitness: "<< population[0][2] << "\n";
                }
                #pragma omp barrier
            }
        }
        omp_destroy_lock(&writelock);
        cout<< inputFile << ";";
        cout<< timelimit << ";";
        cout<< threads_num << ";";
        cout<< repetitions << ";";
        cout<< population_size << ";";
        cout<< cross_prob << ";";
        cout<< maxpopwithoutbetterfit << ";";
        cout<< elite << ";";
        cout<< generation << ";";
        cout<< best_trj1 <<";";
        cout<< best_trj2 <<";";
        cout<< all_time_best << ";";
        cout<< elapsed.count()<< ";"<<endl;
        repetition++;
    }
    return 0;
}