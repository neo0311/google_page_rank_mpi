#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

vector<double> matVec(vector<vector<double>> P, vector<double> r, const int n)
{
    vector<double> q(n);
    double sum = 0;

    for (int i = 0; i < n; i++){

        sum = 0;
        for (int j = 0; j < n; j++)
        {
            sum += P[i][j] * r[j];
        }
        q[i] = sum;
    }

    return q;
}

double L1norm(vector<double> v, int n)
{
    double norm = 0;
    for (int i = 0; i < n; i++)
    {
        norm += abs(v[i]);
    }

    return norm;

}

int main()
{
    int n;
    cout << "Pick a dense matrix size: ";
    cin >> n;

    // Create dense matrix
    vector<vector<double>> L(n);
    for (int i = 0 ; i < n ; i++) {
        L[i].resize(n, 0);
    }
    // fill dense matrix based on web
    L[0][1] = 1;
    L[1][0] = 1;

    L[2][1] = 1;
    L[2][4] = 1;
    L[2][5] = 1;

    L[3][0] = 1;

    L[3][4] = 1;
    L[4][0] = 1;
    L[4][1] = 1;

    L[4][5] = 1;
    L[5][2] = 1;
    L[5][4] = 1;

    // Create page-rank vector
    vector<double> r(n);
    for (int i = 0; i < n ; i++)
    {
        r[i] =1/double(n);
    }

    int nLinks[n] = {0};
    double sum;

    // find number of links from each page
    for (int i = 0; i < n; i++)
    {

        for (int j = 0; j < n; j++)
        {

            if (L[j][i] != 0)
            {

                nLinks[i] += 1;
            }
        }
    }


    // for (int i = 0; i < n; i++)
    // {

    //     sum = 0;
    //     for (int j = 0; j < n; j++)
    //     {

    //         sum += (1 / nLinks[j]) * L[i][j] * r[j];
    //     }
    //     r[i] = sum;
    // }


    // Define Q matrix
    //double Q[n][n] = {};
    vector<vector<double>> Q(n);
    for (int i = 0 ; i < n ; i++) {
        Q[i].resize(n, 0);
    }

    for (int i = 0; i < n; i++)
    {

        for (int j = 0; j < n; j++)
        {

            if (L[i][j] == 0)
            {

                continue;
            }
            Q[i][j] = (1 / double(nLinks[j])) * L[i][j];
        }
    } 

    int e[n] = {1};
    int d[n] = {0};

    //filling d

    for (int column = 0; column < n; column++)
    {   
        bool flag = false;
        for (int row = 0; row < n; row++)
        {   
            if (Q[row][column] != 0)
            {

                flag = true;

            }
        }

        if (flag == false)
        {
            d[column] = 1;
        }
    }

    // e_d outer product matrix
    //double e_d[n][n] = {};
    vector<vector<double>> e_d(n);
    for (int i = 0 ; i < n ; i++) {
        e_d[i].resize(n, 0);
    }

    // fill e_d matrix with 1 in 'zero' columns
    for (int column = 0; column < n; column++)
    {   
        if (d[column] == 1){
            for (int row = 0; row < n; row++)
            {
                e_d[row][column] = 1;
            }
        }
        
    }

    // P matrix
    //double P[n][n] = {};
    vector<vector<double>> P(n);
    for (int i = 0 ; i < n ; i++) {
        P[i].resize(n, 0);
    }

    // fill P matrix
    for (int row = 0; row < n; row++)
    {
        for (int column = 0; column < n; column++)
        {
            P[row][column] = Q[row][column] + (1/ double(n))*e_d[row][column];
        }
    }


    // power iteration
    vector<double> r_k_1(n);
    for (int i = 0 ; i <n ; i++)
    {
        r_k_1[i] = r[i];
    }

    vector<double> r_k(n);
    for (int i = 0 ; i <n ; i++)
    {
        r_k[i] = 0;
    }

    // Create q vector
    vector<double> q_k(n);
    for (int i = 0; i < n ; i++)
    {
        q_k[i] =0;
    }

    vector<double> temp(n);
    for (int i = 0; i < n ; i++)
    {
        temp[i] =0;
    }

    int k = 1;

    double q_k_norm = 0;

     
    while (k<30){
        q_k = matVec(P, r_k_1, n);
        q_k_norm = L1norm(q_k, n);
        printf("\n\nr_k_1_loop\n");
        for (int i = 0; i < n; i++){
            printf("%8.2f  ", double(r_k_1[i]));
        }


        for (int i = 0; i < n; i++) 
        {
            r_k[i] = q_k[i]/q_k_norm;
            temp[i] = r_k_1[i];
            r_k_1[i] = r_k[i];
            
                
        }
        printf("\n\nIteration: %d\n", k);
        printf("\n\nq_k\n");
        for (int i = 0; i < n; i++){
            printf("%8.2f  ", double(q_k[i]));
        }
        printf("\n\nq_k_norm\n");
        for (int i = 0; i < n; i++){
            printf("%8.2f  ", double(q_k_norm));
        }
        printf("\n\nr_k_1\n");
        for (int i = 0; i < n; i++){
            printf("%8.2f  ", double(r_k_1[i]));

        }

        printf("\n\nr_k\n");
        for (int i = 0; i < n; i++){
            printf("%8.2f  ", double(r_k[i]));
        }
        k += 1;

    //} while ((fabs(temp[5] - r_k_1[5]) < 0.0000001) && (k<30));
    } 



    cout << "iterations: " << k << endl;
    double r_sum = 0;
    for (int i = 0; i < n; i++)
    {
        r_sum += double(r_k[i]);

    }
    cout << "r_sum: " << r_sum << endl;

    // for (int i = 0; i < n; i++)
    // {
    //     cout << "r_k " << r_k[i] << endl;
    // }


    // for (int i = 0; i < n; i++) {
    //     printf("\n");
    //     for (int j = 0; j < n; j++)
    //         printf("%8.2f  ", L[i][j]);
    // }
    // printf("\n\nQ");

    // for (int i = 0; i < n; i++) {
    //     printf("\n");
    //     for (int j = 0; j < n; j++)
    //         printf("%8.2f  ", Q[i][j]);
    // }
    // printf("\n\nd\n");
    // for (int i = 0; i < n; i++){
    //     printf("%8.2f  ", double(d[i]));
    // }
    // printf("\n\ne_d\n");
    // for (int i = 0; i < n; i++) {
    //     printf("\n");
    //     for (int j = 0; j < n; j++)
    //         printf("%8.2f  ", double(e_d[i][j]));
    // }
    // printf("\n\nP\n");
    // for (int i = 0; i < n; i++) {
    //     printf("\n");
    //     for (int j = 0; j < n; j++)
    //         printf("%8.2f  ", double(P[i][j]));
    // }
    // printf("\n\nq_k\n");
    // for (int i = 0; i < n; i++){
    //     printf("%8.2f  ", double(q_k[i]));
    // }
    // printf("\n\nq_k_norm\n");
    // for (int i = 0; i < n; i++){
    //     printf("%8.2f  ", double(q_k_norm));
    // }
    // printf("\n\nr_k_1\n");
    // for (int i = 0; i < n; i++){
    //     printf("%8.2f  ", double(r_k_1[i]));

    // }

    // printf("\n\nr_k\n");
    // for (int i = 0; i < n; i++){
    //     printf("%8.2f  ", double(r_k[i]));
    // }

}


