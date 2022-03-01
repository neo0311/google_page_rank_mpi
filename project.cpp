#include <iostream>
#include <vector>

using namespace std;

vector<float> matVec(vector<vector<float>> P, vector<float> r, const int n)
{
    vector<float> q(n);
    float sum = 0;

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

float L1norm(vector<float> v, int n)
{
    float norm = 0;
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
    vector<vector<float>> L(n);
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
    vector<float> r(n);
    for (int i = 0; i < n ; i++)
    {
        r[i] =1/float(n);
    }

    int nLinks[n] = {0};
    float sum;

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
    //float Q[n][n] = {};
    vector<vector<float>> Q(n);
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
            Q[i][j] = (1 / float(nLinks[j])) * L[i][j];
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
    //float e_d[n][n] = {};
    vector<vector<float>> e_d(n);
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
    //float P[n][n] = {};
    vector<vector<float>> P(n);
    for (int i = 0 ; i < n ; i++) {
        P[i].resize(n, 0);
    }

    // fill P matrix
    for (int row = 0; row < n; row++)
    {
        for (int column = 0; column < n; column++)
        {
            P[row][column] = Q[row][column] + (1/ float(n))*e_d[row][column];
        }
    }


    // power iteration
    vector<float> r_k_1(n);
    for (int i = 0 ; i <n ; i++)
    {
        r_k_1[i] = r[i];
    }

    vector<float> r_k(n);
    for (int i = 0 ; i <n ; i++)
    {
        r_k[i] = 0;
    }

    // Create q vector
    vector<float> q_k(n);
    for (int i = 0; i < n ; i++)
    {
        q_k[i] =0;
    }

    vector<float> temp(n);
    for (int i = 0; i < n ; i++)
    {
        temp[i] =0;
    }

    int k = 1;
    float q_k_norm = 0;

    do 
    {
        q_k = matVec(P, r_k_1, n);
        q_k_norm = L1norm(q_k, n);

        for (int i = 0; i < n; i++) 
        {
            r_k[i] = q_k[i]/q_k_norm;
            temp[i] = r_k_1[i];
            r_k_1[i] = r_k[i];
            k++;
                
        }

    } while (abs(temp[0] - r_k_1[0]) < 0.0001);

    cout << "iterations: " << k << endl;
    float r_sum = 0.0;
    for (int i = 0; i < k; i++)
    {
        r_sum += r_k[i];
    }
    cout << "r_sum: " << r_sum << endl;

    for (int i = 0; i < n; i++)
    {
        cout << "r_k " << r_k[i] << endl;
    }

    // // test matrix-vector multiplication function
    // vector<vector<float>> X(3);
    // for (int i = 0 ; i < 3 ; i++) {
    //     X[i].resize(3, 0);
    // }
    // X[1][1] = 8;
    // X[2][2] = 2.5;
    // X[2][1] = 21.23;
    // X[0][0] = 2.6;

    // vector<float> y(3);
    // y[0] = 1;
    // y[1] = 2;
    // y[2] = 3;

    // vector<float> Z(3);
    // int n_ = 3;

    // Z = matVec(X, y, n_);

    // // test L1 norm
    // vector<float> v(n);
    // for (int i = 0; i < n;  i++)
    // {
    //     v[i] = -3;
    // }
    // v[4] = 0;
    // v[5] = 6;
    // float norm;
    // norm = L1norm(v, n);
    // cout << norm << endl;

    // for (int row = 0; row < n; row++)
    // {   
    //     for (int column = 0; column < n; column++)
    //     {
    //         cout << "P " << row << ' ' << column << ' ' << P[row][column] << endl;
    //     }
    // }
}

