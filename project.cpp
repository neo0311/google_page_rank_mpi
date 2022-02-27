#include <iostream>

using namespace std;

int main()
{
    int n;
    cout << "Pick a dense matrix size: ";
    cin >> n;

    // Create dense matrix
    int L[n][n] = {};
    cout << sizeof(L) / sizeof(L[0]) << endl;

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
    float r[n] = {1};

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
    float Q[n][n] = {};

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

    for (int row = 0; row < n; row++)
    {
        cout << d[row] << endl;
        for (int column = 0; column < n; column++)
        {
            //cout << 'Q' << row << ' ' << column << ' ' << Q[row][column] << endl;
        }
    }
}
