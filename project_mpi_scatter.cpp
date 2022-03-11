/**********************************************************************************************

************************************************************************************************/

#include<stdio.h>
#include<mpi.h>
#include<cmath>
#define NUM_PAGES 6
#define M_C_TAG 1 //tag for messages sent from master to childs
#define C_M_TAG 4 //tag for messages sent from childs to master
#define ITERR 1
void printArray(double mat[][NUM_PAGES], int rows); //print the content of output matrix [C];

int rank; //process rank
int size; //number of processes
int i, j, k; //helper variables
int m = 0;
int n = 0;
//double L[NUM_PAGES][NUM_PAGES] = {1}; 
double L[NUM_PAGES][NUM_PAGES] = {0}; //temp
double Q[NUM_PAGES][NUM_PAGES] = {0};
double e_d[NUM_PAGES][NUM_PAGES] = {0};
double P[NUM_PAGES][NUM_PAGES] = {0};
//double r_k_1[NUM_PAGES] = {1 / double(NUM_PAGES)};
double r_k_1[NUM_PAGES] = {0};
double r_k[NUM_PAGES] = {0};
double q_k[NUM_PAGES] = {0};
double q_k_norm;
double temp[NUM_PAGES] = {0};

double r[NUM_PAGES] = {0};
double nLinks[NUM_PAGES] = {0};
double sum;
double e[NUM_PAGES] = {0};
double d[NUM_PAGES] = {0};
double start_time; //hold start time
double end_time; // hold end time
int index_start; //first index of the number of rows of [A] allocated to a child
int index_end; //last index of the number of rows of [A] allocated to a child
int chunk_size; //chunk_size of the number of rows of [A] allocated to a child
int const rnd_zeros = ((NUM_PAGES*NUM_PAGES)*0.25)/NUM_PAGES; //number of positions in each row to be replaced by 0.
int index_arry[rnd_zeros]; 
int rnd_filled = 0;
MPI_Status status;   // store status of a MPI_Recv
MPI_Request request; //capture request of a MPI_Isend

int main(int argc, char *argv[])
{
    srand(17);

    MPI_Init(&argc, &argv);               // initialize MPI operations
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get the rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); //get number of processes

    //size = 4;
    int remainder = NUM_PAGES % size;
    int rows_to_send = (NUM_PAGES / size);
    int local_rows = (NUM_PAGES / size);
    if (rank < remainder) {
        local_rows++;   
    }
    int count = NUM_PAGES / size; // count of each chunk

    int counts_M[size] = {}; // to store chunk sizes
    int indices_M[size] = {}; // to store the indices_M to transfer
    int counts_v[size] = {}; // to store chunk sizes
    int indices_v[size] = {}; // to store the indices_M to transfer
    int counts_M_C[size] = {}; // to store chunk sizes
    int indices_M_C[size] = {}; // to store the indices_M to transfer
    int index = 0; // starting index
    for (i = 0; i < size; i++)
    {
        // Calculate the index locations to transfer
        if(i==0){
            indices_M[i] = 0;
            indices_v[i] = 0;

        }else{
            indices_M[i] = indices_M[i-1]+(NUM_PAGES*rows_to_send);
            indices_v[i] = indices_v[i-1]+(rows_to_send);

            //index += NUM_PAGES;

        }
        // Calculate chunk sizes for each chunk
        if(i+1 == size){

            counts_M[i] = (rows_to_send+remainder)*NUM_PAGES;
            counts_v[i] = (rows_to_send+remainder);

        }else{
            counts_M[i] = rows_to_send*NUM_PAGES;
            counts_v[i] = rows_to_send;

        }
    }

    /* master initializes work*/
    if (rank == 0) {

            printf("\ncounts_M");
        for (i = 0; i < size; i++)
        {
            printf("%8.2d  ", counts_M[i]);
        }
        printf("\nindices_M");
        for (i = 0; i < size; i++)
        {
            printf("%8.2d  ", indices_M[i]);
        }


        // fill dense matrix based on web
        // L[0][1] = 1;
        // L[1][0] = 1;

        // L[2][1] = 1;
        // L[2][4] = 1;
        // L[2][5] = 1;

        // L[3][0] = 1;

        // L[3][4] = 1;
        // L[4][0] = 1;
        // L[4][1] = 1;

        // L[4][5] = 1;
        // L[5][2] = 1;
        // L[5][4] = 1;
        start_time = MPI_Wtime();

    }

    

    double local_data_M[local_rows*NUM_PAGES] = {};
    double local_data_v[local_rows] = {};

    MPI_Scatterv(&L, counts_M, indices_M, MPI_DOUBLE,
                &local_data_M, local_rows * NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    

    // for (j = 0; j < local_rows*NUM_PAGES; j++){
    //     local_data_M[j]+=1;

    //     }
    

    if (rank == 0){

        printf("\n\nL before Gather\n");
        for (i = 0; i < NUM_PAGES; i++) {
            printf("\n\n");
            for (j = 0; j < NUM_PAGES; j++)
                printf("%8.2f  ", L[i][j]);
        }  

        printf("\ncounts_v");
        for (i = 0; i < size; i++)
        {
            printf("%8.2d  ", counts_v[i]);
        }
        printf("\nindices_v");
        for (i = 0; i < size; i++)
        {
            printf("%8.2d  ", indices_v[i]);
        }      
    }

    int row_R = 0;
    int col_R = 0;
    if (rank == 0){
        row_R = 0;
    }else{
        row_R = NUM_PAGES/size;
    }

    for (i = 0; i < local_rows * NUM_PAGES; i++){

        if (((i ) % NUM_PAGES) == 0 && (i!=0))
        {
            row_R += 1;
            col_R = 0;
        }
        if (row_R == col_R) {
            local_data_M[i] = 0;
        }else {

            local_data_M[i] = 1;

                
        }
        col_R += 1;
    }
    row_R = 0;
    col_R = 0;

    if (rank == 0){
        row_R = 0;
    }else{
        row_R = NUM_PAGES/size;
    }

    for (i = 0; i < local_rows * NUM_PAGES; i++)
    {
                if (((i ) % NUM_PAGES) == 0 && (i!=0))
                {
                    row_R += 1;
                    col_R = 0;
                }
                            
                // if (row_R == col_R) {
                //     local_data_M[i] = 0;
                // }else {

                //     local_data_M[i] = 1;

                        
                // }
                if (((i ) % NUM_PAGES) == 0){
                    for (j = 0; j < rnd_zeros;)
                    {

                        index = rand() % ((NUM_PAGES) - 1);
                        if (index != row_R)
                        {

                            local_data_M[i + index] = 0;
                            j++;
                        }
                        printf("\nrank: %d,index: %d, i: %d, row_R; %d, i+index: %d, L: %8.2f\n", rank, index, i, row_R, i+index, local_data_M[i + index]);
                    }
                    //printf("\ni: %d, index: %d, local_data: %f\n", (i+index),index, local_data_M[i + index]);
                }
                
                col_R += 1;
            }

    MPI_Gatherv(local_data_M, local_rows * NUM_PAGES, MPI_DOUBLE, L, counts_M, indices_M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0){

        printf("\n\nL after Gather\n");
        for (i = 0; i < NUM_PAGES; i++) {
            printf("\n\n");
            for (j = 0; j < NUM_PAGES; j++)
                printf("%8.2f  ", L[i][j]);
        }        
    }

    MPI_Scatterv(&r, counts_v, indices_v, MPI_DOUBLE,
                &local_data_v, local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0){

        printf("\n\nr before Gather\n");
        for (i = 0; i < NUM_PAGES; i++) {
            printf("%8.2f  ", r[i]);
        }        
    }
    for (i = 0; i < local_rows; i++) {
            local_data_v[i]=1/double(NUM_PAGES);
        }   

    MPI_Gatherv(local_data_v, local_rows, MPI_DOUBLE, r, counts_v, indices_v, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_data_v, local_rows, MPI_DOUBLE, r_k_1, counts_v, indices_v, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    if (rank == 0){

        printf("\n\nr after Gather\n");
        for (i = 0; i < NUM_PAGES; i++) {
            printf("%8.2f  ", r[i]);
        }        
    }

    // Send assembled L matrix to find number of links
    if (rank == 0){
        for (i = 1; i < size; i++) {//for each child other than the master
            chunk_size = (NUM_PAGES / (size - 1)); // calculate chunk_size without master
            index_start = (i - 1) * chunk_size;
            if (((i + 1) == size) && ((NUM_PAGES % (size - 1)) != 0)) {//if rows of [A] cannot be equally divided among childs
                index_end = NUM_PAGES; //last child gets all the remaining rows
            } else {
                index_end = index_start + chunk_size; //rows of [A] are equally divisable among childs
            }
            
            //send the first index first without blocking, to the intended child
            MPI_Isend(&index_start, 1, MPI_INT, i, M_C_TAG + 3, MPI_COMM_WORLD, &request);
            //next send the last index without blocking, to the intended child
            MPI_Isend(&index_end, 1, MPI_INT, i, M_C_TAG + 4, MPI_COMM_WORLD, &request);
            //finally send the allocated column chunk_size of [A] without blocking, to the intended child
            MPI_Isend(&L[0][index_start], (index_end - index_start) * NUM_PAGES, MPI_DOUBLE, i, M_C_TAG + 5, MPI_COMM_WORLD, &request);
            MPI_Isend(&nLinks[index_start], (index_end - index_start), MPI_DOUBLE, i, M_C_TAG + 6, MPI_COMM_WORLD, &request);
        
        }

        // Receive assembled nLinks vector
        for (i = 1; i < size; i++)
        {   //receive first index from a child
            MPI_Recv(&index_start, 1, MPI_INT, i, C_M_TAG + 3, MPI_COMM_WORLD, &status);
            //receive last index from a child
            MPI_Recv(&index_end, 1, MPI_INT, i, C_M_TAG + 4, MPI_COMM_WORLD, &status);
            // untill all childs have handed back the processed data
            MPI_Recv(&nLinks[index_start], (index_end - index_start), MPI_DOUBLE, i, C_M_TAG + 6, MPI_COMM_WORLD, &status);
        }
    }else{

        // #Links calculation
        //receive first index from the master
        MPI_Recv(&index_start, 1, MPI_INT, 0, M_C_TAG + 3, MPI_COMM_WORLD, &status);
        //next receive last index from the master
        MPI_Recv(&index_end, 1, MPI_INT, 0, M_C_TAG + 4, MPI_COMM_WORLD, &status);
        // recieve columns from master to calculate number of links in each page
        MPI_Recv(&L[0][index_start], (index_end - index_start) * NUM_PAGES, MPI_DOUBLE, 0, M_C_TAG + 5, MPI_COMM_WORLD, &status);
        // recieve chunks of nLInks array to store the number of links.
        MPI_Recv(&nLinks[index_start], (index_end - index_start), MPI_DOUBLE, 0, M_C_TAG + 6, MPI_COMM_WORLD, &status);

        
        
        // find number of links from each page
        for (int i = index_start; i < index_end; i++)
        {

            for (int j = 0; j < NUM_PAGES; j++)
            {

                if (L[j][i] != 0)
                {
                    nLinks[i] += 1;
                }
            }
        }

        // send back the first index first without blocking, to the master
        MPI_Isend(&index_start, 1, MPI_INT, 0, C_M_TAG + 3, MPI_COMM_WORLD, &request);
        //send the last index next without blocking, to the master
        MPI_Isend(&index_end, 1, MPI_INT, 0, C_M_TAG + 4, MPI_COMM_WORLD, &request);
        
        MPI_Isend(&nLinks[index_start], (index_end - index_start), MPI_DOUBLE, 0, C_M_TAG + 6, MPI_COMM_WORLD, &request);
    }


    
    if (rank == 0){

        printf("\n\nnLinks after filling\n");
        for (i = 0; i < NUM_PAGES; i++) {
            printf("%8.2f  ", nLinks[i]);
        }        
    }


    MPI_Bcast(&L, NUM_PAGES*NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nLinks, NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(&Q, counts_M, indices_M, MPI_DOUBLE,
                &local_data_M, local_rows * NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Specify starting point of the global array index
    if (rank == 0){
        m = 0;
    }else{
        m = NUM_PAGES/size;
    }

    for (i = 0; i < local_rows * NUM_PAGES; i++)
    {

        if (((i ) % NUM_PAGES) == 0 && (i!=0))
        {
            m+=1;
            n = 0;
        }

        if (L[m][n] == 0)
            {
                n++;
                continue;
            }
        local_data_M[i] = (1 / double(nLinks[n])) * L[m][n];
        n++;
    }

    MPI_Gatherv(local_data_M, local_rows * NUM_PAGES, MPI_DOUBLE, Q, counts_M, indices_M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0){

        printf("\n\nQ after Gather\n");
        for (i = 0; i < NUM_PAGES; i++) {
            printf("\n\n");
            for (j = 0; j < NUM_PAGES; j++)
                printf("%8.2f  ", Q[i][j]);
        }        
    }
    

    MPI_Datatype MPI_coltype, MPI_coltype2; 
    MPI_Type_vector(NUM_PAGES, 1, NUM_PAGES, MPI_DOUBLE, &MPI_coltype2);
    MPI_Type_create_resized( MPI_coltype2, 0, sizeof(double), &MPI_coltype);
	MPI_Type_commit(&MPI_coltype);

    counts_M_C[0] = counts_M_C[1] = 3;
    indices_M_C[0] = 0;
    indices_M_C[1] = 3;

    double local_data_M_C[NUM_PAGES* local_rows] = {};
    MPI_Scatterv(&Q, counts_M_C, indices_M_C, MPI_coltype,
                &local_data_M_C[0], NUM_PAGES*local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&d, counts_v, indices_v, MPI_DOUBLE,
                &local_data_v, local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0){
        n = 0;
    }else{
        n = NUM_PAGES/size;
    }
    bool only_zeros = true;
    m = 0;
    local_data_v[local_rows] = {};

    int row = 0;
    int col = 0;

    if (rank == 0){
        col = 0;
    }else{
        col = NUM_PAGES/size;
    }

    int vector_index = 0;
    printf("\nQ after scatter column, rank: %d\n", rank);
    for (int i = 0; i <NUM_PAGES*local_rows; i++){

        if (((i ) % NUM_PAGES) == 0 && (i!=0)){

            col += 1;
            row = 0;
            printf("\n");
            if (only_zeros==true){

                local_data_v[vector_index] = 1;
            }
            vector_index += 1;
            only_zeros = true;
        }
        if(local_data_M_C[i]!=0){

            only_zeros = false;
        }
        printf("%8.2f, (%d,%d), %d ", local_data_M_C[i], row, col, only_zeros);

        row += 1;
    }
    printf("\n");


    printf("\nd after scatter column, rank: %d\n", rank);
    for (int i = 0; i <local_rows; i++){

        printf("%8.2f ", local_data_v[i]);
        }
    printf("\n");

    MPI_Gatherv(local_data_v, local_rows, MPI_DOUBLE, d, counts_v, indices_v, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0){

        printf("\n\nd after Gather\n");
        for (i = 0; i < NUM_PAGES; i++) {
            printf("%8.2f  ", d[i]);
        }        
    }

    MPI_Bcast(&d, NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(&e_d, counts_M, indices_M, MPI_DOUBLE,
                 &local_data_M, local_rows * NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    
    col_R = 0;
    if (rank == 0){
        row_R = 0;
    }else{
        row_R = NUM_PAGES/size;
    }
    

    for (int i = 0; i <NUM_PAGES*local_rows; i++){

        if (((i ) % NUM_PAGES) == 0 && (i!=0)){

            row_R += 1;
            col_R = 0;
        }
        if(d[col_R]!=0){

            local_data_M[i] = 1;
        }
        col_R += 1;
    }

    MPI_Gatherv(local_data_M, local_rows * NUM_PAGES, MPI_DOUBLE, e_d, counts_M, indices_M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0){

        printf("\n\ne_d after Gather\n");
        for (i = 0; i < NUM_PAGES; i++) {
            printf("\n\n");
            for (j = 0; j < NUM_PAGES; j++)
                printf("%8.2f  ", e_d[i][j]);
        }        
    }

    MPI_Bcast(&e_d, NUM_PAGES*NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Q, NUM_PAGES*NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Scatterv(&P, counts_M, indices_M, MPI_DOUBLE,
                 &local_data_M, local_rows * NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);


    if (rank == 0){
        row_R = 0;
    }else{
        row_R = NUM_PAGES/size;
    }
    col_R = 0;
    for (int i = 0; i < NUM_PAGES * local_rows; i++)
    {

        if (((i ) % NUM_PAGES) == 0 && (i!=0)){

            row_R += 1;
            col_R = 0;
        }
        local_data_M[i] = Q[row_R][col_R] + (1/ double(NUM_PAGES))*e_d[row_R][col_R];
        
        col_R += 1;
    }

    MPI_Gatherv(local_data_M, local_rows * NUM_PAGES, MPI_DOUBLE, P, counts_M, indices_M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0){

        printf("\n\nP after Gather\n");
        for (i = 0; i < NUM_PAGES; i++) {
            printf("\n\n");
            for (j = 0; j < NUM_PAGES; j++)
                printf("%8.2f  ", P[i][j]);
        }        
    }
    int v = 1;
    int flag_ = 1;
    int stop = 0;

    if (rank == 0){



        while (stop!=1){

            printf("\nStarted: %d", v);

            //q_k vector
            //Send assembled P matrix to find number of links
            for (i = 1; i < size; i++) {//for each child other than the master
                chunk_size = (NUM_PAGES / (size - 1)); // calculate chunk_size without master
                index_start = (i - 1) * chunk_size;
                if (((i + 1) == size) && ((NUM_PAGES % (size - 1)) != 0)) {//if rows of [A] cannot be equally divided among childs
                    index_end = NUM_PAGES; //last child gets all the remaining rows
                } else {
                    index_end = index_start + chunk_size; //rows of [A] are equally divisable among childs
                }
                //send the first index first without blocking, to the intended child
                MPI_Isend(&index_start, 1, MPI_INT, i, M_C_TAG+21, MPI_COMM_WORLD, &request);
                //next send the last index without blocking, to the intended child
                MPI_Isend(&index_end, 1, MPI_INT, i, M_C_TAG + 22, MPI_COMM_WORLD, &request);
                //finally send the allocated column chunk_size of [A] without blocking, to the intended child
                MPI_Isend(&P[index_start][0], (index_end - index_start) * NUM_PAGES, MPI_DOUBLE, i, M_C_TAG + 23, MPI_COMM_WORLD, &request);
                MPI_Isend(&q_k, NUM_PAGES, MPI_DOUBLE, i, M_C_TAG + 24, MPI_COMM_WORLD, &request);
                MPI_Isend(&r_k_1, NUM_PAGES, MPI_DOUBLE, i, M_C_TAG + 96, MPI_COMM_WORLD, &request);


            
            }

            // Receive assembled q_k vector
            for (i = 1; i < size; i++)
            {   
                //receive first index from a child
                MPI_Recv(&index_start, 1, MPI_INT, i, C_M_TAG + 21, MPI_COMM_WORLD, &status);
                //receive last index from a child
                MPI_Recv(&index_end, 1, MPI_INT, i, C_M_TAG + 22, MPI_COMM_WORLD, &status);
                // untill all childs have handed back the processed data
                MPI_Recv(&q_k, NUM_PAGES, MPI_DOUBLE, i, C_M_TAG + 24, MPI_COMM_WORLD, &status);
            }

            q_k_norm = 0;
            // Calculate L1norm
            for (i = 0; i < NUM_PAGES; i++)
            {
                q_k_norm += fabs(q_k[i]);
            }

            // printf("\n\nq_k_norm_m\n");

            // for (i = 0; i < NUM_PAGES; i++){
            //     printf("%8.2f  ",q_k_norm);
            // }
            // printf("\n\n");

            // printf("\n\nq_k_m before send\n");

            // for (i = 0; i < NUM_PAGES; i++){
            //     printf("%8.2f  ", q_k[i]);
            // }
            // printf("\n\n");

            // r_k vector
            // Send assembled vectors
            for (i = 1; i < size; i++) {//for each child other than the master
                chunk_size = (NUM_PAGES / (size - 1)); // calculate chunk_size without master
                index_start = (i - 1) * chunk_size;
                if (((i + 1) == size) && ((NUM_PAGES % (size - 1)) != 0)) {//if rows of [A] cannot be equally divided among childs
                    index_end = NUM_PAGES; //last child gets all the remaining rows
                } else {
                    index_end = index_start + chunk_size; //rows of [A] are equally divisable among childs
                }
                //send the first index first without blocking, to the intended child
                MPI_Isend(&index_start, 1, MPI_INT, i, M_C_TAG+25, MPI_COMM_WORLD, &request);
                //next send the last index without blocking, to the intended child
                MPI_Isend(&index_end, 1, MPI_INT, i, M_C_TAG + 26, MPI_COMM_WORLD, &request);
                //finally send the allocated column chunk_size of [A] without blocking, to the intended child
                MPI_Isend(&r_k[index_start], (index_end - index_start), MPI_DOUBLE, i, M_C_TAG + 27, MPI_COMM_WORLD, &request);
                MPI_Isend(&r_k_1[index_start], (index_end - index_start), MPI_DOUBLE, i, M_C_TAG + 28, MPI_COMM_WORLD, &request);
                MPI_Isend(&temp[index_start], (index_end - index_start), MPI_DOUBLE, i, M_C_TAG + 29, MPI_COMM_WORLD, &request);
                MPI_Isend(&q_k_norm, 1, MPI_DOUBLE, i, M_C_TAG + 97, MPI_COMM_WORLD, &request);
                MPI_Isend(&q_k[index_start], (index_end - index_start), MPI_DOUBLE, i, M_C_TAG + 98, MPI_COMM_WORLD, &request);

            
            }

            // Receive assembled q_k vector
            for (i = 1; i < size; i++)
            {   
                //receive first index from a child
                MPI_Recv(&index_start, 1, MPI_INT, i, C_M_TAG + 25, MPI_COMM_WORLD, &status);
                //receive last index from a child
                MPI_Recv(&index_end, 1, MPI_INT, i, C_M_TAG + 26, MPI_COMM_WORLD, &status);
                // untill all childs have handed back the processed data
                MPI_Recv(&r_k[index_start], (index_end - index_start), MPI_DOUBLE, i, C_M_TAG + 27, MPI_COMM_WORLD, &status);
                MPI_Recv(&r_k_1[index_start], (index_end - index_start), MPI_DOUBLE, i, C_M_TAG + 28, MPI_COMM_WORLD, &status);
                MPI_Recv(&temp[index_start], (index_end - index_start), MPI_DOUBLE, i, C_M_TAG + 29, MPI_COMM_WORLD, &status);
            }

            // printf("\n\nq_k_m after receive\n");

            // for (i = 0; i < NUM_PAGES; i++){
            //     printf("%8.2f  ", q_k[i]);
            // }
            // printf("\n\n");
            printf("\nFinished: %d", v);
            v++;
            if (v>=ITERR){
                stop = 1;
            }
            MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);


        }

        if (rank == 0){

        printf("\n\nr_k after Gather\n");
        for (i = 0; i < NUM_PAGES; i++) {
            printf("%8.2f  ", r_k[i]);
        }        
        }

    }else{
         

        while(stop!=1){

            //**************** Asemble the q_k vector****************************************************+
            //receive first index from the master
            MPI_Recv(&index_start, 1, MPI_INT, 0, M_C_TAG + 21, MPI_COMM_WORLD, &status);
            //next receive last index from the master
            MPI_Recv(&index_end, 1, MPI_INT, 0, M_C_TAG + 22, MPI_COMM_WORLD, &status);
            //finally receive row chunk_size of [A] to be processed from the master
            MPI_Recv(&P[index_start][0], (index_end - index_start) * NUM_PAGES, MPI_DOUBLE, 0, M_C_TAG + 23, MPI_COMM_WORLD, &status);
            MPI_Recv(&q_k, NUM_PAGES, MPI_DOUBLE, 0, M_C_TAG + 24, MPI_COMM_WORLD, &status);
            MPI_Recv(&r_k_1,  NUM_PAGES, MPI_DOUBLE, 0, M_C_TAG + 96, MPI_COMM_WORLD, &status);

            // printf("\n\nr_k_1_nn: %d\n", v);

            //     for (i = 0; i < NUM_PAGES; i++){
            //         printf("%8.2f  ", r_k_1[i]);
            //     }
            //     printf("\n\n");
            
            // printf("\n\nP_c: %d\n", v);

            // for (i = index_start; i < index_end; i++){

            //     printf("\n\n");
            //     for(j = 0; j < NUM_PAGES; j++){

            //         printf("%8.2f  ", P[i][j]);
            //     }
            // }
            // fill P matrix
            for (i = index_start; i < index_end; i++){

                sum = 0;
                for (int j = 0; j < NUM_PAGES; j++)
                {
                    sum += P[i][j] * r_k_1[j];
                    

                }
                q_k[i] = sum;
            }
                
            // printf("\n\nq_k_c\n");

            //     for (i = 0; i < NUM_PAGES; i++){
            //         printf("%8.2f  ", q_k[i]);
            //     }
            //     printf("\n\n");
            // send back the first index first without blocking, to the master
            MPI_Isend(&index_start, 1, MPI_INT, 0, C_M_TAG + 21, MPI_COMM_WORLD, &request);
            //send the last index next without blocking, to the master
            MPI_Isend(&index_end, 1, MPI_INT, 0, C_M_TAG + 22, MPI_COMM_WORLD, &request);
            //finally send the processed chunk_size of data without blocking, to the master
            MPI_Isend(&q_k, NUM_PAGES, MPI_DOUBLE, 0, C_M_TAG + 24, MPI_COMM_WORLD, &request);


            //**************** Asemble the r_k vector****************************************************+
            //receive first index from the master
            MPI_Recv(&index_start, 1, MPI_INT, 0, M_C_TAG + 25, MPI_COMM_WORLD, &status);
            //next receive last index from the master
            MPI_Recv(&index_end, 1, MPI_INT, 0, M_C_TAG + 26, MPI_COMM_WORLD, &status);
            //finally receive row chunk_size of [A] to be processed from the master
            MPI_Recv(&r_k[index_start], (index_end - index_start), MPI_DOUBLE, 0, M_C_TAG + 27, MPI_COMM_WORLD, &status);
            MPI_Recv(&r_k_1[index_start], (index_end - index_start), MPI_DOUBLE, 0, M_C_TAG + 28, MPI_COMM_WORLD, &status);
            MPI_Recv(&temp[index_start], (index_end - index_start), MPI_DOUBLE, 0, M_C_TAG + 29, MPI_COMM_WORLD, &status);
            MPI_Recv(&q_k_norm, 1, MPI_DOUBLE, 0, M_C_TAG + 97, MPI_COMM_WORLD, &status);
            
            MPI_Recv(&q_k[index_start], (index_end - index_start), MPI_DOUBLE, 0, M_C_TAG + 98, MPI_COMM_WORLD, &status);

            // printf("\n\nq_k_norm_c\n");

            //     for (i = 0; i < NUM_PAGES; i++){
            //         printf("%8.2f  ", q_k_norm);
            //     }
            //     printf("\n\n");
            
            // printf("\n\nq_k_c\n");

            //     for (i = index_start; i < index_end; i++){
            //         printf("%8.2f  ", q_k[i]);
            //     }
            //     printf("\n\n");

            // fill vectors
            for (i = index_start; i < index_end; i++) 
            {
                r_k[i] = q_k[i]/q_k_norm;
                temp[i] = r_k_1[i];
                r_k_1[i] = r_k[i];
                    
            }
            // printf("\n\nr_k_1_c\n");

            //     for (i = index_start; i < index_end; i++){
            //         printf("%8.2f  ", r_k_1[i]);
            //     }
            //     printf("\n\n");
        

            // send back the first index first without blocking, to the master
            MPI_Isend(&index_start, 1, MPI_INT, 0, C_M_TAG + 25, MPI_COMM_WORLD, &request);
            //send the last index next without blocking, to the master
            MPI_Isend(&index_end, 1, MPI_INT, 0, C_M_TAG + 26, MPI_COMM_WORLD, &request);
            //finally send the processed chunk_size of data without blocking, to the master
            MPI_Isend(&r_k[index_start], (index_end - index_start), MPI_DOUBLE, 0, C_M_TAG + 27, MPI_COMM_WORLD, &request);
            MPI_Isend(&r_k_1[index_start], (index_end - index_start), MPI_DOUBLE, 0, C_M_TAG + 28, MPI_COMM_WORLD, &request);
            MPI_Isend(&temp[index_start], (index_end - index_start), MPI_DOUBLE, 0, C_M_TAG + 29, MPI_COMM_WORLD, &request);
            MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);

        }
    }
    //printf("\n");
    
    // for (i = 0; i < NUM_PAGES * local_rows; i++)
    // {

    //     if (((i ) % NUM_PAGES) == 0 && (i!=0))
    //     {   
    //         if (only_zeros==true){

                
    //             local_data_v[n] += 1;
    //             printf("\n\n\nHere\n\n%d\n%d\n%d\n", m,n, local_data_v[n]);
    //         }
    //         n+=1;
    //         m = 0;
    //         only_zeros = true;
    //     }

    //     if (local_data_M_C[m] != 0)
    //         {
    //             only_zeros = false;
    //         }
    //     m++;
    // }

    // printf("\n\nd after Scatterv\n");

    // for (i = 0; i < local_rows; i++){

    //     printf("%8.2f ", local_data_v[i]);
    // }
    // printf("\n");

    // if (rank == 0){
    //     j = 0;
    //     printf("\n\nQ after Scatterv\n");
    //     for (i = 0; i < NUM_PAGES*local_rows; i++) {
            
            
    //            if ((i ) % NUM_PAGES == 0)
    //             {
    //                 printf("\n\n");
    //                 j = 0;
    //             }

    //             printf("%8.2f  ", local_data_M_C[i]);
    //     }        
    // }
    

    end_time = MPI_Wtime();
    printf("\nRunning Time = %f\n\n", end_time - start_time);
        
    
    //printf("Process %d received elements: ", rank);
    //printf("\n\nlocalL\n");
    // for (i = 0; i < local_rows; i++) {
    //     printf("\n\n");
    //     for (j = 0; j < NUM_PAGES; j++)
    //         printf("%8.2f  ", local_data_M[j]);
    // }
    // for (j = 0; j < local_rows*NUM_PAGES; j++){
    //     printf("%8.2f  ", local_data_M[j]);
    //     if((j+1)%6 == 0){
    //         printf("\n");
    //     }
    // }

    
    //printArray(local_data_M, local_rows);

    //MPI_Bcast(&nLinks, NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /* work done by childs*/

    MPI_Finalize(); //finalize MPI operations

    return 0;
}

void printArray(double mat[][NUM_PAGES] ,int rows)
{   
    // printf("\n\nL\n");
    // for (i = 0; i < NUM_PAGES; i++) {
    //     printf("\n\n");
    //     for (j = 0; j < NUM_PAGES; j++)
    //         printf("%8.2f  ", L[i][j]);
    // }

    // printf("\n\nlocalL\n");
    // for (i = 0; i < rows; i++) {
    //     printf("\n\n");
    //     for (j = 0; j < NUM_PAGES; j++)
    //         printf("%8.2f  ", mat[i][j]);
    // }

    // for (i = 0; i < NUM_PAGES; i++){
    //     printf("%8.2f  ", mat[i]);
    // }

        // printf("\n\nnLinks\n");

        // for (i = 0; i < NUM_PAGES; i++){
        //     printf("%8.2f  ", nLinks[i]);
        // }
        // printf("\nQ\n");

        // for (i = 0; i < NUM_PAGES; i++) {
        //     printf("\n");
        //     for (j = 0; j < NUM_PAGES; j++)
        //         printf("%8.2f  ", Q[i][j]);
        // }

        // printf("\n\nd\n");

        // for (i = 0; i < NUM_PAGES; i++){
        //     printf("%8.2f  ", d[i]);
        // }
        // printf("\ne_d\n");

        // for (i = 0; i < NUM_PAGES; i++) {
        //     printf("\n");
        //     for (j = 0; j < NUM_PAGES; j++)
        //         printf("%8.2f  ", e_d[i][j]);
        // }
        //  printf("\nP\n");

        // for (i = 0; i < NUM_PAGES; i++) {
        //     printf("\n");
        //     for (j = 0; j < NUM_PAGES; j++)
        //         printf("%8.2f  ", P[i][j]);
        // }
        // printf("\n\nr_k\n");

        // for (i = 0; i < NUM_PAGES; i++){
        //     printf("%8.2f  ", r_k[i]);
        // }
        // printf("\n\n");

        // printf("\n\nr_k_1\n");

        // for (i = 0; i < NUM_PAGES; i++){
        //     printf("%8.2f  ", r_k_1[i]);
        // }
        // printf("\n\n");
        // printf("\n\nq_k\n");

        // for (i = 0; i < NUM_PAGES; i++){
        //     printf("%8.2f  ", q_k[i]);
        // }
        // printf("\n\n");
        // printf("\n\nr\n");

        // for (i = 0; i < NUM_PAGES; i++){
        //     printf("%8.2f  ", r[i]);
        // }
        // printf("\n\n");
}   
