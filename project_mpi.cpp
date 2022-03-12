/**********************************************************************************************

************************************************************************************************/

#include<stdio.h>
#include<mpi.h>
#include<cmath>
#define NUM_PAGES 6
#define M_C_TAG 1 //tag for messages sent from master to childs
#define C_M_TAG 4 //tag for messages sent from childs to master
#define ITERR 30
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
    int testing;
    if (rank < remainder)
    {
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

        printf("\nTest with given web? Then enter 1, else 0: ");
        std::cin >> testing;
        // fill dense matrix based on web
        if (testing ==1){
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
        }
        start_time = MPI_Wtime();

    }

    MPI_Bcast(&testing, 1, MPI_INT, 0, MPI_COMM_WORLD);
    

    double local_data_M[local_rows*NUM_PAGES] = {};
    double local_data_v[local_rows] = {};
    double local_data_v_r_k[local_rows] = {};
    double local_data_v_r_k_1[local_rows] = {};
    double local_data_v_temp[local_rows] = {};
    double local_data_v_q_k[local_rows] = {};

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
        //row_R = NUM_PAGES/size;
        row_R = (NUM_PAGES/size)*rank;
    }
    if (testing == 0){
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
            //row_R = NUM_PAGES/size;
            row_R = (NUM_PAGES/size)*rank;
        }

        for (i = 0; i < local_rows * NUM_PAGES; i++)
        {
                    if (((i ) % NUM_PAGES) == 0 && (i!=0))
                    {
                        row_R += 1;
                        col_R = 0;
                    }
                                

                            
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
                            //printf("\nrank: %d,index: %d, i: %d, row_R; %d, i+index: %d, L: %8.2f\n", rank, index, i, row_R, i+index, local_data_M[i + index]);
                        }
                        //printf("\ni: %d, index: %d, local_data: %f\n", (i+index),index, local_data_M[i + index]);
                    }
                    
                    col_R += 1;
                }

        MPI_Gatherv(local_data_M, local_rows * NUM_PAGES, MPI_DOUBLE, L, counts_M, indices_M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }
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

    MPI_Datatype MPI_coltype, MPI_coltype2; 
    MPI_Type_vector(NUM_PAGES, 1, NUM_PAGES, MPI_DOUBLE, &MPI_coltype2);
    MPI_Type_create_resized( MPI_coltype2, 0, sizeof(double), &MPI_coltype);
	MPI_Type_commit(&MPI_coltype);

    // counts_M_C[0] = counts_M_C[1] = 3;
    // indices_M_C[0] = 0;
    // indices_M_C[1] = 3;

    double local_data_M_C[NUM_PAGES* local_rows] = {};
    local_data_v[local_rows]= {};
    MPI_Scatterv(&L, counts_v, indices_v, MPI_coltype,
                 &local_data_M_C[0], NUM_PAGES * local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&nLinks, counts_v, indices_v, MPI_DOUBLE,
                &local_data_v, local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int row = 0;
    int col = 0;

    if (rank == 0){
        col = 0;
    }else{
        //col = NUM_PAGES/size;
        col = (NUM_PAGES/size)*rank;
    }

    int vector_index = 0;
    for (int i = 0; i <NUM_PAGES*local_rows; i++){

        if (((i ) % NUM_PAGES) == 0 && (i!=0)){

            col += 1;
            row = 0;
            printf("\n");
            // if (only_zeros==true){

            //     local_data_v[vector_index] = 1;
            // }
            vector_index += 1;
        }
        if(local_data_M_C[i]!=0){
            local_data_v[vector_index] += 1;

        }
        //printf("%8.2f, (%d,%d), %d ", local_data_M_C[i], row, col, local_data_v[vector_index]);

        row += 1;
    }
    MPI_Gatherv(local_data_v, local_rows, MPI_DOUBLE, nLinks, counts_v, indices_v, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
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
        //m = NUM_PAGES/size;
        m = (NUM_PAGES/size)*rank;
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
    

    

    // counts_M_C[0] = counts_M_C[1] = 3;
    // indices_M_C[0] = 0;
    // indices_M_C[1] = 3;

    local_data_M_C[NUM_PAGES* local_rows] = {};
    MPI_Scatterv(&Q, counts_v, indices_v, MPI_coltype,
                &local_data_M_C[0], NUM_PAGES*local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&d, counts_v, indices_v, MPI_DOUBLE,
                &local_data_v, local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0){
        n = 0;
    }else{
        n = (NUM_PAGES/size)*rank;
    }
    bool only_zeros = true;
    m = 0;
    local_data_v[local_rows] = {};

    row = 0;
    col = 0;

    if (rank == 0){
        col = 0;
    }else{
        col = (NUM_PAGES/size)*rank;
    }

    vector_index = 0;

    // fill local_data_v (here d with 1)
    for (int i = 0; i < local_rows; i++){

        local_data_v[i] = 1;
    }
        printf("\nQ after scatter column, rank: %d\n", rank);
    for (int i = 0; i <NUM_PAGES*local_rows; i++){

        if (((i ) % NUM_PAGES) == 0 && (i!=0)){

            col += 1;
            row = 0;
            printf("\n");
            // if (only_zeros==true){

            //     local_data_v[vector_index] = 1;
            // }
            vector_index += 1;
            only_zeros = true;
        }
        if(local_data_M_C[i]!=0){

            only_zeros = false;
            local_data_v[vector_index] = 0;
        }
        //printf("[%8.2f, (%d,%d), %d ,%8.2f]", local_data_M_C[i], row, col, only_zeros, local_data_v[vector_index]);

        row += 1;
    }
    //printf("\n");


    // printf("\nd after scatter column, rank: %d\n", rank);
    // for (int i = 0; i <local_rows; i++){

    //     printf("%8.2f ", local_data_v[i]);
    //     }
    // printf("\n");

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
        row_R = (NUM_PAGES/size)*rank;
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
        row_R = (NUM_PAGES/size)*rank;
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

    //if (rank == 0){



    while (stop!=1){

        printf("\nStarted: %d", v);

        //q_k vector

        local_data_M[local_rows*NUM_PAGES] = {};
        local_data_v[local_rows] = {};
        MPI_Scatterv(&P, counts_M, indices_M, MPI_DOUBLE,
                        &local_data_M, local_rows * NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&r_k_1, NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(&q_k, counts_v, indices_v, MPI_DOUBLE,
                &local_data_v, local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        col_R = 0;
        if (rank == 0){
            row_R = 0;
        }else{
            row_R = (NUM_PAGES/size)*rank;
        }
        
        sum = 0;
        vector_index = 0;
        printf("\n");
        for (int i = 0; i < NUM_PAGES * local_rows; i++)
        {

            if (((i ) % NUM_PAGES) == 0 && (i!=0)){

                row_R += 1;
                col_R = 0;
                sum = 0;
                vector_index += 1;

                //printf("\n");
            }

            local_data_v[vector_index] += local_data_M[i]*r_k_1[col_R];
            //printf("[r:%d, ldv:%f, ldM:%f, r_k_1:%f, (%d,%d), vi:%d]", rank, local_data_v[vector_index], local_data_M[i], r_k_1[col_R], row_R,col_R, vector_index);
            col_R += 1;
        }
        MPI_Gatherv(local_data_v, local_rows, MPI_DOUBLE, q_k, counts_v, indices_v, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        //MPI_Bcast(&q_k, NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        if (rank == 0){

        printf("\n\nq_k after Gather\n");
        for (i = 0; i < NUM_PAGES; i++) {
            printf("%8.2f  ", q_k[i]);
        }     
        q_k_norm = 0;
        // Calculate L1norm
        for (i = 0; i < NUM_PAGES; i++)
        {
            q_k_norm += fabs(q_k[i]);
        }
        printf("\nq_k norm: %f\n", q_k_norm);

        }

        MPI_Bcast(&q_k_norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        

        MPI_Scatterv(&r_k, counts_v, indices_v, MPI_DOUBLE,
                &local_data_v_r_k, local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(&r_k_1, counts_v, indices_v, MPI_DOUBLE,
                &local_data_v_r_k_1, local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(&temp, counts_v, indices_v, MPI_DOUBLE,
                &local_data_v_temp, local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(&q_k, counts_v, indices_v, MPI_DOUBLE,
                &local_data_v_q_k, local_rows, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        printf("\n");
        for (int i = 0; i < local_rows; i++)
        {

            local_data_v_r_k[i] = local_data_v_q_k[i]/q_k_norm;
            local_data_v_temp[i] = local_data_v_r_k_1[i];
            local_data_v_r_k_1[i] = local_data_v_r_k[i];

            //printf("\nr: %d, r_k: %f", rank, local_data_v_r_k[i]);
        }

        MPI_Gatherv(local_data_v_q_k, local_rows, MPI_DOUBLE, q_k, counts_v, indices_v, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_data_v_temp, local_rows, MPI_DOUBLE, temp, counts_v, indices_v, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_data_v_r_k_1, local_rows, MPI_DOUBLE, r_k_1, counts_v, indices_v, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_data_v_r_k, local_rows, MPI_DOUBLE, r_k, counts_v, indices_v, MPI_DOUBLE, 0, MPI_COMM_WORLD);


        // MPI_Bcast(&temp, NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // MPI_Bcast(&r_k, NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // MPI_Bcast(&r_k_1, NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);


        if (rank == 0){
        printf("\n\nr_k after Gather_inside_loop\n");
        for (i = 0; i < NUM_PAGES; i++) {
            sum+=r_k[i];
            printf("%8.2f  ", r_k[i]);
        }   
    
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

    
    sum= 0;
    printf("\n\nr_k after Gather\n");
    for (i = 0; i < NUM_PAGES; i++) {
        sum+=r_k[i];
        printf("%8.2f  ", r_k[i]);
    }   
    printf("\nSum: %8.2f\n", sum);
    }

    end_time = MPI_Wtime();
    printf("\nRunning Time = %f\n\n", end_time - start_time);
        
    MPI_Finalize(); //finalize MPI operations

    return 0;
}

