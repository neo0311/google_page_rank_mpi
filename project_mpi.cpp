/**********************************************************************************************
 Finding rank vector and the largest eigen value using parallelised power iteration

 Author: Basil Jose
 TU Bergakademie Freiberg
************************************************************************************************/

#include<stdio.h>
#include<mpi.h>
#include<cmath>

#define NUM_PAGES 10 // number of pages, also the matrix size
#define ITERR 30 // max iterations for power iteration
#define test_and_debug 0 // 1 for enabling verbose output and test using given web, 0 for randomly generated L matrix

int rank; //process rank
int size; //number of processes
double start_time; //hold start time
double end_time; // hold end time

MPI_Status status;   // store status of a MPI_Recv
MPI_Request request; //capture request of a MPI_Isend

int main(int argc, char *argv[])
{   
    
    MPI_Init(&argc, &argv);               //initialize MPI operations
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); //get the rank
    MPI_Comm_size(MPI_COMM_WORLD, &size); //get number of processes

    start_time = MPI_Wtime();

    // remaining portion after dividing the matrix dims equally for processes
    int remainder = NUM_PAGES % size; 
    int rows_to_send = (NUM_PAGES / size);

    // size of local buffer (eg. no. of rows or columns to send per process)
    int local_bfr_size = (NUM_PAGES / size);

    if (rank==size-1){

        local_bfr_size += remainder; // add remainder to last chunk
    }
    int testing; // to decide to test with given web or do random 

    // initialize MPI variables //
    // create a derived MPI datatype to transfer columns using scatterv
    MPI_Datatype MPI_coltype, MPI_coltype2; 
    MPI_Type_vector(NUM_PAGES, 1, NUM_PAGES, MPI_DOUBLE, &MPI_coltype2);
    MPI_Type_create_resized( MPI_coltype2, 0, sizeof(double), &MPI_coltype);
	MPI_Type_commit(&MPI_coltype);

    // initialize variables for Scatterv
    // during row wise transfer
    int counts_M[size] = {}; // to store chunk sizes for matrices
    int indices_M[size] = {}; // to store the indices to transfer (matrix)
    int counts_v[size] = {}; // to store chunk sizes for vectors
    int indices_v[size] = {}; // to store the indices to transfer (vector)
    //(during column wise transfer)
    int counts_M_C[size] = {}; // to store chunk sizes for matrices 
    int indices_M_C[size] = {}; // to store the indices_M to transfer

    //local buffers//
    //during rowwise transfer
    double local_data_M[local_bfr_size*NUM_PAGES] = {};
    double local_data_v[local_bfr_size] = {};
    double local_data_v_r_k[local_bfr_size] = {};
    double local_data_v_r_k_1[local_bfr_size] = {};
    double local_data_v_temp[local_bfr_size] = {};
    double local_data_v_q_k[local_bfr_size] = {};
    //during colum nwise transfer
    double local_data_M_C[NUM_PAGES* local_bfr_size] = {};
    //for MPI_Reduce
    double local_sum = 0;
    double global_sum = 0;
    double vector_sum = 0;
    double duration = 0;
    double total_time = 0;

    // initialize required matrices and vectors
    double L[NUM_PAGES][NUM_PAGES] = {0}; 
    double Q[NUM_PAGES][NUM_PAGES] = {0};
    double e_d[NUM_PAGES][NUM_PAGES] = {0}; // to store vector product of e and d
    double P[NUM_PAGES][NUM_PAGES] = {0};
    double r_k_1[NUM_PAGES] = {0};
    double r_k[NUM_PAGES] = {0};
    double q_k[NUM_PAGES] = {0};
    double q_k_norm;
    double l2_norm = 0.0;
    double lambda_max = 0.0; // maximum eigenvalue for final r vector
    double temp[NUM_PAGES] = {0};
    double r[NUM_PAGES] = {0}; // rank vector
    double nLinks[NUM_PAGES] = {0}; // to store number of links to a page
    double sum;
    double e[NUM_PAGES] = {0};
    double d[NUM_PAGES] = {0};
    int const rnd_zeros = ((NUM_PAGES*NUM_PAGES)*0.30)/NUM_PAGES; //number of random positions in each row to be replaced by 0.


    // initialize variables for loops
    int index = 0; // to store indices
    int i, j, k; //helper variables
    int m = 0;
    int n = 0;
    int row = 0;
    int col = 0;
    int row_R = 0; // row indices if rows are transfered
    int col_R = 0; // column indices if rows are transfered
    int row_C = 0; // row indices if columns are transfered
    int col_C = 0; // column indices if columns are transfered
    int matrix_index = 0; // local row or column indices of matrices
    int vector_index = 0; // local indices of vectors

    // initialize variables for power iteration
    int v = 1;
    int stop = 0;

    // calculate indices and chunk sizes
    for (i = 0; i < size; i++)
    {
        // Calculate the index locations to transfer
        if (i == 0)
        {
            indices_M[i] = 0;
            indices_v[i] = 0;
        }
        else
        {
            indices_M[i] = indices_M[i-1]+(NUM_PAGES*rows_to_send);
            indices_v[i] = indices_v[i-1]+(rows_to_send);
        }
        // Calculate chunk sizes for each chunk
        if (i + 1 == size)
        {
            counts_M[i] = (rows_to_send+remainder)*NUM_PAGES;
            counts_v[i] = (rows_to_send+remainder);
        }
        else
        {
            counts_M[i] = rows_to_send*NUM_PAGES;
            counts_v[i] = rows_to_send;
        }
    }

    // *************************** L Matrix *****************************************************
    /* master initializes work*/
    if (rank == 0) {

        printf("\nNo. of ranks: %d\n", size);

        if (test_and_debug == 1){

            testing = 1;

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

            // fill dense matrix based on given web
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
            
        }else{
            printf("\nNumber of pages: %d\nIterations: %d\n", NUM_PAGES, ITERR);
            testing = 0;
        }
    }

    MPI_Bcast(&testing, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // scatter L to processes for random initilaization
    MPI_Scatterv(&L, counts_M, indices_M, MPI_DOUBLE,
                &local_data_M, local_bfr_size * NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if ((rank == 0) && (test_and_debug == 1)){
        printf("\nL before Gather\n");
        for (i = 0; i < NUM_PAGES; i++) {
            printf("\n\n");
            for (j = 0; j < NUM_PAGES; j++)
                printf("%8.2f  ", L[i][j]);
            }     
    }

    // initialize row and column indices based on rank 
    if (rank == 0){
        row_R = 0;
    }else{
        // to keep global view of row index
        row_R = (NUM_PAGES/size)*rank;
    }

    // random initilaization of L matrix
    if (testing == 0)
    {
        for (i = 0; i < local_bfr_size * NUM_PAGES; i++){
            
            // calculates row and column indices
            if ((i % NUM_PAGES) == 0 && (i!=0))
            {
                row_R += 1;
                col_R = 0;
            }

            // initialises the L matrix with 0s and 1s
            if (row_R == col_R) 
            {
                local_data_M[i] = 0;
            }else 
            {
                local_data_M[i] = 1;      
            }
            col_R += 1;
        }
        
        // reinitialize for random filling of the matrix
        row_R = 0;
        col_R = 0;

        if (rank == 0)
        {
            row_R = 0;
        }else
        {
            // to keep global view of row index
            row_R = (NUM_PAGES/size)*rank;
        }

        matrix_index = 0;
        for (i = 0; i < local_bfr_size * NUM_PAGES; i++)
        {
                    // calculates global row and column indices
                    if ((i % NUM_PAGES) == 0 && (i!=0))
                    {
                        row_R += 1;
                        col_R = 0;
                        matrix_index += 1;
                    }

                    if ((i % NUM_PAGES) == 0){
                        // fills random places in rows with zeros (preset counts).
                        for (j = 0; j < rnd_zeros;)
                        {
                            srand(row_R); // set seed based on global row index
                            // randomly generate indices to be zeroed out
                            index = rand() % ((NUM_PAGES) -1);
                            // to calculate correct index in matrix
                            local_data_M[matrix_index*NUM_PAGES + index] = 0;

                            j++;
                        }
                    }
                    col_R += 1;
        }
        // reassemble the initialised L matrix
        MPI_Gatherv(local_data_M, local_bfr_size * NUM_PAGES, MPI_DOUBLE, L, counts_M, indices_M, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        printf("\nL Matrix\n");
        for (i = 0; i < NUM_PAGES; i++)
        {
            printf("\n\n");
            for (j = 0; j < NUM_PAGES; j++)
            printf("%8.2f  ", L[i][j]);
        }
    }

    if ((rank == 0) && (test_and_debug == 1)){

        printf("\n\nr before Gather\n");
        for (i = 0; i < NUM_PAGES; i++) {
            printf("%8.2f  ", r[i]);
        }        
    }

    //************************************************* r, r_k_1 init **************************************************+
    MPI_Scatterv(&r, counts_v, indices_v, MPI_DOUBLE,
                &local_data_v, local_bfr_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // initialize r vector values
    for (i = 0; i < local_bfr_size; i++) {
            local_data_v[i]=1/double(NUM_PAGES);
        }   

    // gather - also initialize r_k_1 same as r with same values.
    MPI_Gatherv(local_data_v, local_bfr_size, MPI_DOUBLE, r, counts_v, indices_v, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gatherv(local_data_v, local_bfr_size, MPI_DOUBLE, r_k_1, counts_v, indices_v, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if ((rank == 0) && (test_and_debug == 1)){
        printf("\n\nr after Gather\n");
        for (i = 0; i < NUM_PAGES; i++) {
            printf("%8.2f  ", r[i]);
        }        
    }

    // *************************************************** nLinks ***********************************************************
    // calculate no. of links to each page by summing the column in L matrix
    local_data_v[local_bfr_size]= {};
    MPI_Scatterv(&L, counts_v, indices_v, MPI_coltype,
                 &local_data_M_C[0], NUM_PAGES * local_bfr_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&nLinks, counts_v, indices_v, MPI_DOUBLE,
                &local_data_v, local_bfr_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    row_C = 0;
    col_C = 0;
    if (rank == 0){
        col_C = 0;
    }else{
        col_C = (NUM_PAGES/size)*rank;
    }

    vector_index = 0;
    for (int i = 0; i <NUM_PAGES*local_bfr_size; i++){

        if ((i % NUM_PAGES) == 0 && (i!=0)){

            col_C += 1;
            row_C = 0;
            vector_index += 1;
        }
        // column wise addition
        if(local_data_M_C[i]!=0){
            local_data_v[vector_index] += 1;

        }
        row_C += 1;
    }

    MPI_Gatherv(local_data_v, local_bfr_size, MPI_DOUBLE, nLinks, counts_v, indices_v, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    if (rank == 0){
        printf("\n\nnLinks after filling\n");
        for (i = 0; i < NUM_PAGES; i++) {
            printf("%8.2f  ", nLinks[i]);
        }        
    }

    MPI_Bcast(&L, NUM_PAGES*NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&nLinks, NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&Q, counts_M, indices_M, MPI_DOUBLE,
                &local_data_M, local_bfr_size * NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Specify starting point of the global array index
    if (rank == 0){
        m = 0;
    }else{
        m = (NUM_PAGES/size)*rank;
    }

    for (i = 0; i < local_bfr_size * NUM_PAGES; i++)
    {

        if ((i % NUM_PAGES) == 0 && (i!=0))
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

    MPI_Gatherv(local_data_M, local_bfr_size * NUM_PAGES, MPI_DOUBLE, Q, counts_M, indices_M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0){
        printf("\n\nQ after Gather\n");
        for (i = 0; i < NUM_PAGES; i++) {
            printf("\n\n");
            for (j = 0; j < NUM_PAGES; j++)
                printf("%8.2f  ", Q[i][j]);
        }        
    }

    local_data_M_C[NUM_PAGES* local_bfr_size] = {};
    MPI_Scatterv(&Q, counts_v, indices_v, MPI_coltype,
                &local_data_M_C[0], NUM_PAGES*local_bfr_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&d, counts_v, indices_v, MPI_DOUBLE,
                &local_data_v, local_bfr_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0){
        n = 0;
    }else{
        n = (NUM_PAGES/size)*rank;
    }
    m = 0;
    
    local_data_v[local_bfr_size] = {};
    row = 0;
    col = 0;

    if (rank == 0){
        col = 0;
    }else{
        col = (NUM_PAGES/size)*rank;
    }
    
    // flag for columns with only zeros
    bool only_zeros = true;
    vector_index = 0;

    // fill local_data_v (here d with 1) - initialises with 1, changes to 0 if not only zeros in L matrix column
    for (int i = 0; i < local_bfr_size; i++){

        local_data_v[i] = 1;
    }
    
    // calculate d vector
    for (int i = 0; i <NUM_PAGES*local_bfr_size; i++){

        if ((i % NUM_PAGES) == 0 && (i!=0)){

            col += 1;
            row = 0;
            vector_index += 1;
            only_zeros = true;
        }
        //changes to 0 if not only zeros in L matrix column
        if(local_data_M_C[i]!=0){

            only_zeros = false;
            local_data_v[vector_index] = 0;
        }
        row += 1;
    }

    MPI_Gatherv(local_data_v, local_bfr_size, MPI_DOUBLE, d, counts_v, indices_v, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if ((rank == 0) && (test_and_debug == 1)){

        printf("\n\nd after Gather\n");
        for (i = 0; i < NUM_PAGES; i++) {
            printf("%8.2f  ", d[i]);
        }        
    }

    MPI_Bcast(&d, NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ************************************************* e_d matrix ********************************
    // make e_d matrix without vector multiplicatin
    // e_d just contains '1's in columns where the columns are zeros in L matrix

    MPI_Scatterv(&e_d, counts_M, indices_M, MPI_DOUBLE,
                 &local_data_M, local_bfr_size * NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    
    col_R = 0;
    if (rank == 0){
        row_R = 0;
    }else{
        row_R = (NUM_PAGES/size)*rank;
    }
    
    for (int i = 0; i <NUM_PAGES*local_bfr_size; i++){

        if ((i % NUM_PAGES) == 0 && (i!=0)){

            row_R += 1;
            col_R = 0;
        }
        if(d[col_R]!=0){

            local_data_M[i] = 1;
        }
        col_R += 1;
    }

    MPI_Gatherv(local_data_M, local_bfr_size * NUM_PAGES, MPI_DOUBLE, e_d, counts_M, indices_M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if ((rank == 0) && (test_and_debug == 1)){

        printf("\n\ne_d after Gather\n");
        for (i = 0; i < NUM_PAGES; i++) {
            printf("\n\n");
            for (j = 0; j < NUM_PAGES; j++)
                printf("%8.2f  ", e_d[i][j]);
        }        
    }

    MPI_Bcast(&e_d, NUM_PAGES*NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&Q, NUM_PAGES*NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    // ************************************************ P matrix ***********************************************
    
    MPI_Scatterv(&P, counts_M, indices_M, MPI_DOUBLE,
                 &local_data_M, local_bfr_size * NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (rank == 0){
        row_R = 0;
    }else{
        row_R = (NUM_PAGES/size)*rank;
    }
    col_R = 0;
    
    for (int i = 0; i < NUM_PAGES * local_bfr_size; i++)
    {

        if ((i % NUM_PAGES) == 0 && (i!=0)){

            row_R += 1;
            col_R = 0;
        }
        local_data_M[i] = Q[row_R][col_R] + (1/ double(NUM_PAGES))*e_d[row_R][col_R];
        
        col_R += 1;
    }

    MPI_Gatherv(local_data_M, local_bfr_size * NUM_PAGES, MPI_DOUBLE, P, counts_M, indices_M, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if ((rank == 0) && (test_and_debug == 1)){

        printf("\n\nP after Gather\n");
        for (i = 0; i < NUM_PAGES; i++) {
            printf("\n\n");
            for (j = 0; j < NUM_PAGES; j++)
                printf("%8.2f  ", P[i][j]);
        }        
    }
    // **************************************+ Power iteration *************************************************************
    v = 1;
    stop = 0;

    while (stop!=1){

        if (rank == 0) printf("\nStarted iteration: %d", v);
        
        //**************************************** q_k vector and L1 norm *******************************//
        local_data_M[local_bfr_size*NUM_PAGES] = {};
        local_data_v[local_bfr_size] = {};

        MPI_Scatterv(&P, counts_M, indices_M, MPI_DOUBLE,
                        &local_data_M, local_bfr_size * NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&r_k_1, NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(&q_k, counts_v, indices_v, MPI_DOUBLE,
                &local_data_v, local_bfr_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        col_R = 0;
        if (rank == 0){
            row_R = 0;
        }else{
            row_R = (NUM_PAGES/size)*rank;
        }
        
        sum = 0;
        vector_index = 0;
        // matrix vector multiplicatin
        for (int i = 0; i < NUM_PAGES * local_bfr_size; i++)
        {
            if ((i % NUM_PAGES) == 0 && (i!=0))
            {

                row_R += 1;
                col_R = 0;
                sum = 0;
                vector_index += 1;

            }
            local_data_v[vector_index] += local_data_M[i]*r_k_1[col_R];
            col_R += 1;
        }

        // prepare send buffer for MPI_Reduce to find L1 norm
        local_sum = 0;
        for (int i = 0; i < local_bfr_size; i++)
        {
            local_sum += local_data_v[i];
        }
        
        //Gather q_k
        MPI_Gatherv(local_data_v, local_bfr_size, MPI_DOUBLE, q_k, counts_v, indices_v, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        //Reduce the local buffer values of q_k to find norm
        MPI_Reduce(&local_sum, &q_k_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        
        if ((rank == 0) )
        {
            if (test_and_debug == 1){
            printf("\nq_k after Gather\n");
            for (i = 0; i < NUM_PAGES; i++) {
                printf("%8.2f  ", q_k[i]);
            }     
            
            printf("\nq_k norm: %f\n", q_k_norm);
            }

        }

        MPI_Bcast(&q_k_norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        
        MPI_Scatterv(&r_k, counts_v, indices_v, MPI_DOUBLE,
                &local_data_v_r_k, local_bfr_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(&r_k_1, counts_v, indices_v, MPI_DOUBLE,
                &local_data_v_r_k_1, local_bfr_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(&temp, counts_v, indices_v, MPI_DOUBLE,
                &local_data_v_temp, local_bfr_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Scatterv(&q_k, counts_v, indices_v, MPI_DOUBLE,
                &local_data_v_q_k, local_bfr_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        //************************** calculate q_k, r_k_1 and r_k vectors*************************+
        for (int i = 0; i < local_bfr_size; i++)
        {
            local_data_v_r_k[i] = local_data_v_q_k[i]/q_k_norm;
            local_data_v_temp[i] = local_data_v_r_k_1[i];
            local_data_v_r_k_1[i] = local_data_v_r_k[i];

        }
        // check for convergence - commented to do scaling 
        //     if(fabs(local_data_v_temp[i]-local_data_v_r_k[i]) < 0.00001){
        //         stop = 1;
        //     }else stop = 0;
        // }
        MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_data_v_q_k, local_bfr_size, MPI_DOUBLE, 
                q_k, counts_v, indices_v, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_data_v_temp, local_bfr_size, MPI_DOUBLE, 
                temp, counts_v, indices_v, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_data_v_r_k_1, local_bfr_size, MPI_DOUBLE, 
                r_k_1, counts_v, indices_v, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(local_data_v_r_k, local_bfr_size, MPI_DOUBLE, 
                r_k, counts_v, indices_v, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0)
        {   
            if (test_and_debug == 1){
                printf("\nr_k after Gather_inside_loop\n");
                for (i = 0; i < NUM_PAGES; i++) {
                    sum+=r_k[i];
                    printf("%8.2f  ", r_k[i]);
                }  
            } 
            printf("\nFinished iteration: %d\n\n", v);
        }

        v++;
        if (v>ITERR){
            stop = 1;
        }
        MPI_Bcast(&stop, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    if (rank == 0)
    {
        sum= 0;
        printf("\nFinal r_k (r) after Gather:\n");
        for (i = 0; i < NUM_PAGES; i++) {
            sum+=r_k[i];
            printf("%f  ", r_k[i]);
        }   
        printf("\nSum of r vector: %8.2f\n", sum);
    }

    // *********************************************** largest eigenvalue calculation *******************************
    local_data_M[local_bfr_size*NUM_PAGES] = {};
    local_data_v_temp[local_bfr_size] = {};
    local_data_v_r_k[local_bfr_size] = {};
    temp[NUM_PAGES] = {};

    //Scatter P, r_k 
    MPI_Scatterv(&P, counts_M, indices_M, MPI_DOUBLE,
                 &local_data_M, local_bfr_size * NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&temp, counts_v, indices_v, MPI_DOUBLE,
                &local_data_v_temp, local_bfr_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(&r_k, counts_v, indices_v, MPI_DOUBLE,
                &local_data_v_r_k, local_bfr_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&r_k, NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    col_R = 0;
    if (rank == 0){
        row_R = 0;
    }else{
        row_R = (NUM_PAGES/size)*rank;
    }
    
    sum = 0;
    vector_index = 0;
    local_data_v_temp[vector_index] = 0;
    // matrix vector multiplicatin
    for (int i = 0; i < NUM_PAGES * local_bfr_size; i++)
    {
        if ((i % NUM_PAGES) == 0 && (i!=0))
        {
            row_R += 1;
            col_R = 0;
            sum = 0;
            vector_index += 1;
            local_data_v_temp[vector_index] = 0;
        }
        // matrix vector multiplication (P*r_k)
        sum += ((local_data_M[i]*r_k[col_R]));
        // element wise product (r_k * P * r_k) before summation
        local_data_v_temp[vector_index] = sum*r_k[row_R];
        col_R += 1;
    }

    local_sum = 0;
    // find local sum of r_k * P * r_k vector - to get dot product
    for(i = 0; i < local_bfr_size; i++){

        local_sum+=local_data_v_temp[i];
    }
    // gather numerator for Rayleigh-Quotient
    MPI_Gatherv(local_data_v_temp, local_bfr_size, MPI_DOUBLE, temp, counts_v, indices_v, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    vector_sum = 0;
    //Reduce the local buffer values of r_k * P * r_k vector 
    MPI_Reduce(&local_sum, &vector_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // find l2 norm of r_k
    local_sum = 0;
    for(i = 0; i < local_bfr_size; i++){
        local_sum+=local_data_v_r_k[i]*local_data_v_r_k[i];
    }

    //Reduce the local buffer to l2 norm
    MPI_Reduce(&local_sum, &l2_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0)
    {
        // Rayleigh-Quotient
        lambda_max = vector_sum/l2_norm;
        printf("\nlambda_max: %8.2f\n", lambda_max);
        printf("\nNo. of ranks: %d\n", size);
        
    }

    MPI_Barrier(MPI_COMM_WORLD);
    end_time = MPI_Wtime();
    duration = end_time - start_time;
    MPI_Reduce(&duration,&total_time,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

    if (rank == 0) printf("Running Time = %f\n", total_time);
    MPI_Finalize(); //finalize MPI operations

    return 0;
}

