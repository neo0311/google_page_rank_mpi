/**********************************************************************************************
* Matrix Multiplication Program using MPI.
*
* Viraj Brian Wijesuriya - University of Colombo School of Computing, Sri Lanka.
* 
* Works with any type of two matrixes [A], [B] which could be multiplied to produce a matrix [c].
*
* Master process initializes the multiplication operands, distributes the muliplication 
* operation to worker processes and reduces the worker results to construct the final output.
*  
************************************************************************************************/

#include<stdio.h>
#include<mpi.h>
#define NUM_PAGES 6
#define M_C_TAG 1 //tag for messages sent from master to childs
#define C_M_TAG 4 //tag for messages sent from childs to master
void makeAB(); //makes the [A] and [B] matrixes
void printArray(); //print the content of output matrix [C];

int rank; //process rank
int size; //number of processes
int i, j, k; //helper variables
//double L[NUM_PAGES][NUM_PAGES] = {1}; //declare input [A]
double L[NUM_PAGES][NUM_PAGES] = {0}; //temp
double Q[NUM_PAGES][NUM_PAGES] = {0};



double r[NUM_PAGES] = {1 / double(NUM_PAGES)};
double nLinks[NUM_PAGES] = {0};
double sum;
double e[NUM_PAGES] = {1};
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

    /* master initializes work*/
    if (rank == 0) {

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
        start_time = MPI_Wtime();

        // Send the initialised L matrix to child processes.
        for (i = 1; i < size; i++) {
            chunk_size = (NUM_PAGES / (size - 1)); // calculate chunk_size without master
            index_start = (i - 1) * chunk_size;
            if (((i + 1) == size) && ((NUM_PAGES % (size - 1)) != 0)) {//if rows cannot be equally divided among childs
                index_end = NUM_PAGES; //last child gets all the remaining rows
            } else {
                index_end = index_start + chunk_size; //rows are equally divisable among childs
            }
            //send the first index first without blocking, to the intended child
            MPI_Isend(&index_start, 1, MPI_INT, i, M_C_TAG, MPI_COMM_WORLD, &request);
            //next send the last index without blocking, to the intended child
            MPI_Isend(&index_end, 1, MPI_INT, i, M_C_TAG + 1, MPI_COMM_WORLD, &request);
            //finally send the allocated row chunk_size of [A] without blocking, to the intended child
            MPI_Isend(&L[index_start][0], (index_end - index_start) * NUM_PAGES, MPI_DOUBLE, i, M_C_TAG + 2, MPI_COMM_WORLD, &request);

        }

        // Receive processed data from child processes and assemble it
        for (i = 1; i < size; i++) {

            //receive first index from a child
            MPI_Recv(&index_start, 1, MPI_INT, i, C_M_TAG, MPI_COMM_WORLD, &status);
            //receive last index from a child
            MPI_Recv(&index_end, 1, MPI_INT, i, C_M_TAG + 1, MPI_COMM_WORLD, &status);
            //receive processed data from a child
            MPI_Recv(&L[index_start][0], (index_end - index_start) * NUM_PAGES, MPI_DOUBLE, i, C_M_TAG + 2, MPI_COMM_WORLD, &status);

        }

        // Send assembled L matrix to find number of links
        for (i = 1; i < size; i++) {//for each child other than the master
            chunk_size = (NUM_PAGES / (size - 1)); // calculate chunk_size without master
            index_start = (i - 1) * chunk_size;
            if (((i + 1) == size) && ((NUM_PAGES % (size - 1)) != 0)) {//if rows of [A] cannot be equally divided among childs
                index_end = NUM_PAGES; //last child gets all the remaining rows
            } else {
                index_end = index_start + chunk_size; //rows of [A] are equally divisable among childs
            }
            //finally send the allocated column chunk_size of [A] without blocking, to the intended child
            MPI_Isend(&L[0][index_start], (index_end - index_start) * NUM_PAGES, MPI_DOUBLE, i, M_C_TAG + 3, MPI_COMM_WORLD, &request);
            MPI_Isend(&nLinks[index_start], (index_end - index_start), MPI_DOUBLE, i, M_C_TAG + 4, MPI_COMM_WORLD, &request);
        
        }

        // Receive assembled nLinks vector
        for (i = 1; i < size; i++)
        { // untill all childs have handed back the processed data
            MPI_Recv(&nLinks[index_start], (index_end - index_start), MPI_DOUBLE, i, C_M_TAG + 4, MPI_COMM_WORLD, &status);
        }

        // Send nLinks vector to all processes
        MPI_Bcast(&nLinks, NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        // Send assembled L matrix to all processes
        MPI_Bcast(&L, NUM_PAGES*NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Send the initialised Q matrix to child processes.
        for (i = 1; i < size; i++) {
            chunk_size = (NUM_PAGES / (size - 1)); // calculate chunk_size without master
            index_start = (i - 1) * chunk_size;
            if (((i + 1) == size) && ((NUM_PAGES % (size - 1)) != 0)) {//if rows cannot be equally divided among childs
                index_end = NUM_PAGES; //last child gets all the remaining rows
            } else {
                index_end = index_start + chunk_size; //rows are equally divisable among childs
            }
            //send the first index first without blocking, to the intended child
            MPI_Isend(&index_start, 1, MPI_INT, i, M_C_TAG+5, MPI_COMM_WORLD, &request);
            //next send the last index without blocking, to the intended child
            MPI_Isend(&index_end, 1, MPI_INT, i, M_C_TAG + 6, MPI_COMM_WORLD, &request);
            //finally send the allocated row chunk_size of [A] without blocking, to the intended child
            MPI_Isend(&Q[index_start][0], (index_end - index_start) * NUM_PAGES, MPI_DOUBLE, i, M_C_TAG + 7, MPI_COMM_WORLD, &request);

        }

        // Receive processed Q matrix from child processes and assemble it
        for (i = 1; i < size; i++) {

            //receive first index from a child
            MPI_Recv(&index_start, 1, MPI_INT, i, C_M_TAG + 5, MPI_COMM_WORLD, &status);
            //receive last index from a child
            MPI_Recv(&index_end, 1, MPI_INT, i, C_M_TAG + 6, MPI_COMM_WORLD, &status);
            //receive processed data from a child
            MPI_Recv(&Q[index_start][0], (index_end - index_start) * NUM_PAGES, MPI_DOUBLE, i, C_M_TAG + 7, MPI_COMM_WORLD, &status);

        }

        // d vector
        // Send assembled L matrix to find number of links
        for (i = 1; i < size; i++) {//for each child other than the master
            chunk_size = (NUM_PAGES / (size - 1)); // calculate chunk_size without master
            index_start = (i - 1) * chunk_size;
            if (((i + 1) == size) && ((NUM_PAGES % (size - 1)) != 0)) {//if rows of [A] cannot be equally divided among childs
                index_end = NUM_PAGES; //last child gets all the remaining rows
            } else {
                index_end = index_start + chunk_size; //rows of [A] are equally divisable among childs
            }
            //finally send the allocated column chunk_size of [A] without blocking, to the intended child
            MPI_Isend(&Q[0][index_start], (index_end - index_start) * NUM_PAGES, MPI_DOUBLE, i, M_C_TAG + 8, MPI_COMM_WORLD, &request);
            MPI_Isend(&d[index_start], (index_end - index_start), MPI_DOUBLE, i, M_C_TAG + 9, MPI_COMM_WORLD, &request);
        
        }

        // Receive assembled q vector
        for (i = 1; i < size; i++)
        { // untill all childs have handed back the processed data
            MPI_Recv(&d[index_start], (index_end - index_start), MPI_DOUBLE, i, C_M_TAG + 9, MPI_COMM_WORLD, &status);
        }


        end_time = MPI_Wtime();
        printf("\nRunning Time = %f\n\n", end_time - start_time);
        printArray();
    }
    //MPI_Bcast(&nLinks, NUM_PAGES, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    /* work done by childs*/
    if (rank > 0) {

        //**************** Asemble the L matrix****************************************************+
        //receive first index from the master
        MPI_Recv(&index_start, 1, MPI_INT, 0, M_C_TAG, MPI_COMM_WORLD, &status);
        //next receive last index from the master
        MPI_Recv(&index_end, 1, MPI_INT, 0, M_C_TAG + 1, MPI_COMM_WORLD, &status);
        //finally receive row chunk_size of [A] to be processed from the master
        MPI_Recv(&L[index_start][0], (index_end - index_start) * NUM_PAGES, MPI_DOUBLE, 0, M_C_TAG + 2, MPI_COMM_WORLD, &status);
        // for (i = index_start; i < index_end; i++) {//iterate through a given set of rows of [A]
        //     for (j = 0; j < NUM_PAGES; j++) 
        //     {
                
                
        //         if (i == j) {
        //             L[i][j] = 0;
        //         }else {

        //             L[i][j] = 1;

        //             }
        //     }
        //     for (j = 0; j < rnd_zeros;)
        //     {

        //         int index = rand() % ((NUM_PAGES) - 1);
        //         std::cout << index << std::endl;
        //         if (index != i)
        //         {

        //             L[i][index] = 0;
        //             j++;
        //         }
        //     }
        // }

        // send back the first index first without blocking, to the master
        MPI_Isend(&index_start, 1, MPI_INT, 0, C_M_TAG, MPI_COMM_WORLD, &request);
        //send the last index next without blocking, to the master
        MPI_Isend(&index_end, 1, MPI_INT, 0, C_M_TAG + 1, MPI_COMM_WORLD, &request);
        //finally send the processed chunk_size of data without blocking, to the master
        MPI_Isend(&L[index_start][0], (index_end - index_start) * NUM_PAGES, MPI_DOUBLE, 0, C_M_TAG + 2, MPI_COMM_WORLD, &request);
        
        // #Links calculation
        // recieve columns from master to calculate number of links in each page
        MPI_Recv(&L[0][index_start], (index_end - index_start) * NUM_PAGES, MPI_DOUBLE, 0, M_C_TAG + 3, MPI_COMM_WORLD, &status);
        // recieve chunks of nLInks array to store the number of links.
        MPI_Recv(&nLinks[index_start], (index_end - index_start), MPI_DOUBLE, 0, M_C_TAG + 4, MPI_COMM_WORLD, &status);

        
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
        for (i = 0; i< index_end; i++){

            std::cout << nLinks[i] << std::endl;
        }
        std::cout << "\n" << std::endl; 

        MPI_Isend(&nLinks[index_start], (index_end - index_start), MPI_DOUBLE, 0, C_M_TAG + 4, MPI_COMM_WORLD, &request);


        //**************** Asemble the Q matrix****************************************************+
        //receive first index from the master
        MPI_Recv(&index_start, 1, MPI_INT, 0, M_C_TAG + 5, MPI_COMM_WORLD, &status);
        //next receive last index from the master
        MPI_Recv(&index_end, 1, MPI_INT, 0, M_C_TAG + 6, MPI_COMM_WORLD, &status);
        //finally receive row chunk_size of [A] to be processed from the master
        MPI_Recv(&Q[index_start][0], (index_end - index_start) * NUM_PAGES, MPI_DOUBLE, 0, M_C_TAG + 7, MPI_COMM_WORLD, &status);

        for (i = index_start; i < index_end; i++) {//iterate through a given set of rows of [A]
            for (j = 0; j < NUM_PAGES; j++) 
            {
                
                
                if (L[i][j] == 0)
                {

                    continue;
                }
                Q[i][j] = (1 / double(nLinks[j])) * L[i][j];
            }

        
        }

        // send back the first index first without blocking, to the master
        MPI_Isend(&index_start, 1, MPI_INT, 0, C_M_TAG + 5, MPI_COMM_WORLD, &request);
        //send the last index next without blocking, to the master
        MPI_Isend(&index_end, 1, MPI_INT, 0, C_M_TAG + 6, MPI_COMM_WORLD, &request);
        //finally send the processed chunk_size of data without blocking, to the master
        MPI_Isend(&Q[index_start][0], (index_end - index_start) * NUM_PAGES, MPI_DOUBLE, 0, C_M_TAG + 7, MPI_COMM_WORLD, &request);


        // #d[] calculation
        // recieve columns from master to calculate number of links in each page
        MPI_Recv(&Q[0][index_start], (index_end - index_start) * NUM_PAGES, MPI_DOUBLE, 0, M_C_TAG + 8, MPI_COMM_WORLD, &status);
        // recieve chunks of nLInks array to store the number of links.
        MPI_Recv(&d[index_start], (index_end - index_start), MPI_DOUBLE, 0, M_C_TAG + 9, MPI_COMM_WORLD, &status);
        //filling d

        for (int column = index_start; column < index_end; column++)
        {   
            bool flag = false;
            for (int row = 0; row < NUM_PAGES; row++)
            {   
                if (Q[row][column] != 0)
                {

                    flag = true;

                }
            }

            if (flag == false)
            {
                d[column] = 1;
                printf("Hello");
            }
        }
        MPI_Isend(&d[index_start], (index_end - index_start), MPI_DOUBLE, 0, C_M_TAG + 9, MPI_COMM_WORLD, &request);
    }
    /* master gathers processed work*/
    // if (rank == 0) {
    //     // for (i = 1; i < size; i++) {// untill all childs have handed back the processed data

    //     //     //receive first index from a child
    //     //     MPI_Recv(&index_start, 1, MPI_INT, i, C_M_TAG, MPI_COMM_WORLD, &status);
    //     //     //receive last index from a child
    //     //     MPI_Recv(&index_end, 1, MPI_INT, i, C_M_TAG + 1, MPI_COMM_WORLD, &status);
    //     //     //receive processed data from a child
    //     //     MPI_Recv(&L[index_start][0], (index_end - index_start) * NUM_PAGES, MPI_DOUBLE, i, C_M_TAG + 2, MPI_COMM_WORLD, &status);

    //     // }

    //     for (i = 1; i < size; i++)
    //     { // untill all childs have handed back the processed data
    //         MPI_Recv(&nLinks[index_start], (index_end - index_start), MPI_DOUBLE, i, C_M_TAG + 4, MPI_COMM_WORLD, &status);
    //     }

    //     end_time = MPI_Wtime();
    //     printf("\nRunning Time = %f\n\n", end_time - start_time);
    //     printArray();
    // }

    MPI_Finalize(); //finalize MPI operations
    return 0;
}

void printArray()
{
    printf("\n\nL\n");
    for (i = 0; i < NUM_PAGES; i++) {
        printf("\n\n");
        for (j = 0; j < NUM_PAGES; j++)
            printf("%8.2f  ", L[i][j]);
    }
    printf("\n\nnLinks\n");

    for (i = 0; i < NUM_PAGES; i++){
        printf("%8.2f  ", nLinks[i]);
    }
    printf("\nQ\n");

    for (i = 0; i < NUM_PAGES; i++) {
        printf("\n");
        for (j = 0; j < NUM_PAGES; j++)
            printf("%8.2f  ", Q[i][j]);
    }

    printf("\n\nd\n");

    for (i = 0; i < NUM_PAGES; i++){
        printf("%8.2f  ", d[i]);
    }
}   
