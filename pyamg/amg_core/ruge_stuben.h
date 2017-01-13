#ifndef RUGE_STUBEN_H
#define RUGE_STUBEN_H

#include <iostream>
#include <vector>
#include <iterator>
#include <cassert>
#include <limits>
#include <algorithm>

#include "linalg.h"
#include "graph.h"


/*
 *  Compute a strength of connection matrix using the classical strength
 *  of connection measure by Ruge and Stuben. Both the input and output
 *  matrices are stored in CSR format.  An off-diagonal nonzero entry
 *  A[i,j] is considered strong if:
 *
 *      -A[i,j] >= theta * max( -A[i,k] )   where k != i
 *
 * Otherwise, the connection is weak.
 *
 *  Parameters
 *      num_rows   - number of rows in A
 *      theta      - stength of connection tolerance
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      Ax[]       - CSR data array
 *      Sp[]       - (output) CSR row pointer
 *      Sj[]       - (output) CSR index array
 *      Sx[]       - (output) CSR data array
 *
 *
 *  Returns:
 *      Nothing, S will be stored in Sp, Sj, Sx
 *
 *  Notes:
 *      Storage for S must be preallocated.  Since S will consist of a subset
 *      of A's nonzero values, a conservative bound is to allocate the same
 *      storage for S as is used by A.
 *
 */
template<class I, class T, class F>
void classical_strength_of_connection(const I n_row,
                                      const F theta,
                                      const I Ap[], const int Ap_size,
                                      const I Aj[], const int Aj_size,
                                      const T Ax[], const int Ax_size,
                                            I Sp[], const int Sp_size,
                                            I Sj[], const int Sj_size,
                                            T Sx[], const int Sx_size)
{
    I nnz = 0;
    Sp[0] = 0;

    for(I i = 0; i < n_row; i++){
        F max_offdiagonal = std::numeric_limits<F>::min();

        const I row_start = Ap[i];
        const I row_end   = Ap[i+1];

        for(I jj = row_start; jj < row_end; jj++){
            if(Aj[jj] != i){
                max_offdiagonal = std::max(max_offdiagonal,mynorm(Ax[jj]));
            }
        }

        F threshold = theta*max_offdiagonal;
        for(I jj = row_start; jj < row_end; jj++){
            F norm_jj = mynorm(Ax[jj]);

            // Add entry if it exceeds the threshold
            if(norm_jj >= threshold){
                if(Aj[jj] != i){
                    Sj[nnz] = Aj[jj];
                    Sx[nnz] = Ax[jj];
                    nnz++;
                }
            }

            // Always add the diagonal
            if(Aj[jj] == i){
                Sj[nnz] = Aj[jj];
                Sx[nnz] = Ax[jj];
                nnz++;
            }
        }

        Sp[i+1] = nnz;
    }
}

/*
 *  Compute the maximum in magnitude row value for a CSR matrix
 *
 *  Parameters
 *      num_rows   - number of rows in A
 *      Ap[]       - CSR row pointer
 *      Aj[]       - CSR index array
 *      Ax[]       - CSR data array
 *       x[]       - num_rows array
 *
 *  Returns:
 *      Nothing, x[i] will hold row i's maximum magnitude entry
 *
 */
template<class I, class T, class F>
void maximum_row_value(const I n_row,
                              T x[], const int  x_size,
                       const I Ap[], const int Ap_size,
                       const I Aj[], const int Aj_size,
                       const T Ax[], const int Ax_size)
{

    for(I i = 0; i < n_row; i++){
        F max_entry = std::numeric_limits<F>::min();

        const I row_start = Ap[i];
        const I row_end   = Ap[i+1];

        // Find this row's max entry
        for(I jj = row_start; jj < row_end; jj++){
            max_entry = std::max(max_entry, mynorm(Ax[jj]) );
        }

        x[i] = max_entry;
    }
}



#define F_NODE 0
#define C_NODE 1
#define U_NODE 2
#define PRE_F_NODE 3

/*
 * Compute a C/F (coarse-fine( splitting using the classical coarse grid
 * selection method of Ruge and Stuben.  The strength of connection matrix S,
 * and its transpose T, are stored in CSR format.  Upon return, the  splitting
 * array will consist of zeros and ones, where C-nodes (coarse nodes) are
 * marked with the value 1 and F-nodes (fine nodes) with the value 0.
 *
 * Parameters:
 *   n_nodes   - number of rows in A
 *   Sp[]      - CSR pointer array
 *   Sj[]      - CSR index array
 *   Tp[]      - CSR pointer array
 *   Tj[]      - CSR index array
 *   influence - array that influences splitting (values stored here are added to lambda for each point)
 *   splitting - array to store the C/F splitting
 *
 * Notes:
 *   The splitting array must be preallocated
 *
 */
template<class I>
void rs_cf_splitting(const I n_nodes,
                     const I Sp[], const int Sp_size,
                     const I Sj[], const int Sj_size,
                     const I Tp[], const int Tp_size,
                     const I Tj[], const int Tj_size,
                     const I influence[], const int influence_size,
                           I splitting[], const int splitting_size)
{
  // printf("Entered CF splitting\n");
    std::vector<I> lambda(n_nodes,0);

    //compute lambdas
    // printf("   Compute lambdas\n");
    I lambda_max = 0;
    for(I i = 0; i < n_nodes; i++){
        lambda[i] = Tp[i+1] - Tp[i] + influence[i];
        if (lambda[i] > lambda_max) lambda_max = lambda[i];
        // if (lambda[i] > n_nodes)
        // {
        //   printf("Lamda was set too large\n");
        //   lambda[i] = n_nodes;
        // }
    }

    //for each value of lambda, create an interval of nodes with that value
    // ptr - is the first index of the interval
    // count - is the number of indices in that interval
    // index to node - the node located at a given index
    // node to index - the index of a given node
    lambda_max = lambda_max*2;
    if (n_nodes+1 > lambda_max) lambda_max = n_nodes+1;
    // printf("lambda_max = %d\n", lambda_max);
    std::vector<I> interval_ptr(lambda_max,0);
    std::vector<I> interval_count(lambda_max,0);
    std::vector<I> index_to_node(n_nodes);
    std::vector<I> node_to_index(n_nodes);

    for(I i = 0; i < n_nodes; i++){
        interval_count[lambda[i]]++;
    }
    for(I i = 0, cumsum = 0; i < lambda_max; i++){
        interval_ptr[i] = cumsum;
        cumsum += interval_count[i];
        interval_count[i] = 0;
    }
    for(I i = 0; i < n_nodes; i++){
        I lambda_i = lambda[i];
        // printf("lambda_i = %d, lambda_max = %d\n", lambda_i, lambda_max);
        I index    = interval_ptr[lambda_i] + interval_count[lambda_i];
        index_to_node[index] = i;
        node_to_index[i]     = index;
        interval_count[lambda_i]++;
    }


    std::fill(splitting, splitting + n_nodes, U_NODE);
    // printf("Did the fill\n");

    // all nodes with no neighbors become F nodes
    for(I i = 0; i < n_nodes; i++){
        if (lambda[i] == 0 || (lambda[i] == 1 && Tj[Tp[i]] == i))
            splitting[i] = F_NODE;
    }

    //Now add elements to C and F, in descending order of lambda
    // printf("   Add elements to C and F in descending order of lambda\n");
    for(I top_index = n_nodes - 1; top_index != -1; top_index--){
        I i        = index_to_node[top_index];
        I lambda_i = lambda[i];
        // printf("lambda_i = %d, lambda_max = %d\n", lambda_i, lambda_max);

        // if (i == 73)
        // {
        //   std::cout << "LOOK HERE: i = 73, top_index = " << top_index << std::endl;
        //   for (I j = n_nodes - 1; j > -1; j--)
        //   {
        //     std::cout << "             index: " << j << ", node: " << index_to_node[j] << ", lamda: " << lambda[index_to_node[j]] << std::endl;
        //   }
        // }

        //remove i from its interval
        interval_count[lambda_i]--;

        if(splitting[i] == F_NODE)
        {        
          // if (n_nodes == 121)
            // std::cout << "Fine node #" << i << " with lambda " << lambda[i] << std::endl;
            continue;
        }
        else
        {
            assert(splitting[i] == U_NODE);

            // Search over this interval to make sure we process nodes in descending node order
            I max_node = i;
            I max_index = top_index;
            for (I j = interval_ptr[lambda_i]; j < interval_ptr[lambda_i] + interval_count[lambda_i]; j++)
            {
              if (index_to_node[j] > max_node)
              {
                max_node = index_to_node[j];
                max_index = j;
              }
            }

            node_to_index[index_to_node[top_index]] = max_index;
            node_to_index[index_to_node[max_index]] = top_index;


            std::swap(index_to_node[top_index], index_to_node[max_index]);
            i = index_to_node[top_index];


            splitting[i] = C_NODE;
            // if (n_nodes == 121)
              // std::cout << "Coarse node #" << i << " with lambda " << lambda[i] << std::endl;

            //For each j in S^T_i /\ U
            for(I jj = Tp[i]; jj < Tp[i+1]; jj++)
            {
                I j = Tj[jj];
                if(splitting[j] == U_NODE) splitting[j] = PRE_F_NODE;
            }

            for(I jj = Tp[i]; jj < Tp[i+1]; jj++)
            {
                I j = Tj[jj];
                if(splitting[j] == PRE_F_NODE)
                {
                    splitting[j] = F_NODE;
                    //For each k in S_j /\ U
                    for(I kk = Sp[j]; kk < Sp[j+1]; kk++){
                        I k = Sj[kk];

                        if(splitting[k] == U_NODE){
                            //move k to the end of its current interval
                            if(lambda[k] >= n_nodes - 1) continue;

                            // TODO make this robust
                            // if(lambda[k] >= n_nodes -1)
                            //    std::cout << std::endl << "lambda[" << k << "]=" << lambda[k] << " n_nodes=" << n_nodes << std::endl;
                            // assert(lambda[k] < n_nodes - 1);//this would cause problems!

                            I lambda_k = lambda[k];
                            // printf("lambda_k = %d, lambda_max = %d\n", lambda_k, lambda_max);
                            I old_pos  = node_to_index[k];
                            I new_pos  = interval_ptr[lambda_k] + interval_count[lambda_k] - 1;

                            node_to_index[index_to_node[old_pos]] = new_pos;
                            node_to_index[index_to_node[new_pos]] = old_pos;
                            std::swap(index_to_node[old_pos], index_to_node[new_pos]);

                            //update intervals
                            interval_count[lambda_k]   -= 1;
                            interval_count[lambda_k+1] += 1; //invalid write!
                            interval_ptr[lambda_k+1]    = new_pos;

                            //increment lambda_k
                            lambda[k]++;
                            // if (lambda[i] > n_nodes)
                            // {
                            //   printf("Lamda was set too large\n");
                            //   lambda[i] = n_nodes;
                            // }
                            // if (n_nodes == 121)
                              // std::cout << "    increment node #" << k << " to lambda " << lambda[k] << std::endl;

                        }
                    }
                }
            }

            //For each j in S_i /\ U
            for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
                I j = Sj[jj];
                if(splitting[j] == U_NODE){            //decrement lambda for node j
                    if(lambda[j] == 0) continue;

                    //assert(lambda[j] > 0);//this would cause problems!

                    //move j to the beginning of its current interval
                    I lambda_j = lambda[j];
                    // printf("lambda_j = %d, lambda_max = %d\n", lambda_j, lambda_max);
                    I old_pos  = node_to_index[j];
                    I new_pos  = interval_ptr[lambda_j];

                    node_to_index[index_to_node[old_pos]] = new_pos;
                    node_to_index[index_to_node[new_pos]] = old_pos;
                    std::swap(index_to_node[old_pos],index_to_node[new_pos]);

                    //update intervals
                    interval_count[lambda_j]   -= 1;
                    interval_count[lambda_j-1] += 1;
                    interval_ptr[lambda_j]     += 1;
                    interval_ptr[lambda_j-1]    = interval_ptr[lambda_j] - interval_count[lambda_j-1];

                    //decrement lambda_j
                    lambda[j]--;
                    // std::cout << "    decrement node #" << j << " to lambda " << lambda[j] << std::endl;
                }
            }
        }
    }
  // printf("Leaving CF splitting\n");
}


/*
 *  Compute a CLJP splitting
 *
 *  Parameters
 *      n          - number of rows in A (number of vertices)
 *      Sp[]       - CSR row pointer (strength matrix)
 *      Sj[]       - CSR index array
 *      Tp[]       - CSR row pointer (transpose of the strength matrix)
 *      Tj[]       - CSR index array
 *      splitting  - array to store the C/F splitting
 *      colorflag  - flag to indicate coloring
 *
 *  Notes:
 *      The splitting array must be preallocated.
 *      CLJP naive since it requires the transpose.
 */

template<class I>
void cljp_naive_splitting(const I n,
                          const I Sp[], const int Sp_size,
                          const I Sj[], const int Sj_size,
                          const I Tp[], const int Tp_size,
                          const I Tj[], const int Tj_size,
                                I splitting[], const int splitting_size,
                          const I colorflag)
{
  // initialize sizes
  int ncolors;
  I unassigned = n;
  I nD;
  int nnz = Sp[n];

  // initialize vectors
  // complexity = 5n
  // storage = 4n
  std::vector<int> edgemark(nnz,1);
  std::vector<int> coloring(n);
  std::vector<double> weight(n);
  std::vector<I> D(n,0);      // marked nodes  in the ind set
  std::vector<I> Dlist(n,0);      // marked nodes  in the ind set
  std::fill(splitting, splitting + n, U_NODE);
  int * c_dep_cache = new int[n];
  std::fill_n(c_dep_cache, n, -1);

  // INITIALIZE WEIGHTS
  // complexity = O(n^2)?!? for coloring
  // or
  // complexity = n for random
  if(colorflag==1){ // with coloring
    //vertex_coloring_jones_plassmann(n, Sp, Sj, &coloring[0],&weight[0]);
    //vertex_coloring_IDO(n, Sp, Sj, &coloring[0]);
    vertex_coloring_mis(n, Sp, Sp_size, Sj, Sj_size, &coloring[0], n);
    ncolors = *std::max_element(coloring.begin(), coloring.end()) + 1;
    for(I i=0; i < n; i++){
      weight[i] = double(coloring[i])/double(ncolors);
    }
  }
  else {
    srand(2448422);
    for(I i=0; i < n; i++){
      weight[i] = double(rand())/RAND_MAX;
    }
  }

  for(I i=0; i < n; i++){
    for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
      I j = Sj[jj];
      if(i != j) {
        weight[j]++;
      }
    }
  }
  // end INITIALIZE WEIGHTS

  // SELECTION LOOP
  I pass = 0;
  while(unassigned > 0){
    pass++;

    // SELECT INDEPENDENT SET
    // find i such that w_i > w_j for all i in union(S_i,S_i^T)
    nD = 0;
    for(I i=0; i<n; i++){
      if(splitting[i]==U_NODE){
        D[i] = 1;
        // check row (S_i^T)
        for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
          I j = Sj[jj];
          if(splitting[j]==U_NODE && weight[j]>weight[i]){
            D[i] = 0;
            break;
          }
        }
        // check col (S_i)
        if(D[i] == 1) {
          for(I jj = Tp[i]; jj < Tp[i+1]; jj++){
            I j = Tj[jj];
            if(splitting[j]==U_NODE && weight[j]>weight[i]){
              D[i] = 0;
              break;
            }
          }
        }
        if(D[i] == 1) {
          Dlist[nD] = i;
          unassigned--;
          nD++;
        }
      }
      else{
        D[i]=0;
      }
    } // end for
    for(I i = 0; i < nD; i++) {
      splitting[Dlist[i]] = C_NODE;
    }
    // end SELECT INDEPENDENT SET

    // UPDATE WEIGHTS
    // P5
    // nbrs that influence C points are not good C points
    for(I iD=0; iD < nD; iD++){
      I c = Dlist[iD];
      for(I jj = Sp[c]; jj < Sp[c+1]; jj++){
        I j = Sj[jj];
        // c <---j
        if(splitting[j]==U_NODE && edgemark[jj] != 0){
          edgemark[jj] = 0;  // "remove" edge
          weight[j]--;
          if(weight[j]<1){
            splitting[j] = F_NODE;
            unassigned--;
          }
        }
      }
    } // end P5

    // P6
    // If k and j both depend on c, a C point, and j influces k, then j is less
    // valuable as a C point.
    for(I iD=0; iD < nD; iD++){
      I c = Dlist[iD];
      for(I jj = Tp[c]; jj < Tp[c+1]; jj++){
        I j = Tj[jj];
        if(splitting[j]==U_NODE)                 // j <---c
          c_dep_cache[j] = c;
      }

      for(I jj = Tp[c]; jj < Tp[c+1]; jj++) {
        I j = Tj[jj];
        for(I kk = Sp[j]; kk < Sp[j+1]; kk++) {
          I k = Sj[kk];
          if(splitting[k] == U_NODE && edgemark[kk] != 0) { // j <---k
            // does c ---> k ?
            if(c_dep_cache[k] == c) {
              edgemark[kk] = 0; // remove edge
              weight[k]--;
              if(weight[k] < 1) {
                splitting[k] = F_NODE;
                unassigned--;
                //kk = Tp[j+1]; // to break second loop
              }
            }
          }
        }
      }
    } // end P6
  }
  // end SELECTION LOOP

  for(I i = 0; i < Sp[n]; i++){
    if(edgemark[i] == 0){
      edgemark[i] = -1;
    }
  }
  for(I i = 0; i < n; i++){
    if(splitting[i] == U_NODE){
      splitting[i] = F_NODE;
    }
  }
  delete[] c_dep_cache;
}


/*
 *   Produce the Ruge-Stuben prolongator using "Direct Interpolation"
 *
 *
 *   The first pass uses the strength of connection matrix 'S'
 *   and C/F splitting to compute the row pointer for the prolongator.
 *
 *   The second pass fills in the nonzero entries of the prolongator
 *
 *   Reference:
 *      Page 479 of "Multigrid"
 *
 */
template<class I>
void rs_direct_interpolation_pass1(const I n_nodes,
                                   const I Sp[], const int Sp_size,
                                   const I Sj[], const int Sj_size,
                                   const I splitting[], const int splitting_size,
                                         I Bp[], const int Bp_size)
{
    I nnz = 0;
    Bp[0] = 0;
    for(I i = 0; i < n_nodes; i++){
        if( splitting[i] == C_NODE ){
            nnz++;
        } else {
            for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
                if ( (splitting[Sj[jj]] == C_NODE) && (Sj[jj] != i) )
                    nnz++;
            }
        }
        Bp[i+1] = nnz;
    }
}


template<class I, class T>
void rs_direct_interpolation_pass2(const I n_nodes,
                                   const I Ap[], const int Ap_size,
                                   const I Aj[], const int Aj_size,
                                   const T Ax[], const int Ax_size,
                                   const I Sp[], const int Sp_size,
                                   const I Sj[], const int Sj_size,
                                   const T Sx[], const int Sx_size,
                                   const I splitting[], const int splitting_size,
                                   const I Bp[], const int Bp_size,
                                         I Bj[], const int Bj_size,
                                         T Bx[], const int Bx_size)
{

    for(I i = 0; i < n_nodes; i++){
        if(splitting[i] == C_NODE){
            Bj[Bp[i]] = i;
            Bx[Bp[i]] = 1;
        } else {
            T sum_strong_pos = 0, sum_strong_neg = 0;
            for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
                if ( (splitting[Sj[jj]] == C_NODE) && (Sj[jj] != i) ){
                    if (Sx[jj] < 0)
                        sum_strong_neg += Sx[jj];
                    else
                        sum_strong_pos += Sx[jj];
                }
            }

            T sum_all_pos = 0, sum_all_neg = 0;
            T diag = 0;
            for(I jj = Ap[i]; jj < Ap[i+1]; jj++){
                if (Aj[jj] == i){
                    diag += Ax[jj];
                } else {
                    if (Ax[jj] < 0)
                        sum_all_neg += Ax[jj];
                    else
                        sum_all_pos += Ax[jj];
                }
            }

            T alpha = sum_all_neg / sum_strong_neg;
            T beta  = sum_all_pos / sum_strong_pos;

            if (sum_strong_pos == 0){
                diag += sum_all_pos;
                beta = 0;
            }

            T neg_coeff = -alpha/diag;
            T pos_coeff = -beta/diag;

            I nnz = Bp[i];
            for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
                if ( (splitting[Sj[jj]] == C_NODE) && (Sj[jj] != i) ){
                    Bj[nnz] = Sj[jj];
                    if (Sx[jj] < 0)
                        Bx[nnz] = neg_coeff * Sx[jj];
                    else
                        Bx[nnz] = pos_coeff * Sx[jj];
                    nnz++;
                }
            }
        }
    }


    std::vector<I> map(n_nodes);
    for(I i = 0, sum = 0; i < n_nodes; i++){
        map[i]  = sum;
        sum    += splitting[i];
    }
    for(I i = 0; i < Bp[n_nodes]; i++){
        Bj[i] = map[Bj[i]];
    }
}


/*
 *   Produce the Ruge-Stuben prolongator using standard interpolation
 *
 *
 *   The first pass uses the strength of connection matrix 'S'
 *   and C/F splitting to compute the row pointer for the prolongator.
 *
 *   The second pass fills in the nonzero entries of the prolongator
 *
 *   Reference:
 *      Page 144 "A Multigrid Tutorial"
 *
 */
template<class I>
void rs_standard_interpolation_pass1(const I n_nodes,
                                   const I Sp[], const int Sp_size,
                                   const I Sj[], const int Sj_size,
                                   const I splitting[], const int splitting_size,
                                         I Bp[], const int Bp_size)
{
    I nnz = 0;
    Bp[0] = 0;
    for(I i = 0; i < n_nodes; i++){
        if( splitting[i] == C_NODE ){
            nnz++;
        } else {
            for(I jj = Sp[i]; jj < Sp[i+1]; jj++){
                if ( (splitting[Sj[jj]] == C_NODE) && (Sj[jj] != i) )
                    nnz++;
            }
        }
        Bp[i+1] = nnz;
    }
}

template<class I, class T>
void rs_standard_interpolation_pass2(const I n_nodes,
                                   const I Ap[], const int Ap_size,
                                   const I Aj[], const int Aj_size,
                                   const T Ax[], const int Ax_size,
                                   const I Sp[], const int Sp_size,
                                   const I Sj[], const int Sj_size,
                                   const T Sx[], const int Sx_size,
                                   const I splitting[], const int splitting_size,
                                   const I Bp[], const int Bp_size,
                                         I Bj[], const int Bj_size,
                                         T Bx[], const int Bx_size)
{

    for(I i = 0; i < n_nodes; i++) {
        // If node i is a C-point, then set interpolation as injection
        if(splitting[i] == C_NODE) {
            Bj[Bp[i]] = i;
            Bx[Bp[i]] = 1;
        } 
        // Otherwise, use RS standard interpolation formula
        else {

            // Calculate denominator
            T denominator = 0;

            // Start by summing entire row of A
            for(I mm = Ap[i]; mm < Ap[i+1]; mm++) {
                denominator += Ax[mm];
            }

            // Then subtract off the strong connections so that you are left with 
            // denominator = a_ii + sum_{m in weak connections} a_im
            for(I mm = Sp[i]; mm < Sp[i+1]; mm++) {
                if ( Sj[mm] != i ) denominator -= Sx[mm]; // making sure to leave the diagonal entry in there
            }

            // Set entries in P (interpolation weights w_ij from strongly connected C-points)
            I nnz = Bp[i];
            for(I jj = Sp[i]; jj < Sp[i+1]; jj++) {
                if ( (splitting[Sj[jj]] == C_NODE) && (Sj[jj] != i) ) {
                    // Set temporary value for Bj to be mapped to appropriate coarse-grid
                    // column index later and get column index j
                    Bj[nnz] = Sj[jj];
                    I j = Sj[jj];

                    // Initialize numerator as a_ij
                    T numerator = Sx[jj];
                    // printf("  numerator initialized to %e\n", numerator);
                    // Sum over strongly connected fine points
                    for(I kk = Sp[i]; kk < Sp[i+1]; kk++) {
                        if ( (splitting[Sj[kk]] == F_NODE) && (Sj[kk] != i) ) {
                            // Get column index k
                            I k = Sj[kk];

                            // Get a_kj (have to search over k'th row in A for connection a_kj)
                            T a_kj = 0;
                            for(I search_ind = Ap[k]; search_ind < Ap[k+1]; search_ind++) {
                                if ( Aj[search_ind] == j ){
                                    a_kj = Ax[search_ind];
                                }
                            }

                            // If a_kj == 0, then we don't need to do any more work, otherwise
                            // proceed to account for node k's contribution
                            if (a_kj != 0) {
                                // Calculate sum for inner denominator (loop over strongly connected C-points)
                                T inner_denominator = 0;
                                I inner_denom_added_to = 0;
                                for(I ll = Sp[i]; ll < Sp[i+1]; ll++) {
                                    if ( (splitting[Sj[ll]] == C_NODE) && (Sj[ll] != i) ) {
                                        // Get column index l
                                        I l = Sj[ll];
                                        // Add connection a_kl if present in matrix (search over kth row in A for connection)
                                        for(I search_ind = Ap[k]; search_ind < Ap[k+1]; search_ind++) {
                                            // Note: we check here to make sure a_kj and a_kl are same sign
                                            if ( Aj[search_ind] == l && a_kj*Ax[search_ind] > 0) {
                                                inner_denom_added_to = 1;
                                                inner_denominator += Ax[search_ind];
                                            }
                                        }
                                    }
                                }

                                // Add a_ik*a_kj/inner_denominator to the numerator 
                                if (inner_denominator == 0 && !inner_denom_added_to) {
                                    printf("Inner denominator was zero: there was a strongly " \
                                        "connected fine point with no connections to points in C_i\n");
                                }
                                if (inner_denominator == 0 && inner_denom_added_to) {
                                    printf("Inner denominator was zero due to cancellations!\n");
                                }
                                numerator += Sx[kk]*a_kj/inner_denominator;
                            }
                        }
                    }
                    // Set w_ij = -numerator/denominator
                    if (denominator == 0) {
                        printf("Outer denominator was zero: diagonal plus sum of weak connections was zero\n");
                    }
                    Bx[nnz] = -numerator/denominator;
                    nnz++;
                }
            }
        }
    }

    std::vector<I> map(n_nodes);
    for(I i = 0, sum = 0; i < n_nodes; i++) {
        map[i]  = sum;
        sum    += splitting[i];
    }
    for(I i = 0; i < Bp[n_nodes]; i++) {
        Bj[i] = map[Bj[i]];
    }
}


/* Helper function for compatible relaxation to perform steps 3.1d - 3.1f
 * in Falgout / Brannick (2010).  
 *
 * Input:
 * ------
 * A_rowptr : const {int array}
 *      Row pointer for sparse matrix in CSR format.
 * A_colinds : const {int array}
 *      Column indices for sparse matrix in CSR format.
 * B : const {float array}
 *      Target near null space vector for computing candidate set measure. 
 * e : {float array}
 *      Relaxed vector for computing candidate set measure.
 * indices : {int array}
 *      Array of indices, where indices[0] = the number of F indices, nf,
 *      followed by F indices in elements 1:nf, and C indices in (nf+1):n.
 * splitting : {int array}
 *      Integer array with current C/F splitting of nodes, 0 = C-point,
 *      1 = F-point. 
 * gamma : {float array}
 *      Preallocated vector to store candidate set measure.  
 * thetacs : const {float}
 *      Threshold for coarse grid candidates from set measure. 
 *
 * Returns:
 * --------  
 * Nothing, updated C/F-splitting and corresponding indices modified in place. 
 */
template<class I, class T>
void cr_helper(const I A_rowptr[], const int A_rowptr_size,
               const I A_colinds[], const int A_colinds_size, 
               const T B[], const int B_size,
               T e[], const int e_size,
               I indices[], const int indices_size,
               I splitting[], const int splitting_size,
               T gamma[], const int gamma_size,
               const T thetacs, 
               T cost[], const int cost_size )
{
    const T &Annz = A_colinds_size;
    const I &n = splitting_size;
    I &num_Fpts = indices[0];

    // Steps 3.1d, 3.1e in Falgout / Brannick (2010)
    // Divide each element in e by corresponding index in initial target vector.
    // Get inf norm of new e.
    T inf_norm = 0;
    for (I i=1; i<(num_Fpts+1); i++) {
        I pt = indices[i];
        e[pt] = std::abs(e[pt] / B[pt]);
        if (e[pt] > inf_norm) {
            inf_norm = e[pt];
        }   
    }
    cost[0] += num_Fpts / Annz;

    // Compute candidate set measure, pick coarse grid candidates.
    std::vector<I> Uindex;
    for (I i=1; i<(num_Fpts+1); i++) {
        I pt = indices[i];
        gamma[pt] = e[pt] / inf_norm; 
        if (gamma[pt] > thetacs) {
            Uindex.push_back(pt);
        }
    }
    I set_size = Uindex.size();
    cost[0] += num_Fpts / Annz;

    // Step 3.1f in Falgout / Brannick (2010)
    // Find weights: omega_i = |N_i\C| + gamma_i
    std::vector<T> omega(n,0);
    for (I i=0; i<set_size; i++) {
        I pt = Uindex[i];
        I num_neighbors = 0;
        I A_ind0 = A_rowptr[pt];
        I A_ind1 = A_rowptr[pt+1];
        for (I j=A_ind0; j<A_ind1; j++) {
            I neighbor = A_colinds[j];
            if (splitting[neighbor] == 0) {
                num_neighbors += 1;
            }
        }
        omega[pt] = num_neighbors + gamma[pt];
    }

    // Form maximum independent set
    while (true) {
        // 1. Add point i in U with maximal weight to C 
        T max_weight = 0;
        I new_pt = -1;
        for (I i=0; i<set_size; i++) {
            I pt = Uindex[i];
            if (omega[pt] > max_weight) {
                max_weight = omega[pt];
                new_pt = pt;
            }
        }
        // If all points have zero weight (index set is empty) break loop
        if (new_pt < 0) {
            break;
        }
        splitting[new_pt] = 1;
        gamma[new_pt] = 0;

        // 2. Remove from candidate set all nodes connected to 
        // new C-point by marking weight zero.
        std::vector<I> neighbors;
        I A_ind0 = A_rowptr[new_pt];
        I A_ind1 = A_rowptr[new_pt+1];
        for (I i=A_ind0; i<A_ind1; i++) {
            I temp = A_colinds[i];
            neighbors.push_back(temp);
            omega[temp] = 0;
        }

        // 3. For each node removed in step 2, set the weight for 
        // each of its neighbors still in the candidate set +1.
        I num_neighbors = neighbors.size();
        for (I i=0; i<num_neighbors; i++) {
            I pt = neighbors[i];
            I A_ind0 = A_rowptr[pt];
            I A_ind1 = A_rowptr[pt+1];
            for (I j=A_ind0; j<A_ind1; j++) {
                I temp = A_colinds[j];
                if (omega[temp] != 0) {
                    omega[temp] += 1;                   
                }
            }
        }
    }

    // Reorder indices array, with the first element giving the number
    // of F indices, nf, followed by F indices in elements 1:nf, and 
    // C indices in (nf+1):n. Note, C indices sorted largest to smallest.
    num_Fpts = 0;
    I next_Find = 1;
    I next_Cind = n;
    for (I i=0; i<n; i++) {
        if (splitting[i] == 0) {
            indices[next_Find] = i;
            next_Find += 1;
            num_Fpts += 1;
        }
        else {
            indices[next_Cind] = i;
            next_Cind -= 1;
        }
    }
}



template<class I, class T>
void approx_ideal_restriction_pass1(      I rowptr[], const int rowptr_size,
                                    const I C_rowptr[], const int C_rowptr_size,
                                    const I C_colinds[], const int C_colinds_size,
                                          T C_data[], const int C_data_size,
                                    const I Cpts[], const int Cpts_size,
                                    const I splitting[], const int splitting_size,
                                    const I max_row = std::numeric_limits<I>::max() )
{
    // Function to sort two pairs by the second argument
    struct sort_2nd {
        bool operator()(const std::pair<I,T> &left, const std::pair<I,T> &right) {
            return left.second < right.second;
        }
    };

    I nnz = 0;
    rowptr[0] = 0;

    // Deterimine number of nonzeros in each row of R.
    for (I row=0; row<Cpts_size; row++) {
        I cpoint = Cpts[row];

        // Determine number of strongly connected F-points in sparsity for R.
        // Store strength values and indices.
        std::vector<std::pair<I,T> > neighborhood;
        for (I i=C_rowptr[cpoint]; i<C_rowptr[cpoint+1]; i++) {
            if ( (splitting[C_colinds[i]] == F_NODE) && (std::abs(C_data[i]) > 1e-16) ) {
                neighborhood.push_back(std::make_pair(i, C_data[i]));
            }
        }

        // If neighborhood is larger than the maximum nonzeros per row, sort
        // F-point neighborhood by strength, set SOC matrix equal to zero in
        // smallest elements (i.e. no longer strongly connected).
        I size = neighborhood.size();
        if (size > max_row) {
            std::sort(neighborhood.begin(), neighborhood.end(), sort_2nd());
            for (I i=max_row; i<size; i++) {
                C_data[neighborhood[i].first] = 0;
            }
        }

        // Set row-pointer for this row of R (including identity on C-points).
        nnz += (1 + std::min(size, max_row));
        rowptr[row+1] = nnz; 
    }
}

template<class I, class T>
void approx_ideal_restriction_pass2(const I rowptr[], const int rowptr_size,
                                          I colinds[], const int colinds_size,
                                          T data[], const int data_size,
                                    const I A_rowptr[], const int A_rowptr_size,
                                    const I A_colinds[], const int A_colinds_size,
                                    const T A_data[], const int A_data_size,
                                    const I C_rowptr[], const int C_rowptr_size,
                                    const I C_colinds[], const int C_colinds_size,
                                    const T C_data[], const int C_data_size,
                                    const I Cpts[], const int Cpts_size,
                                    const I splitting[], const int splitting_size )
{
    // Build column indices and data for each row of R.
    for (I row=0; row<Cpts_size; row++) {

        I cpoint = Cpts[row];
        I ind = rowptr[row];

        // Set column indices for R as strongly connected F-points.
        for (I i=C_rowptr[cpoint]; i<C_rowptr[cpoint+1]; i++) {
            if ( (splitting[C_colinds[i]] == F_NODE) && (std::abs(C_data[i]) > 1e-16) ) {
                colinds[ind] = C_colinds[i];
                ind +=1 ;
            }
        }

        if (ind != (rowptr[row+1]-1)) {
            std::cout << "Error: Row pointer does not agree with neighborhood size.\n";
        }

        // Build local linear system as the submatrix A restricted to the
        // neighborhood, Nf, of strongly connected F-points to the current
        // C-point, that is A0^T = A[Nf, Nf]. System stored in column major
        // for ease of iteation - each column of A0 corresponds to a row in
        // A and A is stored in CSR. 
        I is_col_major = true;
        I size_N = ind - rowptr[row];
        std::vector<T> A0(size_N*size_N);
        I temp_A = 0;
        for (I j=rowptr[row]; j<ind; j++) { 
            I this_ind = colinds[j];
            for (I i=rowptr[row]; i<ind; i++) {
                // Search for indice in row of A
                I found_ind = 0;
                for (I k=A_rowptr[this_ind]; k<A_rowptr[this_ind+1]; k++) {
                    if (colinds[i] == A_colinds[k]) {
                        A0[temp_A] = A_data[k];
                        found_ind = 1;
                        temp_A += 1;
                        break;
                    }
                }
                // If indice not found, set element to zero
                if (found_ind == 0) {
                    A0[temp_A] = 0.0;
                    temp_A += 1;
                }
            }
        }

        // Build local right hand side given by b_j = A_{cpt,N_j}, where N_j
        // is the jth indice in the neighborhood of strongly connected F-points
        // to the current C-point. 
        I temp_b = 0;
        std::vector<T> b0(size_N);
        for (I i=rowptr[row]; i<ind; i++) {
            // Search for indice in row of A
            I found_ind = 0;
            for (I k=A_rowptr[cpoint]; k<A_rowptr[cpoint+1]; k++) {
                if (colinds[i] == A_colinds[k]) {
                    b0[temp_b] = A_data[k];
                    found_ind = 1;
                    temp_b += 1;
                    break;
                }
            }
            // If indice not found, set element to zero
            if (found_ind == 0) {
                b0[temp_b] = 0.0;
                temp_b += 1;
            }
        }

        // Solve linear system (least squares solves exactly when full rank)
        // s.t. (RA)_ij = 0 for (i,j) within the sparsity pattern of R. Store
        // solution in data vector for R.
        least_squares(&A0[0], &b0[0], &data[rowptr[row]], size_N, size_N, is_col_major);

        // Add identity for C-point in this row
        colinds[ind] = cpoint;
        data[ind] = 1.0;
    }
}


#endif
