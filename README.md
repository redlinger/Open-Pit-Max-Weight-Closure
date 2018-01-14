# Open_Pit_Max_Weight_Closure_Python
Open Pit Mining Problem as Max Weight Closure Problem in Python

Setup: Mine Co. has an open pit copper (Cu) mine. The mine is made up of 
blocks of rock, each rock with a varying concentration (aka grade) of Cu. 
To mine a given block, you must mine 1) the block directly above it, 
2) the block above and to the immediate right, and 3) the block above and 
to the immediate left; these are the precedent constraints.

Problem: Determine which blocks of rock should be mined in order to 
maximize profit, while observing the precedent constraints.

Formulation: The problem is formulated as a Maximum Weight Closure Problem and 
solved as a linear program:
 
   min -SUM_i (block_val(i) * x(i))

   s.t. -x(j) + x(i) <= 0 for every (i,j) in Arcs
        0 <= x(i) <= 1 for every i

 notes:
     objective: minimizing negative block value (equiv to maximizing block value)
     choice variable: x(i) =1 if block i is mined; =0 if not mined
     block_val(i): economic value of block i (value of the Cu recovered less 
                    the cost to extract the block)
     Constraints:
     The constraint -x(j) + x(i) <= 0 ensures that if block i is mined (xi=1), 
         then block j must be mined (xj=1).
         If block j must be mined before block i, then there is a directed arc
         from block i to block j.The matrix Arcs is unimodular matrix made up 
         of 1s and -1s; the matrix represents directed arcs between each node 
         (each node is a block)
     The constraint 0 <= x(i) <= 1 ensures x(i) is either mined or not mined.
        Note the matrix A is unimodular, so integrality is ensured (x(i) will
        be zero or one)
