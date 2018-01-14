# -*- coding: utf-8 -*-
"""

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

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog


# INPUTS #####################################################################

price = 3.25 * 2204.62    # copper $/lb x (2204.62lb per metric tons)
block_size = 100   # each ore block is 100 metric tons
cost = 30   # cost to mine $ per metric ton
eff = 90 # recovery efficiency % : % of Cu in each block that is recovered

# pit size= n_row * n_col = n_blocks
# pit must be a square: n_row * n_col
n_row = 8
n_col = n_row
n_blocks = n_row * n_col

# ore grade: weight % Cu 
# grade in blocks is lognormal distributed
# mean, std dev of underlying normal
rng = np.random.RandomState(9)
grade_mean = np.log(0.4)
grade_std = 0.483
# generate grade for all blocks
ore_grade = rng.lognormal(grade_mean, grade_std, n_blocks)

# block values
block_val = (price * (ore_grade/100) * (eff/100) - cost) * block_size



# CONSTRAINTS ################################################################

# Arcs matrix is unimodular matrix made up of 1s and -1s
# the matrix represents directed arcs between each node (each node is a block)
# an arc from block i to block j corresponds to contraint xi - xj <= 0 for arc (i, j)
# that is, if block i is mined (xi=1), then block j must be mined (xj=1)
Arcs = np.zeros((1, n_blocks))

# iterate through each bock in pit and add the arcs for that block's
# precedent constraints- note the first n_row block are on top of pit

for i in range(n_row + 1, n_blocks):
    
    # add arc between block i and the block directly above
    arc_above = np.zeros([1, n_blocks])
    arc_above[0, i - n_col] = -1
    arc_above[0, i] = 1
    #append new arc to arc matrix
    Arcs = np.r_[Arcs, arc_above]

    # if block i is not on right edge of pit, there is a block above &
    # to the right that must be mined before block i
    if (n_blocks % i) != 0:
        # arc with block above & to the right
        arc_right = np.zeros([1, n_blocks])
        arc_right[0, i - n_col + 1] = -1
        arc_right[0, i] = 1
        #append new arc to arc matrix
        Arcs = np.r_[Arcs, arc_right]
        
    # if block is not on left edge of pit
    if (n_blocks % i) != 0:
        # add 2 new arcs
        arc_left = np.zeros([1, n_blocks])
        # arc with block above & left
        arc_left[0, i - n_col - 1] = -1
        arc_left[0, i] = 1
        #append new arcs to arc matrix
        Arcs = np.r_[Arcs, arc_left]

# vector of zeros for precedent constraints
b = np.zeros(Arcs.shape[0])

# x is bounded between 0 and 1
x_bnd = [(0,1)]*n_blocks



# OPTIMIZATION ################################################################

# linear program
# objective: miniziming negative block value (equiv to maximizing block value)
# choice variable: xi =1 if block i is mined; =0 if block is not mined
# constraints: Arcs <= b (precedent constraints)
#              0 <= xi <= 1
# the matrix A is unimodular, so integrality is ensured
open_pit = linprog(-block_val, A_ub=Arcs, b_ub=b, bounds=x_bnd, options={"disp":True})
block_mine = open_pit.x
print(open_pit)



# MAKE PLOTS #################################################################

# plot block values
fig = plt.figure(figsize=(21,7))
plot_block_val = fig.add_subplot(1,2,1)
plot_block_val.set_title('Block Economic Values')
img_block_val = block_val.reshape(n_row, n_col)
plt.imshow(img_block_val, cmap='RdYlGn', interpolation='nearest', aspect='auto')
ax = fig.gca()
ax.set_xticklabels(np.arange(0, n_col+1, 1))
ax.set_yticklabels(np.arange(0, n_col+1, 1))
plt.colorbar()

# plot the optimal blocks to mine
fig = plt.figure(figsize=(8,8))
plot_block_mine = fig.add_subplot(1,1,1)
fig.suptitle('Blocks to Mine')
plot_block_mine.set_title('Mine Black Blocks; Do Not Mine White Blocks')

img_block_mine = block_mine.reshape(n_row, n_col)
plt.imshow(img_block_mine, cmap='binary', interpolation='nearest', aspect='auto')

ax = fig.gca()
# minor axis ticks at right edge of each block
ax.set_xticks(np.arange(0.5, n_col + 0.5, 1), minor=True)
# major axis ticks at center of each block
ax.set_xticks(np.arange(0, n_col, 1))
ax.set_xticklabels(np.arange(1, n_col+1, 1), minor=False)
# minor axis ticks at right edge of each block
ax.set_yticks(np.arange(0.5, n_col + 0.5, 1), minor=True)
# major axis ticks at center of each block
ax.set_yticks(np.arange(0, n_col, 1))
ax.set_yticklabels(np.arange(1, n_col+1, 1), minor=False)
plt.grid(which='minor')
