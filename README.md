(a)
- Platform Used: PySpark on Google Colab.

- Logic:

	- Matrix Addition (Parts I & II): Mapped each matrix element to key (i, j) with its value. Reduced by summing values with the same key, basically adding matrices M and N.

	- Matrix Multiplication (Parts III & IV): Implemented single-pass matrix multiplication. Mapped elements of M and N to keys (i, k), emitting necessary values for multiplication. In the reduce phase, multiplied matching elements and summed the products to compute each element of the result matrix.

	- Some observations: We had to make sure to correctly input the matrix dimensions, otherwise you would get the wrong answers.



(b) Setup:
	1. Ensure matrixaddition.txt and matrixmultiplication.txt are in the same directory.
	2. Run all cells sequentially.
	3. When you get to the cell that asks you for an input, enter 128 by 128.
	4. Output files SumMatrix.txt and ProductMatrix.txt will contain the results.

(c) Runtime:
	- Part II Runtime : Approximately 35 seconds.
	- Part IV Runtime : Approximately 31 seconds.
