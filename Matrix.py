# -*- coding: utf-8 -*-


---
You can assume the matrices are stored in your file as follows. Feel free to use other formats,
but please mention that clearly in your readme file. The provided ‘matrixaddition.txt’ and
‘matrixmultiplication.txt’ files use a format that you can follow.

For matrix M, the format can be: M, i, j, mij where i is the row index, j is the column index and mij is the value.

For example:

M, 0, 0, 1

M, 0, 1, 2

etc.


For matrix N, the format can be: N, j, k, njk where j is the row index, k is the column index and njk is the value.

For example:

N, 0, 0, 5

N, 0, 1, 4

etc
"""

#Code in this section includes work taken or modified from the file "CIS 5570 - Fall 24 - PySpark Starter Notebook.ipynb" provided for the use in this class, author Salem Sharak
#This code is standard setup code required run PySpark in Google Colab
!pip install -q pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

conf = SparkConf().setAppName('SparkMatrix')
sc = SparkContext.getOrCreate(conf = conf)

sqlContext = SparkSession.builder\
        .master("local")\
        .appName("Colab")\
        .config('spark.ui.port', '4050')\
        .getOrCreate()
#these files are owned/shared by danmit ... corrected file links addition/multiplication files IDs were reversed
#Added files multiplication identity & simple to help with debugging the Multiplication Map/Reduce
!wget -q 'https://drive.google.com/uc?export=download&id=15aHWiYRxiYdvmLX3zeQ3IUMxKix_mgQn' -O 'matrixaddition-1.txt'
!wget -q 'https://drive.google.com/uc?export=download&id=1yvPZJxt9ukTVXzQbGMmYLb5-YAYUbhEr' -O 'matrixmultiplication-1.txt'
!wget -q 'https://drive.google.com/uc?export=download&id=1-tl-ai9i1jWlQd_G2xRgNXDTepqll9FP' -O 'matrixmultiplication-identity'
!wget -q 'https://drive.google.com/uc?export=download&id=1tsSum37XkIJ07vxbdsCvOfRJjvokMnUy' -O 'matrixmultiplication-simple'
!wget -q 'https://drive.google.com/uc?export=download&id=15rkzm1awPr_Oq-em1RdiG_HVFyJQDrvB' -O 'smallMatrixMultTest'

input_file_addition = sc.textFile('matrixaddition-1.txt')
input_file_multiplication = sc.textFile('matrixmultiplication-1.txt')
input_file_multiplication_identity = sc.textFile('matrixmultiplication-identity')
input_file_multiplication_simple = sc.textFile('matrixmultiplication-simple')
input_file_multiplication_small = sc.textFile('smallMatrixMultTest')

"""**Part I**

Implement a map-reduce algorithm to add two matrices M and N together.

```
Checkdata for Matrix M, N and resultant matrix P
M 0 0 -157  ...  N 0 0  832  ...  P 0 0 -157 + 832 =   675
M 0 1 -423  ...  N 0 1  353  ...  P 0 1 -423 + 353 =   -70
M 0 2 -547  ...  N 0 2 -672  ...  P 0 2 -547 - 672 = -1219
M 0 3 -845  ...  N 0 3  556  ...  P 0 3 -845 + 556 =  -289
M 0 4 -331  ...  N 0 4    4  ...  P 0 4 -331 +   4 =  -327
M 0 5  -22  ...  N 0 5 -585  ...  P 0 5  -22 - 585 =  -607
```
**Part II**

Using the provided 'matrixaddition.txt' file, which contains 2 matricies (matrix M and matrix N) of size (512,512), find the final matrix addition using your code.  Save the output as a txt file.

"""

#Function removes the "source Matrix" (M/N) identifier, and sets the reduce Key to the row-column
def MatrixMap(line):
  values  = line.split()
  matrix  = values[0]
  row     = int(values[1])
  column  = int(values[2])
  value   = int(values[3])
  return ((row,column),value)

#This will accept any number of matricies (M, N, ...) in a single file.
#The matricies are assumed to be the same SIZE, and the have complete data
#Resultant RDD should be ((row,column),Value)
AddMap = input_file_addition.map(lambda line: MatrixMap(line))
#With the Key (row,column), and the Values together to reduce, sort by the Key
AddReduce = AddMap.reduceByKey(lambda a,b: a+b).sortByKey()

#Code in this section is taken from or inspired by the file "CIS 5570 - Fall 24 - PySpark Starter Notebook.ipynb" provided for the use in this class, author Salem Sharak
#This code is standard output code to create the resultant/required text file
SumMatrix = sqlContext.createDataFrame(AddReduce).withColumnRenamed('_1','P(i,k)').withColumnRenamed('_2','Value')
SumMatrix.toPandas().to_csv('SumMatrix.txt', index=False)
SumMatrix.show()

"""**Part III**

Implement the following algorithm for single-pass matrix multiplication:
The Map function will generate its (key, value) pairs from the input matrices, and the
Reduce function will use the output of the Map function to perform its calculations and
generate its (key, value) pairs as we did in Chapter 2 for the single-pass matrix
multiplication.

**Part IV**

Using the provided ‘matrixmultiplication.txt’ file, which contains 2 matrices (matrix M and
matrix N) of shape (96,256) and (256,128), find the final multiplied matrix output using your
code. Save the output as a txt file.

```
M 0 0 -587
M 127 255 -298
N 0 0 380
N 255 127 -215
```


"""

# ------------------------ Matrix Dimensions from user input -----------------------------

# The number of columns in Matrix M (j) is captured for potential validation
# to ensure that it matches the number of rows in Matrix N,
# which is a requirement for valid matrix multiplication.

i = int(input("Enter the number of rows in matrix M(i,j): "))
k = int(input("Enter the number of columns in matrix N(j,k): "))

# ------------------------ Functions -----------------------------
#NOTE: i, k are hard-coded in MatrixMultMap
def MatrixMultMap(line,i=128,k=128):
  values  = line.split()
  matrix  = values[0]
  row     = int(values[1])  #M(i,j)=i, N(j,k)=j
  column  = int(values[2])  #M(i,j)=j, N(j,k)=k
  value   = int(values[3])
  output  = []
  if matrix == 'M':     # then row col is i and j
    for K in range(k):  # for all k: key (i,k)
      output.append( ( (row,K) , ('M',column,value) ) )
    #Returns the LIST of multiples (column, J) for all J
    return output
  elif matrix == 'N':   # then row col is j and k
    for I in range(i):  # for all i: key (i,k)
      output.append( ( (I,column) , ('N',row,value) ) )
    #Returns the LIST of multiples (J, row) for all J
    return output

def ReduceMultiplySum(key, val):
  # for each (i,k) there will be two J values, one for each matrix
  sum_Pij = 0
  j_map = {}
  for values in val:
    matrix = values[0]
    j = values[1]
    val = values[2]
    if j in j_map:
      # means we have already seen the other matching j, and can multiply it with this one
      j_map[j] = j_map[j] * val
    else:
      # means we haven't see this j value yet
      j_map[j] = val

  for val in j_map.values():
    sum_Pij += val

  return (key, sum_Pij)

# --------------------------- MAP --------------------------------
#This will accept two matricies (M, N) in a single file.
#The matricies are assumed to be the same SIZE as the sample file (i=128, k=128), and the have complete data
#Resultant RDD should be (P(row,column),('M' or 'N', j, Mij or Njk))
MulMap = input_file_multiplication.flatMap(lambda line: MatrixMultMap(line, i, k))

# --------------------------- GROUP/REDUCE ------------------------------------
MulReduce = MulMap.groupByKey().map(lambda pair: ReduceMultiplySum(pair[0],pair[1])).sortByKey()

# --------------------------- OUTPUT ------------------------------------
#Code in this section is taken from or inspired by the file "CIS 5570 - Fall 24 - PySpark Starter Notebook.ipynb" provided for the use in this class, author Salem Sharak
#This code is standard output code to create the resultant/required text file
MulMatrix = sqlContext.createDataFrame(MulReduce).withColumnRenamed('_1','P(i,k)').withColumnRenamed('_2','Value')
MulMatrix.toPandas().to_csv('ProductMatrix.txt', index=False)
MulMatrix.show(20, False)
