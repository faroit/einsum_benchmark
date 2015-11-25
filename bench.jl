using TensorOperations, IndexNotation

A = randn(40, 40, 100, 10)
X = randn(40, 10)
Z = randn(40, 10)

V[l"a,b,f,t,c"] = A[l"a,b,f,j"]*X[l"t,j"]*Z[l"c,j"]
