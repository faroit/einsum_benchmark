using TensorOperations

A = randn(2000, 100)
B = randn(2000, 100)

tic()
for i = 1:100
    @tensor V[a,b] := A[a,k]*B[b,k]
end
toc()
