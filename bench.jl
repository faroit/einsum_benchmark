using TensorOperations

A = randn(1000, 50)
B = randn(1000, 50)
C = randn(100, 50)

V=zeros(100, 100, 100)

tic()
for i = 1:100
    @tensor V[a,b] := A[a,k]*B[b,k]
end
toc()
