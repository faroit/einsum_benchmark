using Einsum


function parafac2(n::Integer, r=10000)
	X = zeros(n,n)
	A = randn(n,r)
	B = randn(n,r)

	return @einsum X[a,b] = A[a,j]*B[b,j]
end


function parafac5(n::Integer, r=10)
	X = zeros(n,n,n,n,n)
	A = randn(n,r)
	B = randn(n,r)
	C = randn(n,r)
	D = randn(n,r)
	E = randn(n,r)

	return @einsum X[a,b,c,d,e] = A[a,j]*B[b,j]*C[c,j]*D[d,j]*E[e,j]
end


function commonfate(n::Integer, r=10)
  P = zeros(n,n,n,n,n);

  A = randn(n,n,n,r);
  H = randn(n,r);
  C = randn(n,r);

  return @einsum P[a,b,f,t,c] = A[a,b,f,j]*H[t,j]*C[c,j]
end


function timeit(n, reps, func_handle)
    time = @elapsed for j in 1:reps
        func_handle(n)
    end
    return(time/reps)
end

outfile = open("julia.csv", "w")

for j in 10:10:60
  write(outfile, join(("parafac2", j, timeit(j, 3, parafac2)), " "), "\n")
  write(outfile, join(("parafac5", j, timeit(j, 3, parafac5)), " "), "\n")
  write(outfile, join(("commonfate", j, timeit(j, 3, commonfate)), " "), "\n")
end

close(outfile)
