using LinearAlgebra
using SparseArrays
using MatrixNetworks

sp = 0.1
A = sparse(erdos_renyi_undirected(1000,sp));
B = our_erdos_renyi(1000,sp);

T = A*A*A;
sum(diag(T))/6
nnz(A)/prod(size(A))
length(collect(triangles(A))) #function used based on correlation coefficients

F = B*B*B;
sum(diag(F))/6
nnz(B)/prod(size(B))
length(collect(triangles(B)))

# check out https://github.com/nassarhuda/MatrixNetworks.jl/blob/master/src/triangles.jl 
# for a cool iterator application in julia