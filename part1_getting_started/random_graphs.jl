using LinearAlgebra
using SparseArrays
using Plots

# disclaimer: not necessairly the best way to write these functions, 
# but it might be useful to write them this way for educational purposes
# for more details, check out: https://github.com/nassarhuda/MatrixNetworks.jl/blob/master/src/generators.jl

function our_erdos_renyi(n::Int64,p::Float64)
    if p < 0 || p >= 1 error("the probablity value must be in the interval [0,1)") end
    A = rand(n,n)
    T = triu(A,1)
    T = sparse(triu(T.<p,1))
    return max.(T,T')
end

function our_pref_attach(n::Int64,n0::Int64,k::Int64)
    A = spzeros(Bool,n,n)
    C = ones(Int,n0,n0)-I
    A[1:n0,1:n0] .= C
    for i = n0+1:n
        degs = sum(A[1:i-1,1:i-1],dims=2)[:]
        degp = degs./sum(degs)
        c = cumsum(degp)
        r = rand(k)
        nodes = map(x->findfirst(c.>x),r)
        A[nodes,i] .= 1
        A[i,nodes] .= 1
    end
    return A
end


A = our_pref_attach(1000,5,5);
sp = nnz(A)/prod(size(A))
degs = sum(A,dims=2)[:];
plot(sort(degs,rev=true),axis=:log,
    linewidth = 2,label="PA")
B = our_erdos_renyi(1000,sp);
degs = sum(B,dims=2)[:];
plot!(sort(degs,rev=true),
    linewidth=2,label="ER")
