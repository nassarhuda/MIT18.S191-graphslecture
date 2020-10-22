function our_first_pagerank(A::SparseMatrixCSC, alpha,niter)
    n = size(A,1)
    d = vec(sum(A,dims=2)) 
    x = ones(n)./n
  
    for iter=1:niter
      x ./= d
      y = A'*x # compute a matrix vector product
      y .*= alpha
      y .+= (1.0-alpha)/n
      x = y
    end
    return x
  end
  
function random_walk(A::SparseMatrixCSC{T,Int64},u::Int,walk_length::Int;seed::Int=-1) where T
    visited = Vector{Int}(undef,walk_length)
    visited[1] = u
    currentnode = u
    for i = 2:walk_length
        currentnode = rand(view(A.rowval,A.colptr[currentnode]:A.colptr[currentnode+1]-1))
        visited[i] = currentnode
    end
    return visited
end