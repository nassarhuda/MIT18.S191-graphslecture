# Turan shadow work:
# XXX force undirected and unweighted
# notes to check: set vs vector accumulations (for T and S), which is better (now I have the fastest configuration)
# this version finds the random cliques as well

using SparseArrays
using LinearAlgebra
using MatrixNetworks
using StatsBase

struct OneShadow
    parents::Vector{Int}
    vertices::Vector{Int}
    k::Int
end

function density(c::Vector{Int},A::SparseMatrixCSC{F,Int64}) where F <: Real
    edges = sum(view(A,c,c))
    vi = length(c)
    d = edges/(vi*(vi-1))
end

function create_shadows!(A::SparseMatrixCSC{F,Int},k::Int,S::Vector{OneShadow}) where F <: Real
    T = Set{OneShadow}()
    n = A.n
    ind = ones(Bool,n)
    newlabels = sortperm(sum(A,dims=2)[:])
    rp = A.colptr
    ci = A.rowval
    @inbounds for i = 1:n
        v = newlabels[i]
        rpi = rp[v]:rp[v+1]-1
        Ci = ci[rpi]
        Vi = Ci[ind[Ci]]
        l = k-1
        if length(Vi) >= l
            if density(Vi,A) > 1-(1/(l-1))
                push!(S,OneShadow([v],Vi,l))
            else
                push!(T,OneShadow([v],Vi,l))
            end
        end
        ind[v] = false # can no longer add it
    end
    return T,newlabels
end

function create_shadows_view!(A::SparseMatrixCSC{F,Int},k::Int,vertices::Vector{Int},T::Set{OneShadow},
    S::Vector{OneShadow},newlabels::Vector{Int},curp::Vector{Int}) where F <: Real
    vn = length(vertices)
    ind = ones(Bool,vn)
    V = view(A,vertices,vertices)
    # newlabels = sortperm(map(i->sum(view(V,i,:)),1:vn))
    sortperm!(view(newlabels,1:vn),map(i->sum(view(V,i,:)),1:vn))
    @inbounds for i = 1:vn
        v = newlabels[i]
        veci = V[:,v]
        Ci = veci.nzind # all neighbors in current set
        Wi = Ci[ind[Ci]] # all neighbors we can connect to
        if length(Wi) >= k-1
            Vi = vertices[Wi]
            vi = vertices[v]
            # curp = vcat(curp,vi)
            # @show show curp
            # @show vi
            # push!(curp,vi)
            newcurp = vcat(curp,vi)
            #####
            if k <=2 || density(Vi,A) > 1-(1/(k-2))
                push!(S,OneShadow(newcurp,Vi,k-1))
            else
                push!(T,OneShadow(newcurp,Vi,k-1))
            end
            #####
            # push!(T,OneShadow(Vi,l))
        end
        ind[v] = false # can no longer add it
    end
end

function shadow_finder(A::SparseMatrixCSC{F,Int},k::Int) where F <: Real
    S = Vector{OneShadow}()
    T,newlabels = create_shadows!(A,k,S)
    while !isempty(T)
        currentShadow = pop!(T)
        vertices = currentShadow.vertices
        l = currentShadow.k
        curp = currentShadow.parents
        create_shadows_view!(A,l,vertices,T,S,newlabels,curp)
        # create_shadows_view!(A,currentShadow.k,currentShadow.vertices,T,S)
    end
    return S
end

function isclique(l::Vector{Int64},A::SparseMatrixCSC{F,Int64}) where F <: Real
    ns = length(l)
    if sum(view(A,l,l)) == ns*ns-ns
        return true
    else
        return false
    end
end

# https://juliastats.github.io/StatsBase.jl/latest/weights.html#FrequencyWeights-1
function sample_shadow(A::SparseMatrixCSC{F,Int64},S::Vector{OneShadow},k::Int,t::Int,exact_clique::Bool=true) where F <: Real
    clique_sets = Vector{Vector{Int}}()
    parent_clique = Vector{Vector{Int}}()
    w = zeros(Int,length(S))
    @inbounds for i = 1:length(S)
        Si = S[i]
        w[i] = binomial(length(Si.vertices),Si.k)
    end
    sweight = sum(w)
    X = ones(Bool,t)
    all_ids = wsample(1:length(w),Float64.(w),t)
    for r = 1:t
        id = all_ids[r] #mysample(p)
        Si = S[id]
        ltuple = StatsBase.sample(Si.vertices,Si.k;replace=false) # need without replacement
        if isclique(ltuple,A)
            X[r] = 1
            if exact_clique
                ltuple = vcat(ltuple,Si.parents)
            else # bottom most clique
                ltuple = vcat(ltuple,Si.parents[end])
            end
            push!(clique_sets,ltuple)
            # push!(parent_clique,Si.parents)
        else
            X[r] = 0
        end
    end
    approxval = (sum(X)/t)*sweight
    return approxval,clique_sets
end

function TuranShadow(A::SparseMatrixCSC{F,Int64},k::Int,t::Int,exact_clique::Bool=true) where F <: Real
    S = shadow_finder(A,k)
    if length(S) == 0
        return 0,Array{Array{Int,1}}(undef,0)
    else
        approxval,clique_sets = sample_shadow(A,S,k,t,exact_clique)
        return approxval,clique_sets
    end
end


# form W out of clique_sets
function create_C(clique_sets)
    lcliques = length.(clique_sets)
    totalnnz = sum(lcliques)
    ei = zeros(Int,totalnnz)
    ej = zeros(Int,totalnnz)
    icounter = 1
    @inbounds for i = 1:length(lcliques)
        curclique = clique_sets[i]
        ei[icounter:icounter+lcliques[i]-1] .= i
        ej[icounter:icounter+lcliques[i]-1] = curclique
        icounter += lcliques[i]
    end
    return ei,ej
end

function TuranShadow_matrix(A::SparseMatrixCSC{T},myfn,from_k,upto_k,scaleA,t) where T
    GW = spzeros(size(A)...)
    for k = from_k:upto_k
        # approxval,clique_sets = TuranShadow(A,k,t)

        approxval,clique_sets = TuranShadow(A,k,t)
        csets_new = copy(clique_sets)
        map(i->csets_new[i] = sort(csets_new[i]),1:length(csets_new))
        csets_new = unique(csets_new)
        clique_sets = copy(csets_new)

        # map!(i->sort(clique_sets[i]),clique_sets,1:length(clique_sets))
        # clique_sets = unique(clique_sets)

        # csets_new = copy(clique_sets)
        # map(i->csets_new[i] = sort(csets_new[i]),1:length(csets_new))
        # csets_new = unique(csets_new)
        # clique_sets = copy(csets_new)
        
        if approxval == 0
            @warn "largest clique size found is $(k-1)"
            break
        end
        ei,ej = create_C(clique_sets)
        C = sparse(ei,ej,1,length(clique_sets),A.n)
        W = C'*C
        W = myfn(k)*W
        W = W - spdiagm(0=>diag(W))
        GW += W # GW = max.(W,GW) #instead of GW += W
    end

    GW += scaleA*A
    
    dropzeros!(GW)
    return GW
end

# cid = 3
# approxval,clique_sets = TuranShadow(A,cid,t)
# ei,ej = create_C(clique_sets)
# C = sparse(ei,ej,1,length(clique_sets),A.n)
# W = C'*C
# F = W - spdiagm(0=>diag(W))

# approxval,clique_sets = TuranShadow(A,cid,t)
# ei,ej = create_C(clique_sets)
# C = sparse(ei,ej,1,length(clique_sets),A.n)
# W = C'*C
# F += W - spdiagm(0=>diag(W))

# F = F+A

# approxval,clique_sets = TuranShadow(A,10,t)
# ei,ej = create_C(clique_sets)
# C = sparse(ei,ej,1,length(lcliques),A.n)
# W = C'*C # => gives NaNs in the vectors
# F = W - spdiagm(0=>diag(W))
# F = F+A
# x2,x3,l = x2_x3_from_spectral_embedding(F)
# xy = hcat(x2,x3)
# using Plots
# pyplot()
# f = plot(leg=false,axis=false)
# scatter!(f,xy[:,1],xy[:,2],color = :black)
# savefig("test4.pdf")

#=
spectral embedding 
=#
import MatrixNetworks._symeigs_smallest_arpack
using Printf
import LinearAlgebra.checksquare

function x2_x3_from_spectral_embedding(A::SparseMatrixCSC{V,Int};
    tol=1e-12,maxiter=300,dense=96,nev=3,checksym=true) where V
    n = checksquare(A)
    if checksym
        if !issymmetric(A)
            throw(ArgumentError("The input matrix must be symmetric."))
        end
    end

    d = vec(sum(A,dims=1))
    d = sqrt.(d)

    if n == 1
        X = zeros(V,1,2)
        lam2 = 0.
    elseif n <= dense
        ai,aj,av = findnz(A)
        L = sparse(ai,aj,-av./((d[ai].*d[aj])),n,n) # applied sqrt above
        L = Matrix(L) + 2I
        F = eigen!(Symmetric(L))
        lam2 = F.values[2]-1.
        X = F.vectors
    else # modifying this branch 1-11-2019
        ai,aj,av = findnz(A)
        L = sparse(ai,aj,-av./((d[ai].*d[aj])),n,n) # applied sqrt above
        L = L + sparse(2.0I,n,n)

        (lams,X,nconv) = _symeigs_smallest_arpack(L,nev,tol,maxiter,d)
        # lam2 = lams[2]-1.
        lams = lams.-1.
    end
    x1err = norm(X[:,1]*sign(X[1,1]) - d/norm(d))
    if x1err >= sqrt(tol)
        s = @sprintf("""
        the null-space vector associated with the normalized Laplacian
        was computed inaccurately (diff=%.3e); the Fiedler vector is
        probably wrong or the graph is disconnected""",x1err)
        @warn s
    end
    @show size(X)

    # modifying here to access x2 and x3
    x2 = vec(X[:,2])
    if n > 1
        x2 = x2./d # applied sqrt above
    end

    x3 = vec(X[:,3])
    if n > 1
        x3 = x3./d # applied sqrt above
    end

    # flip the sign if the number of pos. entries is less than the num of neg. entries
    nneg = sum(x2 .< 0.)
    if n-nneg < nneg
      x2 = -x2;
    end

    nneg = sum(x3 .< 0.)
    if n-nneg < nneg
      x3 = -x3;
    end
    X = X./repeat(d,1,size(X,2))
    # LX = [lams,X]
    return x2,x3,X
end

