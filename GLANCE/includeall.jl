# things we need


using PyCall
using Pkg
Pkg.build("PyCall")
#using Plots
#pyplot()

using ScikitLearn
using SparseArrays
using LinearAlgebra
using MatrixNetworks
#using Plots
#using GraphRecipes
#using NearestNeighbors

#using LightGraphs
#using SimpleWeightedGraphs
#using Node2Vec
#using Tables
#using CSV
#using MAT
#using Distributed
using StatsBase

#addprocs(100-nprocs())
#@everywhere using MatrixNetworks

include("TuranShadow.jl")
#include("n2v.jl")
#include("draw_neighboring_nodes.jl")

# pyplot()

"""
findin_index(x,y) returns a vector v of the same size as that of x
- where v[i] = index of the element x[i] in the vector y
- v[i] = 0 if x[i] does not exist in y
- assumption: If y does not consist of unique elements, the index returned is the last occurence
```
example:
    julia> x = [1,2,3,10,1,4];
    julia> y = [1,2,5,4,3];
    julia> findin_index(x,y)
    6-element Array{Int64,1}:
     1
     2
     5
     0
     1
     4
```
"""
function findin_index(x::Vector{T},y::Vector{T}) where T
  indices_in_y = zeros(Int64,length(x))
  already_exist = findall((in)(y), x)
  donot_exist = setdiff(1:length(x),already_exist)
  funcmap = i -> indices_in_y[findall(x.==y[i])] .= i
  lookfor_indices = findall((in)(x), y)
  map(funcmap,lookfor_indices)
  return indices_in_y
end

spones(W) = sparse(findnz(W)[1:2]...,1,W.n,W.n)

function plot_high_low_deg_nodes!(xy,nodeids,nodeids_small)
scatter!(legend=:topleft,xy[nodeids,1],xy[nodeids,2],
    markeralpha=0.8,
    markerstrokecolor=RGB(255/256,69/256,0/256),
    color=RGB(255/256,69/256,0/256),
    markersize=2,
    legendfontcolor=RGB(0,0,0),
    label="high degree")
scatter!(xy[nodeids_small,1],xy[nodeids_small,2],
    markeralpha=0.8,
    markerstrokecolor=RGB(102/256,255/256,102/256),
    color=RGB(102/256,255/256,102/256),
    markersize=2,
    label="low degree")
end

function plotweighted(A,xy)
  return graphplot(findnz(triu(A,1))[1:2]..., x =xy[:,1], y=xy[:,2],
    markercolor=:black, markerstrokecolor=:white,
    markersize=2, linecolor=1, linealpha=0.2, linewidth=0.7,
    markeralpha=0.2,
    axis_buffer=0.02, background=nothing,label = "")
end

function plotweighted_scatter(A,xy)
  return graphplot(findnz(triu(A,1))[1:2]..., x =xy[:,1], y=xy[:,2],
    markercolor=:black, markerstrokecolor=:white,
    markersize=5, linecolor=1, linealpha=0, linewidth=0,
    markeralpha=0.2,
    axis_buffer=0.02, background=nothing,label = "")
end

function plotweighted_scatter!(A,xy)
  return graphplot!(findnz(triu(A,1))[1:2]..., x =xy[:,1], y=xy[:,2],
    markercolor=:black, markerstrokecolor=:white,
    markersize=2, linecolor=1, linealpha=0, linewidth=0,
    markeralpha=0.2,
    axis_buffer=0.02, background=nothing,label = "")
end

function plotweighted!(A,xy)
  return graphplot!(findnz(triu(A,1))[1:2]..., x =xy[:,1], y=xy[:,2],
    markercolor=:black, markerstrokecolor=:white,
    markersize=2, linecolor=1, linealpha=0.2, linewidth=0.7,
    markeralpha=0.2,
    axis_buffer=0.02, background=nothing,label = "")
end

using PyCall
#igraph = pyimport("igraph")

function igraph_layout(A::SparseMatrixCSC{T}, layoutname::AbstractString="drl") where T
    ei,ej,ew = findnz(A)
    edgelist = [(ei[i]-1,ej[i]-1) for i = 1:length(ei)]
    nverts = size(A)
    G = igraph.Graph(nverts, edges=edgelist, directed=true)
    xy = G.layout(layoutname)
    xy = [Float64(xy[i][j]) for i in 1:length(xy),  j in 1:length(xy[1])]
    # xy = [Float64(get(xy,i-1,j-1)) for i in 1:length(xy),  j in 1:length(xy[1])]
    # get(o, i - 1)
end

# evaluation metric 1
function evaluate_degree_exact(A,xy,degs,multfactor)
    data = copy(xy')
    kdtree = KDTree(data)
    TP = 0
    for i = 1:A.n
        d = degs[i]+1
        point = xy[i,:]
        idxs, dists = knn(kdtree, point, floor(Int,multfactor*d), true)
        true_neighbors = A[i,:].nzind
        deduced_neignbors = idxs
        correct_deduced_neighbors = intersect(true_neighbors,deduced_neignbors)
        TP += length(correct_deduced_neighbors)
    end
    return TP/sum(degs)
end

# evaluation metric 2
function evaluate_degree_relaxed(A,xy,filename)
    data = copy(xy')
    kdtree = KDTree(data)
    distsplot = plot(legend=false)
    mygif = @animate for i = 1:A.n
        point = xy[i,:]
        idxs, dists = knn(kdtree, point, A.n, true)
        true_neighbors = A[i,:].nzind
        plot!(distsplot,dists)
        plot(dists,legend=false)
        xvals = findin_index(true_neighbors,idxs)
        distvals = dists[xvals]
        scatter!(xvals,distvals,color=:red)
    end
    giffile = filename*".gif"
    pdffile = filename*".png"
    savefig(distsplot,pdffile)
    gif(mygif, fps = 10, giffile)
end

function TuranShadow_W(A::SparseMatrixCSC{T},myfn,from_k,upto_k,scaleA,t) where T
    GW = spzeros(size(A)...)
    for k = from_k:upto_k
        approxval,clique_sets = TuranShadow(A,k,t)
        csets_new = copy(clique_sets)
        map(i->csets_new[i] = sort(csets_new[i]),1:length(csets_new))
        csets_new = unique(csets_new)
        clique_sets = copy(csets_new)
        
        if approxval == 0
            @warn "largest clique size found is $(k-1)"
            break
        end
        ei,ej = create_C(clique_sets)
        C = sparse(ei,ej,1,length(clique_sets),A.n)
        W = C'*C
        W = myfn(k)*W
        W = W - spdiagm(0=>diag(W))
        GW = max.(W,GW) #instead of GW += W
    end

    GW += scaleA*A
    
    dropzeros!(GW)
    return GW
end

function TuranShadow_layout(A::SparseMatrixCSC{T},myfn,from_k,upto_k,scaleA,t) where T
    GW = spzeros(size(A)...)
    # refcliquesets = []
    for k = from_k:upto_k
        approxval,clique_sets = TuranShadow(A,k,t)

        # map!(i->sort(clique_sets[i]),clique_sets,1:length(clique_sets))
        # clique_sets = unique(clique_sets)

        csets_new = copy(clique_sets)
        map(i->clique_sets[i] = sort(clique_sets[i]),1:length(clique_sets))
        map!(i->sort(clique_sets[i]),clique_sets,1:length(clique_sets))
        clique_sets = unique(clique_sets)
        clique_sets = copy(csets_new)
        
        if approxval == 0
            @warn "largest clique size found is $(k-1)"
            break
        end

        # refcliquesets = vcat(refcliquesets,clique_sets)
        ei,ej = create_C(clique_sets)
        C = sparse(ei,ej,1,length(clique_sets),A.n)
        W = C'*C
        W = myfn(k)*W
        W = W - spdiagm(0=>diag(W))
        GW += W #GW = max.(W,GW) #instead of GW += W
    end

    # ei,ej = create_C(clique_sets)
    # C = sparse(ei,ej,1,length(clique_sets),A.n)
    # W = C'*C
    # W = myfn(k)*W
    # W = W - spdiagm(0=>diag(W))
    # GW += W #GW = max.(W,GW) #instead of GW += W

    GW += scaleA*A
    
    dropzeros!(GW)
    x2,x3,l = x2_x3_from_spectral_embedding(GW)
    xy = hcat(x2,x3)
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

function quick_scatter(xy,labels,labelsgt)
    p = plot()
    for i = 1:length(labelsgt)
        ids = findall(labels[:].==labelsgt[i])
        scatter!(xy[ids,1],xy[ids,2])
    end
    return p
end

function quick_scatter(xy,labels,labelsgt,constraintnb)
    p = plot()
    for i = 1:length(labelsgt)
        ids = findall(labels[:].==labelsgt[i])
        if length(ids) >= constraintnb
            scatter!(xy[ids,1],xy[ids,2])
        end
    end
    return p
end

function my_plot_graph(A,xy,displayedges;labels=[])
    if displayedges
        #plot edges first
        graphplot(findnz(triu(A,1))[1:2]..., x =xy[:,1], y=xy[:,2],
            markercolor=nothing, markerstrokecolor=nothing,linecolor=:black, linealpha=0.2, linewidth=0.7,
            axis_buffer=0.02, background=nothing,label = "")
    else
        plot(background=nothing,legend=false,axis=false)
    end
    if isempty(labels)
        scatter!(xy[:,1],xy[:,2],markercolor=1,markerstrokecolor=nothing,markersize=2,markeralpha=0.2)
    else
        labelsgt = unique(labels)
        for i = 1:length(labelsgt)
            ids = findall(labels[:].==labelsgt[i])
            if labelsgt[i] == 0
                # do nothing
                # scatter!(xy[ids,1],xy[ids,2],markercolor=:gray,markerstrokecolor=nothing,markersize=2,markeralpha=0.8)
            else
                scatter!(xy[ids,1],xy[ids,2],markercolor=i,markerstrokecolor=nothing,markersize=2,markeralpha=0.8)
            end
        end
    end
    return plot!()
end 


@sk_import manifold : TSNE
# @pyimport numpy as np
np = pyimport("numpy")