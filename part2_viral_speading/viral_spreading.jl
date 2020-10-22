#code (modified slightly) and notes from:
# https://www.cs.purdue.edu/homes/dgleich/cs515-2020/julia/viral-spreading.html
# https://www.cs.purdue.edu/homes/dgleich/cs515-2020/notes/viral-spread-and-matrices.pdf


using Random
Random.seed!(0) # ensure repeatable results...
using NearestNeighbors, Distributions, SparseArrays

function spatial_graph_edges(n::Integer,d::Integer;degreedist=LogNormal(log(4),1))
  xy = rand(d,n)
  T = BallTree(xy)
  # form the edges for sparse
  ei = Int[]
  ej = Int[]
  for i=1:n
    deg = min(ceil(Int,rand(degreedist)),n-1)
    idxs, dists = knn(T, xy[:,i], deg+1)
    for j in idxs
      if i != j
        push!(ei,i)
        push!(ej,j)
      end
    end
  end
  return xy, ei, ej
end
function spatial_network(n::Integer, d::Integer; degreedist=LogNormal(log(3),1))
  xy, ei, ej = spatial_graph_edges(n, d;degreedist=degreedist)
  A = sparse(ei,ej,1,n,n)
  return max.(A,A'), xy
end
A,xy = spatial_network(1000, 2)

##
using Plots
function plotgraph(A::SparseMatrixCSC,xy::AbstractArray{T,2};kwargs...) where T
  px,py = zeros(T,0),zeros(T,0)
  P = [px,py]
  rows = rowvals(A)
  skip = NaN.*xy[:,begin] # first row
  for j=1:size(A,2) # for each column
    for nzi in nzrange(A, j)
      i = rows[nzi]
      if i > j
        push!.(P, @view xy[:,i])
        push!.(P, @view xy[:,j])
        push!.(P, skip)
      end
    end
  end
  plot(px,py;framestyle=:none,legend=false,kwargs...)
end

plotgraph(A,xy,alpha=0.25); 
scatter!(xy[1,:],xy[2,:],
  markersize=2, markerstrokewidth=0, color=1)

p = 0.2
function evolve(x::Vector,  p::Real, A::AbstractMatrix)
    log_not_infected = log.(1 .- p.*x)
    y = 1 .- exp.(A*log_not_infected)
    y = max.(y, x)
end
  
x = zeros(size(A,1))
x[2] = 1.0
anim = @animate for i=1:100
    global x = evolve(x, p, A)
    plotgraph(A,xy,alpha=0.25, size=(600,600))
    scatter!(xy[1,:],xy[2,:],
    markersize=5, markerstrokewidth=0,
    color=1, marker_z = x, clim=(0,1))
    title!("$i")
end
gif(anim, "viral-first.gif", fps=10)

function evolve_self(x::Vector,  p::Real, A::AbstractMatrix)
    log_not_infected = log.(1 .- p.*x)
    y = (1 .- exp.(A*log_not_infected).*(1 .- x))
    y = max.(y, x)
end
p = 0.2
x = zeros(size(A,1))
x[2] = 1.0
anim = @animate for i=1:100
global x = evolve_self(x, p, A)
plotgraph(A,xy,alpha=0.25, size=(600,600))
scatter!(xy[1,:],xy[2,:],
    markersize=5, markerstrokewidth=0,
    color=1, marker_z = x, clim=(0,1))
title!("$i")
end
gif(anim, "viral-self.gif", fps=10)

##
function evolve_self(x::Vector,  p::Real, A::AbstractMatrix)
    log_not_infected = log.(1 .- p.*x)
    y = (1 .- exp.(A*log_not_infected).*(1 .- x))
    y = max.(y, x)
  end
  p = 0.01
  x = zeros(size(A,1))
  x[2] = 1.0
  anim = @animate for i=1:500
    global x = evolve_self(x, p, A)
    plotgraph(A,xy,alpha=0.25, size=(600,600))
    scatter!(xy[1,:],xy[2,:],
      markersize=5, markerstrokewidth=0,
      color=1, marker_z = x, clim=(0,1))
    title!("$i")
  end
  gif(anim, "viral-self-slow-long.gif", fps=10)
