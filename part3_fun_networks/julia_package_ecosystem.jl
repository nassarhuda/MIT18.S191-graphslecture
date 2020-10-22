# fun graphs
using MatrixNetworks
using Plots
using CSV

# paste link XXX
labels = CSV.read("../Julia-dependency-graph/Julia-dependency-graph.labels",header=false)
A = MatrixNetworks.readSMAT("../Julia-dependency-graph/Julia-dependency-graph.smat")
A = max.(A,A')
A = A.>0
include("../GLANCE/includeall.jl")
include("../GLANCE/vis_paper.jl")

#### plotting the graph and running pagerank and other methods...

function GLANCE(A;TSfunction=x->x,fromk=3,tok=20,trialnb=50000,mydims = 25)
    cc = scomponents(A)
    @assert length(cc.sizes) == 1
    @assert issymmetric(A)
    tfn = TSNE(n_components=2)
    # step 1: Turan Shadow reweighting
    G = TuranShadow_matrix(A,TSfunction,fromk,tok,1,trialnb)
    # step 2: Embedding on the new weighted matrix
    x2,x3,X = x2_x3_from_spectral_embedding(G;tol=1e-12,maxiter=300,dense=96,nev=mydims,checksym=true)
    # step 3: transform via t-sne
    xy_coords = tfn.fit_transform(X)
    return xy_coords
end

A,lccv = largest_component(A);
attributes = labels[!,:Column2][lccv]

degsA = sum(A,dims=2);
pkgs_to_annotate = sortperm(degsA[:],rev=true)[1:20]

xy = GLANCE(A;mydims=15);
generate_nice_plot(xy,A,"packages_layout.png";labels=1,invpermval=true,mymarkeralpha=0.6,mymarkersize=2)
x = xy[:,1]
y = xy[:,2]
x = invperm(sortperm(x))
y = invperm(sortperm(y))
xyINV = Float64.(hcat(x,y))

for i = 1:length(pkgs_to_annotate)
    annotate!([(xyINV[pkgs_to_annotate[i],1],xyINV[pkgs_to_annotate[i],2],
        text(attributes[pkgs_to_annotate[i]],8,:left))])
end
savefig("packages_network_julia.png")
#------


n = size(A,1)
D = zeros(n,n);
map(i->(D[:,i],pred) = dijkstra(A,i),1:size(A,1))
maximum(D)

include("pagerank_and_randomwalks.jl")
pr_ours = our_first_pagerank(A,0.85,1000);
pr_MN = MatrixNetworks.pagerank(A,0.85);

pkgs_to_annotate = sortperm(pr_MN[:],rev=true)[1:10]

generate_nice_plot(xy,A,"packages_layout.png";labels=1,invpermval=true,mymarkeralpha=0.6,mymarkersize=2)
for i = 1:length(pkgs_to_annotate)
    annotate!([(xyINV[pkgs_to_annotate[i],1],xyINV[pkgs_to_annotate[i],2],
        text(attributes[pkgs_to_annotate[i]],8,:left))])
end
savefig("packages_network_julia_pagerank_annotated.png")

rw = random_walk(A,1,1_000_000)
h = fit(Histogram, rw,nbins=n)
pp = sortperm(h.weights,rev=true)[1:10]

## clustering coefficient

cc = clustercoeffs(A)
cc[findall(isnan.(cc))] .=0
mean(cc)
