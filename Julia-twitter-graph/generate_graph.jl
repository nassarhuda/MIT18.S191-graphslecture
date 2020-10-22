using LinearAlgebra
using PyCall
using CSV
using LinearAlgebra
using SparseArrays
using MatrixNetworks
using Plots
@pyimport twint as twint

# first, get all the tweets + save them
c = twint.Config()
c.Hide_output = true
c.Store_csv = true
c.Output = "julialang_tweets.csv"
c.Search = "#julialang"
c.Min_likes = 5
curtweets = twint.run.Search(c)

# second, get all the usernames that tweeted those tweets
julia_tweets = CSV.read("julialang_tweets.csv")
unique_usernames = unique(julia_tweets[!,:username])

# third, get the list of people these usernames follow
for (curi,cur_username) in enumerate(unique_usernames)
    @show curi, 1032
    h = twint.Config()
    h.Username = cur_username
    h.Hide_output = true
    h.Store_csv = true
    h.Output = join(["following_",cur_username,".csv"])
    curtweets = twint.run.Following(h)
end

# fourth build the social network
n = length(unique_usernames)
A = spzeros(Int,n,n)
for i = 1:n
    curusername = unique_usernames[i]
    try
        curfollowing = CSV.read(join(["following_",curusername,".csv"]))[:username]
        vi = map(i->in(unique_usernames[i],curfollowing),1:length(unique_usernames))
        A[:,i] = vi
    catch e
        println("This person doesn't follow anyone, moving on...")
    end
end
A

J = max.(A,A')

J,lccv = largest_component(J)
labels = unique_usernames[lccv]

########### get the bios
@pyimport tweepy as tweepy
auth = tweepy.OAuthHandler("M2zSxJtxokO2NI8nmETvJQ","XmpkkEeE6qR4W70KVpawtiVrYLxtsQOczDEnMP3RYk");
auth.set_access_token("24695635-t4eOQ21l0myDq4F6jVshJJR9ZcX46Vq8njFBwHfvU","2e36yHmzXEY3E6IjA03VKdCd77J8aqB0vT4mXywYajuH5");
api = tweepy.API(auth)
bios = similar(labels)
for i = 1:length(labels)
    try
        user = api.get_user(labels[i])
        bios[i] = user.description
    catch e
        println("Person not found, using empty bio")
        bios[i] = ""
    end
end
########### end of getting the bios
findcounts = x->[length(findall("julia",lowercase(x))),
                           length(findall("python",lowercase(x))),
                           length(findall("rstats",lowercase(x)))+length(findall("rlang",lowercase(x)))]
allcounts = findcounts.(bios)

# define some colors
juliacolor = RGB(0.706,0.322,0.804)
pythoncolor = RGB(1.0, 0.870,0.341)
rcolor = RGB(0.0862,0.360,0.666)
# juliapythoncolor = RGB(218/255,152/255,146/255)
juliapythoncolor = RGB(255/255,0/255,0/255)
# juliaRcolor = RGB(101/255,87/255,188/255)
juliaRcolor = RGB(255/255,165/255,0/255)
# pythonRcolor = RGB(139/255,157/255,129/255)
pythonRcolor = RGB(0/255,255/255,0/255)
alllangs = RGB(0/255,0/255,0/255)

singlecolor = [juliacolor,pythoncolor,rcolor]
doublenotcolor = [pythonRcolor,juliaRcolor,juliapythoncolor]

selectcolor = x->
       length(findall(x.>0)) == 0 ? RGB(0.827,0.827,0.827) :
       length(findall(x.>0)) == 1 ? singlecolor[findfirst(x.>0)] :
       length(findall(x.>0)) == 2 ? doublenotcolor[findfirst(x.==0)] :
       alllangs

scatternodecolors = selectcolor.(allcounts)

degsA = sum(A,dims=2)[:][lccv]
degsJ = sum(J,dims=2)[:]
plot(sort(degsJ,rev=true))

include("../../../github/GLANCE/includeall.jl")
include("../../../github/GLANCE/vis_paper.jl")
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

xy = GLANCE(J;mydims = 15);
generate_nice_plot(xy,J,"julialang_twitter_plot_colored2.png";labels=[],mymarkeralpha=0.95,mymarkersize = 3,invpermval=true,rgbvals=scatternodecolors)

plot!(Shape([],[]),color=juliacolor,label="Julia")
plot!(Shape([],[]),color=pythoncolor,label="Python")
plot!(Shape([],[]),color=rcolor,label="R")

plot!(Shape([],[]),color=juliapythoncolor,label="Julia/Python")
plot!(Shape([],[]),color=pythonRcolor,label="Python/R")
plot!(Shape([],[]),color=juliaRcolor,label="Julia/R")

plot!(Shape([],[]),color=alllangs,label="all")
# plot!(legend=:outerright)
savefig("julialang_twitter_plot_colored_labeled2.png")

x = xy[:,1]
y = xy[:,2]
x = invperm(sortperm(x))
y = invperm(sortperm(y))
xyINV = Float64.(hcat(x,y))

# usrs_to_annotate = sortperm(degsJ[:],rev=true)[1:20]
usrs_to_annotate = sortperm(degsA[:],rev=true)[1:200]
xdiff = zeros(200)
ydiff = zeros(200)
# ydiff[3]= 9
# xdiff[3]= 15
# ydiff[2]= -5
# xdiff[2]= 5
# ydiff[4] = -10
# xdiff[17] = -30
for i = 1:length(usrs_to_annotate)
    annotate!([(xdiff[i]+xyINV[usrs_to_annotate[i],1],ydiff[i]+xyINV[usrs_to_annotate[i],2],
        text(labels[usrs_to_annotate[i]],5,:left))])
end


savefig("julialang_twitter_plot_colored_labeled_annotated200.png")

togdegs = sortperm(degsJ,rev=true)[1:30]  
labels[togdegs]


