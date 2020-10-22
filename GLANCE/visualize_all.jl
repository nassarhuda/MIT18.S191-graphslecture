include("ecc.jl")
include("evaluation_metrics.jl")
function visualize_all(filename,A,displayedges;labels=[],TSfunction=x->x,fromk=3,tok=20,trialnb=50000,mydims = [5,10,25])

    # just a couple assertions
    cc = scomponents(A)
    @assert length(cc.sizes) == 1
    @assert issymmetric(A)

    # just force compilation:
    Atemp = sprand(20,20,0.2);
    Atemp = Atemp-Diagonal(Atemp);
    Atemp = spones(Atemp);
    Atemp = max.(Atemp,Atemp');
    x2_x3_from_spectral_embedding(Atemp); 
    igraph_layout(Atemp,"drl"); 
    igraph_layout(Atemp,"lgl"); 
    TuranShadow_layout(Atemp,TSfunction,fromk,tok,1,trialnb); 
    x2_x3_from_spectral_embedding(Atemp;tol=1e-12,maxiter=300,dense=96,nev=mydims[end],checksym=true);
    TuranShadow_matrix(Atemp,TSfunction,fromk,tok,1,trialnb);
    n2v(Atemp);
    # ecc(Atemp,tok,1,TSfunction)
    println("real experiment starting now")


    @show mydims # these are the TSNE dimensions used.
    alltimes = zeros(6+3*length(mydims))
    allmetrics = zeros(6+3*length(mydims),5)
    n = size(A,1)
    pair_nodes = hcat(rand(1:n,10000),rand(1:n,10000));

	## figure 1 - normalized laplacian
	curtime = @elapsed x2,x3,l = x2_x3_from_spectral_embedding(A); 
    alltimes[1] = curtime
    allmetrics[1,:] = evaluate_drawing(A,hcat(x2,x3),pair_nodes)
	my_plot_graph(A,hcat(x2,x3),displayedges;labels=labels)
	savefig(join([filename,"_Laplacian_A.png"]))

	##figure 2 - DRL
	curtime = @elapsed xy_drl = igraph_layout(A,"drl"); 
    alltimes[2] = curtime
    allmetrics[2,:] = evaluate_drawing(A,xy_drl,pair_nodes)
	my_plot_graph(A,xy_drl,displayedges;labels=labels)
	savefig(join([filename,"_DRL.png"]))

    ##figure 3 - LGL
    # curtime = @elapsed xy_lgl = igraph_layout(A,"lgl"); 
    # alltimes[3] = curtime
    # allmetrics[3,:] = evaluate_drawing(A,xy_lgl,pair_nodes)
    # my_plot_graph(A,xy_lgl,displayedges;labels=labels)
    # savefig(join([filename,"_LGL.png"]))

	##figure 4 - TuranShadow
	curtime = @elapsed xyTR = TuranShadow_layout(A,TSfunction,fromk,tok,1,trialnb); 
    alltimes[4] = curtime
    allmetrics[4,:] = evaluate_drawing(A,xyTR,pair_nodes)
	my_plot_graph(A,xyTR,displayedges;labels=labels)
	savefig(join([filename,"_LapTuranShadow.png"]))

    tfn = TSNE(n_components=2)

	##figure set 5 - TSNE on L(A)
	pl = Array{Any}(undef,length(mydims))
    curtimeref = @elapsed x2,x3,Xref = x2_x3_from_spectral_embedding(A;tol=1e-12,maxiter=300,dense=96,nev=mydims[end],checksym=true)
	for (ni,nd) in enumerate(mydims)	
        X = Xref[:,1:nd]
    	# X = np.array(Matrix(X))
    	curtime = curtimeref + @elapsed xytsneL4 = tfn.fit_transform(X)
    	pl_ni = my_plot_graph(A,xytsneL4[:,1:2],displayedges;labels=labels)
    	Plots.title!(join(["ndims = ",nd]),titlecolor=:black)
    	alltimes[4+ni] = curtime
        allmetrics[4+ni,:] = evaluate_drawing(A,xytsneL4[:,1:2],pair_nodes)
        savefig(join([filename,"_",nd,"_TSNE_LapA_.png"]))
    end
    # plot(pl[1],pl[2],pl[3],pl[4],layout=(2,2),size = (900,400))
    # savefig(join([filename,"_TSNE_LapA.png"]))

    ##figure set 6 - TSNE on L(TS(A))
    curtimeref = @elapsed G = TuranShadow_matrix(A,TSfunction,fromk,tok,1,trialnb)
    curtimeref += @elapsed x2,x3,Xref = x2_x3_from_spectral_embedding(G;tol=1e-12,maxiter=300,dense=96,nev=mydims[end],checksym=true)
    for (ni,nd) in enumerate(mydims)
		X = Xref[:,1:nd]
    	# X = np.array(Matrix(X))
    	curtime = curtimeref + @elapsed xytsneL4 = tfn.fit_transform(X)
    	pl_ni = my_plot_graph(A,xytsneL4[:,1:2],displayedges;labels=labels)
    	Plots.title!(join(["ndims = ",nd]))
    	alltimes[4+length(mydims)+ni] = curtime
        allmetrics[4+length(mydims)+ni,:] = evaluate_drawing(A,xytsneL4[:,1:2],pair_nodes)
        savefig(join([filename,"_", nd,"_TSNE_LapTuranShadow_.png"]))
    end
    # plot(pl[1],pl[2],pl[3],pl[4],layout=(2,2),size = (900,400))
    # savefig(join([filename,"_TSNE_LapTuranShadow.png"]))

    # figure 6 - Node2Vec
    curtime = @elapsed X = n2v(A)
    curtime += @elapsed xyN2V = tfn.fit_transform(X)
    my_plot_graph(A,xyN2V,displayedges;labels=labels)
    savefig(join([filename,"_N2V.png"]))
    alltimes[4+length(mydims)+length(mydims)+1] = curtime
    allmetrics[4+length(mydims)+length(mydims)+1,:] = evaluate_drawing(A,xyN2V,pair_nodes)


    # ## figure 7 - ecc from Shweta
    # curtimeref = @elapsed GW = ecc(A,tok,1,TSfunction)
    # curtimeref += @elapsed x2,x3,Xref = x2_x3_from_spectral_embedding(GW;tol=1e-12,maxiter=300,dense=96,nev=mydims[end],checksym=true)
    # xycoord = hcat(x2,x3)
    # my_plot_graph(A,xycoord,displayedges;labels=labels)
    # savefig(join([filename,"_ECC.png"]))
    # alltimes[4+length(mydims)+length(mydims)+2] = curtimeref
    # allmetrics[4+length(mydims)+length(mydims)+2,:] = evaluate_drawing(A,xycoord,pair_nodes)

    # ## figure 8
    # for (ni,nd) in enumerate(mydims)
    #     X = Xref[:,1:nd]
    #     # X = np.array(Matrix(X))
    #     # tfn = TSNE(n_components=2)
    #     curtime = curtimeref + @elapsed xytsneL4 = tfn.fit_transform(X)
    #     pl_ni = my_plot_graph(A,xytsneL4[:,1:2],displayedges;labels=labels)
    #     Plots.title!(join(["ndims = ",nd]))
    #     alltimes[6+length(mydims)+length(mydims)+ni] = curtime
    #     allmetrics[6+length(mydims)+length(mydims)+ni,:] = evaluate_drawing(A,xytsneL4[:,1:2],pair_nodes)
    #     savefig(join([filename,"_", nd,"_TSNE_Lap_ECC_TuranShadow_.png"]))
    # end

    # plot(alltimes,xticks = (1:length(alltimes),
        methods = ["L(A)", "DRL", "LGL", "L(TS(A))", "L(A)-TSNE1", "L(A)-TSNE2", "L(A)-TSNE3", "L(TS(A))-TSNE1", "L(TS(A))-TSNE2", "L(TS(A))-TSNE3", "N2V",
        "ECC","ECC-TSNE1", "ECC-TSNE2", "ECC-TSNE3"]
        methodstats = hcat(methods,alltimes,allmetrics)

    CSV.write(join([filename,"_stats.txt"]),Tables.table(methodstats),header=["method","time","Kendall","Spearman",
                                                                "closeness_metric","closeness_metric_weighted","randomwalks_metric"])
    # savefig(join([filename,"_alltimes.png"]))
    mycmd = `find . -name "$filename*" -exec mv "{}" ./visualization-results \;`
    Base.run(mycmd)

    return methodstats
end