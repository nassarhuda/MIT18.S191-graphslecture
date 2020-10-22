include("hexbingraphplots.jl")

function generate_nice_plot(xy,A,imagefilename;labels=[],mymarkeralpha=0.1,mymarkersize = 5,invpermval=false,colorscale=[],rgbvals=[])
	gr()
	if invpermval
        x = xy[:,1]
        y = xy[:,2]
        x = invperm(sortperm(x))
        y = invperm(sortperm(y))
        xy = Float64.(hcat(x,y))
    end

    if !isempty(labels)
		labels = Int.(labels)
		ulabels = unique(labels)
		d = Dict(sort(ulabels) .=> 1:length(ulabels))
		color_palette = distinguishable_colors(length(ulabels));color_palette[1] = RGB(1,1,1);
		colorstouse = map(i->color_palette[d[labels[i]]],1:length(labels))
	end
	src,dst = findnz(triu(A,1))[1:2]
	xd = xy[:,1]
	yd = xy[:,2]
	@time hedges1 = hexbingraphplot(src,dst,xd,yd,nbins=1000)
	##
	xsh,ysh,vh = shapecoords(hedges1)
	##
	axform = vh
	avals = axform/maximum(axform)
	avals = map(x -> x <= 1/4 ? sqrt(x) : ((sqrt(1/4)-1/4) + x), avals )
	avals = 1/1.25.*avals
	plot()
	if isempty(labels) && isempty(colorscale) && !isempty(rgbvals)
		scatter!(xd,yd,color=rgbvals,alpha=mymarkeralpha,markersize=mymarkersize,colorbar=false,markerstrokewidth=0,background=:white,label="")
	elseif isempty(labels) && isempty(colorscale)
		scatter!(xd,yd,color=2,alpha=mymarkeralpha,markersize=mymarkersize,legend=false,colorbar=false,markerstrokewidth=0,background=:white)
	elseif isempty(colorscale)
		ids = findall(labels.!=0)
		scatter!(xd[ids],yd[ids],color=colorstouse[ids],alpha=mymarkeralpha,markersize=mymarkersize,legend=false,colorbar=false,markerstrokewidth=0,background=:white)
	else
		cscale = colorscale ./ maximum(colorscale)
		rgbvals = get(ColorSchemes.valentine,cscale)
		ids = findall(colorscale.!=0)
		scatter!(xd[ids,:],yd[ids,:],color=rgbvals,alpha=mymarkeralpha,markersize=mymarkersize,legend=false,colorbar=false,markerstrokewidth=0,background=:white)
	end
	p = plot!(xsh, ysh, seriestype=:shape, alpha=avals, fillcolor=:darkblue,linealpha=0, framestyle=:none,background=:white,label="")
	##
	plot!(dpi=300,size=(800,800))
	savefig(imagefilename)
end