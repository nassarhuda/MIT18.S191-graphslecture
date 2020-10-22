using TOML
using Printf

# this variable needs to be changed based on your own installation directory
path = joinpath(homedir(),".julia","registries","General")
packages_dict = TOML.parsefile(joinpath(path,"Registry.toml"))["packages"]
# this variable needs to be changed based on your own installation directory
const STDLIB_DIR = "/Applications/Julia-1.5.app/Contents/Resources/julia/share/julia/stdlib/v1.5/"
const STDLIBS = readdir(STDLIB_DIR)

for (i, stdlib) in enumerate(STDLIBS)
    if isfile(joinpath(STDLIB_DIR, stdlib, "Project.toml"))
        proj = TOML.parsefile(joinpath(STDLIB_DIR, stdlib, "Project.toml"))
        packages_dict[proj["uuid"]] = proj
    end
end
pkg_keys = collect(keys(packages_dict))
pkg_ids = Dict(pkg_keys[i] => i-1 for i = 1:length(pkg_keys))

ei,ej = [],[]
for i = 1:length(pkg_keys)
    pkg_id = pkg_ids[pkg_keys[i]]
    if haskey(packages_dict[pkg_keys[i]],"path")
        dep_path = joinpath(path,packages_dict[pkg_keys[i]]["path"],"Deps.toml")
        if isfile(dep_path)
            dep_dict = TOML.parsefile(dep_path)
            for key in keys(dep_dict)
                tmp_dict = dep_dict[key]
                for pkg_name in keys(tmp_dict)
                    push!(ei,pkg_ids[tmp_dict[pkg_name]])
                    push!(ej,pkg_id)
                end
            end
        end
    else
        if haskey(packages_dict[pkg_keys[i]],"deps")
            for key in packages_dict[pkg_keys[i]]["deps"]
                push!(ei,pkg_ids[key[2]])
                push!(ej,pkg_id)
            end
        end
    end
end

wptr = open("Julia-dependency-graph.smat","w")
@printf(wptr,"%d\t%d\t%d\n",length(pkg_ids),length(pkg_ids),length(ei))
for k = 1:length(ei)
    @printf(wptr,"%d\t%d\t%d\n",ei[k],ej[k],1)
end
close(wptr)

pkg_ids_rev = Dict(pkg_ids[key]=>key for key in keys(pkg_ids))
wptr = open("Julia-dependency-graph.labels","w")
for k = 0:length(pkg_keys)-1
    uuid = pkg_ids_rev[k]
    name = packages_dict[uuid]["name"]
    @printf(wptr,"%s\t%s\n",uuid,name)
end
close(wptr)
