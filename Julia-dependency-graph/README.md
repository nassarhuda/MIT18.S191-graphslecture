## Data Description
This is the Julia packages dependency graph generated on 2019-10-10. If package A depends on package B, then there will be a directed link from B to A.

## Files
+ Julia-dependency-graph.jl
   * the actual Julia scipt we use to generate the graph. To run it, "Julia-1.1" and the "TOML" package are required. Refer to [https://julialang.org/]() for more details on how to install Julia as well as how to install packages in Julia.
   *  You also need to change the "path" variable to point to the installation directory of your own Julia application.
+ Julia-dependency-graph.labels
	* stores the uuid and package name for each node in the graph sorted by the node index increasingly
+ Julia-dependency-graph.smat
	* An .smat representation of the dependency graph.
