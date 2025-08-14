# These are the default values
ENV["COLORSCHEME"] = "JuliaDynamics" # or others, see docs
ENV["BGCOLOR"] = :white        # anything for `backgroundcolor` of Makie
ENV["AXISCOLOR"] = :black            # color of all axis elements (labels, spines, ticks)

using MakieForProjects # this has now set the theme already!
using CairoMakie # enable backend

# you may further edit the set theme by using
Makie.update_theme!(;
    linewidth = 1.0,
    # ScatterLines = (markersize = 5),
    # size = (figwidth, figheight),
)