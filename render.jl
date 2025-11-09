using GLMakie, GeometryBasics

function render_wireframe_makie(vertices::AbstractMatrix,
                                edges::Vector{<:Tuple{Int,Int}};
                                width::Int=256, height::Int=256,
                                azimuth::Real=pi, elevation::Real=pi/6,
                                linewidth::Real=2,
                                linecolor=:black,
                                bgcolor=:white,
                                show::Bool=false)
    fig = Figure(size=(width, height), backgroundcolor=bgcolor)
    ax  = Axis3(fig[1,1]; aspect=:data, perspectiveness=0.9, backgroundcolor=bgcolor)
    hidedecorations!(ax); hidespines!(ax)
    pts  = Point3f.(eachrow(vertices))
    segs = [pts[i] => pts[j] for (i,j) in edges]
    linesegments!(ax, segs; linewidth, color=linecolor)
    ax.azimuth[] = azimuth
    ax.elevation[] = elevation
    autolimits!(ax)
    show && display(fig)
    img = colorbuffer(fig.scene)
    return img, fig
end

V = Float32.([0 0 0; 1 0 0; 1 1 0; 0 1 0; 0 0 1; 1 0 1; 1 1 1; 0 1 1])
E = [(1,2),(2,3),(3,4),(4,1),(5,6),(6,7),(7,8),(8,5),(1,5),(2,6),(3,7),(4,8)]

img, fig = render_wireframe_makie(V, E; width=256, height=256, azimuth=1*pi, elevation=0.5*pi)
save("obs_fig.png", fig)
save("obs_pixels.png", img)
