using GLMakie, GeometryBasics, Random

#######################################
##   Render Wireframe w/ GLMakie     ##
#######################################

function render_wireframe_makie(vertices::AbstractMatrix,
                                edges::Vector{<:Tuple{Int,Int}};
                                width::Int=256, height::Int=256,
                                azimuth::Real=pi, elevation::Real=pi/6,
                                linewidth::Real=2,
                                linecolor=:black,
                                bgcolor=:white)
    fig = Figure(size=(width, height), backgroundcolor=bgcolor)
    ax  = Axis3(fig[1,1]; aspect=:data, perspectiveness=0.9, backgroundcolor=bgcolor)
    hidedecorations!(ax); hidespines!(ax)
    pts  = Point3f.(eachrow(vertices))
    segs = [pts[i] => pts[j] for (i,j) in edges]
    linesegments!(ax, segs; linewidth, color=linecolor)
    ax.azimuth[] = azimuth
    ax.elevation[] = elevation
    img = colorbuffer(fig.scene)
    return img, fig
end

az = pi/2
el = pi/3
V2D = Float32.([
    0.0000 0.3156 0.0000;
    0.5187 0.0057 0.0000;
    0.9982 0.3726 0.0000;
    0.5152 0.0038 0.0000;
    0.4385 0.9962 0.0000;
    1.0000 0.3726 0.0000;
    0.0000 0.3137 0.0000;
    0.4670 0.8574 0.0000;
    0.4385 1.0000 0.0000;
    0.2513 0.5494 0.0000;
    0.4670 0.8593 0.0000;
    0.2460 0.5456 0.0000;
    0.5045 0.3498 0.0000;
    0.5169 0.0000 0.0000;
    0.7326 0.5798 0.0000;
    0.5045 0.3460 0.0000;
    1.0000 0.3745 0.0000;
    0.7308 0.5837 0.0000;
    0.4635 0.8593 0.0000;
    0.7291 0.5837 0.0000;
    0.2496 0.5456 0.0000;
    0.5045 0.3479 0.0000
])
E2D = [(1,2),(1,10),(3,4),(5,6),(5,7),(8,9),(11,12),(13,14),(15,16),(17,18),(19,20),(21,22)]

img, fig = render_wireframe_makie(V2D, E2D; azimuth=az, elevation=el)
save("test.png", fig)