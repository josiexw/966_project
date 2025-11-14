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

#######################################
##            Make Shapes            ##
#######################################

square_pts = Float32.([0 0; 1 0; 1 1; 0 1])
tri_pts    = Float32.([0 0; 1 0; 0.5 0.8660254])
l_pts      = Float32.([0 0; 1 0; 1 0.3; 0.3 0.3; 0.3 1; 0 1])
hex_pts = Array{Float32}(undef, 6, 2)
for k in 0:5
    θ = 2f0 * Float32(pi) * Float32(k) / 6f0
    x = cos(θ)
    y = sin(θ)
    hex_pts[k + 1, 1] = (x + 1f0) / 2f0
    hex_pts[k + 1, 2] = (y + 1f0) / 2f0
end

function make_polygon(points::AbstractMatrix, h::Real)
    n = size(points, 1)
    V = Array{Float32}(undef, 2n, 3)
    for i in 1:n
        V[i, 1] = Float32(points[i, 1])
        V[i, 2] = Float32(points[i, 2])
        V[i, 3] = 0f0
        V[i + n, 1] = Float32(points[i, 1])
        V[i + n, 2] = Float32(points[i, 2])
        V[i + n, 3] = Float32(h)
    end
    base_edges = [(i, i % n + 1) for i in 1:n]
    top_edges  = [(i + n, (i % n) + 1 + n) for i in 1:n]
    side_edges = [(i, i + n) for i in 1:n]
    return V, vcat(base_edges, top_edges, side_edges)
end

function make_polygon_2d(points::AbstractMatrix)
    n = size(points, 1)
    V = Array{Float32}(undef, n, 3)
    for i in 1:n
        V[i, 1] = Float32(points[i, 1])
        V[i, 2] = Float32(points[i, 2])
        V[i, 3] = 0f0
    end
    edges = [(i, i % n + 1) for i in 1:n]
    return V, edges
end

V = Float32.([0 0 0; 1 0 0; 1 1 0; 0 1 0; 0 0 1; 1 0 1; 1 1 1; 0 1 1])
E = [(1,2),(2,3),(3,4),(4,1),(5,6),(6,7),(7,8),(8,5),(1,5),(2,6),(3,7),(4,8)]

square_V, square_E = make_polygon(square_pts, 1.0)
tri_V,   tri_E   = make_polygon(tri_pts, 1.0)
hex_V,   hex_E   = make_polygon(hex_pts, 1.0)
l_V,     l_E     = make_polygon(l_pts, 1.0)

square2d_V, square2d_E = make_polygon_2d(square_pts)
tri2d_V,   tri2d_E     = make_polygon_2d(tri_pts)
hex2d_V,   hex2d_E     = make_polygon_2d(hex_pts)
l2d_V,     l2d_E       = make_polygon_2d(l_pts)

shape_library = Dict(
    "cube"  => (square_V, square_E),
    "tri_prism"     => (tri_V, tri_E),
    "hex_prism"     => (hex_V, hex_E),
    "l_prism"       => (l_V, l_E),
    "square_2d"     => (square2d_V, square2d_E),
    "tri_2d"        => (tri2d_V, tri2d_E),
    "hex_2d"        => (hex2d_V, hex2d_E),
    "l_2d"          => (l2d_V, l2d_E),
)

views = [1, 2, 3, 4]

function random_view()
    az = 2pi * rand()
    el = (pi / 2) * rand()
    return az, el
end

function render_canonical_shapes(shape_library, views; max_imgs::Int=20)
    mkpath("figures")
    n_saved = 0

    for (shape_name, (Vshape, Eshape)) in shape_library
        for _ in views
            azv, elv = random_view()
            img, fig = render_wireframe_makie(
                Vshape, Eshape;
                width=256, height=256,
                azimuth=azv, elevation=elv
            )
            az_str = string(round(azv, digits=2))
            el_str = string(round(elv, digits=2))
            fname = joinpath("figures", "$(shape_name)_$(az_str)_$(el_str).png")
            save(fname, img)
            n_saved += 1
            if n_saved >= max_imgs
                return
            end
        end
    end
end

render_canonical_shapes(shape_library, views; max_imgs=40)
