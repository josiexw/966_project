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

wireframes = [
    (1, "cube",       square_V,   square_E),
    (2, "tri_prism",  tri_V,      tri_E),
    (3, "hex_prism",  hex_V,      hex_E),
    (4, "l_prism",    l_V,        l_E),
    (5, "square_2d",  square2d_V, square2d_E),
    (6, "tri_2d",     tri2d_V,    tri2d_E),
    (7, "hex_2d",     hex2d_V,    hex2d_E),
    (8, "l_2d",       l2d_V,      l2d_E),
]

print(wireframes)