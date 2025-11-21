using GLMakie, GeometryBasics
using Statistics
using LinearAlgebra
using Images

function render_wireframe_makie(vertices::AbstractMatrix,
                                edges::Vector{<:Tuple{Int,Int}};
                                width::Int=256, height::Int=256,
                                azimuth::Real=pi, elevation::Real=pi/6,
                                perspectiveness::Real=0.5,
                                linewidth::Real=2,
                                linecolor=:black,
                                bgcolor=:white)
    fig = Figure(size=(width, height), backgroundcolor=bgcolor)
    ax  = Axis3(fig[1,1]; aspect=:data, perspectiveness=perspectiveness, backgroundcolor=bgcolor)
    hidedecorations!(ax); hidespines!(ax)
    pts  = Point3f.(eachrow(vertices))
    segs = [pts[i] => pts[j] for (i,j) in edges]
    linesegments!(ax, segs; linewidth, color=linecolor)
    ax.azimuth[] = azimuth
    ax.elevation[] = elevation
    autolimits!(ax)
    img = colorbuffer(fig.scene)
    return img, fig
end

function img_to_gray_array(img)
    Float32.(channelview(Gray.(img)))
end

function edge_mask(img; thresh::Real=0.5)
    gray = img_to_gray_array(img)
    BitMatrix(gray .< thresh)
end

function edge_coords(edges::BitMatrix)
    idxs = findall(edges)
    n = length(idxs)
    xs = Array{Float64}(undef, n)
    ys = Array{Float64}(undef, n)
    for (i, idx) in enumerate(idxs)
        ys[i] = idx[1]
        xs[i] = idx[2]
    end
    hcat(xs, ys)
end

# Make scale and transition invariant
function normalize_points(pts::AbstractMatrix{<:Real})
    if size(pts, 1) == 0
        return Array{Float64}(undef, 0, 2)
    end
    ptsf = Array{Float64}(pts)
    μ = vec(mean(ptsf, dims=1))
    pts0 = ptsf .- μ'
    r = maximum(sqrt.(sum(abs2, pts0; dims=2)))
    if r == 0.0
        return pts0
    end
    pts0 ./ r
end

function chamfer_distance(obs_edges::BitMatrix, pred_edges::BitMatrix)
    As_raw = edge_coords(obs_edges)
    Bs_raw = edge_coords(pred_edges)

    if size(As_raw, 1) == 0 && size(Bs_raw, 1) == 0
        return 0.0
    elseif size(As_raw, 1) == 0 || size(Bs_raw, 1) == 0
        return Inf
    end

    As = normalize_points(As_raw)
    Bs = normalize_points(Bs_raw)

    dA = 0.0
    for i in 1:size(As, 1)
        ax = As[i, 1]
        ay = As[i, 2]
        min_d2 = Inf
        for j in 1:size(Bs, 1)
            bx = Bs[j, 1]
            by = Bs[j, 2]
            dx = ax - bx
            dy = ay - by
            d2 = dx*dx + dy*dy
            if d2 < min_d2
                min_d2 = d2
            end
        end
        dA += sqrt(min_d2)
    end
    dA /= size(As, 1)

    dB = 0.0
    for j in 1:size(Bs, 1)
        bx = Bs[j, 1]
        by = Bs[j, 2]
        min_d2 = Inf
        for i in 1:size(As, 1)
            ax = As[i, 1]
            ay = As[i, 2]
            dx = bx - ax
            dy = by - ay
            d2 = dx*dx + dy*dy
            if d2 < min_d2
                min_d2 = d2
            end
        end
        dB += sqrt(min_d2)
    end
    dB /= size(Bs, 1)

    (dA + dB) / 2
end

function logsumexp(v::AbstractVector{<:Real})
    m = maximum(v)
    m + log(sum(exp.(v .- m)))
end

function make_pose_grid(num_pose_samples::Int, p_vals::Vector{<:Real})
    n_az = floor(Int, sqrt(num_pose_samples))
    n_el = cld(num_pose_samples, n_az)
    az_vals = collect(range(0, 2pi; length=n_az+1))[1:end-1]
    el_vals = collect(range(-pi/2, pi/2; length=n_el))
    [(az, el, p) for az in az_vals for el in el_vals for p in p_vals]
end

function compare_shapes_marginal_with_topk(obs_img, shapes;
                                           width::Int=256, height::Int=256,
                                           sig_d_squared::Real=1.0,
                                           num_pose_samples::Int=500,
                                           priors::AbstractVector{<:Real}=fill(1/length(shapes), length(shapes)),
                                           k_top::Int=5,
                                           basename::AbstractString="shape",
                                           p_vals::Vector{<:Real} = [0.9])
    poses = make_pose_grid(num_pose_samples, p_vals)
    obs_edges = edge_mask(obs_img)

    S = length(shapes)
    logps = Vector{Float64}(undef, S)

    for (s, ((V, E), prior)) in enumerate(zip(shapes, priors))
        logLs = fill(-Inf, length(poses))

        for (i, (az, el, p)) in pairs(poses)
            pred_img, _ = render_wireframe_makie(V, E;
                                                 width=width, height=height,
                                                 azimuth=az, elevation=el, perspectiveness=p)
            pred_edges = edge_mask(pred_img)
            D = chamfer_distance(obs_edges, pred_edges)
            if isfinite(D)
                logLs[i] = -(D^2) / (2 * sig_d_squared)
            end
        end

        logps[s] = log(prior) + logsumexp(logLs) - log(length(poses))

        idxs = sortperm(logLs; rev=true)[1:min(k_top, length(logLs))]

        open("$(basename)_$(s+1)D_top$(length(idxs)).txt", "w") do io
            for (rank, idx) in enumerate(idxs)
                az, el, p = poses[idx]
                logL = logLs[idx]
                fname = "$(basename)_$(s+1)D_rank$(rank).png"
                pred_img, fig = render_wireframe_makie(V, E;
                                                       width=width, height=height,
                                                       azimuth=az, elevation=el,
                                                       perspectiveness=p)
                save(fname, fig)
                println(io, "rank=$(rank) idx=$(idx) az=$(az) el=$(el) p=$(p) logL=$(logL) file=$(fname)")
            end
        end
    end

    m = maximum(logps)
    unnorm = exp.(logps .- m)
    unnorm ./ sum(unnorm)
end

V2D = Float32.([
    0.0027 0.0049 0.0000;
    0.9973 0.0073 0.0000;
    0.4545 0.9976 0.0000;
    0.4251 0.0024 0.0000;
    0.8422 0.8439 0.0000;
    0.4519 0.9976 0.0000;
    0.1310 0.7902 0.0000;
    0.4519 1.0000 0.0000;
    0.0000 0.0000 0.0000;
    0.1337 0.7951 0.0000;
    1.0000 0.0049 0.0000;
    0.8449 0.8463 0.0000;
    0.7781 0.6195 0.0000;
    0.8449 0.8488 0.0000;
    0.8583 0.0049 0.0000;
    0.7834 0.6268 0.0000;
    0.5321 0.5390 0.0000;
    0.7861 0.6268 0.0000;
    0.5294 0.0049 0.0000;
    0.5321 0.5366 0.0000;
    0.2594 0.5927 0.0000;
    0.5348 0.5390 0.0000;
    0.1310 0.7927 0.0000;
    0.2647 0.5878 0.0000;
    0.1925 0.0073 0.0000;
    0.2620 0.5854 0.0000
])
E2D = [(1,2),(3,4),(5,6),(7,8),(9,10),(11,12),(13,14),(15,16),(17,18),(19,20),(21,22),(23,24),(25,26)]

V3D = Float32.([1.0 0.5 0.0; 0.75 0.9330127 0.0; 0.24999997 0.9330127 0.0; 0.0 0.49999997 0.0; 0.25000006 0.066987276 0.0; 0.7500002 0.066987395 0.0; 1.0 0.5 1.0; 0.75 0.9330127 1.0; 0.24999997 0.9330127 1.0; 0.0 0.49999997 1.0; 0.25000006 0.066987276 1.0; 0.7500002 0.066987395 1.0])
E3D = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 1), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 7), (1, 7), (2, 8), (3, 9), (4, 10), (5, 11), (6, 12)]

az_true = 0.1
el_true = 0.38

obs_img, _ = render_wireframe_makie(V3D, E3D;
                                    azimuth=az_true,
                                    elevation=el_true,
                                    perspectiveness=0.9)
save("obs_img.png", obs_img)

shapes = [(V2D, E2D), (V3D, E3D)]

post = compare_shapes_marginal_with_topk(obs_img, shapes;
                                         width=256, height=256,
                                         sig_d_squared=0.001,
                                         num_pose_samples=1000,
                                         k_top=5,
                                         basename="shape")

println("\nBAYESIAN MARGINAL-LIKELIHOOD ANALYSIS")
println("P(2D | image) = $(round(post[1], digits=4))")
println("P(3D | image) = $(round(post[2], digits=4))")
