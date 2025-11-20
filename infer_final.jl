using GLMakie, GeometryBasics
using Statistics
using LinearAlgebra
using Images

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

function img_to_gray_array(img)
    Float32.(channelview(Gray.(img)))
end

function edge_mask(img; thresh::Real=0.5)
    gray = img_to_gray_array(img)
    BitMatrix(gray .< thresh)
end

function chamfer_distance(obs_edges::BitMatrix, pred_edges::BitMatrix)
    As = findall(obs_edges)
    Bs = findall(pred_edges)

    if isempty(As) && isempty(Bs)
        return 0.0
    elseif isempty(As) || isempty(Bs)
        return Inf
    end

    dA = 0.0
    for a in As
        ax, ay = Tuple(a)
        min_d2 = Inf
        for b in Bs
            bx, by = Tuple(b)
            dx = ax - bx
            dy = ay - by
            d2 = dx*dx + dy*dy
            if d2 < min_d2
                min_d2 = d2
            end
        end
        dA += sqrt(min_d2)
    end
    dA /= length(As)

    dB = 0.0
    for b in Bs
        bx, by = Tuple(b)
        min_d2 = Inf
        for a in As
            ax, ay = Tuple(a)
            dx = bx - ax
            dy = by - ay
            d2 = dx*dx + dy*dy
            if d2 < min_d2
                min_d2 = d2
            end
        end
        dB += sqrt(min_d2)
    end
    dB /= length(Bs)

    (dA + dB) / 2
end

function logsumexp(v::AbstractVector{<:Real})
    m = maximum(v)
    m + log(sum(exp.(v .- m)))
end

function log_marginal_likelihood_shape_chamfer(vertices, edges, obs_img, poses;
                                               width::Int=256, height::Int=256,
                                               sig_d_squared::Real=1.0)
    obs_edges = edge_mask(obs_img)
    logLs = Vector{Float64}(undef, length(poses))
    for (i, (az, el)) in pairs(poses)
        pred_img, _ = render_wireframe_makie(vertices, edges;
                                             width=width, height=height,
                                             azimuth=az, elevation=el)
        pred_edges = edge_mask(pred_img)
        D = chamfer_distance(obs_edges, pred_edges)
        if isfinite(D)
            logLs[i] = -(D^2) / (2sig_d_squared)
        else
            logLs[i] = -Inf
        end
    end
    logsumexp(logLs) - log(length(poses))
end

function make_pose_grid(num_pose_samples::Int)
    n_az = floor(Int, sqrt(num_pose_samples))
    n_el = cld(num_pose_samples, n_az)
    az_vals = collect(range(0, 2pi; length=n_az+1))[1:end-1]
    el_vals = collect(range(-pi/2, pi/2; length=n_el))
    [(az, el) for az in az_vals for el in el_vals]
end

function compare_shapes_marginal(obs_img, shapes;
                                 width::Int=256, height::Int=256,
                                 sig_d_squared::Real=1.0,
                                 num_pose_samples::Int=500,
                                 priors::AbstractVector{<:Real}=fill(1/length(shapes), length(shapes)))
    poses = make_pose_grid(num_pose_samples)
    logps = Vector{Float64}(undef, length(shapes))
    for (k, ((V, E), prior)) in enumerate(zip(shapes, priors))
        logps[k] = log(prior) + log_marginal_likelihood_shape_chamfer(V, E, obs_img, poses;
                                                                      width=width,
                                                                      height=height,
                                                                      sig_d_squared=sig_d_squared)
    end
    m = maximum(logps)
    unnorm = exp.(logps .- m)
    unnorm ./ sum(unnorm)
end

V3D = Float32.([0 0 0; 1 0 0; 1 1 0; 0 1 0; 0 0 1; 1 0 1; 1 1 1; 0 1 1])
E3D = [(1,2),(2,3),(3,4),(4,1),
       (5,6),(6,7),(7,8),(8,5),
       (1,5),(2,6),(3,7),(4,8)]

V2D = Float32.([
    0.0 0.0 0.0;
    1.0 0.0 0.0;
    1.0 1.0 0.0;
    0.0 1.0 0.0;
    0.25 0.25 0.0;
    0.75 0.25 0.0;
    0.75 0.75 0.0;
    0.25 0.75 0.0
])
E2D = [(1,2),(2,3),(3,4),(4,1),
       (5,6),(6,7),(7,8),(8,5),
       (1,5),(2,6),(3,7),(4,8)]

az_true = pi
el_true = 0.5pi
obs_img, _ = render_wireframe_makie(V3D, E3D;
                                    width=256, height=256,
                                    azimuth=az_true,
                                    elevation=el_true)
save("obs_img.png", obs_img)

shapes = [(V3D, E3D), (V2D, E2D)]

post = compare_shapes_marginal(obs_img, shapes;
                               width=256, height=256,
                               sig_d_squared=1.0,
                               num_pose_samples=1000)

println("\nBAYESIAN MARGINAL-LIKELIHOOD ANALYSIS")
println("P(3D | image) = $(round(post[1], digits=4))")
println("P(2D | image) = $(round(post[2], digits=4))")
