using GLMakie, GeometryBasics
using Plots, ImageCore, ImageFiltering, Statistics, FileIO
using Images

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

# 3D wireframe
V3D = Float32.([0 0 0; 1 0 0; 1 1 0; 0 1 0; 0 0 1; 1 0 1; 1 1 1; 0 1 1])
E3D = [(1,2),(2,3),(3,4),(4,1),
       (5,6),(6,7),(7,8),(8,5),
       (1,5),(2,6),(3,7),(4,8)]

# TODO: not the correct size, inner square needs to be smaller
# 2D wireframe
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
el_true = 0.5*pi
obs_img, _ = render_wireframe_makie(V3D, E3D; width=256, height=256, azimuth=az_true, elevation=el_true)
save("obs_img.png", obs_img)

#######################################
##         Scoring Functions          ##
#######################################

function _edge_mask(a)
    gray_vals = Float32.(channelview(Gray.(a)))
    return gray_vals .< 0.5f0
end

const OBS_EDGE_MASK = Ref{Union{Nothing, Matrix{Float32}}}(nothing)
const OBS_EDGE_COUNT = Ref{Float32}(0.0f0)
const OBS_BLURRED = Ref{Union{Nothing, Matrix{Float32}}}(nothing)

function cache_observation!(obs_img; sig::Real=2.0)
    m_obs = _edge_mask(obs_img)
    obs_f = Float32.(m_obs)
    
    OBS_EDGE_MASK[] = obs_f
    OBS_EDGE_COUNT[] = sum(obs_f)
    OBS_BLURRED[] = imfilter(obs_f, Kernel.gaussian((sig, sig)))
    nothing
end

function edge_proximity_score(pred_img; sig::Real=2.0, verbose::Bool=false)
    if OBS_BLURRED[] === nothing
        error("Observation not cached!")
    end
    
    m_pred = _edge_mask(pred_img)
    
    H = min(size(OBS_EDGE_MASK[], 1), size(m_pred, 1))
    W = min(size(OBS_EDGE_MASK[], 2), size(m_pred, 2))
    
    obs_f = @view OBS_EDGE_MASK[][1:H, 1:W]
    blurred_obs = @view OBS_BLURRED[][1:H, 1:W]
    m_pred_view = @view m_pred[1:H, 1:W]
    pred_f = Float32.(m_pred_view)
    
    blurred_pred = imfilter(pred_f, Kernel.gaussian((sig, sig)))
    
    obs_edge_pixels = findall(x -> x > 0.5, obs_f)
    pred_edge_pixels = findall(x -> x > 0.5, pred_f)
    
    if isempty(pred_edge_pixels) || isempty(obs_edge_pixels)
        return 0.0
    end
    
    pred_to_obs = mean([blurred_obs[idx] for idx in pred_edge_pixels])
    obs_to_pred = mean([blurred_pred[idx] for idx in obs_edge_pixels])
    
    if pred_to_obs < 1e-8 || obs_to_pred < 1e-8
        score = 0.0
    else
        score = 2 * (pred_to_obs * obs_to_pred) / (pred_to_obs + obs_to_pred)
    end
    
    return score
end

#######################################
##         Optimization              ##
#######################################

function camera_ransac(vertices, edges;
                       width::Int=256, height::Int=256,
                       num_candidates::Int=500)
    best_score = -Inf
    best_az = 0.0
    best_el = 0.0

    for i in 1:num_candidates
        az = 2pi * rand()
        el = -pi/2 + pi * rand()
        pred_img, _ = render_wireframe_makie(vertices, edges;
                                             width=width, height=height,
                                             azimuth=az, elevation=el)
        score = edge_proximity_score(pred_img; verbose=false)
        
        if score > best_score
            best_score, best_az, best_el = score, az, el
        end
    end
    
    best_az, best_el, best_score
end

function optimize_camera(vertices, edges, az_init, el_init;
                        width::Int=256, height::Int=256,
                        num_iterations::Int=100, initial_step::Float64=0.3)
    
    best_az = az_init
    best_el = el_init
    pred_img, _ = render_wireframe_makie(vertices, edges;
                                         width=width, height=height,
                                         azimuth=best_az, elevation=best_el)
    best_score = edge_proximity_score(pred_img; verbose=false)
    
    step_size = initial_step
    no_improvement_count = 0
    
    az_hist = Float64[best_az]
    el_hist = Float64[best_el]
    score_hist = Float64[best_score]
    
    for iter in 1:num_iterations
        improved = false
        
        for _ in 1:30
            daz = randn() * step_size
            del = randn() * step_size
            
            new_az = best_az + daz
            new_el = clamp(best_el + del, -pi/2, pi/2)
            new_az = mod(new_az, 2*pi)
            
            pred_img, _ = render_wireframe_makie(vertices, edges;
                                                 width=width, height=height,
                                                 azimuth=new_az, elevation=new_el)
            new_score = edge_proximity_score(pred_img; verbose=false)
            
            if new_score > best_score
                best_az = new_az
                best_el = new_el
                best_score = new_score
                improved = true
                no_improvement_count = 0
                break
            end
        end
        
        push!(az_hist, best_az)
        push!(el_hist, best_el)
        push!(score_hist, best_score)
        
        if !improved
            no_improvement_count += 1
            if no_improvement_count > 10
                step_size *= 0.7
                no_improvement_count = 0
                if step_size < 0.001
                    break
                end
            end
        end
    end
    
    (; az_hist, el_hist, score_hist, best_az, best_el, best_score)
end

function run_optimization_chain(label::String, vertices, edges, obs_img;
                               width::Int=256, height::Int=256,
                               num_iterations::Int=100)

    mkpath("frames_$label")

    # RANSAC initialization
    az_init, el_init, score_init = camera_ransac(vertices, edges;
                                                  width=width, height=height,
                                                  num_candidates=500)
    
    # Optimize: only updates if the score is better
    result = optimize_camera(vertices, edges, az_init, el_init;
                            width=width, height=height,
                            num_iterations=num_iterations,
                            initial_step=0.3)
    
    for (i, (az, el)) in enumerate(zip(result.az_hist, result.el_hist))
        pred_img, _ = render_wireframe_makie(vertices, edges;
                                             width=width, height=height,
                                             azimuth=az, elevation=el)
        
        fname = joinpath("frames_$label", "frame_" * lpad(string(i), 4, '0') * ".png")
        save(fname, pred_img)
    end
    
    # Save CSV
    csv_path = "chain_$label.csv"
    open(csv_path, "w") do io
        println(io, "iter,azimuth,elevation,edge_score")
        for i in 1:length(result.az_hist)
            println(io, string(i, ",",
                               result.az_hist[i], ",",
                               result.el_hist[i], ",",
                               result.score_hist[i]))
        end
    end
    
    idx_best = argmax(result.score_hist)
    
    println("\n[$label] RESULTS:")
    println("  Best iteration: $idx_best")
    println("  Best azimuth: $(round(result.best_az, digits=4)) (true: $az_true)")
    println("  Best elevation: $(round(result.best_el, digits=4)) (true: $el_true)")
    println("  Best edge score: $(round(result.best_score, digits=4))")

    (az_hist=result.az_hist, el_hist=result.el_hist, score_hist=result.score_hist,
     best_az=result.best_az, best_el=result.best_el, best_score=result.best_score,
     csv_path=csv_path)
end

#######################################
##             Main                  ##
#######################################

cache_observation!(obs_img; sig=2.0)

res3d = run_optimization_chain("3d", V3D, E3D, obs_img;
                              width=256, height=256,
                              num_iterations=100)

res2d = run_optimization_chain("2d", V2D, E2D, obs_img;
                              width=256, height=256,
                              num_iterations=100)

prior_3d = 0.5
prior_2d = 1 - prior_3d

logp3 = log(prior_3d) + log(res3d.best_score + 1e-10)
logp2 = log(prior_2d) + log(res2d.best_score + 1e-10)
m = max(logp3, logp2)
p3d = exp(logp3 - m) / (exp(logp3 - m) + exp(logp2 - m))
p2d = 1 - p3d

println("\n" * "="^60)
println("FINAL RESULTS")
println("="^60)
println("Final edge score 3D: $(round(res3d.best_score, digits=4))")
println("Final edge score 2D: $(round(res2d.best_score, digits=4))")
println("Score difference: $(round(res3d.best_score - res2d.best_score, digits=4))")
println("\nP(3D | image) = $(round(p3d, digits=4))")
println("P(2D | image) = $(round(p2d, digits=4))")
println("\n3D chain CSV: $(res3d.csv_path)")
println("2D chain CSV: $(res2d.csv_path)")
println("="^60)
