using GLMakie, GeometryBasics
using Gen, ImageCore, ImageFiltering, Statistics, FileIO
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
##         Scoring Functions         ##
#######################################

function _edge_mask(a)
    gray_vals = Float32.(channelview(Gray.(a)))
    return gray_vals .< 0.5f0
end

const OBS_EDGE_MASK = Ref{Union{Nothing, Matrix{Float32}}}(nothing)
const OBS_EDGE_COUNT = Ref{Float32}(0.0f0)
const OBS_BLURRED = Ref{Union{Nothing, Matrix{Float32}}}(nothing)

function cache_observation!(obs_img; σ::Real=2.0)
    m_obs = _edge_mask(obs_img)
    obs_f = Float32.(m_obs)
    
    OBS_EDGE_MASK[] = obs_f
    OBS_EDGE_COUNT[] = sum(obs_f)
    OBS_BLURRED[] = imfilter(obs_f, Kernel.gaussian((σ, σ)))
    nothing
end

function edge_proximity_score(pred_img; σ::Real=2.0)
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
    
    blurred_pred = imfilter(pred_f, Kernel.gaussian((σ, σ)))
    
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
##            Gen Model              ##
#######################################

struct EdgeScore <: Gen.Distribution{Any} end
const edge_score = EdgeScore()

Gen.logpdf(::EdgeScore, obs_img, renderer::Function, lambda::Real) = begin
    pred_img = renderer()
    s = edge_proximity_score(pred_img; σ=2.0)
    lambda * s
end

Gen.random(::EdgeScore, renderer::Function, lambda::Real) = renderer()
Gen.has_output_grad(::EdgeScore) = false
Gen.has_argument_grads(::EdgeScore) = (false, false)
Gen.is_discrete(::EdgeScore) = false

@gen function wireframe_camera_model(vertices, edges, width::Int, height::Int, lambda::Real)
    az ~ uniform_continuous(0, 2*pi)
    el ~ uniform_continuous(-pi/2, pi/2)

    renderer = () -> begin
        img, _ = render_wireframe_makie(vertices, edges;
                                        width=width, height=height,
                                        azimuth=az, elevation=el)
        img
    end

    img ~ edge_score(renderer, lambda)
    return (az, el)
end

#######################################
##           RANSAC + Drift          ##
#######################################

function camera_ransac(vertices, edges;
                       width::Int=256, height::Int=256,
                       lambda::Real=10.0,
                       num_candidates::Int=500)
    best_ll = -Inf
    best_az = 0.0
    best_el = 0.0

    for i in 1:num_candidates
        az = 2pi * rand()
        el = -pi/2 + pi * rand()
        pred_img, _ = render_wireframe_makie(vertices, edges;
                                             width=width, height=height,
                                             azimuth=az, elevation=el)
        score = edge_proximity_score(pred_img)
        ll = lambda * score
        
        if ll > best_ll
            best_ll, best_az, best_el = ll, az, el
        end
    end
    
    best_az, best_el
end

@gen function ransac_proposal(prev_trace, vertices, edges,
                              width::Int, height::Int, lambda::Real)
    az_guess, el_guess = camera_ransac(vertices, edges;
                                       width=width, height=height, 
                                       lambda=lambda,
                                       num_candidates=500)
    {:az} ~ normal(az_guess, 0.1)
    {:el} ~ normal(el_guess, 0.1)
end

@gen function drift_proposal(prev_trace, step_size::Float64)
    az_prev = prev_trace[:az]
    el_prev = prev_trace[:el]
    {:az} ~ normal(az_prev, step_size)
    {:el} ~ normal(el_prev, step_size)
end

#######################################
##                MCMC               ##
#######################################

function run_mcmc_chain(label::String, vertices, edges, obs_img;
                       width::Int=256, height::Int=256,
                       lambda::Real=10.0,
                       num_iterations::Int=100)

    mkpath("frames_$label")

    # Initialize trace
    cons = Gen.choicemap()
    cons[:img] = obs_img
    (tr, _) = Gen.generate(wireframe_camera_model,
                           (vertices, edges, width, height, lambda), cons)
    
    println("\n[$label] Initial state:")
    println("  az=$(round(tr[:az], digits=3)), el=$(round(tr[:el], digits=3))")
    
    # RANSAC initialization
    (tr, _) = mh(tr, ransac_proposal, (vertices, edges, width, height, lambda))
    println("[$label] After RANSAC:")
    println("  az=$(round(tr[:az], digits=3)), el=$(round(tr[:el], digits=3))")
    
    # Drift
    for i in 1:50
        (tr, accepted) = mh(tr, drift_proposal, (0.15,))
        if i <= 5 || i == 50
            println("  Refine step $i: accepted=$accepted, az=$(round(tr[:az], digits=3)), el=$(round(tr[:el], digits=3))")
        end
    end
    
    az_hist = Float64[]
    el_hist = Float64[]
    score_hist = Float64[]
    accept_count = 0
    
    step_size = 0.2
    
    for iter in 1:num_iterations
        # MCMC drift
        iter_accepts = 0
        for _ in 1:10
            (tr, accepted) = mh(tr, drift_proposal, (step_size,))
            if accepted
                iter_accepts += 1
                accept_count += 1
            end
        end
        
        az = tr[:az]
        el = tr[:el]
        pred_img, _ = render_wireframe_makie(vertices, edges;
                                             width=width, height=height,
                                             azimuth=az, elevation=el)
        current_score = edge_proximity_score(pred_img)
        
        push!(az_hist, az)
        push!(el_hist, el)
        push!(score_hist, current_score)
        
        if iter % 20 == 0 || iter <= 5
            println("[$label] Iter $iter: accepts=$iter_accepts/10, score=$(round(current_score, digits=3)), az=$(round(az, digits=3)), el=$(round(el, digits=3))")
        end
        
        fname = joinpath("frames_$label", "frame_" * lpad(string(iter), 4, '0') * ".png")
        save(fname, pred_img)
    end
    
    println("[$label] Total acceptance rate: $(accept_count)/($(num_iterations*10)) = $(round(accept_count/(num_iterations*10), digits=3))")
    
    # Save CSV
    csv_path = "chain_$label.csv"
    open(csv_path, "w") do io
        println(io, "iter,azimuth,elevation,edge_score")
        for i in 1:length(az_hist)
            println(io, string(i, ",",
                               az_hist[i], ",",
                               el_hist[i], ",",
                               score_hist[i]))
        end
    end
    
    idx_best = argmax(score_hist)
    az_best = az_hist[idx_best]
    el_best = el_hist[idx_best]
    score_best = score_hist[idx_best]
    
    println("\n[$label] RESULTS:")
    println("  Best iteration: $idx_best")
    println("  Best azimuth: $(round(az_best, digits=4)) (true: $az_true)")
    println("  Best elevation: $(round(el_best, digits=4)) (true: $el_true)")
    println("  Best edge score: $(round(score_best, digits=4))")

    (az_hist=az_hist, el_hist=el_hist, score_hist=score_hist,
     best_az=az_best, best_el=el_best, best_score=score_best,
     csv_path=csv_path)
end

#######################################
##             Main                  ##
#######################################

cache_observation!(obs_img; σ=2.0)

res3d = run_mcmc_chain("3d", V3D, E3D, obs_img;
                      width=256, height=256,
                      lambda=10.0,
                      num_iterations=100)

res2d = run_mcmc_chain("2d", V2D, E2D, obs_img;
                      width=256, height=256,
                      lambda=10.0,
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
