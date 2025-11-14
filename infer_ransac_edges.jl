using GLMakie, GeometryBasics

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

V = Float32.([0 0 0; 1 0 0; 1 1 0; 0 1 0; 0 0 1; 1 0 1; 1 1 1; 0 1 1])
E = [(1,2),(2,3),(3,4),(4,1),(5,6),(6,7),(7,8),(8,5),(1,5),(2,6),(3,7),(4,8)]

az = pi
el = 0.5*pi
obs_img, obs_fig = render_wireframe_makie(V, E; width=256, height=256, azimuth=az, elevation=el);
println("True azimuth: ", az);
println("True elevation: ", el);
save("obs_img.png", obs_img);

#######################################
##         RANSAC + Drift MH         ##
#######################################
using Gen, Plots, ImageCore

function _edge_mask(a)
    img = a isa AbstractMatrix{<:Colorant} ? a :
          a isa AbstractArray{<:Real,3}    ? colorview(RGB, permutedims(a, (2,3,1))) :
                                            a
    g = channelview(Gray.(img))
    g_ch = @view g[1, :, :]
    g_ch .< 0.5f0
end

struct EdgeScore <: Gen.Distribution{Any} end
const edge_score = EdgeScore()

Gen.logpdf(::EdgeScore, y, renderer::Function, lambda::Real) = begin
    m_obs = _edge_mask(y)
    m_mu  = _edge_mask(renderer())

    H = min(size(m_obs, 1), size(m_mu, 1))
    W = min(size(m_obs, 2), size(m_mu, 2))
    m_obs = @view m_obs[1:H, 1:W]
    m_mu  = @view m_mu[1:H, 1:W]

    inter = sum(m_obs .& m_mu)
    union = sum(m_obs .| m_mu) + 1e-6
    iou = inter / union

    lambda * iou
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

    {:img} ~ edge_score(renderer, lambda)
    return (az, el)
end

function view_loglik(vertices, edges, obs_img;
                     width::Int=256, height::Int=256, lambda::Real=10.0,
                     az::Real=0.0, el::Real=0.0)
    renderer = () -> begin
        img, _ = render_wireframe_makie(vertices, edges;
                                        width=width, height=height,
                                        azimuth=az, elevation=el)
        img
    end
    Gen.logpdf(edge_score, obs_img, renderer, lambda)
end

function camera_ransac(vertices, edges, obs_img;
                       width::Int=256, height::Int=256, lambda::Real=10.0,
                       num_candidates::Int=200)
    best_ll = -Inf
    best_az = 0.0
    best_el = 0.0

    for _ in 1:num_candidates
        az = 2pi * rand()
        el = -pi/2 + pi * rand()
        ll = view_loglik(vertices, edges, obs_img;
                         width=width, height=height, lambda=lambda,
                         az=az, el=el)
        if ll > best_ll
            best_ll, best_az, best_el = ll, az, el
        end
    end
    best_az, best_el
end

const SIG_AZ = 0.05
const SIG_EL = 0.05

@gen function camera_drift_proposal(prev_trace)
    az_prev = prev_trace[:az]
    el_prev = prev_trace[:el]
    az ~ normal(az_prev, SIG_AZ)
    el ~ normal(el_prev, SIG_EL)
end

function gaussian_drift_update(tr)
    (tr, _) = mh(tr, camera_drift_proposal, ())
    tr
end

@gen function ransac_proposal(prev_trace, vertices, edges, obs_img,
                              width::Int, height::Int, lambda::Real)
    az_guess, el_guess = camera_ransac(vertices, edges, obs_img;
                                       width=width, height=height, lambda=lambda)
    az ~ normal(az_guess, 0.1)
    el ~ normal(el_guess, 0.1)
end

function ransac_update(tr, vertices, edges, obs_img;
                       width::Int=256, height::Int=256, lambda::Real=10.0)
    (tr, _) = mh(tr, ransac_proposal, (vertices, edges, obs_img, width, height, lambda))
    for _ in 1:20
        tr = gaussian_drift_update(tr)
    end
    tr
end

function gaussian_drift_inference(vertices, edges, obs_img;
                                  width::Int=256, height::Int=256, lambda::Real=10.0,
                                  steps::Int=1000)
    cons = Gen.choicemap()
    cons[:img] = obs_img
    (tr, _) = Gen.generate(wireframe_camera_model,
                           (vertices, edges, width, height, lambda), cons)
    for _ in 1:steps
        tr = gaussian_drift_update(tr)
    end
    tr
end

function ransac_inference(vertices, edges, obs_img;
                          width::Int=256, height::Int=256, lambda::Real=10.0,
                          steps::Int=200)
    cons = Gen.choicemap()
    cons[:img] = obs_img
    (tr, _) = Gen.generate(wireframe_camera_model,
                           (vertices, edges, width, height, lambda), cons)
    tr = ransac_update(tr, vertices, edges, obs_img;
                       width=width, height=height, lambda=lambda)
    for _ in 1:steps
        tr = gaussian_drift_update(tr)
    end
    tr
end

visualize_trace(tr; title="") = begin
    az = tr[:az]
    el = tr[:el]
    img, _ = render_wireframe_makie(V, E; width=256, height=256, azimuth=az, elevation=el)
    Plots.plot(img, axis=false, border=false, title=title)
end

function animate_drift(tr; steps=500, path="vis.gif", fps=20)
    anim = Plots.@animate for i in 1:steps
        tr = gaussian_drift_update(tr)
        visualize_trace(tr; title="Iteration $i/$steps")
    end
    Plots.gif(anim, path, fps=fps)
    tr
end

tr = ransac_inference(V, E, obs_img; width=256, height=256, lambda=0.05, steps=500);
println("Estimated azimuth:", tr[:az]);
println("Estimated elevation:", tr[:el]);
animate_drift(tr; steps=500, path="vis.gif", fps=20);
