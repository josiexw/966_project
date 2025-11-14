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
using Gen, ImageCore, Plots

_to_chw3(a) = begin
    if a isa AbstractMatrix{<:Colorant}
        Float32.(channelview(a)[1:3, :, :])
    elseif a isa AbstractArray{<:Real,3}
        A = Float32.(a)
        size(A,1) == 4 ? @view(A[1:3, :, :]) : A
    else
        Float32.(channelview(a)[1:3, :, :])
    end
end

struct ImageGaussian <: Gen.Distribution{Any} end
const image_gaussian = ImageGaussian()

Gen.logpdf(::ImageGaussian, y, renderer::Function, sig::Real) = begin
    mu = _to_chw3(renderer())
    yy = _to_chw3(y)

    Hμ, Wμ = size(mu,2), size(mu,3)
    Hy, Wy = size(yy,2), size(yy,3)
    H = min(Hμ, Hy); W = min(Wμ, Wy)
    mu = @view mu[:, 1:H, 1:W]
    yy = @view yy[:, 1:H, 1:W]

    s2 = sig^2
    n  = length(yy)
    -0.5f0*n*log(2f0*Float32(pi)*s2) - sum(abs2, yy .- mu) / (2f0*s2)
end

Gen.random(::ImageGaussian, renderer::Function, sig::Real) = renderer()
Gen.has_output_grad(::ImageGaussian) = false
Gen.has_argument_grads(::ImageGaussian) = (false, false)
Gen.is_discrete(::ImageGaussian) = false

@gen function wireframe_camera_model(vertices, edges, width::Int, height::Int, sig::Real)
    az ~ uniform_continuous(0, 2*pi)
    el ~ uniform_continuous(-pi/2, pi/2)

    renderer = () -> begin
        img, _ = render_wireframe_makie(vertices, edges; width=width, height=height,
                                        azimuth=az, elevation=el)
        img
    end

    {:img} ~ image_gaussian(renderer, sig)
    return (az, el)
end

function view_loglik(vertices, edges, obs_img;
                     width::Int=256, height::Int=256, sig::Real=0.05,
                     az::Real=0.0, el::Real=0.0)
    renderer = () -> begin
        img, _ = render_wireframe_makie(vertices, edges;
                                        width=width, height=height,
                                        azimuth=az, elevation=el)
        img
    end
    Gen.logpdf(image_gaussian, obs_img, renderer, sig)
end

function camera_ransac(vertices, edges, obs_img;
                       width::Int=256, height::Int=256, sig::Real=0.05,
                       num_candidates::Int=200)
    best_ll = -Inf
    best_az = 0.0
    best_el = 0.0

    for _ in 1:num_candidates
        az = 2pi * rand()
        el = -pi/2 + pi * rand()
        ll = view_loglik(vertices, edges, obs_img;
                         width=width, height=height, sig=sig,
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

@gen function ransac_proposal(prev_trace, vertices, edges, obs_img, width::Int, height::Int, sig::Real)
    az_guess, el_guess = camera_ransac(vertices, edges, obs_img;
                                       width=width, height=height, sig=sig)
    az ~ normal(az_guess, 0.1)
    el ~ normal(el_guess, 0.1)
end

function ransac_update(tr, vertices, edges, obs_img;
                       width::Int=256, height::Int=256, sig::Real=0.05)
    (tr, _) = mh(tr, ransac_proposal, (vertices, edges, obs_img, width, height, sig))
    for _ in 1:20
        tr = gaussian_drift_update(tr)
    end
    tr
end

function gaussian_drift_inference(vertices, edges, obs_img;
                                  width::Int=256, height::Int=256, sig::Real=0.05,
                                  steps::Int=1000)
    cons = Gen.choicemap()
    cons[:img] = obs_img
    (tr, _) = Gen.generate(wireframe_camera_model, (vertices, edges, width, height, sig), cons)
    for _ in 1:steps
        tr = gaussian_drift_update(tr)
    end
    tr
end

function ransac_inference(vertices, edges, obs_img;
                          width::Int=256, height::Int=256, sig::Real=0.05,
                          steps::Int=200)
    cons = Gen.choicemap()
    cons[:img] = obs_img
    (tr, _) = Gen.generate(wireframe_camera_model, (vertices, edges, width, height, sig), cons)
    tr = ransac_update(tr, vertices, edges, obs_img; width=width, height=height, sig=sig)
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

tr = ransac_inference(V, E, obs_img; width=256, height=256, sig=0.05, steps=500);
println("Estimated azimuth:", tr[:az]);
println("Estimated elevation:", tr[:el]);
animate_drift(tr; steps=500, path="vis.gif", fps=20);
