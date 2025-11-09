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
obs_img, obs_fig = render_wireframe_makie(V, E; width=256, height=256, azimuth=az, elevation=el)
println("True azimuth: ", az)
println("True elevation: ", el)
save("obs_img.png", obs_img)


#######################################
##         Integrate w/ Gen          ##
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

function infer_camera(vertices, edges, obs_img; width=size(obs_img,2), height=size(obs_img,1), sig=0.05, steps=500)
    cons = Gen.choicemap()
    cons[:img] = obs_img

    tr, _ = Gen.generate(wireframe_camera_model, (vertices, edges, width, height, sig), cons)

    best = tr
    best_score = Gen.get_score(tr)
    for _ in 1:steps
        tr, _ = Gen.mh(tr, Gen.select(:az))
        tr, _ = Gen.mh(tr, Gen.select(:el))
        s = Gen.get_score(tr)
        if s > best_score
            best = tr
            best_score = s
        end
    end
    return best, best_score
end

trace, score = infer_camera(V, E, obs_img; sig=0.05, steps=500)
az = trace[:az]
el = trace[:el]
best_img, best_fig = render_wireframe_makie(V, E; width=256, height=256, azimuth=az, elevation=el)
println("Predicted azimuth: ", az)
println("Predicted elevation: ", el)
save("infer_img.png", best_img)

# Visualize
cons = Gen.choicemap()
cons[:img] = obs_img

visualize_trace(tr; title="") = begin
    az = tr[:az]
    el = tr[:el]
    img, _ = render_wireframe_makie(V, E; width=256, height=256, azimuth=az, elevation=el)
    Plots.plot(img, axis=false, border=false, title=title)
end

function animate(t; steps=500, path="mh.gif", fps=20)
    viz = Plots.@animate for i in 1:steps
        t, _ = Gen.mh(t, Gen.select(:az))
        t, _ = Gen.mh(t, Gen.select(:el))
        visualize_trace(t; title = "Iteration $i/$steps")
    end
    Plots.gif(viz, path, fps=fps)
end

cons = Gen.choicemap(); cons[:img] = obs_img
t, _ = Gen.generate(wireframe_camera_model, (V, E, 256, 256, 0.05), cons)
animate(t; steps=500, path="vis.gif", fps=20)