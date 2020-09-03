using LinearAlgebra
using Zygote, BackwardsLinalg

function get_energy(K::Matrix{T}, samples) where T <: Real
    energy = sum(samples .* (K*samples), dims=1)
end

function free_energy(m, model::AutoRegressiveModel, samples) where T <: Real
    return mean(FF(samples, m) .+ get_logp(model, samples))
end

function free_energy_exact(m, model::AutoRegressiveModel, samples) where T <: Real
    p = exp.(get_logp(model, samples))
    return sum(p.*(FF(samples, m) .+ get_logp(model, samples)))[1]
end

function loss(m, model::AutoRegressiveModel, nbatch::Int) where T <: Real
    samples = gen_samples(model, nbatch)
    free_energy(m, model, samples)
end

function loss_reinforce(m, model::AutoRegressiveModel, samples) where T <: Real
    e = FF(samples, m)
    logp = get_logp(model, samples)
    f = e .+ logp
    b = mean(f)
    return mean(logp.* (f .- b))
end

function grad_model(η, model::AutoRegressiveModel, samples) where T <: Real
    model_grad = gradient(loss_reinforce, η, model, samples)[2]
    (model_grad.W..., model_grad.b...)
end
#page state
function page(samples, η)
    L = size(samples, 1)
    samples = 2*(samples.-0.5)
    x = sum(samples, dims = 1)
    -log.((cosh.(η*x))./cosh(η*L))
end

function page_generate(model::AutoRegressiveModel, η)
    configs = bitarray(collect(1:(1<<model.nbits)), model.nbits)
    p = get_logp(model, configs)
    -p .- exact_page(model.nbits, η)
end

function page_generate_rate(model::AutoRegressiveModel, η)
    configs = bitarray(collect(1:(1<<model.nbits)), model.nbits)
    f = free_energy_exact(η, model, configs)
    configs = configs[:, 1:(size(configs, 2)-2)]
    x = sum(configs, dims=1)
    p = get_logp(model, configs)
    (-p .- f)./(-x.+model.nbits)
end
#free fermion model
function FF_H(n::Int, m)
    H = zeros(ComplexF64,n, n)
    for i = 1:n-1
        H[i,i+1] = 1im*(1+m*(-1)^i)
    end
    H[1,n] = -1im*(1+m*(-1)^n)
    Hermitian(H)
end

function FF_C(n::Int, m)
    H = FF_H(n, m)
    F = eigen(H)
    cut = 0
    for i = 1:n-1
        if F.values[i]<0&&F.values[i+1]>0
            cut = i
            break
        end
    end

    C = zeros(ComplexF64, n, n)
    for i = 1:cut
        ψ = F.vectors[:,i]
        C.=C.+(ψ*conj.(transpose(ψ)))
    end
    C
end

function get_S(M)
    v, U = BackwardsLinalg.symeigen(M)
    l = length(v)
    S = -sum(real.(log.(((v.*v).+(-v.+1).*(-v.+1)).+fill(1E-7,(l)))))[1]
end

function FF(samples, m)
    n = size(samples, 1)
    l = size(samples, 2)
    C = FF_C(n ,m)
    Id = Matrix{ComplexF64}(I,n,n)

    A = [((samples[:,i]*transpose(samples[:,i]))).*C for i=1:l]
    S = [get_S(A[i]) for i=1:l]
    S
end

function FF_generate(model::AutoRegressiveModel, m)
    configs = bitarray(collect(1:(1<<model.nbits)), model.nbits)
    p = get_logp(model, configs)
    -p .- exact_FF(model.nbits, m)
end
