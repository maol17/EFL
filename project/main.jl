using Flux.Optimise: ADAM, update!, gradient
using StatsBase
using Random
Random.seed!(42)

using PyCall
using PyPlot

push!(LOAD_PATH,"E:/暑研/VAN.jl/src")
using VAN

function train_page(nbits, nsamples, nhiddens, niter, η, lr)
    model = VAN.build_model(nbits, nhiddens)
    exact = zeros(niter)
    exact .= VAN.exact_page(nbits, η)

    θ = VAN.model_parameters(model)
    f = zeros(niter)
    for i = 1:niter
        _, g, _ = gradient(VAN.loss, η, model, nsamples)
        #g = VAN.unpack_gradient(model, g)
        for j = 1:length(g)
            update!(ADAM(lr), θ[j], g[j])
        end
        VAN.model_dispatch!(model, θ)
        loss = VAN.loss(η, model, nsamples)
        f[i] = loss
        println("$i $loss")
    end

    pygui(:true)
    fig,ax=plt.subplots()
    ax[:plot](1:niter,f,linewidth=1,label=L"$loss$")
    ax[:plot](1:niter,exact,linewidth=1,label=L"$exact$")
    plt.ylabel("niter")
    plt.xlabel("EF")
    plt.title("page state, nbits = $nbits ,η = $η")
    ax[:legend]()

    model
end

function page_entropy(model::VAN.AutoRegressiveModel, η)
    nbits = model.nbits

    exact = VAN.exact_page_entropy(nbits, η)
    learnt = VAN.page_generate(model, η)

    x = 1:(2^model.nbits)
    fig,ax=plt.subplots()
    ax[:plot](x,learnt',linewidth=1,label=L"$learned$")
    ax[:plot](x,exact',linewidth=1,label=L"$exact$")
    plt.ylabel("entropy")
    plt.xlabel("configuration")
    plt.title("page state, nbits = $nbits ,η = $η")
    ax[:legend]()
#=
    fig,ax2=plt.subplots()
    ax2[:plot](x,(learnt'-exact'),linewidth=1,label=L"$deviation$")
    plt.ylabel("deviation")
    plt.xlabel("configuration")
    plt.title("page state, nbits = $nbits ,η = $η")
    ax[:legend]()
=#
end

function train_FF(nbits, nsamples, nhiddens, niter, m, lr)
    model = VAN.build_model(nbits, nhiddens)
    exact = zeros(niter)
    exact .= VAN.exact_FF(nbits, m)

    θ = VAN.model_parameters(model)
    f = zeros(niter)
    for i = 1:niter
        _, g, _ = gradient(VAN.loss, m, model, nsamples)
        #g = VAN.unpack_gradient(model, g)
        for j = 1:length(g)
            update!(ADAM(lr), θ[j], g[j])
        end
        VAN.model_dispatch!(model, θ)
        loss = VAN.loss(m, model, nsamples)
        f[i] = loss
        println("$i $loss")
    end

    pygui(:true)
    fig,ax=plt.subplots()
    ax[:plot](1:niter,f,linewidth=1,label=L"$loss$")
    ax[:plot](1:niter,exact,linewidth=1,label=L"$exact$")
    plt.ylabel("niter")
    plt.xlabel("EF")
    plt.title("free fermion state, nbits = $nbits ,η = $η")
    ax[:legend]()

    model
end

function FF_entropy(model::VAN.AutoRegressiveModel, η)
    nbits = model.nbits

    exact = VAN.exact_FF_entropy(nbits, η)
    learnt = VAN.FF_generate(model, η)

    x = 1:(2^model.nbits)
    fig,ax=plt.subplots()
    ax[:plot](x,learnt',linewidth=1,label=L"$learned$")
    ax[:plot](x,exact',linewidth=1,label=L"$exact$")
    plt.ylabel("entropy")
    plt.xlabel("configuration")
    plt.title("free energy state, nbits = $nbits ,η = $η")
    ax[:legend]()
end

nbits = 5
nsamples = 1000
nhiddens = [5]
niter = 3000
η = 0

model = train_FF(nbits, nsamples, nhiddens, niter, η, 0.001)
FF_entropy(model, η)
