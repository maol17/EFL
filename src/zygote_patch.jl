using Zygote
using Zygote: @adjoint, @nograd

@adjoint function free_energy(η, model::AutoRegressiveModel, samples)
    free_energy(η, model, samples), function (adjy)
        adjmodel = grad_model(η, model, samples) .* adjy
        return (nothing, adjmodel, nothing)
    end
end
@nograd gen_samples, createmasks, FF_C
