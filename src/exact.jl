export exact_free_energy
export exact_page

function bitarray(v::Vector{T}, num_bit::Int)::BitArray{2} where T<:Number
    #ba = BitArray{2}(0, 0)
    ba = BitArray(undef, 0, 0)
    ba.len = 64*length(v)
    ba.chunks = UInt64.(v)
    ba.dims = (64, length(v))
    view(ba, 1:num_bit, :)
end
#=
function exact_free_energy(K)
    nbits = size(K, 1)
    configs = bitarray(collect(0:(1<<nbits-1)), nbits)
    energy = sum(configs .* (K*configs), dims=1)
    -log( sum( exp.(-energy)))
end
=#
function exact_page(nbits,η)
    configs = bitarray(collect(1:(1<<nbits)), nbits)
    energy = page(configs, η)
    -log( sum( exp.(-energy)))
end

function exact_page_entropy(nbits, η)
    configs = bitarray(collect(1:(1<<nbits)), nbits)
    page(configs, η)
end

function exact_page_entropyrate(nbits, η)
    configs = bitarray(collect(1:(1<<nbits)), nbits)
    configs = configs[:, 1:(size(configs, 2)-2)]
    x = sum(configs, dims=1)
    page(configs, η)./(-x.+nbits)
end

function exact_FF(nbits, m)
    configs = bitarray(collect(1:(1<<nbits)), nbits)
    energy = FF(configs, m)
    -log( sum( exp.(-energy)))
end

function exact_FF_entropy(nbits, m)
    configs = bitarray(collect(1:(1<<nbits)), nbits)
    FF(configs, m)
end
