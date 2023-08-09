using ITensors
using Plots

# assume k = 6
function blockade_radius(rabi_f)
    C6 = 2*pi * 275 * 10^9 # Interaction Coefficient for n (quantum number) = 61 (Hz)
    return (C6/rabi_f)^(1/6)
end

function interaction_strength(a, b, N)
    C6 = 2*pi * 275 * 10^9 # Interaction Coefficient for n (quantum number) = 61 (Hz)
    lattice_spacing = 3.6 # lattice constant

    V_nn = 2*pi*60*(10^6)
    V_nnn = 2*pi*2.3*(10^6)

    return C6 / (abs(a - b) * lattice_spacing)^6
    # if abs(a-b) == 1
    #     return V_nn
    # elseif abs(a-b) == 2
    #     return V_nnn
    # else
    #     return 0
    # end
end

function on_site_detuning(site, N)
    det = 0
    for j=1:N
        if j != site
            det += interaction_strength(site, j, N)
        end
    end
    return -0.5 * det
end

function interactions(site, N)
    os = OpSum()
    for j=1:N
        if j != site
            os += interaction_strength(site, j, N), "Sz", site, "Sz", j
        end
    end
    return os
end

function rydberg(N, sites, rabi_f, delta)
    os = OpSum()
    for site=1:N
        os += 0.5 * rabi_f, "Sx", site
        os += -0.5 * (delta + on_site_detuning(site, N)), "Sz", site
        os += 0.25 * interactions(site, N)
    end
    H = MPO(os, sites)
    
    # println(combiner(sites)*contract(H))

    return H
end

function rydberg3(sites, rabi_f, delta)
    V_nn = 2*pi*60*(10^6)
    V_nnn = 2*pi*2.3*(10^6)
    os = OpSum()
    os+= 0.25*V_nn, "Sz", 1, "Sz", 2
    os+= 0.25*V_nn, "Sz", 2, "Sz", 1
    os+= 0.25*V_nn, "Sz", 2, "Sz", 3
    os+= 0.25*V_nn, "Sz", 3, "Sz", 2
    os+= 0.25*V_nnn, "Sz", 1, "Sz", 3
    os+= 0.25*V_nnn, "Sz", 3, "Sz", 1
    os+= rabi_f, "Sx", 1
    os+= rabi_f, "Sx", 2
    os+= rabi_f, "Sx", 3
    os+= (-delta+0.5*(V_nn + V_nnn)), "Sz", 1
    os+= (-delta+0.5*(V_nn + V_nn)), "Sz", 2
    os+= (-delta+0.5*(V_nn + V_nnn)), "Sz", 3

    H = MPO(os, sites)

    return H
end

function bipartite_entropy(psi)
    b = Int(ceil(length(psi)/2))
    # from ITensor docs
    orthogonalize!(psi, b)
    U,S,V = svd(psi[b], (linkind(psi, b-1), siteind(psi,b)))
    SvN = 0.0
    for n in 1:dim(S, 1)
        p = S[n,n]^2
        SvN -= p * log2(p)
    end
    return SvN
end

function ground_state_entropy(N, rabi_f, delta)
    sites = siteinds("S=1/2", N)

    H = rydberg(N, sites, rabi_f, delta)

    psi0 = randomMPS(sites, 10)

    nsweeps = 5
    maxdim = [10,20,100,100,200]
    cutoff = [1E-10]

    energy, psi = dmrg(H, psi0; nsweeps, maxdim, cutoff)
    # combin = combiner(sites)
    # println(combin * contract(psi))
    # println(energy)
    entropy = bipartite_entropy(psi)
    return entropy
end

function main()
    N = 100
    V_nn = 2*pi*60*(10^6)
    V_nnn = 2*pi*2.3*(10^6)
    rabi_f = 2*pi*6.4*(10^6)

    freq = (10 .^ range(-4, stop=1, length=10)) .* V_nn
    deltas = (10 .^ range(-4, stop=1, length=10)) .* V_nn
    entropies = Array{Float64}(undef, length(freq), length(deltas))

    lattice_spacing = 3.6 # lattice constant

    # println(ground_state_entropy(N, rabi_f, 0))
    # os = OpSum()
    # os+=0.25*V_nn, "Sz", 1, "Sz", 2
    # os+=0.25*V_nn, "Sz", 2, "Sz", 3
    # os+=0.25*V_nnn, "Sz", 1, "Sz", 3
    # os+=0.5*rabi_f, "Sx", 1
    # os+=0.5*rabi_f, "Sx", 2
    # os+=0.5*rabi_f, "Sx", 3
    # os+=0.25*(V_nn + V_nnn), "Sz", 1
    # os+=0.25*(V_nn + V_nn), "Sz", 2
    # os+=0.25*(V_nn + V_nnn), "Sz", 3

    # H = MPO(os, siteinds("S=1/2", 3))
    # println(contract(H))
    # sites = siteinds("S=1/2", 2)
    # println(combiner(sites) * rydberg(2, sites, rabi_f, 0))

    # for i = 1:length(freq)
    #     for j = 1:length(deltas)
    #         entropies[i,j] = ground_state_entropy(N, freq[i], deltas[j])
    #     end
    # end

    # X = log10.(freq)
    # Y = log10.(deltas)

    Y = freq
    X = collect(range(0.1, stop=3, length=10))

    entropies = Array{Float64}(undef, length(Y), length(X))

    for i = 1:length(Y)
        for j = 1:length(X)
            entropies[i,j] = ground_state_entropy(N, Y[i], X[j]*Y[i])
        end
    end

    Y = sort(Y.^(-1/6))
    entropies = reverse(entropies)
    # Compute the step size for X and Y
    deltaX = (maximum(X) - minimum(X)) / length(X)
    deltaY = (maximum(Y) - minimum(Y)) / length(Y)

    # Adjust the plot limits to ensure cells are not cut off
    adjusted_xlims = (minimum(X) - 0.5 * deltaX, maximum(X) + 0.5 * deltaX)
    adjusted_ylims = (minimum(Y) - 0.5 * deltaY, maximum(Y) + 0.5 * deltaY)

    # heatmap(
    #     X, Y, entropies,
    #     xlims=adjusted_xlims,
    #     ylims=adjusted_ylims,
    #     color=:viridis,
    #     aspect_ratio=:auto,
    #     xlabel="log Ω",
    #     ylabel="log Δ",
    #     colorbar_title="Bipartite Entropy"
    # )

    heatmap(
        X, Y, entropies,
        xlims=adjusted_xlims,
        ylims=adjusted_ylims,
        color=:viridis,
        aspect_ratio=:auto,
        xlabel="Δ/Ω",
        ylabel="Ω^(-1/6) ~ R_b/a",
        colorbar_title="Bipartite Entropy"
    )
    savefig("filename.png")

    return
end

main()