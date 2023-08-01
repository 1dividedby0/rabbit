using ITensors

function main()
    N = 2
    s = siteinds(2,N)
    chi = 4
    psi = MPS([1, 0, 0, 1], s)
    # psi = randomMPS(s;linkdims=chi)

    # Make an array of integers of the element we
    # want to obtain
    # el = [1,2,1,1,2,1,2,2,2,1]

    # V = ITensor(1.)
    # for j=1:N
    #     V *= (psi[j]*state(s[j],el[j]))
    # end
    # v = scalar(V)

    # v is the element we wanted to obtain:
    @show psi[1]
    @show psi[2]
end

main()