function GammaISIplot(dt, fr, k, n=1000)
    theta = 1/(k*(fr*dt*1e-3)) # fr = 1/(k*theta)
    isi = rand(Gamma(k, theta), n)
    gamma_pdf = pdf.(Gamma(k, theta), minimum(isi):maximum(isi))

    hist(isi, bins=20, density=true, alpha=0.5, ec="black"); 
    plot(minimum(isi):maximum(isi), gamma_pdf, color="black"); 
    xlabel("ISI (ms)"); ylabel("Density");
end