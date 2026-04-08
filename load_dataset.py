from resram_ng.resram_core import  resram_data

output = resram_data("example")
print(output.abs)
output.plot()
output.fig_raman.savefig("example/example_ramanpng")
output.fig_absfl.savefig("example/example_absfl.png")
output.fig_profs.savefig("example/example_rep.png")