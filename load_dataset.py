from resram_ng.resram_core import  resram_data

output = resram_data()  # load default data from src/resram_ng/example/. Can also specify dir with resram_data(dir="path/to/dir/")
print(output.abs)
output.plot()
output.fig_raman.savefig("example_ramanpng")
output.fig_absfl.savefig("example_absfl.png")
output.fig_profs.savefig("example_rep.png")