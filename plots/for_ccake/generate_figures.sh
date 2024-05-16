#!/usr/bin/ev bash

#Please, make sure to have pandas and matplotlib installed in your environment
#before running this script

./plot_gubser_figure_4.py data/analytical_solutions \
    data/output_gubser_without-regulator figure-4.png
./plot_gubser_figure_4.py data/analytical_solutions \
    data/output_gubser_without-regulator figure-4.pdf

./plot_gubser_figure_5.py data/analytical_solutions \
    data/output_gubser_regulators figure-5.png
./plot_gubser_figure_5.py data/analytical_solutions \
    data/output_gubser_regulators figure-5.pdf

./freezeout_surface.py data/output_gubser_without-regulator/freeze_out.dat