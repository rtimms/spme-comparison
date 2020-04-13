import pybamm
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# change working directory to the root of pybamm
os.chdir(pybamm.root_dir())

# set style
matplotlib.rc_file("examples/scripts/_matplotlibrc", use_default_template=True)

# Models ----------------------------------------------------------------------
print("Setting up models")

# set up models
models = {
    "SP": pybamm.lithium_ion.BasicSPM(),
    "SPMe (Linear)": pybamm.lithium_ion.BasicSPMe(linear_diffusion=True),
    "SPMe (Nonlinear)": pybamm.lithium_ion.BasicSPMe(
        linear_diffusion=False, use_log=False
    ),
    "cSP": pybamm.lithium_ion.BasicCSP(),
    "DFN": pybamm.lithium_ion.BasicDFN(),
}

# pick parameters, keeping C-rate as an input to be changed for each solve
parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Ecker2015)
parameter_values.update({"C-rate": "[input]"})

# set up number of points for discretisation
var = pybamm.standard_spatial_vars
var_pts = {
    var.x_n: int(parameter_values.evaluate(pybamm.geometric_parameters.L_n / 1e-6)),
    var.x_s: int(parameter_values.evaluate(pybamm.geometric_parameters.L_s / 1e-6)),
    var.x_p: int(parameter_values.evaluate(pybamm.geometric_parameters.L_p / 1e-6)),
    var.r_n: int(parameter_values.evaluate(pybamm.geometric_parameters.R_n / 1e-7)),
    var.r_p: int(parameter_values.evaluate(pybamm.geometric_parameters.R_p / 1e-7)),
}

# set up simulations
sims = {}
for name, model in models.items():
    sims[name] = pybamm.Simulation(
        model, parameter_values=parameter_values, var_pts=var_pts
    )

# pick C_rates and times to integrate over
C_rates = [1, 2.5, 5, 7.5]
t_evals = [
    np.linspace(0, 3800, 1000),
    np.linspace(0, 1510, 1000),
    np.linspace(0, 720, 1000),
    np.linspace(0, 440, 1000),
]

# loop over C-rates
solutions = {
    "SP": [None] * len(C_rates),
    "SPMe (Linear)": [None] * len(C_rates),
    "SPMe (Nonlinear)": [None] * len(C_rates),
    "cSP": [None] * len(C_rates),
    "DFN": [None] * len(C_rates),
}

print("Start solving")

for i, C_rate in enumerate(C_rates):
    print("C-rate = {}".format(C_rate))

    # solve models
    t_eval = t_evals[i]

    for name, sim in sims.items():
        print("Solving {}...".format(name))
        sim.solve(
            t_eval=t_eval,
            solver=pybamm.CasadiSolver(mode="fast"),
            inputs={"C-rate": C_rate},
        )
        solutions[name][i] = sim.solution

print("Finished")


# Tables -----------------------------------------------------------------------
print("Generating tables")

spm_errors = [None] * len(C_rates)
spme_LD_errors = [None] * len(C_rates)
spme_ND_errors = [None] * len(C_rates)
csp_errors = [None] * len(C_rates)

spm_solve_times = [None] * len(C_rates)
spme_LD_solve_times = [None] * len(C_rates)
spme_ND_solve_times = [None] * len(C_rates)
csp_solve_times = [None] * len(C_rates)

# Compute RMSE at each C-rate
for i, C_rate in enumerate(C_rates):
    # The SPM of Richardson et. al. is just the OCV
    spm_voltage = solutions["SP"][i]["Measured open circuit voltage [V]"](
        solutions["SP"][i].t
    )
    spme_LD_voltage = solutions["SPMe (Linear)"][i]["Terminal voltage [V]"](
        solutions["SPMe (Linear)"][i].t
    )
    spme_ND_voltage = solutions["SPMe (Nonlinear)"][i]["Terminal voltage [V]"](
        solutions["SPMe (Nonlinear)"][i].t
    )
    csp_voltage = solutions["cSP"][i]["Terminal voltage [V]"](solutions["cSP"][i].t)
    dfn_voltage = solutions["DFN"][i]["Terminal voltage [V]"](solutions["DFN"][i].t)

    spm_errors[i] = pybamm.rmse(dfn_voltage, spm_voltage) * 1e3
    spme_LD_errors[i] = pybamm.rmse(dfn_voltage, spme_LD_voltage) * 1e3
    spme_ND_errors[i] = pybamm.rmse(dfn_voltage, spme_ND_voltage) * 1e3
    csp_errors[i] = pybamm.rmse(dfn_voltage, csp_voltage) * 1e3

    spm_solve_times[i] = round(solutions["SP"][i].solve_time * 1000)
    spme_LD_solve_times[i] = round(solutions["SPMe (Linear)"][i].solve_time * 1000)
    spme_ND_solve_times[i] = round(solutions["SPMe (Nonlinear)"][i].solve_time * 1000)
    csp_solve_times[i] = round(solutions["cSP"][i].solve_time * 1000)


# print error table -- could be prettier...
print("RMSE(mV) at 1C, 2.5C, 5C and 7.5C")
print("SP", spm_errors)
print("SPMe (Linear)", spme_LD_errors)
print("SPMe (Nonlinear)", spme_ND_errors)
print("cSP", csp_errors)

# print solve times table -- could be prettier...
print("Solve times (ms) at 1C, 2.5C, 5C and 7.5C")
print("SP", spm_solve_times)
print("SPMe (Linear)", spme_LD_solve_times)
print("SPMe (Nonlinear)", spme_ND_solve_times)
print("cSP", csp_solve_times)


# Plots -----------------------------------------------------------------------
print("Generating plots")

# plot -- could be generated more efficiently...
fig, ax = plt.subplots(2, 2, figsize=(10, 6))
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.76, top=0.93, wspace=0.33, hspace=0.5)
linestyles = {
    "SP": "-",
    "DFN": "-",
    "cSP": "--",
    "SPMe (Linear)": "--",
    "SPMe (Nonlinear)": ":",
}

colors = {
    "SP": "blue",
    "DFN": "black",
    "cSP": "red",
    "SPMe (Linear)": "green",
    "SPMe (Nonlinear)": "purple",
}

markers = {
    "SP": None,
    "cSP": "s",
    "SPMe (Linear)": "o",
    "SPMe (Nonlinear)": "x",
    "DFN": None,
}

markeverys = {
    "SP": 50,
    "cSP": 40,
    "SPMe (Linear)": 50,
    "SPMe (Nonlinear)": 70,
    "DFN": 50,
}

V_major_ticks = np.arange(2.4, 4.21, 0.2)
V_minor_ticks = np.arange(2.5, 4.11, 0.2)

error_major_ticks = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e-2, 1e-1, 1]
error_minor_ticks = []

# plot discharge curves
models_to_plot = ["DFN", "SP", "cSP", "SPMe (Linear)", "SPMe (Nonlinear)"]

# 1C
for model in models_to_plot:
    t = solutions[model][0]["Time [s]"](solutions[model][0].t)
    # The SPM of Richardson et. al. is just the OCV
    if model == "SP":
        V = solutions[model][0]["Measured open circuit voltage [V]"](
            solutions[model][0].t
        )
    else:
        V = solutions[model][0]["Terminal voltage [V]"](solutions[model][0].t)
    ax[0, 0].plot(
        t,
        V,
        label=model,
        linestyle=linestyles[model],
        color=colors[model],
        marker=markers[model],
        markevery=markeverys[model],
        fillstyle="none",
    )
ax[0, 0].set_xlabel("Time [s]")
ax[0, 0].set_ylabel("Voltage [V]")
ax[0, 0].set_xlim([0, 4000])
ax[0, 0].set_ylim([2.4, 4.2])
t_major_ticks = np.arange(0, 4001, 500)
t_minor_ticks = np.arange(250, 3751, 500)
ax[0, 0].set_xticks(t_major_ticks)
ax[0, 0].set_xticks(t_minor_ticks, minor=True)
ax[0, 0].set_yticks(V_major_ticks)
ax[0, 0].set_yticks(V_minor_ticks, minor=True)
ax[0, 0].grid(which="major")
ax[0, 0].title.set_text("1C")

# 2.5C
for model in models_to_plot:
    t = solutions[model][1]["Time [s]"](solutions[model][1].t)
    # The SPM of Richardson et. al. is just the OCV
    if model == "SP":
        V = solutions[model][1]["Measured open circuit voltage [V]"](
            solutions[model][1].t
        )
    else:
        V = solutions[model][1]["Terminal voltage [V]"](solutions[model][1].t)
    ax[0, 1].plot(
        t,
        V,
        label=model,
        linestyle=linestyles[model],
        color=colors[model],
        marker=markers[model],
        markevery=markeverys[model],
        fillstyle="none",
    )

ax[0, 1].set_xlabel("Time [s]")
ax[0, 1].set_ylabel("Voltage [V]")
ax[0, 1].set_xlim([0, 1600])
ax[0, 1].set_ylim([2.4, 4.2])
t_major_ticks = np.arange(0, 1601, 200)
t_minor_ticks = np.arange(100, 1501, 200)
ax[0, 1].set_xticks(t_major_ticks)
ax[0, 1].set_xticks(t_minor_ticks, minor=True)
ax[0, 1].set_yticks(V_major_ticks)
ax[0, 1].set_yticks(V_minor_ticks, minor=True)
ax[0, 1].grid(which="major")
ax[0, 1].title.set_text("2.5C")

ax[0, 1].legend(loc="upper left", bbox_to_anchor=(1, 1))

# 5C
for model in models_to_plot:
    t = solutions[model][2]["Time [s]"](solutions[model][2].t)
    # The SPM of Richardson et. al. is just the OCV
    if model == "SP":
        V = solutions[model][2]["Measured open circuit voltage [V]"](
            solutions[model][2].t
        )
    else:
        V = solutions[model][2]["Terminal voltage [V]"](solutions[model][2].t)
    ax[1, 0].plot(
        t,
        V,
        label=model,
        linestyle=linestyles[model],
        color=colors[model],
        marker=markers[model],
        markevery=markeverys[model],
        fillstyle="none",
    )
ax[1, 0].set_xlabel("Time [s]")
ax[1, 0].set_ylabel("Voltage [V]")
ax[1, 0].set_xlim([0, 800])
ax[1, 0].set_ylim([2.4, 4.2])
t_major_ticks = np.arange(0, 801, 100)
t_minor_ticks = np.arange(50, 751, 100)
ax[1, 0].set_xticks(t_major_ticks)
ax[1, 0].set_xticks(t_minor_ticks, minor=True)
ax[1, 0].set_yticks(V_major_ticks)
ax[1, 0].set_yticks(V_minor_ticks, minor=True)
ax[1, 0].grid(which="major")
ax[1, 0].title.set_text("5C")

# 5C
for model in models_to_plot:
    t = solutions[model][3]["Time [s]"](solutions[model][3].t)
    # The SP of Richardson et. al. is just the OCV
    if model == "SP":
        V = solutions[model][3]["Measured open circuit voltage [V]"](
            solutions[model][3].t
        )
    else:
        V = solutions[model][3]["Terminal voltage [V]"](solutions[model][3].t)
    ax[1, 1].plot(
        t,
        V,
        label=model,
        linestyle=linestyles[model],
        color=colors[model],
        marker=markers[model],
        markevery=markeverys[model],
        fillstyle="none",
    )
ax[1, 1].set_xlabel("Time [s]")
ax[1, 1].set_ylabel("Voltage [V]")
ax[1, 1].set_xlim([0, 500])
ax[1, 1].set_ylim([2.4, 4.2])
t_major_ticks = np.arange(0, 501, 100)
t_minor_ticks = np.arange(50, 451, 100)
ax[1, 1].set_xticks(t_major_ticks)
ax[1, 1].set_xticks(t_minor_ticks, minor=True)
ax[1, 1].set_yticks(V_major_ticks)
ax[1, 1].set_yticks(V_minor_ticks, minor=True)
ax[1, 1].grid(which="major")
ax[1, 1].title.set_text("7.5C")

plt.savefig("discharge_curves.pdf", format="pdf", dpi=1000)


fig_err, ax_err = plt.subplots(2, 2, figsize=(10, 6))
fig_err.subplots_adjust(
    left=0.1, bottom=0.1, right=0.76, top=0.93, wspace=0.33, hspace=0.5
)

# plot RMS volatge errors
models_to_plot = ["SP", "cSP", "SPMe (Linear)", "SPMe (Nonlinear)"]

# 1C
true_voltage = solutions["DFN"][0]["Terminal voltage [V]"](solutions["DFN"][0].t)
for model in models_to_plot:
    t = solutions[model][0]["Time [s]"](solutions[model][0].t)
    # The SPM of Richardson et. al. is just the OCV
    if model == "SP":
        V = solutions[model][0]["Measured open circuit voltage [V]"](
            solutions["DFN"][0].t
        )
    else:
        V = solutions[model][0]["Terminal voltage [V]"](solutions["DFN"][0].t)

    error = np.abs(V - true_voltage)

    ax_err[0, 0].semilogy(
        t,
        error,
        label=model + " vs. DFN",
        linestyle=linestyles[model],
        color=colors[model],
        marker=markers[model],
        markevery=markeverys[model],
        fillstyle="none",
    )

ax_err[0, 0].set_xlabel("Time [s]")
ax_err[0, 0].set_ylabel("Absolute error [V]")
ax_err[0, 0].set_xlim([0, 4000])
ax_err[0, 0].set_ylim([1e-5, 1])
t_major_ticks = np.arange(0, 4001, 500)
t_minor_ticks = np.arange(250, 3751, 500)
ax_err[0, 0].set_xticks(t_major_ticks)
ax_err[0, 0].set_xticks(t_minor_ticks, minor=True)
ax_err[0, 0].set_yticks(error_major_ticks)
ax_err[0, 0].set_yticks(error_minor_ticks, minor=True)
ax_err[0, 0].grid(which="major")
ax_err[0, 0].title.set_text("1C")

# 2.5C
true_voltage = solutions["DFN"][1]["Terminal voltage [V]"](solutions["DFN"][1].t)
for model in models_to_plot:
    t = solutions[model][1]["Time [s]"](solutions[model][1].t)
    # The SPM of Richardson et. al. is just the OCV
    if model == "SP":
        V = solutions[model][1]["Measured open circuit voltage [V]"](
            solutions["DFN"][1].t
        )
    else:
        V = solutions[model][1]["Terminal voltage [V]"](solutions["DFN"][1].t)

    error = np.abs(V - true_voltage)

    ax_err[0, 1].semilogy(
        t,
        error,
        label=model + " vs. DFN",
        linestyle=linestyles[model],
        color=colors[model],
        marker=markers[model],
        markevery=markeverys[model],
        fillstyle="none",
    )

ax_err[0, 1].set_xlabel("Time [s]")
ax_err[0, 1].set_ylabel("Absolute error [V]")
ax_err[0, 1].set_xlim([0, 1600])
ax_err[0, 1].set_ylim([1e-5, 1])
t_major_ticks = np.arange(0, 1601, 200)
t_minor_ticks = np.arange(100, 1501, 200)
ax_err[0, 1].set_xticks(t_major_ticks)
ax_err[0, 1].set_xticks(t_minor_ticks, minor=True)
ax_err[0, 1].set_yticks(error_major_ticks)
ax_err[0, 1].set_yticks(error_minor_ticks, minor=True)
ax_err[0, 1].grid(which="major")
ax_err[0, 1].title.set_text("2.5C")

ax_err[0, 1].legend(loc="upper left", bbox_to_anchor=(1, 1))

# 5C
true_voltage = solutions["DFN"][2]["Terminal voltage [V]"](solutions["DFN"][2].t)
for model in models_to_plot:
    t = solutions[model][2]["Time [s]"](solutions[model][2].t)
    # The SPM of Richardson et. al. is just the OCV
    if model == "SP":
        V = solutions[model][2]["Measured open circuit voltage [V]"](
            solutions["DFN"][2].t
        )
    else:
        V = solutions[model][2]["Terminal voltage [V]"](solutions["DFN"][2].t)

    error = np.abs(V - true_voltage)

    ax_err[1, 0].semilogy(
        t,
        error,
        label=model + " vs. DFN",
        linestyle=linestyles[model],
        color=colors[model],
        marker=markers[model],
        markevery=markeverys[model],
        fillstyle="none",
    )

ax_err[1, 0].set_xlabel("Time [s]")
ax_err[1, 0].set_ylabel("Absolute error [V]")
ax_err[1, 0].set_xlim([0, 800])
ax_err[1, 0].set_ylim([1e-5, 1])

t_major_ticks = np.arange(0, 801, 100)
t_minor_ticks = np.arange(50, 751, 100)
ax[1, 0].set_xticks(t_major_ticks)
ax[1, 0].set_xticks(t_minor_ticks, minor=True)
ax_err[1, 0].set_yticks(error_major_ticks)
ax_err[1, 0].set_yticks(error_minor_ticks, minor=True)
ax_err[1, 0].grid(which="major")
ax_err[1, 0].title.set_text("5C")

# 7.5C
true_voltage = solutions["DFN"][3]["Terminal voltage [V]"](solutions["DFN"][3].t)
for model in models_to_plot:
    t = solutions[model][3]["Time [s]"](solutions[model][3].t)
    # The SPM of Richardson et. al. is just the OCV
    if model == "SP":
        V = solutions[model][3]["Measured open circuit voltage [V]"](
            solutions["DFN"][3].t
        )
    elif model != "SPMe (Linear)":
        V = solutions[model][3]["Terminal voltage [V]"](solutions["DFN"][3].t)

    if model == "SPMe (Linear)":
        # in this case need to just use the spme (ld) solution time

        V = solutions[model][3]["Terminal voltage [V]"](solutions["SPMe (Linear)"][3].t)
        tV = solutions["DFN"][3]["Terminal voltage [V]"](
            solutions["SPMe (Linear)"][3].t
        )
        error = np.abs(V - tV)

    else:
        error = np.abs(V - true_voltage)

    ax_err[1, 1].semilogy(
        t,
        error,
        label=model + " vs. DFN",
        linestyle=linestyles[model],
        color=colors[model],
        marker=markers[model],
        markevery=markeverys[model],
        fillstyle="none",
    )

ax_err[1, 1].set_xlabel("Time [s]")
ax_err[1, 1].set_ylabel("Absolute error [V]")
ax_err[1, 1].set_xlim([0, 500])
ax_err[1, 1].set_ylim([1e-5, 1])
t_major_ticks = np.arange(0, 501, 100)
t_minor_ticks = np.arange(50, 451, 100)
ax[1, 1].set_xticks(t_major_ticks)
ax[1, 1].set_xticks(t_minor_ticks, minor=True)
ax_err[1, 1].set_yticks(error_major_ticks)
ax_err[1, 1].set_yticks(error_minor_ticks, minor=True)
ax_err[1, 1].grid(which="major")
ax_err[1, 1].title.set_text("7.5C")

plt.savefig("RMS_voltage_error.pdf", format="pdf", dpi=1000)
plt.show()
