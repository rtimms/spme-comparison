import pybamm
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# set style
matplotlib.rc_file("_matplotlibrc", use_default_template=True)

# Models ----------------------------------------------------------------------
print("Setting up models")

# set up models (we will update the parameters later so that the diffusivity is
# constant in the linear model)
models = {
    "SPM": pybamm.lithium_ion.SPM(),
    "SPMe (linear)": pybamm.lithium_ion.SPMe(),
    "SPMe (nonlinear)": pybamm.lithium_ion.SPMe(),
    "DFN": pybamm.lithium_ion.DFN(),
}

# pick parameters, keeping C-rate as an input to be changed for each solve
parameter_values = pybamm.ParameterValues(chemistry=pybamm.parameter_sets.Ecker2015)
parameter_values.update({"C-rate": "[input]"})

# for the linear SPMe we use the same parameters, but with constant diffusivity
# in the electrolyte
c_e_typ = parameter_values["Typical electrolyte concentration [mol.m-3]"]
T_inf = parameter_values["Ambient temperature [K]"]
D_e_const = parameter_values.evaluate(
    pybamm.standard_parameters_lithium_ion.D_e_dimensional(c_e_typ, T_inf)
)
linear_parameter_values = pybamm.ParameterValues(
    chemistry=pybamm.parameter_sets.Ecker2015
)
linear_parameter_values.update(
    {"C-rate": "[input]", "Electrolyte diffusivity [m2.s-1]": D_e_const}
)

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
    if name == "SPMe (linear)":
        params = linear_parameter_values
    else:
        params = parameter_values
    sims[name] = pybamm.Simulation(model, parameter_values=params, var_pts=var_pts)

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
    "SPM": [None] * len(C_rates),
    "SPMe (linear)": [None] * len(C_rates),
    "SPMe (nonlinear)": [None] * len(C_rates),
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


# Plots -----------------------------------------------------------------------
print("Generating table")

spm_errors = [None] * len(C_rates)
linear_spme_errors = [None] * len(C_rates)
nonlinear_spme_errors = [None] * len(C_rates)

# Compute RMSE at each C-rate
for i, C_rate in enumerate(C_rates):
    # The SPM of Richardson et. al. is just the OCV
    spm_voltage = solutions["SPM"][i]["Measured open circuit voltage [V]"](
        solutions["SPM"][i].t
    )
    linear_spme_voltage = solutions["SPMe (linear)"][i]["Terminal voltage [V]"](
        solutions["SPMe (linear)"][i].t
    )
    nonlinear_spme_voltage = solutions["SPMe (nonlinear)"][i]["Terminal voltage [V]"](
        solutions["SPMe (nonlinear)"][i].t
    )
    dfn_voltage = solutions["DFN"][i]["Terminal voltage [V]"](solutions["DFN"][i].t)

    spm_errors[i] = pybamm.rmse(dfn_voltage, spm_voltage) * 1e3
    linear_spme_errors[i] = pybamm.rmse(dfn_voltage, linear_spme_voltage) * 1e3
    nonlinear_spme_errors[i] = pybamm.rmse(dfn_voltage, nonlinear_spme_voltage) * 1e3

# print table -- could be prettier...
print("RMSE(mV) at 1C, 2.5C, 5C and 7.5C")
print("SPM", spm_errors)
print("SPMe (linear)", linear_spme_errors)
print("SPMe (nonlinear)", nonlinear_spme_errors)

# Plots -----------------------------------------------------------------------
print("Generating plots")

# plot -- could probably be generated more efficiently...
fig, ax = plt.subplots(2, 2, figsize=(6.4, 6))
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.85, wspace=0.3, hspace=0.5)
linestyles = ["solid", "dashed", "dashdot", "solid"]
colors = ["blue", "green", "red", "black"]
markers = [None, "s", "o", None]
markeverys = [50, 40, 50, 50]
V_major_ticks = np.arange(2.4, 4.2, 0.2)
V_minor_ticks = np.arange(2.5, 4.1, 0.2)

# 1C
for i, key in enumerate(solutions.keys()):
    t = solutions[key][0]["Time [s]"](solutions[key][0].t)
    # The SPM of Richardson et. al. is just the OCV
    if key == "SPM":
        V = solutions[key][0]["Measured open circuit voltage [V]"](solutions[key][0].t)
    else:
        V = solutions[key][0]["Terminal voltage [V]"](solutions[key][0].t)
    ax[0, 0].plot(
        t,
        V,
        label=key,
        linestyle=linestyles[i],
        color=colors[i],
        marker=markers[i],
        markevery=markeverys[i],
        fillstyle="none",
    )
ax[0, 0].set_xlabel("Time [s]")
ax[0, 0].set_ylabel("Voltage [V]")
ax[0, 0].set_xlim([0, 4000])
ax[0, 0].set_ylim([2.4, 4.2])
t_major_ticks = np.arange(0, 4000, 500)
t_minor_ticks = np.arange(250, 3750, 500)
ax[0, 0].set_xticks(t_major_ticks)
ax[0, 0].set_xticks(t_minor_ticks, minor=True)
ax[0, 0].set_yticks(V_major_ticks)
ax[0, 0].set_yticks(V_minor_ticks, minor=True)
ax[0, 0].grid(which="major")
ax[0, 0].legend(loc="lower left")
ax[0, 0].title.set_text("1C")

# 2.5C
for i, key in enumerate(solutions.keys()):
    t = solutions[key][1]["Time [s]"](solutions[key][1].t)
    # The SPM of Richardson et. al. is just the OCV
    if key == "SPM":
        V = solutions[key][1]["Measured open circuit voltage [V]"](solutions[key][1].t)
    else:
        V = solutions[key][1]["Terminal voltage [V]"](solutions[key][1].t)
    ax[0, 1].plot(
        t,
        V,
        label=key,
        linestyle=linestyles[i],
        color=colors[i],
        marker=markers[i],
        markevery=markeverys[i],
        fillstyle="none",
    )
ax[0, 1].set_xlabel("Time [s]")
ax[0, 1].set_ylabel("Voltage [V]")
ax[0, 1].set_xlim([0, 1600])
ax[0, 1].set_ylim([2.4, 4.2])
t_major_ticks = np.arange(0, 1600, 200)
t_minor_ticks = np.arange(100, 1500, 200)
ax[0, 1].set_xticks(t_major_ticks)
ax[0, 1].set_xticks(t_minor_ticks, minor=True)
ax[0, 1].set_yticks(V_major_ticks)
ax[0, 1].set_yticks(V_minor_ticks, minor=True)
ax[0, 1].grid(which="major")
ax[0, 1].legend(loc="lower left")
ax[0, 1].title.set_text("2.5C")

# 5C
for i, key in enumerate(solutions.keys()):
    t = solutions[key][2]["Time [s]"](solutions[key][2].t)
    # The SPM of Richardson et. al. is just the OCV
    if key == "SPM":
        V = solutions[key][2]["Measured open circuit voltage [V]"](solutions[key][2].t)
    else:
        V = solutions[key][2]["Terminal voltage [V]"](solutions[key][2].t)
    ax[1, 0].plot(
        t,
        V,
        label=key,
        linestyle=linestyles[i],
        color=colors[i],
        marker=markers[i],
        markevery=markeverys[i],
        fillstyle="none",
    )
ax[1, 0].set_xlabel("Time [s]")
ax[1, 0].set_ylabel("Voltage [V]")
ax[1, 0].set_xlim([0, 800])
ax[1, 0].set_ylim([2.4, 4.2])
t_major_ticks = np.arange(0, 800, 100)
t_minor_ticks = np.arange(50, 750, 100)
ax[1, 0].set_xticks(t_major_ticks)
ax[1, 0].set_xticks(t_minor_ticks, minor=True)
ax[1, 0].set_yticks(V_major_ticks)
ax[1, 0].set_yticks(V_minor_ticks, minor=True)
ax[1, 0].grid(which="major")
ax[1, 0].legend(loc="lower left")
ax[1, 0].title.set_text("5C")

# 7.5C
for i, key in enumerate(solutions.keys()):
    t = solutions[key][3]["Time [s]"](solutions[key][3].t)
    # The SPM of Richardson et. al. is just the OCV
    if key == "SPM":
        V = solutions[key][3]["Measured open circuit voltage [V]"](solutions[key][3].t)
    else:
        V = solutions[key][3]["Terminal voltage [V]"](solutions[key][3].t)
    ax[1, 1].plot(
        t,
        V,
        label=key,
        linestyle=linestyles[i],
        color=colors[i],
        marker=markers[i],
        markevery=markeverys[i],
        fillstyle="none",
    )
ax[1, 1].set_xlabel("Time [s]")
ax[1, 1].set_ylabel("Voltage [V]")
ax[1, 1].set_xlim([0, 500])
ax[1, 1].set_ylim([2.4, 4.2])
t_major_ticks = np.arange(0, 500, 100)
t_minor_ticks = np.arange(50, 450, 100)
ax[1, 1].set_xticks(t_major_ticks)
ax[1, 1].set_xticks(t_minor_ticks, minor=True)
ax[1, 1].set_yticks(V_major_ticks)
ax[1, 1].set_yticks(V_minor_ticks, minor=True)
ax[1, 1].grid(which="major")
ax[1, 1].legend(loc="lower left")
ax[1, 1].title.set_text("7.5C")

plt.savefig("ecker_c_rates.pdf", format="pdf", dpi=1000)
plt.show()
