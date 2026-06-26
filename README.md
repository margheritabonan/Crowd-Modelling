# Crowd Modelling Project: User Guide

In this repository we collected all the notebooks to run the different simulations in our project. Here in order we give a small user guide for each of them.

## 1D LWR model: crowd_modelling_1D.ipynb 

This notebook implements the simple one-dimensional crowd flow model based on the Lighthill-Whitham-Richards equation. The density is evolved with a finite-volume scheme and a Lax-Friedrichs flux.

It is used as a basic test case for the crowd modelling project. The notebook includes constant or time-dependent inflow, free outflow, bottlenecks with reduced capacity, shock-type initial data, density plots, diagnostics, and optional GIF animations.

To use it, first run the cells defining the solver. Then choose an initial density function and call `solve_lwr_corridor(...)`. The main parameters are `rho_left` for the inflow density, `T` for the final time, `num_time_steps` for the time discretisation, and `capacity` if a bottleneck is included.

The solver returns the spatial grid, saved density snapshots, and diagnostics such as total mass, boundary fluxes, cumulative outflow, and time-step information. The snapshots can be visualised with `plot_density_snapshots(...)` or animated with `create_density_animation(...)`.


## 2D Hughes model: Hughes_model_2D.ipynb

This notebook implements the two-dimensional Hughes crowd model. At each time step it computes the density-dependent speed, solves an eikonal equation for the potential to the exit, uses the direction `-\nabla phi`, and updates the density with a finite-volume RK3 scheme. Obstacles are represented by boolean masks and treated as walls.

Run the notebook cells in order. The first part defines the mesh, boundary handling, speed/cost functions, eikonal solver, flux computation, and density update. The marathon part simulates a long corridor with four inflow waves and different obstacle geometries. A single run can be started with

`frames, obs, h = run_marathon_sim("rectangle", frame_dt=2.0)`

where the geometry name must be one of the keys in `GEOMETRIES`. The saved frames can then be animated with `make_animation(...)` or exported as snapshots with `save_snapshots(...)`.

The last part repeats the Hughes solver on the same room geometries used for the Payne-Whitham comparison. A single geometry can be run with

`res = run_room_sim("thick_middle_block")`

The returned result contains density frames, the obstacle mask, mass diagnostics, congestion and speed-loss terms, and the final loss value. The main plotting functions are `plot_4panel(...)`, `plot_snapshot_3panel(...)`, `plot_conservation(...)`, and `plot_loss_ranking(...)`.


## 2D Payne-Whitham model: payne_model.ipynb

This notebook implements the two-dimensional Payne-Whitham crowd model on triangular meshes. The state contains density and momentum, and the motion is guided by a potential field pointing pedestrians toward the exit. The model is used to compare different room and obstacle geometries.

Run the notebook cells in order. The first part defines the mesh construction, boundary detection, potential solver, fluxes, time stepping, and plotting functions. Then choose a geometry, build the mesh, set the initial condition, and call `run_simulation(...)`.

The main inputs are the mesh data, the initial state `U_initial`, the final time `t_final`, the time step `dt`, and optionally an inflow density and inflow velocity. Obstacles are included through the chosen geometry before the mesh is generated.

The simulation returns density and velocity snapshots, together with diagnostics such as total mass, exited pedestrians, congestion, speed loss, and total loss. The saved snapshots can be plotted directly or used to make animations of the crowd evolution.



## Stochastic model: Stochastic_model.ipynb

This notebook implements a stochastic agent-based crowd model. Pedestrians are simulated as individual particles with discrete directions and speed classes. Their motion is influenced by the target exit, nearby agents inside a field of view, walls, obstacles, and optional inflow rules.

Run the notebook cells in order. The first part defines the geometry objects (`Boundary`, `Exit`, `Obstacle`, `TriangleObstacle`, `MultiObstacle`), the angle and speed discretisations, the `Simulation` class, and the `Visualizer` class. A simulation is created by setting the domain, exit, start area, obstacles, field of view, speed classes, number of agents, final time, and number of frames.

A typical run has the form

`sim.run(); viz = Visualizer(sim); viz.setup(); anim = viz.animate()`

The notebook includes several predefined geometries, such as open road, bottlenecks, triangular obstacles, central divider, and multi-obstacle layouts. Inflow can be prescribed with `inflow_schedule`, while `max_exit_rate` limits how many agents can leave per second.

The main outputs are the stored agent positions `sim.xs`, `sim.ys`, the active-agent array `sim.active`, and the generated animations. Extra tools include `plot_pov(...)` for visualising one agent’s field of view, `plot_snapshots(...)` for selected particle snapshots, and `plot_conservation(...)` for counting entered, active, and exited agents.