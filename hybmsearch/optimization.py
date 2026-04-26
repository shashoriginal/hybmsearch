"""
Optimization Framework

Genetic Algorithm with Bayesian optimization for automatic parameter tuning.
Uses DEAP for GA operations and scikit-learn for Gaussian Process Regression.
"""

import random
import math
import json
import logging
import multiprocessing
from typing import Dict, Tuple, Any

import numpy as np
from deap import base, creator, tools
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from scipy.stats import norm
from scipy.optimize import minimize, Bounds

from .core import SearchConfig, perform_search


logger = logging.getLogger(__name__)


def initialize_deap(array_size=None):
    """
    Initialize DEAP framework for genetic algorithm.
    
    Args:
        array_size: Size of the array being optimized for (enables size-adaptive constraints)
    """
    # Ensure FitnessMin and Individual are created only once
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # Minimize time (lower is better)
    if not hasattr(creator, "Individual"):
        # Use dict for individual to easily access parameters by name
        creator.create("Individual", dict, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()

    # Define attribute generators for the individual dictionary
    toolbox.register("attr_bool", random.choice, [False, True])
    toolbox.register("attr_levels", random.choice, [1, 2])
    
    # Size-adaptive parameter constraints
    if array_size is not None:
        # New policy (per paper replication discussion):
        # - Use only logical lower bounds; derive adaptive upper bounds from array size
        # - Keep search space sane to avoid pathological slow trials on massive arrays

        # Logical lower bounds
        min_chunk_size = 512
        min_sub_chunk_size = 256
        min_pivots = 4

        # Adaptive upper bounds
        # chunk_size upper ≈ n / 32 (tighter option n / 64). Never exceed n.
        chunk_upper_cap = max(1, array_size // 32)
        max_chunk_size = min(array_size, chunk_upper_cap)
        # Ensure range valid
        if max_chunk_size < min_chunk_size:
            max_chunk_size = min_chunk_size

        # sub_chunk_size upper ≈ chunk_size / 4 (based on upper estimate)
        max_sub_chunk_size = max(min_sub_chunk_size + 1, max_chunk_size // 4)

        # Estimate target_chunks for pivot bound: n / chunk_size (use upper-bound estimate to be conservative)
        target_chunks_est = max(1, array_size // max(min_chunk_size, max_chunk_size))
        pivot_upper_est = int(round(8 * math.sqrt(target_chunks_est)))
        max_pivots = max(min_pivots + 1, min(256, pivot_upper_est))

        logger.info(f"Size-adaptive constraints for array size {array_size:,}:")
        logger.info(f"  Chunk size: {min_chunk_size:,} to {max_chunk_size:,} (cap≈n/32)")
        logger.info(f"  Sub-chunk size: {min_sub_chunk_size:,} to {max_sub_chunk_size:,} (≈chunk/4)")
        logger.info(f"  Pivot count: {min_pivots} to {max_pivots} (≈8*sqrt(target_chunks))")

        toolbox.register("attr_chunk_size", random.randint, min_chunk_size, max_chunk_size)
        toolbox.register("attr_sub_chunk_size", random.randint, min_sub_chunk_size, max_sub_chunk_size)
        toolbox.register("attr_pivot_count", random.randint, min_pivots, max_pivots)
    else:
        # Fallback to original fixed ranges if no array size provided
        toolbox.register("attr_chunk_size", random.randint, 1_000, 50_000_000)
        toolbox.register("attr_sub_chunk_size", random.randint, 1_000, 5_000_000)
        toolbox.register("attr_pivot_count", random.randint, 2, 128)
    
    max_threads = multiprocessing.cpu_count()
    toolbox.register("attr_num_threads", random.randint, 1, max_threads)

    # Define the structure of an individual
    def init_individual(icls, bool_gen, level_gen, chunk_gen, sub_chunk_gen, thread_gen, pivot_gen):
        return icls({
            "use_merge_search": bool_gen(),
            "use_interpolation": bool_gen(),
            "num_levels": level_gen(),
            "chunk_size": chunk_gen(),
            "sub_chunk_size": sub_chunk_gen(),
            "num_threads": thread_gen(),
            "use_vector_pivot": bool_gen(),
            "pivot_count": pivot_gen(),
        })

    # Register the individual initializer
    toolbox.register("individual", init_individual, creator.Individual,
                     toolbox.attr_bool, toolbox.attr_levels, toolbox.attr_chunk_size,
                     toolbox.attr_sub_chunk_size, toolbox.attr_num_threads, toolbox.attr_pivot_count)

    # Register the population initializer
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    return toolbox


def evaluate_individual(ind: Dict[str, Any], arr: np.ndarray, targets: np.ndarray, 
                       cache: Dict[Tuple, Tuple], generation: int = 0) -> Tuple[float]:
    """
    Evaluate an individual's performance (search time). Uses a cache.
    
    Args:
        ind: Individual parameter dictionary
        arr: Array to search in
        targets: Target values to search for
        cache: Cache dictionary for storing results
        generation: Generation number
        
    Returns:
        Tuple containing the fitness value (time)
    """
    # Create a cache key from the individual's parameters.
    # Sorting ensures consistent key regardless of dictionary order.
    # We exclude 'generation' from the key to reuse results across generations
    # if the parameters are identical.
    key_items = tuple(sorted(ind.items()))
    key = key_items # Simpler key for now

    if key in cache:
        # Return cached fitness value (time)
        return cache[key]
    else:
        try:
            # Create config from individual parameters
            config = SearchConfig(**ind)
            # Execute the search with this configuration
            _, elapsed = perform_search(arr, targets, generation=generation, config=config)
            
            # Store the result (time) in the cache
            # The fitness is a tuple (as required by DEAP)
            fitness = (elapsed,)
            cache[key] = fitness
            logger.info(f"Evaluated (Gen={generation}): {ind}, Time={elapsed:.6f}s")
            return fitness
        except Exception as e:
            # Handle potential errors during evaluation (e.g., invalid parameters, memory issues)
            logger.error(f"Exception during evaluation (Gen={generation}) for {ind}: {e}", exc_info=True)
            # Assign a very high fitness (bad score) for failed evaluations
            fitness = (1e6,) # Use a large float for penalty
            cache[key] = fitness # Cache the failure too
            return fitness


def expected_improvement(X, X_sample, Y_sample, gpr, xi=0.01):
    """
    Computes the Expected Improvement per point X based on existing samples X_sample, Y_sample
    and a Gaussian Process Regressor gpr.
    
    Args:
        X: Points to evaluate EI at
        X_sample: Sample points used to train GP
        Y_sample: Sample values used to train GP
        gpr: Trained Gaussian Process Regressor
        xi: Exploration parameter
        
    Returns:
        Expected improvement values
    """
    mu, sigma = gpr.predict(X, return_std=True)
    sigma = sigma.reshape(-1, 1) # sigma is (N, 1)

    # Ensure mu is also consistently shaped if needed (though predict usually returns 1D)
    mu = mu.reshape(-1, 1) # mu is (N, 1)

    # Get the best observed actual value
    mu_sample_opt = np.min(Y_sample) # Y_sample should be (M, 1), min is scalar

    with np.errstate(divide='warn', invalid='ignore'):
        imp = mu_sample_opt - mu - xi # mu is (N, 1), imp becomes (N, 1)

        # Calculate Z = imp / sigma. Both are (N, 1), Z is (N, 1)
        # Avoid division by zero where sigma is close to zero
        sigma_safe = np.maximum(sigma, 1e-9) # Use a small positive value instead of 0
        Z = imp / sigma_safe

        # Calculate EI, both terms result in (N, 1)
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)

        # Handle cases where sigma was essentially zero - EI should be max(0, imp)
        # Use the original sigma for the mask comparison
        zero_sigma_mask = (sigma <= 1e-9) # Mask for where sigma was zero or very small
        ei[zero_sigma_mask] = np.maximum(0, imp[zero_sigma_mask]) # Apply correction

    return ei


def mutate_bayesian(individual, arr, targets, cache, generation=0):
    """
    Mutates an individual using Bayesian Optimization principles
    to suggest promising new parameter values based on past evaluations.
    Focuses on numerical parameters: chunk_size, sub_chunk_size, num_threads, pivot_count.
    Boolean parameters are toggled randomly afterwards.
    
    Args:
        individual: Individual to mutate
        arr: Array being searched
        targets: Target values
        cache: Evaluation cache
        generation: Current generation
        
    Returns:
        Tuple containing the mutated individual
    """
    # Extract evaluated samples (X) and their results (Y) from the cache
    # Focus on the numerical parameters relevant for optimization
    param_keys = ["chunk_size", "sub_chunk_size", "num_threads", "pivot_count"]
    X_sample = []
    Y_sample = []

    for k_tuple, fitness in cache.items():
        # Reconstruct dict from sorted tuple key
        dd = dict(k_tuple)
        y = fitness[0]
        # Include only finite and reasonably small results (avoid penalties)
        if np.isfinite(y) and y < 1e5: # Filter out failed/penalized runs
            try:
                # Ensure all keys exist before appending
                sample_point = [dd[key] for key in param_keys]
                X_sample.append(sample_point)
                Y_sample.append(y)
            except KeyError:
                # Skip if a cached entry is somehow missing expected keys
                logger.warning(f"Skipping cached entry with missing keys: {dd}")
                continue

    # Need sufficient data points to build a reliable GP model
    if len(Y_sample) < 10: # Increased threshold for better GP model
        # Fallback to random mutation if not enough data
        individual["chunk_size"] = random.randint(1_000, 50_000_000)
        individual["sub_chunk_size"] = random.randint(1_000, 5_000_000)
        individual["num_threads"] = random.randint(1, multiprocessing.cpu_count())
        individual["pivot_count"] = random.randint(2, 128)
        # Also randomly toggle boolean/categorical params
        if random.random() < 0.3: individual["use_merge_search"] = not individual["use_merge_search"]
        if random.random() < 0.3: individual["use_interpolation"] = not individual["use_interpolation"]
        if random.random() < 0.3: individual["num_levels"] = 1 if individual["num_levels"] == 2 else 2
        if random.random() < 0.3: individual["use_vector_pivot"] = not individual["use_vector_pivot"]
        return (individual,) # Return as tuple as required by DEAP

    # --- Bayesian Optimization Step ---
    # Define parameter bounds for optimization
    max_threads = multiprocessing.cpu_count()
    # Define bounds using scipy.optimize.Bounds object
    bnds = Bounds(
        [1_000, 1_000, 1, 2], # Lower bounds for chunk, sub_chunk, threads, pivot
        [100_000_000, 10_000_000, max_threads, 128], # Upper bounds (increased range)
        keep_feasible=True # Ensure bounds are respected by L-BFGS-B
    )

    # Prepare data for Gaussian Process Regressor
    X_sample = np.array(X_sample, dtype=np.float64)
    Y_sample = np.array(Y_sample, dtype=np.float64).reshape(-1, 1)

    # Define and fit the GPR model
    # Matern kernel is generally robust
    kernel = Matern(nu=2.5)
    gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, # Small noise level
                                   normalize_y=True, # Important if Y varies significantly
                                   n_restarts_optimizer=5) # More restarts for better fit
    try:
        gpr.fit(X_sample, Y_sample)
    except Exception as e:
        logger.error(f"GPR fitting failed: {e}. Falling back to random mutation.")
        # Fallback to random mutation if GPR fails
        individual["chunk_size"] = random.randint(int(bnds.lb[0]), int(bnds.ub[0]))
        individual["sub_chunk_size"] = random.randint(int(bnds.lb[1]), int(bnds.ub[1]))
        individual["num_threads"] = random.randint(int(bnds.lb[2]), int(bnds.ub[2]))
        individual["pivot_count"] = random.randint(int(bnds.lb[3]), int(bnds.ub[3]))
        return (individual,)

    # Define the acquisition function to minimize (negative EI)
    def neg_expected_improvement(x):
        # Reshape x for GPR prediction
        x = np.atleast_2d(x)
        # Calculate EI and return its negative
        ei = expected_improvement(x, X_sample, Y_sample, gpr, xi=0.01) # xi balances exploration/exploitation
        return -ei.flatten()

    # Find the point that maximizes Expected Improvement using L-BFGS-B
    # Use multiple random starts to avoid local optima
    best_x_new = None
    min_neg_ei = np.inf

    n_restarts = 5 # Number of random starts for optimization
    for _ in range(n_restarts):
        # Generate a random starting point within bounds
        x0 = np.array([random.uniform(low, high) for low, high in zip(bnds.lb, bnds.ub)])

        try:
            res = minimize(neg_expected_improvement, x0, method="L-BFGS-B", bounds=bnds)

            if res.success and res.fun < min_neg_ei:
                min_neg_ei = res.fun
                best_x_new = res.x
        except Exception as e:
             logger.warning(f"Optimizer failed for start x0={x0}: {e}")
             continue # Try next random start

    # If optimization succeeded, update the numerical parameters
    if best_x_new is not None:
        x_new_int = np.round(best_x_new).astype(int)
        # Apply bounds strictly after rounding
        x_new_int = np.maximum(bnds.lb, np.minimum(x_new_int, bnds.ub)).astype(int)

        # Update the individual's numerical parameters
        individual["chunk_size"] = x_new_int[0]
        individual["sub_chunk_size"] = x_new_int[1]
        individual["num_threads"] = x_new_int[2]
        individual["pivot_count"] = x_new_int[3]

    else:
        # Fallback if optimizer failed for all starts
        logger.warning(f"Gen {generation}: Bayesian optimizer failed, using random mutation for numerical params.")
        individual["chunk_size"] = random.randint(int(bnds.lb[0]), int(bnds.ub[0]))
        individual["sub_chunk_size"] = random.randint(int(bnds.lb[1]), int(bnds.ub[1]))
        individual["num_threads"] = random.randint(int(bnds.lb[2]), int(bnds.ub[2]))
        individual["pivot_count"] = random.randint(int(bnds.lb[3]), int(bnds.ub[3]))

    # Apply small random chance to toggle boolean/categorical parameters
    # This adds diversity beyond the numerical optimization
    if random.random() < 0.2: individual["use_merge_search"] = not individual["use_merge_search"]
    if random.random() < 0.2: individual["use_interpolation"] = not individual["use_interpolation"]
    if random.random() < 0.2: individual["num_levels"] = 1 if individual["num_levels"] == 2 else 2
    if random.random() < 0.2: individual["use_vector_pivot"] = not individual["use_vector_pivot"]

    # Return the mutated individual (as a tuple)
    return (individual,)


def mate_dict(ind1, ind2, toolbox):
    """
    Performs crossover between two individuals (dictionaries).
    Randomly swaps parameter values between the two parents.
    
    Args:
        ind1, ind2: Parent individuals
        toolbox: DEAP toolbox
        
    Returns:
        Tuple of two child individuals
    """
    child1, child2 = toolbox.clone(ind1), toolbox.clone(ind2) # Use toolbox.clone
    for k in child1.keys(): # Iterate through keys of the dictionary
        if random.random() < 0.5: # Probability of swapping this parameter
            # Swap the value for this key between the two children
            child1[k], child2[k] = child2[k], child1[k]
            # Invalidate fitness after crossover
            del child1.fitness.values
            del child2.fitness.values
    return child1, child2


def optimize_search_parameters(arr: np.ndarray, targets: np.ndarray, 
                              pop_size: int = 50, ngen: int = 10,
                              cxpb: float = 0.7, mutpb: float = 0.3) -> Tuple[Dict[str, Any], Dict]:
    """
    Run Genetic Algorithm to find optimal search parameters.
    
    Args:
        arr: Array to search in
        targets: Target values to search for
        pop_size: Population size for GA
        ngen: Number of generations
        cxpb: Crossover probability
        mutpb: Mutation probability
        
    Returns:
        Tuple of (best_parameters_dict, evaluation_cache)
    """
    toolbox = initialize_deap(array_size=len(arr))
    eval_cache = {} # Cache evaluation results {parameter_key: fitness_tuple}

    # Register evaluation, mate, mutate, select operators with the toolbox
    def eval_wrapper(ind, gen):
        return evaluate_individual(ind, arr, targets, eval_cache, generation=gen)

    toolbox.register("evaluate", eval_wrapper)
    toolbox.register("mate", mate_dict, toolbox=toolbox)
    toolbox.register("mutate", mutate_bayesian, arr=arr, targets=targets, cache=eval_cache)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Initialize population
    population = toolbox.population(n=pop_size)

    # Statistics tracking (min, avg, max fitness)
    stats = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Hall of Fame to keep track of the best individual found so far
    hof = tools.HallOfFame(1)

    # Run the GA generations
    for gen in range(ngen):
        logger.info(f"\n-- Generation {gen} --")

        # Evaluate individuals with invalid fitness (e.g., new individuals)
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = [toolbox.evaluate(ind, gen=gen) for ind in invalid_ind]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update Hall of Fame
        hof.update(population)

        # Log generation statistics
        record = stats.compile(population)
        logger.info(
            f"Generation {gen}: "
            f"Min Time = {record['min']:.6f}s, "
            f"Avg Time = {record['avg']:.6f}s, "
            f"Max Time = {record['max']:.6f}s, "
            f"Std Dev = {record['std']:.6f}s"
        )
        logger.info(f"Best Individual So Far (Gen {gen}): {hof[0]} Fitness: {hof[0].fitness.values[0]:.6f}")

        # Select the next generation population
        offspring = toolbox.select(population, len(population))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover (mate)
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cxpb:
                toolbox.mate(child1, child2)
                # Fitness becomes invalid after crossover
                del child1.fitness.values
                del child2.fitness.values

        # Apply mutation
        for mutant in offspring:
            if random.random() < mutpb:
                toolbox.mutate(mutant, generation=gen)
                del mutant.fitness.values

        # The population for the next generation is the offspring
        population[:] = offspring

    # Get the best individual from the Hall of Fame
    best_individual = hof[0]
    logger.info("\n--- GA Optimization Complete ---")
    logger.info(f"Best Individual Found: {best_individual}")
    logger.info(f"Best Time (Fitness): {best_individual.fitness.values[0]:.6f}s")

    # Convert to regular dict and handle numpy types for JSON serialization
    best_params_serializable = {}
    for k, v in best_individual.items():
        if isinstance(v, (np.integer, np.int64)):
            best_params_serializable[k] = int(v)
        elif isinstance(v, (np.floating, np.float64)):
            best_params_serializable[k] = float(v)
        elif isinstance(v, np.bool_):
             best_params_serializable[k] = bool(v)
        else:
            best_params_serializable[k] = v

    try:
        with open("best_params_hybmsearch.json", "w") as f:
            json.dump(best_params_serializable, f, indent=4)
        logger.info("Best parameters saved to best_params_hybmsearch.json")
    except Exception as e:
        logger.error(f"Failed to save best parameters to JSON: {e}")

    return best_params_serializable, eval_cache
