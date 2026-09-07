"""Generate input instances and run the required TSP experiments."""

import csv
from pathlib import Path
import random
import time

from TSP import greedyTSP, readInput, simulatedAnnealing


CITY_COUNTS = [10, 20, 50, 100, 200]
BASE_SEED = 2026
INPUT_FOLDER = Path(__file__).parent / "inputs"
RESULTS_FOLDER = Path(__file__).parent / "results"
SA_SEEDS = [1, 2, 3, 4, 5]
INITIAL_TEMPERATURES = [100, 1000, 5000]
COOLING_RATES = [0.90, 0.95, 0.995]
INITIAL_METHODS = ["random", "greedy"]
TUNING_INITIALIZATION = "greedy"
SELECTED_INITIAL_TEMPERATURE = 100
SELECTED_COOLING_RATE = 0.995
SELECTED_INITIALIZATION = "random"


def generateCostMatrix(numberOfCities, seed):
    """Generate a symmetric matrix containing random costs from 1 to 1000."""
    randomGenerator = random.Random(seed)
    costMatrix = [[0] * numberOfCities for _ in range(numberOfCities)]

    for firstCity in range(numberOfCities):
        for secondCity in range(firstCity + 1, numberOfCities):
            cost = randomGenerator.randint(1, 1000)
            costMatrix[firstCity][secondCity] = cost
            costMatrix[secondCity][firstCity] = cost

    return costMatrix


def saveInputFile(fileName, costMatrix):
    """Save one cost matrix using the assignment's required input format."""
    with open(fileName, "w") as outputFile:
        outputFile.write(f"{len(costMatrix)}\n")

        for row in costMatrix:
            outputFile.write(" ".join(map(str, row)) + "\n")


def generateInputFiles():
    INPUT_FOLDER.mkdir(exist_ok=True)

    for numberOfCities in CITY_COUNTS:
        seed = BASE_SEED + numberOfCities
        costMatrix = generateCostMatrix(numberOfCities, seed)
        fileName = INPUT_FOLDER / f"test_{numberOfCities:03}.txt"
        saveInputFile(fileName, costMatrix)
        print(f"Generated {fileName.name} with {numberOfCities} cities")


def runExperiment1():
    """Compare Greedy with the best and average of five SA runs."""
    results = []

    for testNumber, numberOfCities in enumerate(CITY_COUNTS, start=1):
        fileName = INPUT_FOLDER / f"test_{numberOfCities:03}.txt"
        numberOfCities, costMatrix = readInput(fileName)

        startTime = time.perf_counter()
        _, greedyCost = greedyTSP(numberOfCities, costMatrix)
        greedyTime = time.perf_counter() - startTime

        saCosts = []
        saTimes = []

        for seed in SA_SEEDS:
            startTime = time.perf_counter()
            saResult = simulatedAnnealing(
                numberOfCities,
                costMatrix,
                initialTemperature=SELECTED_INITIAL_TEMPERATURE,
                coolingRate=SELECTED_COOLING_RATE,
                initialMethod=SELECTED_INITIALIZATION,
                randomSeed=seed,
            )
            saTimes.append(time.perf_counter() - startTime)
            saCosts.append(saResult["bestCost"])

        saBest = min(saCosts)
        saAverage = sum(saCosts) / len(saCosts)
        improvement = (greedyCost - saBest) / greedyCost * 100

        results.append(
            {
                "Instance": f"Test {testNumber}",
                "Cities": numberOfCities,
                "Runs": len(SA_SEEDS),
                "Initial Temperature": SELECTED_INITIAL_TEMPERATURE,
                "Cooling Rate": SELECTED_COOLING_RATE,
                "Initialization": SELECTED_INITIALIZATION.capitalize(),
                "Greedy Cost": greedyCost,
                "SA Best": saBest,
                "SA Average": round(saAverage, 2),
                "Improvement (%)": round(improvement, 2),
                "Greedy Time (s)": round(greedyTime, 8),
                "SA Average Time (s)": round(sum(saTimes) / len(saTimes), 6),
            }
        )

        print(f"Completed Test {testNumber}: {numberOfCities} cities")

    RESULTS_FOLDER.mkdir(exist_ok=True)
    resultFile = RESULTS_FOLDER / "experiment1.csv"

    with open(resultFile, "w", newline="") as outputFile:
        writer = csv.DictWriter(outputFile, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nEXPERIMENT 1: GREEDY VS. SIMULATED ANNEALING")
    print(
        f"SA configuration: T={SELECTED_INITIAL_TEMPERATURE} | "
        f"Alpha={SELECTED_COOLING_RATE} | "
        f"Initialization: {SELECTED_INITIALIZATION.capitalize()} | "
        f"Runs per instance: {len(SA_SEEDS)}"
    )

    for result in results:
        print(
            f"{result['Instance']} ({result['Cities']} cities): "
            f"Greedy={result['Greedy Cost']}, "
            f"SA Best={result['SA Best']}, "
            f"SA Average={result['SA Average']}, "
            f"Improvement={result['Improvement (%)']}%"
        )

    print(f"\nSaved results to {resultFile}")

    return results


def runExperiment2():
    """Compare three SA initial temperatures on all five instances."""
    results = []

    for testNumber, cityCount in enumerate(CITY_COUNTS, start=1):
        fileName = INPUT_FOLDER / f"test_{cityCount:03}.txt"
        numberOfCities, costMatrix = readInput(fileName)
        _, initialCost = greedyTSP(numberOfCities, costMatrix)

        for initialTemperature in INITIAL_TEMPERATURES:
            runs = []
            runTimes = []

            for seed in SA_SEEDS:
                startTime = time.perf_counter()
                result = simulatedAnnealing(
                    numberOfCities,
                    costMatrix,
                    initialTemperature=initialTemperature,
                    coolingRate=SELECTED_COOLING_RATE,
                    initialMethod=TUNING_INITIALIZATION,
                    randomSeed=seed,
                )
                runTimes.append(time.perf_counter() - startTime)
                runs.append(result)

            costs = [run["bestCost"] for run in runs]
            results.append(
                {
                    "Instance": f"Test {testNumber}",
                    "Cities": numberOfCities,
                    "Runs": len(SA_SEEDS),
                    "Initial Temperature": initialTemperature,
                    "Cooling Rate": SELECTED_COOLING_RATE,
                    "Initialization": TUNING_INITIALIZATION.capitalize(),
                    "Average Initial Cost": initialCost,
                    "Best Cost": min(costs),
                    "Average Cost": round(sum(costs) / len(costs), 2),
                    "Worst Cost": max(costs),
                    "Average Accepted Moves": round(
                        sum(run["acceptedMoves"] for run in runs) / len(runs), 2
                    ),
                    "Average Worse Moves Accepted": round(
                        sum(run["worseMovesAccepted"] for run in runs) / len(runs), 2
                    ),
                    "Average Iterations": round(
                        sum(run["iterations"] for run in runs) / len(runs), 2
                    ),
                    "Average Time (s)": round(sum(runTimes) / len(runTimes), 6),
                }
            )

            print(
                f"Completed {numberOfCities} cities, "
                f"initial temperature {initialTemperature}"
            )

    RESULTS_FOLDER.mkdir(exist_ok=True)
    resultFile = RESULTS_FOLDER / "experiment2_temperatures.csv"

    with open(resultFile, "w", newline="") as outputFile:
        writer = csv.DictWriter(outputFile, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nEXPERIMENT 2: INITIAL TEMPERATURE")
    print(
        f"Instances: {CITY_COUNTS} cities | "
        f"Alpha={SELECTED_COOLING_RATE} | "
        f"Initialization: {TUNING_INITIALIZATION.capitalize()} | "
        f"Runs per value: {len(SA_SEEDS)}"
    )

    for result in results:
        print(
            f"{result['Cities']} cities, T={result['Initial Temperature']}: "
            f"Best={result['Best Cost']}, "
            f"Average={result['Average Cost']}, "
            f"Worse moves accepted={result['Average Worse Moves Accepted']}, "
            f"Time={result['Average Time (s)']}s"
        )

    print(f"\nSaved results to {resultFile}")

    return results


def runExperiment3():
    """Compare three SA cooling rates on all five instances."""
    results = []

    for testNumber, cityCount in enumerate(CITY_COUNTS, start=1):
        fileName = INPUT_FOLDER / f"test_{cityCount:03}.txt"
        numberOfCities, costMatrix = readInput(fileName)
        _, initialCost = greedyTSP(numberOfCities, costMatrix)

        for coolingRate in COOLING_RATES:
            runs = []
            runTimes = []

            for seed in SA_SEEDS:
                startTime = time.perf_counter()
                result = simulatedAnnealing(
                    numberOfCities,
                    costMatrix,
                    initialTemperature=SELECTED_INITIAL_TEMPERATURE,
                    coolingRate=coolingRate,
                    initialMethod=TUNING_INITIALIZATION,
                    randomSeed=seed,
                )
                runTimes.append(time.perf_counter() - startTime)
                runs.append(result)

            costs = [run["bestCost"] for run in runs]
            results.append(
                {
                    "Instance": f"Test {testNumber}",
                    "Cities": numberOfCities,
                    "Runs": len(SA_SEEDS),
                    "Initial Temperature": SELECTED_INITIAL_TEMPERATURE,
                    "Cooling Rate": coolingRate,
                    "Initialization": TUNING_INITIALIZATION.capitalize(),
                    "Average Initial Cost": initialCost,
                    "Best Cost": min(costs),
                    "Average Cost": round(sum(costs) / len(costs), 2),
                    "Worst Cost": max(costs),
                    "Average Accepted Moves": round(
                        sum(run["acceptedMoves"] for run in runs) / len(runs), 2
                    ),
                    "Average Worse Moves Accepted": round(
                        sum(run["worseMovesAccepted"] for run in runs) / len(runs), 2
                    ),
                    "Average Iterations": round(
                        sum(run["iterations"] for run in runs) / len(runs), 2
                    ),
                    "Average Time (s)": round(sum(runTimes) / len(runTimes), 6),
                }
            )

            print(
                f"Completed {numberOfCities} cities, "
                f"cooling rate {coolingRate}"
            )

    RESULTS_FOLDER.mkdir(exist_ok=True)
    resultFile = RESULTS_FOLDER / "experiment3_cooling_rates.csv"

    with open(resultFile, "w", newline="") as outputFile:
        writer = csv.DictWriter(outputFile, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nEXPERIMENT 3: COOLING RATE")
    print(
        f"Instances: {CITY_COUNTS} cities | "
        f"T={SELECTED_INITIAL_TEMPERATURE} | "
        f"Initialization: {TUNING_INITIALIZATION.capitalize()} | "
        f"Runs per value: {len(SA_SEEDS)}"
    )

    for result in results:
        print(
            f"{result['Cities']} cities, Alpha={result['Cooling Rate']}: "
            f"Best={result['Best Cost']}, "
            f"Average={result['Average Cost']}, "
            f"Iterations={result['Average Iterations']}, "
            f"Time={result['Average Time (s)']}s"
        )

    print(f"\nSaved results to {resultFile}")

    return results


def runExperiment4():
    """Compare Random and Greedy initialization on all five instances."""
    results = []

    for testNumber, cityCount in enumerate(CITY_COUNTS, start=1):
        fileName = INPUT_FOLDER / f"test_{cityCount:03}.txt"
        numberOfCities, costMatrix = readInput(fileName)

        for initialMethod in INITIAL_METHODS:
            runs = []
            runTimes = []

            for seed in SA_SEEDS:
                startTime = time.perf_counter()
                result = simulatedAnnealing(
                    numberOfCities,
                    costMatrix,
                    initialTemperature=SELECTED_INITIAL_TEMPERATURE,
                    coolingRate=SELECTED_COOLING_RATE,
                    initialMethod=initialMethod,
                    randomSeed=seed,
                )
                runTimes.append(time.perf_counter() - startTime)
                runs.append(result)

            initialCosts = [run["initialCost"] for run in runs]
            finalCosts = [run["bestCost"] for run in runs]
            results.append(
                {
                    "Instance": f"Test {testNumber}",
                    "Cities": numberOfCities,
                    "Runs": len(SA_SEEDS),
                    "Initial Temperature": SELECTED_INITIAL_TEMPERATURE,
                    "Cooling Rate": SELECTED_COOLING_RATE,
                    "Initialization": initialMethod.capitalize(),
                    "Average Initial Cost": round(
                        sum(initialCosts) / len(initialCosts), 2
                    ),
                    "Best Cost": min(finalCosts),
                    "Average Cost": round(sum(finalCosts) / len(finalCosts), 2),
                    "Worst Cost": max(finalCosts),
                    "Average Accepted Moves": round(
                        sum(run["acceptedMoves"] for run in runs) / len(runs), 2
                    ),
                    "Average Worse Moves Accepted": round(
                        sum(run["worseMovesAccepted"] for run in runs) / len(runs), 2
                    ),
                    "Average Iterations": round(
                        sum(run["iterations"] for run in runs) / len(runs), 2
                    ),
                    "Average Time (s)": round(sum(runTimes) / len(runTimes), 6),
                }
            )

            print(
                f"Completed {numberOfCities} cities, "
                f"{initialMethod} initialization"
            )

    RESULTS_FOLDER.mkdir(exist_ok=True)
    resultFile = RESULTS_FOLDER / "experiment4_initialization.csv"

    with open(resultFile, "w", newline="") as outputFile:
        writer = csv.DictWriter(outputFile, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nEXPERIMENT 4: INITIALIZATION METHOD")
    print(
        f"Instances: {CITY_COUNTS} cities | "
        f"T={SELECTED_INITIAL_TEMPERATURE} | "
        f"Alpha={SELECTED_COOLING_RATE} | "
        f"Runs per method: {len(SA_SEEDS)}"
    )

    for result in results:
        print(
            f"{result['Cities']} cities, {result['Initialization']}: "
            f"Initial average={result['Average Initial Cost']}, "
            f"Best={result['Best Cost']}, "
            f"Final average={result['Average Cost']}, "
            f"Time={result['Average Time (s)']}s"
        )

    print(f"\nSaved results to {resultFile}")

    return results


if __name__ == "__main__":
    generateInputFiles()
    runExperiment1()
    runExperiment2()
    runExperiment3()
    runExperiment4()
