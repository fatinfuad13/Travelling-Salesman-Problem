"""Basic TSP helper functions."""

import math
from pathlib import Path
import random
import sys
import time


# Simulated Annealing parameters - change these values for experiments.
INITIAL_TEMPERATURE = 1000
COOLING_RATE = 0.995
MIN_TEMPERATURE = 0.001
ITERATIONS_PER_TEMPERATURE = 100
MAX_ITERATIONS = 100000
INITIAL_METHOD = "random"
RANDOM_SEED = 42


def readInput(fileName): #
    """Read the number of cities and cost matrix from a valid input file."""
    with open(fileName, "r") as inputFile:
        numberOfCities = int(inputFile.readline())
        costMatrix = [
            list(map(int, inputFile.readline().split()))
            for _ in range(numberOfCities)
        ]

    return numberOfCities, costMatrix


def validateTour(tour, numberOfCities): #
    """Check that a tour starts at 0 and contains every city exactly once."""
    return len(tour) == numberOfCities and tour[0] == 0 and sorted(tour) == list(range(numberOfCities))


def calculateTourCost(tour, costMatrix): #
    """Calculate the complete tour cost, including the return to the start."""
    totalCost = 0

    for index in range(len(tour) - 1):
        totalCost += costMatrix[tour[index]][tour[index + 1]]

    totalCost += costMatrix[tour[-1]][tour[0]] #* Add the last node -> initial node cost

    return totalCost


def greedyTSP(numberOfCities, costMatrix):
    """Build a tour by repeatedly visiting the nearest unvisited city."""
    tour = [0]
    unvisitedCities = list(range(1, numberOfCities))

    while unvisitedCities:
        nextCity = min(unvisitedCities, key=lambda city: costMatrix[tour[-1]][city]) #* sort based on min cost from last city on the tour
        tour.append(nextCity)
        unvisitedCities.remove(nextCity)

    return tour, calculateTourCost(tour, costMatrix)


def generateRandomTour(numberOfCities):
    """Generate a random tour while keeping City 0 at the beginning."""
    remainingCities = list(range(1, numberOfCities))
    random.shuffle(remainingCities)
    return [0] + remainingCities


def generateNeighbor(tour):
    """Generate a neighbor by reversing a randomly selected segment."""
    neighbor = tour.copy() #* avoid changing the tour

    if len(tour) == 2:
        return neighbor

    first, second = sorted(random.sample(range(1, len(tour)), 2))
    neighbor[first : second + 1] = reversed(neighbor[first : second + 1])

    return neighbor


def simulatedAnnealing(
    numberOfCities,
    costMatrix,
    initialTemperature=INITIAL_TEMPERATURE,
    coolingRate=COOLING_RATE,
    minTemperature=MIN_TEMPERATURE,
    iterationsPerTemperature=ITERATIONS_PER_TEMPERATURE,
    maxIterations=MAX_ITERATIONS,
    initialMethod=INITIAL_METHOD,
    randomSeed=RANDOM_SEED,
):
    """Improve a TSP tour using Simulated Annealing."""
    random.seed(randomSeed)

    if initialMethod == "greedy":
        currentTour, currentCost = greedyTSP(numberOfCities, costMatrix)
    else:
        currentTour = generateRandomTour(numberOfCities)
        currentCost = calculateTourCost(currentTour, costMatrix)

    initialTour = currentTour.copy()
    initialCost = currentCost
    bestTour = currentTour.copy()
    bestCost = currentCost

    temperature = initialTemperature
    totalIterations = 0
    acceptedMoves = 0
    worseMovesAccepted = 0

    while (
        temperature > minTemperature
        and totalIterations < maxIterations
        and numberOfCities > 2
    ):
        for _ in range(iterationsPerTemperature):
            if totalIterations >= maxIterations:
                break

            neighbor = generateNeighbor(currentTour)
            neighborCost = calculateTourCost(neighbor, costMatrix)
            difference = neighborCost - currentCost

            if difference <= 0 or random.random() < math.exp(-difference / temperature):
                currentTour = neighbor
                currentCost = neighborCost
                acceptedMoves += 1

                if difference > 0:
                    worseMovesAccepted += 1

                if currentCost < bestCost:
                    bestTour = currentTour.copy()
                    bestCost = currentCost

            totalIterations += 1

        temperature *= coolingRate

    return {
        "initialTour": initialTour,
        "initialCost": initialCost,
        "bestTour": bestTour,
        "bestCost": bestCost,
        "initialTemperature": initialTemperature,
        "coolingRate": coolingRate,
        "iterations": totalIterations,
        "acceptedMoves": acceptedMoves,
        "worseMovesAccepted": worseMovesAccepted,
    }


def tourToString(tour):
    """Convert an internal tour into a closed tour for display."""
    return " -> ".join(map(str, tour + [tour[0]]))


def printResults(numberOfCities, greedyTour, greedyCost, greedyTime, saResult, saTime):
    """Print the results required by the assignment."""
    print("=" * 40)
    print("TRAVELLING SALESMAN PROBLEM")
    print("=" * 40)
    print(f"Number of Cities: {numberOfCities}")

    print("\n---------- GREEDY METHOD ----------")
    print("Tour:")
    print(tourToString(greedyTour))
    print(f"Total Cost: {greedyCost}")
    print(f"Execution Time: {greedyTime:.6f} seconds")

    print("\n---------- SIMULATED ANNEALING ----------")
    print("Initial Solution:")
    print(tourToString(saResult["initialTour"]))
    print(f"Initial Cost: {saResult['initialCost']}")
    print("\nBest Tour Found:")
    print(tourToString(saResult["bestTour"]))
    print(f"Best Cost: {saResult['bestCost']}")
    print(f"Initial Temperature: {saResult['initialTemperature']}")
    print(f"Cooling Rate: {saResult['coolingRate']}")
    print(f"Total Iterations: {saResult['iterations']}")
    print(f"Accepted Moves: {saResult['acceptedMoves']}")
    print(f"Worse Moves Accepted: {saResult['worseMovesAccepted']}")
    print(f"Execution Time: {saTime:.6f} seconds")

    print("\n---------- COMPARISON ----------")
    print(f"Greedy Cost: {greedyCost}")
    print(f"Simulated Annealing Cost: {saResult['bestCost']}")

    if greedyCost < saResult["bestCost"]:
        print("Best Method: Greedy")
    elif saResult["bestCost"] < greedyCost:
        print("Best Method: Simulated Annealing")
    else:
        print("Best Method: Both produced the same cost")

    print("=" * 40)


def main():
    if len(sys.argv) > 1:
        inputFiles = sys.argv[1:]
    else:
        inputFolder = Path(__file__).parent / "inputs"
        inputFiles = sorted(inputFolder.glob("*.txt"))

    for inputFileName in inputFiles:
        numberOfCities, costMatrix = readInput(inputFileName)

        startTime = time.perf_counter()
        greedyTour, greedyCost = greedyTSP(numberOfCities, costMatrix)
        greedyTime = time.perf_counter() - startTime

        startTime = time.perf_counter()
        saResult = simulatedAnnealing(numberOfCities, costMatrix)
        saTime = time.perf_counter() - startTime

        print(f"\nINPUT FILE: {Path(inputFileName).name}")
        printResults(
            numberOfCities,
            greedyTour,
            greedyCost,
            greedyTime,
            saResult,
            saTime,
        )


if __name__ == "__main__":
    main()
