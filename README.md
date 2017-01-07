# ub201617_iota

Project description:
	- Implementing a genetic algorithm to evolve agents that move through a hilly terrain. 
	Genetic algorithm is a metaheuristic inspired by the process of natural selection that belongs to the larger class of evolutionary algorithms.  They are commonly used to generate high-quality solutions to optimization and search problems by relying on bio-inspired operations such as mutation crossover and selection.
	A typical genetic algorithm requires:
		1.	a genetic representation of the solution domain,
		2.	a fitness function to evaluate the solution domain.

	In our case we have a population of agents and each agent contains its own set of commands or set of properties which can be mutated and altered.
	So first we create an initial population/generation with random number of agents where each agents contains a predefined number of program combinations.  After we have created out population of agents we continue to the next step which is calculating the fitness of each member(Agent) of our population. The fitness value is calculated by how well it fits with our desired requirements. Where in our case fitness is the coverage of the agent with a given number of moves. Selection is important because we want to improve our population overall fitness. So after the fitness is evaluated there is a selection process where those agents with higher fitness have a bigger probability to be selected in this process. With the two selected agents we are doing the crossover in order to create a new agents by combining aspects of our selected agents. But we also did some mutation to our population to randomize a bit the combination of solution(the set of commands) so it wouldn't be the same as the initial population. Mutation typically works by making very small changes at random to an individual's genome. And final we are repeating these steps until we reach a termination condition.
