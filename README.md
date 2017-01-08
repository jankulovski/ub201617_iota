##Motivation

Implement a genetic algorithm to evolve agents that move through a hilly terrain. <br/>
The terrain is a 2-dimensional array where the value of each cell represents the height.<br/>
Agents can only move forward or backward and turn left or right.<br/>
As the agents execute moves, the move_counter increases depending on the difference in height between the starting and ending cell.<br/>
The agents can also perform tests and execute moves depending on the result.<br/>
Define N possible movement commands and number them 0-N.<br/>
Their goal is to cover as much ground as possible with the given number of moves.<br/>

A typical genetic algorithm can be defined with following:

- Generates a population of points at each iteration. The best individual in the population approaches an optimal solution.<br/>
- Selects the next population by evolving the current population using crossover and mutation functions.<br/>
- Every individual have its own fitness evaluation function which defines his efficiency in the given domain.<br/>
- Fitness function is used in evolution of the population.<br/>
- Is using predefined thresholds such population size, number of iterations, tournament size, mutation rate etc.<br/>

##Implementation

Genetic alghorithm @hill/evolution.py@:
- Generating first generation of agents @ with random program contained of commands/actions for each agent. <br/>
- Each agent runs on his terrain with the predefined program. <br/>

## Synopsis

Predefined thresholds that can be changed @[Initialization.py](hill/initialization.py)@ :<br/>
- Max iterations<br/>
- Money<br/>
- Population size<br/>
- Num of generation<br/>
- Tournament size<br/>
- Mutation rate<br/>
- Hill(terrain)<br/>

Agents are defined as objects with following attributes @hill/simulator.py - Agent class@:
- Commands 
	CMDS = [
		"nop()",
		"move_forward()",
		"move_backward()",
		"turn_left()",
		"turn_right()",
	    "set_flag()",
	    "clear_flag()",
		"mark_position()",
		"unmark_position()"]
	TESTS = [
    "coverage_improved()",
    "can_move_forward",
    "can_move_backward",
    "marked_ahead",
    "marked_behind",
    "marked_current",
    "flag"]
- Program - Random generated commands
- Terrain or Hill (in 2D)
- Max moves allowed
- Money or Cost
- Fitness

Every Agent can do series of commands under 2D terrain and each cell in that terrain holds its own height as value. Terrains can be found in @hill/hills/hill_*.txt@. They are randomly generated by @hill/generator.py@. <br/>
Fitness function of every command performed by the agent is computed depending on the difference in height between his current and his next state. <br/>
Fitness function is defined in @hill/simulator.py - Agent class@.

